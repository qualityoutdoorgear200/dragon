import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .config import load_config
from .dataset import build_dataset, collate
from .dataset_token import build_token_dataset
from .losses import (
    coverage_loss,
    disagreement_penalty,
    entropy_penalty,
    overconfidence_penalty,
    rate_penalty,
    soft_monotonic_dp_loss,
)
from .model import StreamLM
from .modulator import Modulator
from .results_logger import dump_config, log_metrics
from .tokenize import load_tokenizer
from .utils import batch_entropy, seed_everything

BYTES_PER_GB = 1024 ** 3


def _clone_gate_cfg(gate_cfg):
    cloned = {}
    for key, value in gate_cfg.items():
        cloned[key] = dict(value) if isinstance(value, dict) else value
    return cloned


def _cuda_usage_gb(device_obj):
    allocated = torch.cuda.max_memory_allocated(device_obj) / BYTES_PER_GB
    reserved = torch.cuda.max_memory_reserved(device_obj) / BYTES_PER_GB
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)
    resident = (total_bytes - free_bytes) / BYTES_PER_GB
    return max(allocated, reserved, resident)


def _forward_components(model, x, y, cfg, gate_cfg, ei_cfg, reentry_mode, entropy_floor):
    logits, aux_logits, _ = model(x, gate_cfg=gate_cfg, ei_cfg=ei_cfg, reentry_mode=reentry_mode)
    with autocast("cuda", enabled=False):
        logits_fp32 = logits.float()
        aux_fp32 = [aux.float() for aux in aux_logits] if aux_logits else []
        if cfg.timing.loss == "softDP":
            dp_loss = soft_monotonic_dp_loss(logits_fp32, y, tau=1.0)
        else:
            dp_loss = soft_monotonic_dp_loss(logits_fp32, y, tau=1.0)
        cov = coverage_loss(logits_fp32, y, temperature=1.0) * cfg.timing.lambda_coverage
        rate = rate_penalty(logits_fp32, weight=cfg.timing.lambda_rate)
        ent = entropy_penalty(logits_fp32, target_entropy_floor=entropy_floor, weight=cfg.timing.lambda_entropy)
        over = overconfidence_penalty(logits_fp32, y, entropy_floor=entropy_floor, weight=cfg.dk.overconfidence)
        dis = disagreement_penalty(logits_fp32, aux_fp32, weight=cfg.dk.disagreement, conf_threshold=0.7)
        loss = dp_loss + cov + rate + ent + over + dis
    return loss, dp_loss, cov, rate, ent, over, dis, logits_fp32


def _prepare_probe_batch(dataset, effective_batch, collate_fn):
    try:
        dataset_len = len(dataset)
    except TypeError:
        dataset_len = None
    batch_size = effective_batch if dataset_len is None else min(effective_batch, dataset_len)
    loader_kwargs = dict(batch_size=batch_size, shuffle=False)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader = DataLoader(dataset, **loader_kwargs)
    return next(iter(loader))


def _measure_micro_usage(model, sample_x, sample_y, micro, cfg, gate_cfg, ei_cfg, reentry_mode, entropy_floor, device, amp_enabled):
    if device != "cuda":
        return 0.0
    device_obj = torch.device(device)
    was_training = model.training
    model.train(True)
    try:
        x = sample_x[:micro].to(device_obj)
        y = sample_y[:micro].to(device_obj)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)
        model.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            loss, *_ = _forward_components(model, x, y, cfg, gate_cfg, ei_cfg, reentry_mode, entropy_floor)
        loss.backward()
        torch.cuda.synchronize(device_obj)
        peak = _cuda_usage_gb(device_obj)
    except RuntimeError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        raise
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        model.train(was_training)
    return peak


def _determine_micro_batch(cfg, model, sample_batch, gate_cfg, ei_cfg, reentry_mode, entropy_floor, device, amp_enabled):
    effective = cfg.train.batch_size
    sample_x, sample_y = sample_batch
    max_micro = max(1, min(sample_x.size(0), effective))
    if device != "cuda" or not torch.cuda.is_available() or not cfg.train.max_vram_gb:
        return max_micro, 0.0

    max_vram = float(cfg.train.max_vram_gb) * 0.85

    def measure(micro):
        return _measure_micro_usage(
            model,
            sample_x,
            sample_y,
            micro,
            cfg,
            _clone_gate_cfg(gate_cfg),
            dict(ei_cfg),
            reentry_mode,
            entropy_floor,
            device,
            amp_enabled,
        )

    micro = 1
    usage = measure(micro)
    if usage > max_vram:
        raise RuntimeError(
            f"Even micro-batch size 1 exceeds the VRAM cap ({usage:.2f} GB > {max_vram:.2f} GB)."
        )

    best_micro = micro
    best_usage = usage

    while micro < max_micro:
        candidate = min(max_micro, micro * 2)
        usage = measure(candidate)
        if usage <= max_vram:
            best_micro = candidate
            best_usage = usage
            if candidate == max_micro:
                break
            micro = candidate
            continue

        lo = micro
        hi = candidate
        while hi - lo > 1:
            mid = (hi + lo) // 2
            usage = measure(mid)
            if usage <= max_vram:
                best_micro = mid
                best_usage = usage
                lo = mid
            else:
                hi = mid
        break

    return best_micro, best_usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base_bdh.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    steps_override = os.environ.get("TRAIN_STEPS_OVERRIDE")
    if steps_override:
        cfg.train.steps = int(steps_override)
    seed_everything(cfg.train.seed)
    device = "cuda" if (cfg.train.device == "auto" and torch.cuda.is_available()) else "cpu"
    device_obj = torch.device(device)
    amp_enabled = device == "cuda" and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=True) if amp_enabled else GradScaler("cpu", enabled=False)

    save_root = cfg.train.save_dir or "runs/default"
    os.makedirs(save_root, exist_ok=True)
    fid, cfg_path = dump_config(cfg, save_root)

    tokenizer_path = getattr(cfg.train, "tokenizer_path", None)
    if tokenizer_path:
        tok = load_tokenizer(tokenizer_path)
        ds, vocab_size = build_token_dataset(
            cfg.train.text_path or "data/mix_lite_train.txt",
            tokenizer=tok,
            seq_len=cfg.train.seq_len,
            cache_path=getattr(cfg.train, "tokenized_cache_path", None),
            seed=cfg.train.seed,
            mode="train",
        )
        collate_fn = None
    else:
        ds, stoi = build_dataset(
            cfg.train.dataset,
            cfg.train.text_path,
            seq_len=cfg.train.seq_len,
            num_samples=2048,
        )
        vocab_size = len(stoi)
        collate_fn = collate
    is_streaming = getattr(ds, "is_streaming", False)
    override_micro = getattr(cfg.train, "micro_batch_size", None)
    if override_micro is not None:
        override_micro = max(1, min(int(override_micro), cfg.train.batch_size))

    extras = {
        "use_phase_gated_attn": cfg.extras.use_phase_gated_attn,
        "use_ffn_refine": cfg.extras.use_ffn_refine,
        "refine_steps": cfg.extras.refine_steps,
    }
    fastcfg = {
        "use_fastweights": cfg.fastweights.use_fastweights,
        "rank": cfg.fastweights.rank,
        "eta": cfg.fastweights.eta,
        "gamma": cfg.fastweights.gamma,
        "event": cfg.fastweights.event,
    }

    model = StreamLM(
        vocab_size=vocab_size,
        d_model=cfg.attention.d_model,
        n_layers=cfg.attention.n_layers,
        n_heads=cfg.attention.n_heads,
        d_head=cfg.attention.d_head,
        n_freq=cfg.resonator.n_freq,
        use_B=cfg.rails.use_B,
        use_C=cfg.rails.use_C,
        use_retention_maps=cfg.resonator.use_retention_maps,
        use_threshold_maps=cfg.resonator.use_threshold_maps,
        atoms_per_filter=cfg.resonator.atoms_per_filter,
        use_continuous_atoms=cfg.resonator.use_continuous_atoms,
        extras=extras,
        fastweights=fastcfg,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    gate_cfg = {
        "rho_B": cfg.gate.rho_B,
        "rho_C": cfg.gate.rho_C,
        "alpha": cfg.gate.alpha,
        "mix": dict(cfg.gate.mix),
        "additive_nudge": cfg.reentry.additive_nudge,
    }
    ei_cfg = {
        "use_EI": cfg.ei.use_EI,
        "norm_strength": cfg.ei.norm_strength,
        "phase_corr_scale": cfg.ei.phase_corr_scale,
    }
    reentry_mode = cfg.reentry.mode
    entropy_floor = 0.8 * math.log(vocab_size)

    mod = Modulator().to(device) if cfg.modulator.use_modulator else None
    ema_entropy = 3.0

    sample_batch = _prepare_probe_batch(ds, cfg.train.batch_size, collate_fn)
    sample_x, sample_y = sample_batch
    micro_batch, peak_usage = _determine_micro_batch(
        cfg,
        model,
        sample_batch,
        gate_cfg,
        ei_cfg,
        reentry_mode,
        entropy_floor,
        device,
        amp_enabled,
    ) 
    if override_micro is not None and micro_batch > override_micro:
        micro_batch = override_micro
        peak_usage = _measure_micro_usage(
            model, sample_x, sample_y, micro_batch,
            cfg, gate_cfg, ei_cfg, reentry_mode, entropy_floor, device, amp_enabled
        )
    micro_batch = max(1, min(micro_batch, cfg.train.batch_size))
    grad_accum = max(1, math.ceil(cfg.train.batch_size / micro_batch))

    loader_kwargs = dict(batch_size=micro_batch)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if is_streaming:
        loader = DataLoader(ds, **loader_kwargs)
    else:
        loader = DataLoader(ds, shuffle=True, **loader_kwargs)
    loader_iter = iter(loader)
    ckpt_interval = int(getattr(cfg.train, "checkpoint_interval", 0) or 0)

    print(
        f"[grad_accum] effective_batch={cfg.train.batch_size} micro_batch={micro_batch} grad_accum={grad_accum} peak_usage={peak_usage:.2f}GB"
    )

    for step in range(1, cfg.train.steps + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device_obj)
        if mod is not None:
            ent_norm = torch.tensor(ema_entropy / 5.0, device=device).clamp(0, 1)
            disagree = torch.tensor(0.0, device=device)
            novelty = torch.tensor(0.5, device=device)
            b_rate = torch.tensor(0.1, device=device)
            knobs = mod(ent_norm, disagree, novelty, b_rate)
            gate_cfg["rho_B"] = float(knobs["rho_B"].mean().item())
            gate_cfg["rho_C"] = float(knobs["rho_C"].mean().item())

        opt.zero_grad(set_to_none=True)

        total_loss = total_dp = total_cov = total_rate = total_ent = total_over = total_dis = 0.0
        entropy_weighted = 0.0
        samples_accum = 0

        for _ in range(grad_accum):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            bsz = x.size(0)
            samples_accum += bsz
            if device == "cuda":
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)

            with autocast("cuda", enabled=amp_enabled):
                loss, dp_loss, cov, rate, ent, over, dis, logits = _forward_components(
                    model, x, y, cfg, gate_cfg, ei_cfg, reentry_mode, entropy_floor
                )

            scaling = bsz / cfg.train.batch_size
            scaled_loss = loss * scaling
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            total_loss += loss.item() * bsz
            total_dp += dp_loss.item() * bsz
            total_cov += cov.item() * bsz
            total_rate += rate.item() * bsz
            total_ent += ent.item() * bsz
            total_over += over.item() * bsz
            total_dis += dis.item() * bsz
            entropy_weighted += batch_entropy(logits).mean().item() * bsz

        if samples_accum == 0:
            continue

        if grad_accum > 1 and samples_accum != cfg.train.batch_size:
            scale = cfg.train.batch_size / samples_accum
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

        if amp_enabled:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        if amp_enabled:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        avg_loss = total_loss / samples_accum
        avg_dp = total_dp / samples_accum
        avg_cov = total_cov / samples_accum
        avg_rate = total_rate / samples_accum
        avg_ent = total_ent / samples_accum
        avg_over = total_over / samples_accum
        avg_dis = total_dis / samples_accum
        avg_entropy = entropy_weighted / samples_accum
        ema_entropy = 0.9 * ema_entropy + 0.1 * avg_entropy

        print(
            f"step {step:5d}/{cfg.train.steps:5d} | loss {avg_loss:.3f} | dp {avg_dp:.3f} | cov {avg_cov:.3f} | "
            f"rate {avg_rate:.3f} | ent {avg_ent:.3f} | over {avg_over:.3f} | dis {avg_dis:.3f}",
            flush=True,
        )

        if step % 50 == 0 or step == cfg.train.steps:
            log_metrics(
                {
                    "step": step,
                    "loss": float(avg_loss),
                    "dp": float(avg_dp),
                    "cov": float(avg_cov),
                    "rate": float(avg_rate),
                    "ent": float(avg_ent),
                    "over": float(avg_over),
                    "dis": float(avg_dis),
                },
                save_root,
                fid,
                mode="train",
            )

        if ckpt_interval and (step % ckpt_interval == 0 or step == cfg.train.steps):
            ckpt_path = os.path.join(save_root, f"ckpt_step_{step:05d}.pt")
            payload = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "ema_entropy": ema_entropy,
            }
            if mod is not None:
                payload["modulator"] = mod.state_dict()
            torch.save(payload, ckpt_path)

        if device == "cuda" and cfg.train.max_vram_gb:
            torch.cuda.synchronize(device_obj)
            runtime_peak = _cuda_usage_gb(device_obj)
            max_vram = float(cfg.train.max_vram_gb)
            trigger = max_vram * 0.98
            if runtime_peak > 0 and runtime_peak >= trigger and micro_batch > 1:
                while micro_batch > 1:
                    old_micro = micro_batch
                    micro_batch = max(1, micro_batch // 2)
                    if override_micro is not None:
                        micro_batch = min(micro_batch, override_micro)
                    grad_accum = max(1, math.ceil(cfg.train.batch_size / micro_batch))
                    loader_kwargs = dict(batch_size=micro_batch)
                    if collate_fn is not None:
                        loader_kwargs["collate_fn"] = collate_fn
                    if is_streaming:
                        loader = DataLoader(ds, **loader_kwargs)
                    else:
                        loader = DataLoader(ds, shuffle=True, **loader_kwargs)
                    loader_iter = iter(loader)
                    peak_usage = _measure_micro_usage(
                        model,
                        sample_x,
                        sample_y,
                        micro_batch,
                        cfg,
                        gate_cfg,
                        ei_cfg,
                        reentry_mode,
                        entropy_floor,
                        device,
                        amp_enabled,
                    )
                    print(
                        f"[grad_accum_adjust] peak_usage={runtime_peak:.2f}GB -> {peak_usage:.2f}GB micro_batch={micro_batch} grad_accum={grad_accum} (was {old_micro})"
                    )
                    if peak_usage <= max_vram * 0.90 or micro_batch == 1:
                        break
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_obj)


if __name__ == "__main__":
    main()


