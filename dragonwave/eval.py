
import argparse, torch
from torch.utils.data import DataLoader
from .config import load_config
from .dataset import build_dataset, collate
from .dataset_token import build_token_dataset
from .model import StreamLM
from .metrics import perplexity, ece, brier, false_confidence_rate
from .tokenize import load_tokenizer
from .utils import seed_everything
from .results_logger import dump_config, log_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--text_path", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.text_path:
        cfg.train.text_path = args.text_path

    seed_everything(cfg.train.seed)
    device = "cuda" if (cfg.train.device=="auto" and torch.cuda.is_available()) else "cpu"

    tokenizer_path = getattr(cfg.train, "tokenizer_path", None)
    if tokenizer_path:
        tok = load_tokenizer(tokenizer_path)
        ds, vocab_size = build_token_dataset(
            cfg.train.text_path or "data/mix_lite_train.txt",
            tokenizer=tok,
            seq_len=cfg.train.seq_len,
            cache_path=getattr(cfg.train, "tokenized_cache_path", None),
            mode="eval",
            num_samples=1024,
        )
        collate_fn = None
    else:
        ds, stoi = build_dataset(
            cfg.train.dataset,
            cfg.train.text_path,
            seq_len=cfg.train.seq_len,
            num_samples=1024,
        )
        vocab_size = len(stoi)
        collate_fn = collate

    loader_kwargs = dict(batch_size=cfg.train.batch_size, shuffle=False)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader = DataLoader(ds, **loader_kwargs)

    model = StreamLM(vocab_size=vocab_size,
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
                     extras={"use_phase_gated_attn": cfg.extras.use_phase_gated_attn,
                             "use_ffn_refine": cfg.extras.use_ffn_refine,
                             "refine_steps": cfg.extras.refine_steps},
                     fastweights={"use_fastweights": cfg.fastweights.use_fastweights,
                                  "rank": cfg.fastweights.rank,
                                  "eta": cfg.fastweights.eta,
                                  "gamma": cfg.fastweights.gamma,
                                  "event": cfg.fastweights.event}).to(device)

    gate_cfg = {"rho_B": cfg.gate.rho_B, "rho_C": cfg.gate.rho_C, "alpha": cfg.gate.alpha, "mix": cfg.gate.mix,
                "additive_nudge": cfg.reentry.additive_nudge}
    ei_cfg   = {"use_EI": cfg.ei.use_EI, "norm_strength": cfg.ei.norm_strength, "phase_corr_scale": cfg.ei.phase_corr_scale}
    reentry_mode = cfg.reentry.mode

    all_logits=[]; all_targets=[]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits, aux_logits, di = model(x, gate_cfg=gate_cfg, ei_cfg=ei_cfg, reentry_mode=reentry_mode)
            all_logits.append(logits.cpu()); all_targets.append(y.cpu())
    logits = torch.cat(all_logits, dim=0); targets = torch.cat(all_targets, dim=0)
    ppl = perplexity(logits, targets)
    cal = ece(logits, targets)
    br  = brier(logits, targets)
    fcr = false_confidence_rate(logits, targets, thresh=0.7)

    print(f"Perplexity: {ppl:.3f} | ECE: {cal:.4f} | Brier: {br:.4f} | FCR@0.7: {fcr:.4f}")

    # Log
    from .config import Config
    fid, _ = dump_config(cfg, cfg.train.save_dir or "runs/default")
    log_metrics({"ppl": ppl, "ece": cal, "brier": br, "fcr@0.7": fcr}, cfg.train.save_dir or "runs/default", fid, mode="eval")

if __name__ == "__main__":
    main()
