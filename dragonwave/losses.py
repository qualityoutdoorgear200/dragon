
import torch
import torch.nn.functional as F


@torch.jit.script
def _gather_token_nll(logp: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logp: [B, T, V], targets: [B, J]
    B, T, _ = logp.shape
    target_ids = targets.unsqueeze(1).expand(-1, T, -1)
    ce = -torch.gather(logp, dim=-1, index=target_ids)  # [B, T, J]
    return ce.transpose(1, 2).contiguous()  # [B, J, T]


@torch.jit.script
def _soft_monotonic_dp_loss_impl(C: torch.Tensor, tau: float) -> torch.Tensor:
    B, J, T = C.shape
    big = torch.tensor(1.0e6, dtype=C.dtype, device=C.device)
    D = torch.empty((B, J + 1, T + 1), dtype=C.dtype, device=C.device)
    D.fill_(big)
    D[:, 0, :] = 0.0
    for j in range(1, J + 1):
        row = D[:, j]
        prev_row = D[:, j - 1]
        row[:, 0] = big
        for t in range(1, T + 1):
            stay = row[:, t - 1]
            adv = prev_row[:, t - 1]
            soft = -tau * torch.logaddexp(-stay / tau, -adv / tau)
            row[:, t] = C[:, j - 1, t - 1] + soft
    return D[:, J, T]


def soft_monotonic_dp_loss(logits, targets, tau: float = 1.0):
    if logits.dtype != torch.float32:
        logits = logits.float()
    logp = logits.log_softmax(dim=-1)
    C = _gather_token_nll(logp, targets)
    return _soft_monotonic_dp_loss_impl(C, tau).mean()


def coverage_loss(logits, targets, temperature: float = 1.0):
    if logits.dtype != torch.float32:
        logits = logits.float()
    logp = logits.log_softmax(dim=-1)
    ce = _gather_token_nll(logp, targets)  # [B, J, T]
    softmin_t = -temperature * torch.logsumexp(-ce / temperature, dim=-1)
    margin = 0.1
    loss = torch.relu(softmin_t - margin)
    return loss.sum(dim=1).mean()


def rate_penalty(logits, weight: float = 1.0):
    deltas = logits[:, 1:, :] - logits[:, :-1, :]
    return weight * deltas.abs().mean()


def entropy_penalty(logits, target_entropy_floor: float, weight: float = 1.0):
    p = logits.softmax(dim=-1)
    H = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)
    ent_shortfall = torch.relu(target_entropy_floor - H)
    return weight * ent_shortfall.mean()


def overconfidence_penalty(logits, targets, entropy_floor: float, weight: float = 1.0):
    B, T, V = logits.shape
    B2, J = targets.shape
    assert B == B2
    p = logits.softmax(dim=-1)
    idx = targets.unsqueeze(1).expand(B, T, J)
    p_sel = torch.gather(p, dim=-1, index=idx)
    max_p = p_sel.max(dim=-1).values
    H = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)
    penal = torch.relu((entropy_floor - H)) * (1.0 - max_p)
    return weight * penal.mean()


def disagreement_penalty(main_logits, aux_logits_list, weight: float = 1.0, conf_threshold: float = 0.7):
    p_main = main_logits.softmax(dim=-1)
    H = -(p_main * (p_main.clamp_min(1e-8).log())).sum(dim=-1)
    conf = 1.0 - H / torch.log(torch.tensor(p_main.size(-1), device=H.device, dtype=H.dtype))
    if len(aux_logits_list) == 0:
        kl_avg = torch.zeros_like(conf)
    else:
        p_aux = torch.stack([aux.softmax(dim=-1) for aux in aux_logits_list], dim=0)
        kl = (p_main.unsqueeze(0) * (p_main.clamp_min(1e-8).log().unsqueeze(0) - p_aux.clamp_min(1e-8).log())).sum(dim=-1)
        kl_avg = kl.mean(dim=0)
    penal = torch.relu(conf - conf_threshold) * kl_avg
    return weight * penal.mean()
