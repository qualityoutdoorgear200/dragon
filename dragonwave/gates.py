
import torch

def compute_plv(unit_complex, beta=0.1, prev=None):
    uc = unit_complex.mean(dim=0)            # [F] complex
    if prev is None:
        ema = uc
    else:
        ema = (1-beta)*prev + beta*uc
    plv = ema.abs().clamp(0,1)               # [F]
    return plv, ema

def adaptive_percentile_threshold(scores, target_mass, alpha=5.0):
    if scores.numel() == 0:
        return torch.empty_like(scores), 0.0
    B,F = scores.shape
    sflat = scores.reshape(-1).detach()
    k = max(1, int((1-target_mass)*sflat.numel()))
    tau = torch.kthvalue(sflat, k).values
    gate = torch.sigmoid(alpha*(scores - tau))
    return gate, tau.item()
