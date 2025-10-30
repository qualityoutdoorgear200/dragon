
import torch

def apply_rope(q, k, rope_theta=10000.0):
    """
    Minimal rotary position embedding for Q,K of shape [B,T,H,d].
    """
    B,T,H,d = q.shape
    half = d // 2
    device=q.device; dtype=q.dtype
    idx = torch.arange(half, device=device, dtype=dtype)
    freqs = (1.0 / (rope_theta ** (idx / max(1,half)))).unsqueeze(0)  # [1,half]
    t = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)      # [T,1]
    angles = t * freqs                                                # [T,half]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)                 # [1,T,1,half]
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)
    def rot(x):
        x1, x2 = x[..., :half], x[..., half:half*2]
        xr = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        if d%2==1:
            xr = torch.cat([xr, x[..., -1:].clone()], dim=-1)
        return xr
    return rot(q), rot(k)
