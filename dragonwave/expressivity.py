
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positives import pos, pos_unit

class PhaseGatedAdapter(nn.Module):
    def __init__(self, n_feat: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads; self.d_head = d_head
        self.gater = nn.Linear(n_feat, n_heads*d_head, bias=False)
    def forward(self, res_feat, k_pre, q_pre):
        B,T,H,d = k_pre.shape
        gates = F.softplus(self.gater(res_feat)).view(B,T,H,d)
        eps = 1e-6
        return k_pre * (gates + eps), q_pre * (gates + eps)

class SharedWeightRefiner(nn.Module):
    def __init__(self, block: nn.Module, refine_steps: int = 2, residual_scale: float = 0.5):
        super().__init__()
        self.block = block
        self.refine_steps = refine_steps
        self.residual_scale = residual_scale
    def forward(self, x):
        y = x
        for _ in range(max(0, self.refine_steps)):
            y = y + self.residual_scale * self.block(y)
        return y
