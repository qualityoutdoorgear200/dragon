
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLULowRank(nn.Module):
    """
    BDH-GPU style ReLU-lowrank FFN:
      y = W_dec(ReLU(W_enc x))
    Optionally keep outputs in positive orthant via softplus on residual.
    """
    def __init__(self, d_model: int, rank: int = 256, positive_out: bool = True):
        super().__init__()
        self.enc = nn.Linear(d_model, rank, bias=False)
        self.dec = nn.Linear(rank, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.positive_out = positive_out
        nn.init.xavier_uniform_(self.enc.weight, gain=0.5)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.5)
    def forward(self, x):
        h = F.relu(self.enc(x))
        y = self.dec(h)
        if self.positive_out:
            y = F.softplus(y)
        return self.ln(x + y)
