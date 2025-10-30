
import torch
import torch.nn as nn

class FastWeightsState(nn.Module):
    def __init__(self, d_model: int, rank: int = 4, gamma: float = 0.98):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank), requires_grad=False)
        self.V = nn.Parameter(torch.zeros(d_model, rank), requires_grad=False)
        self.gamma = gamma
    def decay(self):
        with torch.no_grad():
            self.U.mul_(self.gamma)
            self.V.mul_(self.gamma)
    def write(self, x, eta: float):
        with torch.no_grad():
            m = x.mean(dim=0, keepdim=True)    # [1,D]
            self.U.add_(eta * m.T @ torch.ones(1, self.U.size(1), device=m.device))
            self.V.add_(eta * m.T @ torch.ones(1, self.V.size(1), device=m.device))
    def apply(self, x, scale: float = 1.0):
        M = self.U @ self.V.T                  # [D,D]
        return x + scale * (x @ M)
