
import torch
import torch.nn as nn
import torch.nn.functional as F

class Modulator(nn.Module):
    def __init__(self, k_out=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(),
            nn.Linear(32, k_out)
        )
    def forward(self, entropy_norm, disagree_norm, novelty_norm, b_rate):
        x = torch.stack([entropy_norm, disagree_norm, novelty_norm, b_rate], dim=-1)
        y = torch.sigmoid(self.fc(x))
        return {
            "dtheta_scale": 0.5*y[...,0] + 0.75,
            "rho_B": 0.05 + 0.45*y[...,1],
            "rho_C": 0.05 + 0.45*y[...,2],
            "lambdaB": 0.90 + 0.095*y[...,3],
            "lambdaC": 0.95 + 0.045*y[...,3],
            "norm_strength": 0.05 + 0.45*y[...,0]
        }
