
import torch.nn.functional as F
def pos(x):
    return F.softplus(x)
def pos_unit(x, eps=1e-6):
    y = F.softplus(x) + eps
    return y / (y.sum(dim=-1, keepdim=True) + eps)
