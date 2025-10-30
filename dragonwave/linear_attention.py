
import torch
import torch.nn as nn
from .positives import pos_unit

class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_head: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, n_heads*d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads*d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads*d_head, bias=False)
        self.o_proj = nn.Linear(n_heads*d_head, d_model, bias=False)
    def forward(self, x, state=None, k_override=None, q_override=None, v_override=None):
        B,T,D = x.shape
        H,d = self.n_heads, self.d_head
        if k_override is None or q_override is None or v_override is None:
            q = self.q_proj(x).view(B,T,H,d)
            k = self.k_proj(x).view(B,T,H,d)
            v = self.v_proj(x).view(B,T,H,d)
        else:
            q, k, v = q_override, k_override, v_override
        q = pos_unit(q); k = pos_unit(k)
        eps=1e-6
        y_list=[]
        if state is None:
            kv_cum = x.new_zeros(B,H,d,d)
            k_cum  = x.new_zeros(B,H,d)
        else:
            kv_cum = state["kv"]
            k_cum  = state["k"]
        for t in range(T):
            kt = k[:,t]; vt = v[:,t]
            kv = torch.einsum("bhd,bhj->bhjd", kt, vt)
            kv_cum = kv_cum + kv
            k_cum  = k_cum + kt
            qt = q[:,t]
            num = torch.einsum("bhd,bhjd->bhj", qt, kv_cum)
            den = torch.einsum("bhd,bhd->bh", qt, k_cum).unsqueeze(-1) + eps
            yt = num/den
            y_list.append(yt)
        y = torch.stack(y_list, dim=1).reshape(B,T,H*d)
        y = self.o_proj(y)
        new_state = {"kv": kv_cum.detach(), "k": k_cum.detach()}
        return y, new_state
