
import torch
import torch.nn as nn

class QuadraticAttention(nn.Module):
    """
    Repo-style attention (bdh.py-like): RoPE-prepared Q,K, causal lower-tri mask,
    no softmax, then scores@V. Heads are feature groups.
    """
    def __init__(self, d_model: int, n_heads: int = 4, d_head: int = 64, scale: bool = False):
        super().__init__()
        self.n_heads=n_heads; self.d_head=d_head; self.scale=scale
        self.q_proj = nn.Linear(d_model, n_heads*d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads*d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads*d_head, bias=False)
        self.o_proj = nn.Linear(n_heads*d_head, d_model, bias=False)

    def forward(self, x, rope_fn=None):
        B,T,D = x.shape
        H,d = self.n_heads, self.d_head
        q = self.q_proj(x).view(B,T,H,d)
        k = self.k_proj(x).view(B,T,H,d)
        v = self.v_proj(x).view(B,T,H,d)
        if rope_fn is not None:
            q,k = rope_fn(q,k)
        if self.scale and d>0:
            q = q / (d ** 0.5)
        scores = torch.einsum("bthd,bshd->bhts", q, k)
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril(diagonal=-1)
        scores = scores.masked_fill(~mask, 0.0)
        out = torch.einsum("bhts, bshd -> bthd", scores, v)
        out = out.reshape(B,T,H*d)
        return self.o_proj(out)
