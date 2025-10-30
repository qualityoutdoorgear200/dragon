
import torch
import torch.nn as nn
from .layer import BDHResonatorLayer

class StreamLM(nn.Module):
    """
    Supports:
      - shared_layers (single block unrolled n_layers times) OR separate layers
      - attention_mode: "linear" (BDH-GPU-like) or "quadratic" (repo-like)
    """
    def __init__(self, vocab_size: int,
                 d_model=256, n_layers=2, n_heads=4, d_head=64,
                 n_freq=64,
                 use_B=False, use_C=False,
                 use_retention_maps=False, use_threshold_maps=True,
                 atoms_per_filter=2, use_continuous_atoms=False,
                 extras=None, fastweights=None,
                 shared_layers: bool = True,
                 relu_lowrank_rank: int = 256,
                 use_per_head_norm: bool = True,
                 use_rope: bool = False,
                 attention_mode: str = "linear"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(8192, d_model)
        self.shared_layers = shared_layers
        self.n_layers = n_layers

        kwargs = dict(d_model=d_model, n_heads=n_heads, d_head=d_head,
                      n_freq=n_freq, use_B=use_B, use_C=use_C,
                      use_retention_maps=use_retention_maps, use_threshold_maps=use_threshold_maps,
                      atoms_per_filter=atoms_per_filter, use_continuous_atoms=use_continuous_atoms,
                      extras=extras, fastweights=fastweights,
                      relu_lowrank_rank=relu_lowrank_rank,
                      use_per_head_norm=use_per_head_norm,
                      use_rope=use_rope,
                      attention_mode=attention_mode)

        if shared_layers:
            self.block = BDHResonatorLayer(**kwargs)
        else:
            self.layers = nn.ModuleList([BDHResonatorLayer(**kwargs) for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)
        self.canvas_leak = nn.Parameter(torch.tensor(0.95), requires_grad=False)
        self.canvas_proj = nn.Linear(d_model, d_model)
        self.readout = nn.Sequential(nn.GELU(), nn.Linear(d_model, vocab_size, bias=False))
        self.aux_readouts = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(2)])
        self.readout[-1].weight = self.tok_emb.weight

    def forward(self, idx, gate_cfg, ei_cfg, reentry_mode="threshold_only"):
        B,T = idx.shape
        device = idx.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        if self.shared_layers:
            state = self.block.init_state(B, device=device)
            last_di=None
            for _ in range(self.n_layers):
                x, state, di = self.block(x, state, gate_cfg=gate_cfg, ei_cfg=ei_cfg, reentry_mode=reentry_mode)
                last_di = di
        else:
            attn_states = [None for _ in self.layers]
            res_states  = [lyr.init_state(B, device=device)["res"] for lyr in self.layers]
            last_di=None
            for i, lyr in enumerate(self.layers):
                x, st, di = lyr(x, {"res": res_states[i], "attn": attn_states[i]},
                                gate_cfg=gate_cfg, ei_cfg=ei_cfg, reentry_mode=reentry_mode)
                res_states[i], attn_states[i] = st["res"], st["attn"]
                last_di = di

        x = self.norm(x)

        C = torch.zeros(B, x.size(-1), device=device)
        logits_list = []
        aux_logits_list = [[] for _ in self.aux_readouts]
        for t in range(T):
            C = self.canvas_leak * C + self.canvas_proj(x[:,t])
            lg = self.readout(C)
            logits_list.append(lg.unsqueeze(1))
            for k, head in enumerate(self.aux_readouts):
                aux_logits_list[k].append(head(C).unsqueeze(1))
        logits = torch.cat(logits_list, dim=1)
        aux_logits = [torch.cat(seq, dim=1) for seq in aux_logits_list]
        return logits, aux_logits, last_di
