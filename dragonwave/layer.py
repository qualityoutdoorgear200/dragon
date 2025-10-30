
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resonator import ResonatorABC
from .linear_attention import LinearAttention
from .quadratic_attention import QuadraticAttention
from .positives import pos
from .expressivity import PhaseGatedAdapter, SharedWeightRefiner
from .fastweights import FastWeightsState
from .relu_lowrank import ReLULowRank
try:
    from .rope import apply_rope
except Exception:
    def apply_rope(q,k,rope_theta=10000.0):
        return q,k

class HeadwiseLayerNorm(nn.Module):
    def __init__(self, n_heads:int, d_head:int, eps:float=1e-5):
        super().__init__()
        self.n_heads=n_heads; self.d_head=d_head
        self.weight = nn.Parameter(torch.ones(n_heads, 1, d_head))
        self.bias   = nn.Parameter(torch.zeros(n_heads, 1, d_head))
        self.eps=eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True); var  = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias

class BDHResonatorLayer(nn.Module):
    """
    Unified BDH layer:
      - attention_mode: "linear" (BDH-GPU-like) or "quadratic" (repo-like, RoPE + no-softmax)
      - ReLU-lowrank FFN by default (BDH-GPU)
      - optional per-head LayerNorm
      - optional RoPE on (Q,K) for quadratic mode
      - resonator sidecar (A/B/C rails) remains optional & orthogonal
    """
    def __init__(self, d_model=256, n_heads=4, d_head=64,
                 n_freq=64,
                 use_B=False, use_C=False,
                 use_retention_maps=False, use_threshold_maps=True,
                 atoms_per_filter=2, use_continuous_atoms=False,
                 extras=None, fastweights=None,
                 relu_lowrank_rank: int = 256,
                 use_per_head_norm: bool = True,
                 use_rope: bool = False,
                 attention_mode: str = "linear"):
        super().__init__()
        self.n_freq = n_freq
        self.use_res = n_freq > 0
        self.extras = extras or {}
        self.fast_cfg = fastweights or {}
        self.use_rope = use_rope
        self.use_per_head_norm = use_per_head_norm
        self.n_heads=n_heads; self.d_head=d_head
        self.attention_mode = attention_mode  # "linear" or "quadratic"

        if self.use_res:
            self.res = ResonatorABC(n_freq=n_freq, use_B=use_B, use_C=use_C,
                                    use_retention_maps=use_retention_maps,
                                    use_threshold_maps=use_threshold_maps,
                                    atoms_per_filter=atoms_per_filter,
                                    use_continuous_atoms=use_continuous_atoms)
            self.to_amp = nn.Linear(d_model, n_freq)
            self.to_phi = nn.Linear(d_model, n_freq)
            self.feat_proj = nn.Linear(3*n_freq, d_model)
        else:
            self.res = None; self.to_amp = None; self.to_phi = None; self.feat_proj = None

        if attention_mode == "quadratic":
            self.attn = QuadraticAttention(d_model=d_model, n_heads=n_heads, d_head=d_head, scale=False)
        else:
            self.attn = LinearAttention(d_model=d_model, n_heads=n_heads, d_head=d_head)

        self.per_head_ln = HeadwiseLayerNorm(n_heads, d_head) if use_per_head_norm else None

        if self.extras.get("use_phase_gated_attn", False) and self.use_res and attention_mode=="linear":
            self.phase_gate = PhaseGatedAdapter(n_feat=3*n_freq, n_heads=n_heads, d_head=d_head)
            self.k_pre = nn.Linear(d_model, n_heads*d_head, bias=False)
            self.q_pre = nn.Linear(d_model, n_heads*d_head, bias=False)
            self.v_lin = nn.Linear(d_model, n_heads*d_head, bias=False)
        else:
            self.phase_gate = None

        # BDH-GPU style ReLU-lowrank FFN
        self.ffn = ReLULowRank(d_model=d_model, rank=relu_lowrank_rank, positive_out=True)
        self.norm = nn.LayerNorm(d_model)

        if self.fast_cfg.get("use_fastweights", False):
            self.fast_state = FastWeightsState(d_model=d_model, rank=self.fast_cfg.get("rank",4),
                                               gamma=self.fast_cfg.get("gamma",0.98))
            self.fast_eta = self.fast_cfg.get("eta", 1e-3)
            self.fast_event = self.fast_cfg.get("event","B")
        else:
            self.fast_state = None

    def init_state(self, B, device):
        if self.use_res:
            return {"res": self.res.init_state(B, device), "attn": None}
        else:
            return {"res": None, "attn": None}

    def forward(self, x, state, gate_cfg, ei_cfg, reentry_mode):
        B,T,D = x.shape
        attn_state = state["attn"]

        if self.use_res:
            res_state = state["res"]
            amp = torch.nn.functional.softplus(self.to_amp(x))
            phi = torch.pi * torch.tanh(self.to_phi(x))
            in_field_seq = torch.polar(amp, phi)
            feats = []; last_di=None
            for t in range(T):
                res_state, di = self.res.step(res_state, in_field_seq[:,t], gate_cfg=gate_cfg, ei_cfg=ei_cfg, reentry_mode=reentry_mode)
                X = res_state["X"]
                feats.append(torch.cat([X.real, X.imag, X.abs()], dim=-1))
                last_di = di
            feat = torch.stack(feats, dim=1)
            h = pos(self.feat_proj(feat))
        else:
            last_di=None; h = x

        if self.phase_gate is not None and self.attention_mode=="linear" and self.use_res:
            k_pre = self.k_pre(h).view(B,T,self.attn.n_heads,self.attn.d_head)
            q_pre = self.q_pre(h).view(B,T,self.attn.n_heads,self.attn.d_head)
            k_g, q_g = self.phase_gate(feat, k_pre, q_pre)
            v = self.v_lin(h).view(B,T,self.attn.n_heads,self.attn.d_head)
            y, attn_state = self.attn(h, state=attn_state, k_override=k_g, q_override=q_g, v_override=v)
        else:
            if self.attention_mode == "quadratic" and self.use_rope:
                rope_fn = lambda q,k: apply_rope(q,k)
            else:
                rope_fn = None
            if self.attention_mode == "quadratic":
                y = self.attn(h, rope_fn=rope_fn)
            else:
                y, attn_state = self.attn(h, state=attn_state)

        if self.per_head_ln is not None:
            try:
                H = self.attn.n_heads; d = self.attn.d_head
            except:
                H = self.n_heads; d = self.d_head
            y = y.view(B,T,H,d)
            y = self.per_head_ln(y).reshape(B,T,H*d)

        if self.fast_state is not None:
            ev_map = {"A": last_di.get("evA"), "B": last_di.get("evB"), "C": last_di.get("evC")}
            ev = ev_map.get(self.fast_event, None)
            if ev is not None and ev.numel()>0 and (ev.mean() > 0).item():
                self.fast_state.write(h[:, -1, :], eta=self.fast_eta)
            self.fast_state.decay()
            y = self.fast_state.apply(y[:, -T:, :])

        y = self.ffn(y)
        y = self.norm(x + y)
        new_state = {"res": state.get("res", None) if not self.use_res else res_state, "attn": attn_state}
        return y, new_state, {"res": last_di}
