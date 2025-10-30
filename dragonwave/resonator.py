
import torch
import torch.nn as nn
import math
from .gates import compute_plv, adaptive_percentile_threshold

def complex_from_polar(amplitude, phase):
    return torch.polar(amplitude, phase)

class ResonatorABC(nn.Module):
    def __init__(self, n_freq=64, use_B=False, use_C=False,
                 use_retention_maps=False, use_threshold_maps=True,
                 atoms_per_filter=2, use_continuous_atoms=False,
                 alpha_base=0.97, lambda_base=0.95, theta_base=0.10, beta_theta=0.01,
                 nudge_alpha=0.002, nudge_lambda=0.002, nudge_theta=0.01):
        super().__init__()
        self.F = n_freq
        self.use_B = use_B
        self.use_C = use_C
        self.use_retention_maps = use_retention_maps
        self.use_threshold_maps = use_threshold_maps
        self.use_continuous_atoms = use_continuous_atoms
        self.atoms = atoms_per_filter

        self.register_buffer("alpha_base", torch.tensor(alpha_base))
        self.register_buffer("lambda_base", torch.tensor(lambda_base))
        if use_retention_maps:
            self.alpha_map = nn.Parameter(0.0 * torch.randn(n_freq))
            self.lambda_map = nn.Parameter(0.0 * torch.randn(n_freq))
        else:
            self.alpha_map = None
            self.lambda_map = None

        self.register_buffer("theta_base", torch.full((n_freq,), theta_base))
        self.register_buffer("beta_theta", torch.full((n_freq,), beta_theta))

        self.r_re = nn.Parameter(0.05*torch.randn(n_freq))
        self.r_im = nn.Parameter(0.05*torch.randn(n_freq))

        if use_continuous_atoms:
            self.centers = nn.Parameter(torch.linspace(0.1, 1.4, self.atoms))
            self.bandw   = nn.Parameter(0.1*torch.ones(self.atoms))

        self.register_buffer("nudge_alpha", torch.tensor(nudge_alpha))
        self.register_buffer("nudge_lambda", torch.tensor(nudge_lambda))
        self.register_buffer("nudge_theta", torch.tensor(nudge_theta))

    def init_state(self, B, device):
        X0 = torch.zeros(B, self.F, dtype=torch.cfloat, device=device)
        sA = torch.zeros(B, self.F, dtype=torch.float32, device=device)
        state = {
            "X": X0,
            "sA": sA,
            "theta": self.theta_base.clone().to(device),
            "ema_amp_mu": torch.zeros(self.F, device=device),
            "ema_amp_var": torch.ones(self.F, device=device)*1e-2,
            "ema_unit": torch.zeros(self.F, dtype=torch.cfloat, device=device),
            "b_usage": torch.zeros(self.F, device=device),
            "c_usage": torch.zeros(self.F, device=device),
        }
        if self.use_B:
            state["sB"] = torch.zeros(B, self.F, device=device)
        if self.use_C:
            state["sC"] = torch.zeros(B, self.F, device=device)
        return state

    def _build_filter(self):
        if self.F == 0:
            return torch.zeros(0, dtype=torch.cfloat, device=self.r_re.device if hasattr(self, "r_re") else "cpu")
        if not self.use_continuous_atoms:
            return torch.complex(self.r_re, self.r_im)  # [F]
        f = torch.linspace(0, 1, self.F, device=self.r_re.device)
        r = 0
        for c,b in zip(self.centers, self.bandw):
            gauss = torch.exp(-0.5*((f - c)/(b.abs()+1e-3))**2)
            phase = torch.exp(1j*2*math.pi*f*c)
            r = r + gauss * phase
        r = r / max(1, self.atoms)
        return r

    def step(self, state, in_field, gate_cfg=None, ei_cfg=None, reentry_mode="threshold_only"):
        if self.F == 0:
            return state, {"evA": None, "evB": None, "evC": None}
        B,F = in_field.shape
        X = state["X"]
        alpha = self.alpha_base
        lam   = self.lambda_base
        if self.use_retention_maps and self.alpha_map is not None:
            alpha = (alpha + 0.05*torch.tanh(self.alpha_map)).clamp(0.90, 0.995)
            lam   = (lam   + 0.05*torch.tanh(self.lambda_map)).clamp(0.90, 0.995)

        X_mid = alpha * X + in_field

        amp = X_mid.abs()
        unit = torch.exp(1j*X_mid.angle())
        with torch.no_grad():
            mu = state["ema_amp_mu"]
            var = state["ema_amp_var"]
            mu = 0.9*mu + 0.1*amp.mean(dim=0)
            var = 0.9*var + 0.1*((amp - mu.unsqueeze(0))**2).mean(dim=0)
            plv, ema_unit = compute_plv(unit, beta=0.1, prev=state["ema_unit"])
            state["ema_amp_mu"] = mu
            state["ema_amp_var"] = var
            state["ema_unit"] = ema_unit

        zscore = (amp - state["ema_amp_mu"].unsqueeze(0)) / (state["ema_amp_var"].sqrt().unsqueeze(0)+1e-3)
        instab = (1.0 - plv).unsqueeze(0).expand_as(amp)

        gB = gC = None; tauB = tauC = None
        if self.use_B:
            mix = gate_cfg.get("mix", {"amp":1.0,"novelty":0.5,"instability":0.5,"residual":0.0})
            scoreB = mix.get("amp",1.0)*amp + mix.get("novelty",0.5)*zscore.abs() + mix.get("instability",0.5)*instab
            gB, tauB = adaptive_percentile_threshold(scoreB, gate_cfg.get("rho_B",0.25), alpha=gate_cfg.get("alpha",5.0))
        if self.use_C:
            mix = gate_cfg.get("mix", {"amp":1.0,"novelty":0.5,"instability":0.5,"residual":0.0})
            scoreC = mix.get("amp",1.0)*amp + mix.get("novelty",0.5)*zscore.abs() + mix.get("instability",0.5)*instab
            gC, tauC = adaptive_percentile_threshold(-scoreC, gate_cfg.get("rho_C",0.25), alpha=gate_cfg.get("alpha",5.0))

        XA = X_mid
        XB = X_mid * gB if gB is not None else None
        XC = X_mid * gC if gC is not None else None

        r = self._build_filter()
        zA = (XA * torch.conj(r)).real
        sA = state["sA"] = lam * state["sA"] + zA
        evA = (sA >= state["theta"]).float()
        evB = evC = None
        if self.use_B:
            sB = state["sB"] = lam * state["sB"] + ( (XB * torch.conj(r)).real )
            evB = (sB >= state["theta"]).float()
        if self.use_C:
            sC = state["sC"] = lam * state["sC"] + ( (XC * torch.conj(r)).real )
            evC = (sC >= state["theta"]).float()

        with torch.no_grad():
            theta = state["theta"]
            if evB is not None and ei_cfg and ei_cfg.get("use_EI", False):
                theta = theta + 0.01 * (evB.mean(dim=0))
            if evC is not None and ei_cfg and ei_cfg.get("use_EI", False):
                theta = theta - 0.01 * (evC.mean(dim=0))
            theta = theta - self.beta_theta * (theta - self.theta_base.to(theta.device))
            state["theta"] = theta.clamp(0.01, 10.0)

        if reentry_mode in ("threshold_phase","full") and self.use_B and ei_cfg and ei_cfg.get("use_EI", False):
            corr = (evB.mean(dim=0) > 0).float().unsqueeze(0)
            phase = torch.exp(1j*ei_cfg.get("phase_corr_scale",0.02) * corr)
            XA = XA * phase

        if reentry_mode == "full" and gate_cfg.get("additive_nudge", 0.0) > 0 and evB is not None:
            nudge = gate_cfg["additive_nudge"]
            mask = (evB.mean(dim=0) > 0).float().unsqueeze(0)
            XA = XA + nudge * (XB * mask)

        state["X"] = XA
        with torch.no_grad():
            if evB is not None:
                state["b_usage"] = 0.99*state["b_usage"] + 0.01*evB.mean(dim=0)
            if evC is not None:
                state["c_usage"] = 0.99*state["c_usage"] + 0.01*evC.mean(dim=0)

        di = {"amp": amp.detach(), "plv": plv.detach(),
              "gB": gB.detach() if gB is not None else None,
              "gC": gC.detach() if gC is not None else None,
              "evA": evA.detach(),
              "evB": evB.detach() if evB is not None else None,
              "evC": evC.detach() if evC is not None else None,
              "tauB": tauB, "tauC": tauC}
        return state, di
