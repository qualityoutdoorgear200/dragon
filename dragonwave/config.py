
from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml

@dataclass
class GateConfig:
    rho_B: float = 0.25
    rho_C: float = 0.25
    alpha: float = 5.0
    mix: Dict[str, float] = field(default_factory=lambda: {"amp": 1.0, "novelty": 0.5, "instability": 0.5, "residual": 0.0})
    occupancy_balance: float = 0.0
    temperature: float = 1.0

@dataclass
class ResonatorConfig:
    n_freq: int = 64
    atoms_per_filter: int = 2
    use_continuous_atoms: bool = False
    use_retention_maps: bool = False
    use_threshold_maps: bool = True
    alpha: float = 0.97
    lambda_int: float = 0.95
    theta_base: float = 0.1
    beta_theta: float = 0.01
    event_nudge_alpha: float = 0.002
    event_nudge_lambda: float = 0.002
    event_nudge_theta: float = 0.01

@dataclass
class RailsConfig:
    use_B: bool = False
    use_C: bool = False

@dataclass
class EIConfig:
    use_EI: bool = False
    norm_strength: float = 0.2
    phase_corr_scale: float = 0.02

@dataclass
class ReentryConfig:
    mode: str = "threshold_only"    # threshold_only | threshold_phase | full
    additive_nudge: float = 0.0

@dataclass
class DKConfig:
    overconfidence: float = 0.0
    disagreement: float = 0.0

@dataclass
class TimingConfig:
    use_canvas: bool = True
    loss: str = "softDP"
    lambda_rate: float = 1e-3
    lambda_entropy: float = 1e-3
    lambda_coverage: float = 1e-2

@dataclass
class ModulatorConfig:
    use_modulator: bool = False
    smoothing: float = 0.1

@dataclass
class AttentionExtras:
    use_phase_gated_attn: bool = False
    use_ffn_refine: bool = False
    refine_steps: int = 2

@dataclass
class FastWeightsConfig:
    use_fastweights: bool = False
    rank: int = 4
    eta: float = 1e-3
    gamma: float = 0.98
    event: str = "B"   # "A" | "B" | "C"

@dataclass
class AttentionConfig:
    d_model: int = 256
    n_heads: int = 4
    d_head: int = 64
    n_layers: int = 2

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 64
    seq_len: int = 128
    steps: int = 400
    checkpoint_interval: int = 50
    micro_batch_size: Optional[int] = None
    max_vram_gb: Optional[float] = None
    device: str = "auto"
    seed: int = 42
    dataset: str = "text"         # text | synthetic_copy | synthetic_paren | synthetic_count
    text_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    tokenized_cache_path: Optional[str] = None
    save_dir: Optional[str] = "runs/default"

@dataclass
class Config:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    extras: AttentionExtras = field(default_factory=AttentionExtras)
    fastweights: FastWeightsConfig = field(default_factory=FastWeightsConfig)
    resonator: ResonatorConfig = field(default_factory=ResonatorConfig)
    rails: RailsConfig = field(default_factory=RailsConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    ei: EIConfig = field(default_factory=EIConfig)
    reentry: ReentryConfig = field(default_factory=ReentryConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    dk: DKConfig = field(default_factory=DKConfig)
    modulator: ModulatorConfig = field(default_factory=ModulatorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config()
    def merge(dc, dd):
        for k,v in dd.items():
            if hasattr(dc, k):
                if isinstance(v, dict):
                    merge(getattr(dc, k), v)
                else:
                    setattr(dc, k, v)
    merge(cfg, data)
    return cfg
