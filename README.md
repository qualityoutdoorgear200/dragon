
# dragonwave v3 — BDH-based Resonator Research Stack (Toggle-All)

This stack keeps a **Dragon Hatchling (BDH‑GPU)** baseline and adds *modular*, *composable*
options you can A/B test one by one: wave resonator (A/B/C rails), EI micro-circuits,
retention/threshold maps, continuous atoms (fluid spectrum), phase‑gated attention,
shared-weight refiners, optional event‑gated fast-weights, a slow neuromodulator,
and an ablation & eval harness with core metrics (perplexity, ECE, Brier, false-confidence).

> **Default = BDH-only** (`configs/base_bdh.yaml`). Flip flags in YAML to add features.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pyyaml tqdm
```

## Quick start
```bash
# 1) Pure BDH baseline
python -m dragonwave.train --config configs/base_bdh.yaml

# 2) A-only resonator (simple frequency-based neuron)
python -m dragonwave.train --config configs/a_only.yaml

# 3) Add B (salient) then C (subtle) + EI + threshold re-entry
python -m dragonwave.train --config configs/a_b.yaml
python -m dragonwave.train --config configs/a_c.yaml
python -m dragonwave.train --config configs/a_b_c_ei.yaml

# 4) Everything on (stress; tune if unstable)
python -m dragonwave.train --config configs/full_all.yaml
```

## Evaluate (held-out stream)
```bash
python -m dragonwave.eval --config configs/a_b_c_ei.yaml --text_path data/heldout.txt
```

## Ablation grid
```bash
python -m dragonwave.run_ablate --grid configs/grids/small_grid.yaml --out runs/small_grid.csv
```

## Synthetic tasks (algorithmic/WM)
Toggle via `train.dataset`:
- `text` (default): char LM from `train.text_path` or built-in toy text.
- `synthetic_copy`: copy/associative recall
- `synthetic_paren`: balanced parentheses
- `synthetic_count`: running digit sum / parity

See `dragonwave/synthetic_tasks.py` for generation details.

---

## Flags overview (see `config.py`)

- **Resonator**: `resonator.*` — A-rail; optional `use_retention_maps`, `use_threshold_maps`,
  `use_continuous_atoms`, `atoms_per_filter`
- **Rails**: `rails.use_B`, `rails.use_C`; gate params in `gate.*`
- **EI**: `ei.use_EI`, `ei.phase_corr_scale`, `ei.norm_strength`
- **Re-entry**: `reentry.mode` ∈ {`threshold_only`, `threshold_phase`, `full`}, `reentry.additive_nudge`
- **Expressivity**: `extras.use_phase_gated_attn`; `extras.use_ffn_refine`, `extras.refine_steps`
- **Fast-weights**: `fastweights.use_fastweights`, `rank`, `eta`, `gamma`, `event`
- **Neuromodulator**: `modulator.use_modulator`, `modulator.smoothing`
- **Timing & DK-aware**: `timing.*`, `dk.*`
- **Train dataset**: `train.dataset` ∈ {`text`, `synthetic_copy`, `synthetic_paren`, `synthetic_count`}
  and `train.text_path` for custom corpora.

---

## Logging & metrics
- **Training** prints: loss, softDP, coverage, rate, entropy, overconfidence, disagreement.
- **Eval** prints: Perplexity, **ECE**, **Brier**, **False-Confidence Rate**.
- **Results logger** writes JSONL/CSV with a config hash for reproducibility.

---

## Roadmap (see `ROADMAP.md`)
A step-by-step plan: from BDH baseline to A-only, then add B, then C, EI, maps, atoms,
expressivity adapters, fast-weights, and neuromodulation—plus which **datasets** and
**metrics** to use at each step and what to look for.

---

## Notes
- All updates are **token-step** (no analog time). “Events” = conditional within-step updates.
- Attention stays **non‑negative** (BDH invariant). Threshold & gain changes are multiplicative or additive to Θ.
- Keep **nudge magnitudes tiny** and **clamped**. Use EMA fades for stability.
