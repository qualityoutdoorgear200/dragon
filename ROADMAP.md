
# Roadmap: Implementation & Testing Order

## Phase 0 — Baselines
1. **BDH-only baseline** (`configs/base_bdh.yaml`)
   - Dataset: WikiText-2 (or the built-in toy to sanity-check code path).
   - Metrics: PPL, ECE, Brier, FCR.
   - Purpose: anchor for all ablations; verify logging and eval.

2. **A-only resonator** (`configs/a_only.yaml`)
   - Dataset: same as baseline.
   - Expectation: small PPL change; richer features; no calibration harm.

## Phase 1 — Subconscious factorization
3. **Add B (salient)** (`configs/a_b.yaml`)
   - Datasets: WikiText-2; HellaSwag zero-shot.
   - Metrics: ECE↓, FCR↓ without PPL↑. Check thresholds rise on obvious bands.

4. **Add C (subtle)** (`configs/a_c.yaml`)
   - Datasets: Long contexts (PG-19 excerpt); LAMBADA.
   - Metrics: long-context PPL drop; earlier detection of faint signals.

5. **A+B+C + EI + threshold re-entry** (`configs/a_b_c_ei.yaml`)
   - Datasets: mix of above; add GSM8K (reasoning) small sample.
   - Metrics: Calibration improves; no babble (rate low).

## Phase 2 — Timescale shaping & fluid spectrum
6. **Retention maps** (`resonator.use_retention_maps: true`)
   - Synthetic: copy/recall; parentheses; counting.
   - Metrics: memory curve vs lag improves; no saturations.

7. **Continuous atoms** (`use_continuous_atoms: true`)
   - Datasets: WikiText-103 subset.
   - Metrics: stable performance; inspect learned centers/bandwidths.

## Phase 3 — Expressivity & compute shaping
8. **Phase-gated attention** (`extras.use_phase_gated_attn: true`)
   - HellaSwag / ARC.
   - Metrics: accuracy lift at same width; calibration holds.

9. **Shared-weight refiner** (`extras.use_ffn_refine: true`)
   - Any LM set; measure wall-clock / PPL.
   - Metrics: PPL↓ with modest compute↑; good Pareto.

## Phase 4 — Short-term binding & modulation
10. **Fast-weights** (`fastweights.use_fastweights: true`)
    - Synthetic: copy/associative; arithmetic short chains.
    - Metrics: better short-term binding; no drift.

11. **Neuromodulator** (`modulator.use_modulator: true`)
    - Mixed datasets; track ρ_B/ρ_C shifts vs entropy.
    - Metrics: patience–accuracy tradeoff improves; calibration stable.

## Phase 5 — Full ablation grid
12. Run `run_ablate` on a curated set of configs and record metrics.
    - Compare to **GPT-2 small** and **BDH-only** on identical token budgets.
    - Export CSV and reliability diagrams.

---

## Metrics & Considerations (per step)
- **Perplexity (PPL)**: standard LM metric; track vs sequence length for long-context.
- **ECE & Brier**: calibration; ensure EI and DK losses push in the right direction.
- **False-Confidence Rate (FCR)**: wrong with confidence > τ (e.g., 0.7).
- **Latency to emit**: if you later add an explicit “emit decision”, track latency.
- **Sparsity / monosemanticity proxies**: active bands per event; CCA between rails.
- **Efficiency**: wall-clock, VRAM, FLOPs proxy (tokens/sec).

## Datasets per capability
- **LM**: WikiText-2/103, PG-19, C4/Pile subsets.
- **Long-context**: PG-19, arXiv slices, NarrativeQA summaries.
- **Commonsense**: HellaSwag, PIQA, ARC.
- **Reasoning**: GSM8K, SVAMP.
- **Algorithmic/WM**: synthetic tasks in this repo; Long Range Arena (listops/text).

## Comparison standards
- **GPT-2 small** (or your small decoder) and **BDH-only** as baselines.
- Match token budget and optimizer; log compute/V RAM for fairness.
