from dataclasses import dataclass
import json
from typing import Dict, List, Tuple

BATCH_SIZE = 515.0  # grams


@dataclass
class Ingredient:
    phase: str
    name: str
    grams: float

    @property
    def percent(self) -> float:
        return self.grams / BATCH_SIZE * 100.0


def build_formula() -> List[Ingredient]:
    """Return the baseline ingredient list scaled for 1 g addition of the MB liposome."""
    ingredients: List[Ingredient] = []

    def add_many(phase: str, items: Dict[str, float]):
        for name, grams in items.items():
            ingredients.append(Ingredient(phase=phase, name=name, grams=grams))

    # Phase A
    add_many("A", {
        "Distilled Water": 43.73,
        "Sodium Phytate": 0.25,
        "Poloxamer 188": 5.00,
        "Tetrasodium Glutamate Diacetate": 0.25,
        "Hydrolysed Silk Protein": 5.00,
        "Aloe Barbadensis Leaf Juice Powder (200x)": 0.75,
        "Inulin": 5.00,
        "Fructo-Oligosaccharides": 10.00,
        "Beta-Glucan": 5.00,
        "Sodium Hyaluronate 3-5 kDa": 2.50,
        "Sodium Hyaluronate ~300 kDa": 2.50,
        "Colloidal Oat Flour": 10.00,
        "Xanthan Gum": 2.00,
        "Sepimax ZEN": 0.60,
    })

    # Phase B1
    add_many("B1", {
        "Cupuaçu Butter": 17.50,
        "Shea Butter": 12.50,
        "Olive Squalane": 29.75,
        "Caprylic/Capric Triglyceride": 30.00,
        "Isopropyl Myristate": 10.00,
        "Olivem 1000": 10.00,
        "Montanov 202": 10.00,
        "Glyceryl Stearate": 5.00,
        "Pomegranate Sterols": 3.75,
        "Stearic Acid": 1.50,
        "Bisabolol": 2.50,
        "Rosemary Oleoresin": 0.50,
    })

    # Phase B2
    add_many("B2", {
        "Granactive Retinoid 10%": 5.00,
        "Bakuchiol": 10.00,
        "Coenzyme Q10": 5.00,
        "Mixed Tocopherols 90%": 2.50,
        "Ethyl Ferulate": 5.00,
        "Astaxanthin Oleoresin 10%": 2.50,
        "Acetyl Zingerone": 5.00,
        "Tetrahexyldecyl Ascorbate": 10.00,
    })

    # Phase C base (assumed fully transferred across sub-batches)
    add_many("C-base", {
        "Hydrogenated Lecithin": 40.00,
        "DSPE-PEG 2000": 1.25,
        "Dicetyl Phosphate": 1.50,
        "Alpha-Tocopherol": 1.00,
        "1,3-Propanediol (liposome base)": 5.00,
        "Pentylene Glycol (liposome base)": 11.67,
        "Ceramide NP": 1.80,
        "Ceramide AP": 1.35,
        "Ceramide EOP": 1.35,
        "Cholesterol (liposome base)": 1.50,
        "Phytosphingosine": 0.50,
    })

    # Sub-batch A
    add_many("C-A", {
        "Palmitoyl Pentapeptide-4": 0.025,
        "Tetrapeptide-21": 0.025,
        "Tripeptide-10 Citrulline": 0.025,
        "Galactomyces Ferment Filtrate (A)": 12.50,
    })

    # Sub-batch C
    add_many("C-C", {
        "GHK-Cu": 1.000,
        "AHK-Cu": 5.000,
        "Palmitoyl Tripeptide-1": 0.025,
        "Palmitoyl Tetrapeptide-7": 0.025,
        "Palmitoyl Tripeptide-38": 0.025,
        "Galactomyces Ferment Filtrate (C)": 12.50,
    })

    # Sub-batch D
    add_many("C-D", {
        "Madecassoside": 0.50,
        "Asiaticoside": 0.50,
        "Galactomyces Ferment Filtrate (D)": 12.50,
    })

    # Sub-batch E (1 g addition assumed)
    sub_e_total = 0.69 + 0.46 + 0.10 + 0.03 + 25.00
    scale = 1.0 / sub_e_total
    add_many("C-E", {
        "DSPC": 0.69 * scale,
        "Cholesterol (C-E)": 0.46 * scale,
        "DSPE-PEG 2000 (C-E)": 0.10 * scale,
        "Methylene Blue": 0.03 * scale,
        "pH 5.5 Citrate Buffer": 25.00 * scale,
    })

    # Phase D
    add_many("D", {
        "Lactobacillus Ferment Lysate": 20.00,
        "Bifidobacterium Longum Lysate": 5.00,
        "Saccharide Isomerate": 5.00,
        "Honeyquat PF": 5.00,
        "DL-Panthenol": 10.00,
        "Allantoin": 2.50,
        "Pentylene Glycol": 8.34,
        "1,3-Propanediol": 5.00,
        "Tetrahydropiperine": 3.75,
        "Caprylyl Glycol": 2.50,
        "Dimethyl Isosorbide": 5.50,
        "Glyceryl Caprylate": 0.50,
        "PE 9010 Preservative": 5.00,
        "Niacinamide": 15.00,
        "N-Acetyl Glucosamine": 10.00,
        "Sodium PCA": 5.00,
        "Sodium Polyglutamate": 5.00,
        "Ectoin": 5.00,
        "Ergothioneine (0.05%)": 0.25,
    })

    return ingredients


def build_iteration1_formula() -> List[Ingredient]:
    """Iteration 1 adjustments (accepted)."""
    base = build_formula()
    adjustments = {
        "Distilled Water": 63.39,
        "Saccharide Isomerate": 3.00,
        "Sodium Polyglutamate": 3.00,
        "Sodium PCA": 3.00,
        "Sodium Phytate": 0.20,
        "Tetrasodium Glutamate Diacetate": 0.15,
        "Olivem 1000": 10.50,
        "Glyceryl Stearate": 4.50,
    }
    updated: List[Ingredient] = []
    for item in base:
        grams = adjustments.get(item.name, item.grams)
        updated.append(Ingredient(phase=item.phase, name=item.name, grams=grams))
    return updated


def build_iteration2_formula() -> List[Ingredient]:
    """Candidate iteration 2 adjustments."""
    iter1 = build_iteration1_formula()
    adjustments = {
        "Isopropyl Myristate": 0.0,
        "C12-15 Alkyl Benzoate": 10.00,
        "Tetrasodium Glutamate Diacetate": 0.10,
        "Saccharide Isomerate": 2.00,
        "Sodium Polyglutamate": 2.50,
        "Sodium PCA": 2.50,
        "Olivem 1000": 11.00,
        "Glyceryl Stearate": 4.00,
    }
    # Ensure C12-15 Alkyl Benzoate exists
    has_benzoate = any(item.name == "C12-15 Alkyl Benzoate" for item in iter1)
    updated: List[Ingredient] = []
    for item in iter1:
        if item.name == "Isopropyl Myristate" and adjustments["Isopropyl Myristate"] == 0.0:
            continue
        grams = adjustments.get(item.name, item.grams)
        updated.append(Ingredient(phase=item.phase, name=item.name, grams=grams))
    if not has_benzoate:
        updated.append(Ingredient(phase="B1", name="C12-15 Alkyl Benzoate", grams=adjustments["C12-15 Alkyl Benzoate"]))
    # Recalculate distilled water to maintain batch size
    total_mass = mass_balance(updated)
    water_delta = BATCH_SIZE - total_mass
    for idx, item in enumerate(updated):
        if item.name == "Distilled Water":
            updated[idx] = Ingredient(phase=item.phase, name=item.name, grams=item.grams + water_delta)
            break
    return updated


def build_iteration3_formula() -> List[Ingredient]:
    """Iteration 3 adjustments per optimization brief."""
    iter2 = build_iteration2_formula()
    adjustments = {
        "Stearic Acid": 0.50,
        "Glyceryl Stearate": 5.50,
        "Montanov 202": 11.00,
        "Bakuchiol": 7.50,
        "Tetrahydropiperine": 2.50,
        "Glyceryl Caprylate": 0.70,
    }
    updated: List[Ingredient] = []
    for item in iter2:
        grams = adjustments.get(item.name, item.grams)
        updated.append(Ingredient(phase=item.phase, name=item.name, grams=grams))
    total_mass = mass_balance(updated)
    water_delta = BATCH_SIZE - total_mass
    for idx, item in enumerate(updated):
        if item.name == "Distilled Water":
            updated[idx] = Ingredient(phase=item.phase, name=item.name, grams=item.grams + water_delta)
            break
    return updated


def mass_balance(ingredients: List[Ingredient]) -> float:
    return sum(item.grams for item in ingredients)


def compute_nlc_fraction(formula: List[Ingredient]) -> float:
    """Estimate solid:liquid lipid fraction using Phase B1 components."""
    weight_lookup = {item.name: item.grams for item in formula}
    solid_lipids = {
        "Cupuaçu Butter",
        "Shea Butter",
        "Pomegranate Sterols",
        "Stearic Acid",
        "Olivem 1000",
        "Montanov 202",
        "Glyceryl Stearate",
    }
    liquid_lipids = {
        "Olive Squalane",
        "Caprylic/Capric Triglyceride",
        "Isopropyl Myristate",
        "C12-15 Alkyl Benzoate",
        "Bisabolol",
        "Rosemary Oleoresin",
    }
    solid_total = sum(weight_lookup.get(name, 0.0) for name in solid_lipids)
    liquid_total = sum(weight_lookup.get(name, 0.0) for name in liquid_lipids)
    return solid_total / (solid_total + liquid_total)


def compute_hlb(formula: List[Ingredient]) -> Tuple[float, float]:
    """Return required HLB (assumed) and emulsifier blend HLB."""
    weight_lookup = {item.name: item.grams for item in formula}
    # Required HLB assumptions (ASSUMPTION tag)
    required_hlb_inputs = {
        "Cupuaçu Butter": (weight_lookup.get("Cupuaçu Butter", 0.0), 8.0),
        "Shea Butter": (weight_lookup.get("Shea Butter", 0.0), 8.0),
        "Olive Squalane": (weight_lookup.get("Olive Squalane", 0.0), 11.0),
        "Caprylic/Capric Triglyceride": (weight_lookup.get("Caprylic/Capric Triglyceride", 0.0), 11.0),
        "Isopropyl Myristate": (weight_lookup.get("Isopropyl Myristate", 0.0), 11.0),
        "C12-15 Alkyl Benzoate": (weight_lookup.get("C12-15 Alkyl Benzoate", 0.0), 10.0),
        "Bisabolol": (weight_lookup.get("Bisabolol", 0.0), 10.0),
        "Rosemary Oleoresin": (weight_lookup.get("Rosemary Oleoresin", 0.0), 9.0),
        "Pomegranate Sterols": (weight_lookup.get("Pomegranate Sterols", 0.0), 8.5),
        "Stearic Acid": (weight_lookup.get("Stearic Acid", 0.0), 15.0),
    }
    numer = sum(weight * hlb for weight, hlb in required_hlb_inputs.values())
    denom = sum(weight for weight, _ in required_hlb_inputs.values())
    required_hlb = numer / denom if denom else 0.0

    emulsifier_blend = {
        "Olivem 1000": (weight_lookup.get("Olivem 1000", 0.0), 12.0),
        "Montanov 202": (weight_lookup.get("Montanov 202", 0.0), 10.0),
        "Glyceryl Stearate": (weight_lookup.get("Glyceryl Stearate", 0.0), 3.8),
    }
    blend_numer = sum(weight * hlb for weight, hlb in emulsifier_blend.values())
    blend_denom = sum(weight for weight, _ in emulsifier_blend.values())
    blend_hlb = blend_numer / blend_denom if blend_denom else 0.0
    return required_hlb, blend_hlb


def preservation_metrics(formula: List[Ingredient]) -> Dict[str, float]:
    weight_lookup = {item.name: item.grams for item in formula}
    pe9010 = weight_lookup.get("PE 9010 Preservative", 0.0)
    phenoxy = pe9010 * 0.90 / BATCH_SIZE * 100.0
    ehg = pe9010 * 0.10 / BATCH_SIZE * 100.0
    cap_glycol = weight_lookup.get("Caprylyl Glycol", 0.0) / BATCH_SIZE * 100.0
    gly_cap = weight_lookup.get("Glyceryl Caprylate", 0.0) / BATCH_SIZE * 100.0
    # Simple coverage heuristic
    coverage = 6.5
    # penalise for high bioload (ferments, lysates)
    bio_burden = (
        weight_lookup.get("Lactobacillus Ferment Lysate", 0.0)
        + weight_lookup.get("Bifidobacterium Longum Lysate", 0.0)
        + weight_lookup.get("Saccharide Isomerate", 0.0)
        + weight_lookup.get("Honeyquat PF", 0.0)
        + weight_lookup.get("Fructo-Oligosaccharides", 0.0)
        + weight_lookup.get("Inulin", 0.0)
    )
    if bio_burden / BATCH_SIZE * 100.0 > 10.0:
        coverage -= 1.5
    if cap_glycol > 0.45:
        coverage += 0.5
    if gly_cap < 0.1:
        coverage -= 0.5
    coverage = max(0.0, min(10.0, coverage))
    return {
        "phenoxyethanol_%": round(phenoxy, 3),
        "ehg_%": round(ehg, 3),
        "caprylyl_glycol_%": round(cap_glycol, 3),
        "glyceryl_caprylate_%": round(gly_cap, 3),
        "coverage_score": round(coverage, 1),
    }


def sensory_score(formula: List[Ingredient]) -> float:
    humectant_names = {
        "Saccharide Isomerate",
        "Honeyquat PF",
        "DL-Panthenol",
        "Pentylene Glycol",
        "1,3-Propanediol",
        "Niacinamide",
        "N-Acetyl Glucosamine",
        "Sodium PCA",
        "Sodium Polyglutamate",
        "Ectoin",
        "Fructo-Oligosaccharides",
        "Inulin",
        "Beta-Glucan",
        "Sodium Hyaluronate 3-5 kDa",
        "Sodium Hyaluronate ~300 kDa",
        "Galactomyces Ferment Filtrate (A)",
        "Galactomyces Ferment Filtrate (C)",
        "Galactomyces Ferment Filtrate (D)",
        "Lactobacillus Ferment Lysate",
        "Bifidobacterium Longum Lysate",
    }
    total_humectant = sum(item.grams for item in formula if item.name in humectant_names)
    humectant_pct = total_humectant / BATCH_SIZE * 100.0
    score = 7.0
    if humectant_pct > 15.0:
        score -= 2.0
    elif humectant_pct > 12.0:
        score -= 1.5
    elif humectant_pct > 10.0:
        score -= 1.0
    # penalise for heavy wax load (soaping risk)
    waxy = (
        17.50 + 12.50 + 10.00 + 10.00 + 5.00 + 3.75 + 1.50
    ) / BATCH_SIZE * 100.0
    if waxy > 12.0:
        score -= 1.0
    return max(0.0, min(10.0, score))


def cost_score() -> float:
    # Placeholder heuristic: high-actives formula, assume 6/10
    return 6.0


def ee_proxy_scores() -> Dict[str, float]:
    # Heuristic proxies (0-10)
    peptide_score = 6.5  # baseline assumption
    cu_score = 7.5       # due to flash-pulse protocol
    return {"peptide": peptide_score, "cu": cu_score}


def color_stability_proxy() -> float:
    # High antioxidant load but copper risk; assign 6.5/10
    return 6.5


def net_score(preservation: Dict[str, float], sensory: float, cost: float,
              ee: Dict[str, float], color: float, rheology_risk: str,
              mass_ok: bool, hlb_delta: float, nlc_fraction: float) -> float:
    score = 50.0  # baseline
    if not mass_ok:
        score -= 5.0
    if rheology_risk == "high":
        score -= 7.0
    elif rheology_risk == "medium":
        score -= 3.0
    score += sensory - 5.0  # normalized
    score += cost - 5.0
    score += ee["peptide"] - 5.0
    score += ee["cu"] - 5.0
    score += color - 5.0
    hlb_penalty = max(0.0, abs(hlb_delta) - 0.5) * 1.5
    score -= min(4.0, hlb_penalty)
    if not (0.40 <= nlc_fraction <= 0.50):
        score -= 3.0
    score += preservation["coverage_score"] - 5.0
    return round(max(0.0, min(100.0, score)), 1)


def electrolyte_equivalent_risk(formula: List[Ingredient]) -> str:
    electrolyte_names = {
        "Sodium PCA",
        "Sodium Polyglutamate",
        "Sodium Hyaluronate 3-5 kDa",
        "Sodium Hyaluronate ~300 kDa",
    }
    electrolyte_weight = sum(item.grams for item in formula if item.name in electrolyte_names)
    electrolyte_pct = electrolyte_weight / BATCH_SIZE * 100.0
    if electrolyte_pct > 4.0:
        return "high"
    if electrolyte_pct > 2.0:
        return "medium"
    return "low"


def compat_flags(formula: List[Ingredient]) -> List[str]:
    flags: List[str] = []
    chelator_pct = sum(
        item.grams for item in formula
        if item.name in {"Sodium Phytate", "Tetrasodium Glutamate Diacetate"}
    ) / BATCH_SIZE * 100.0
    if chelator_pct >= 0.10:
        flags.append("Cu×chelator(>=0.10%)")
    # Assume thickeners added after liposome payload per latest SOP
    polymer_pre_exposure = False
    if polymer_pre_exposure:
        flags.append("Cu×anionic_polymer(sequence)")
    bakuchiol_pct = next((item.grams for item in formula if item.name == "Bakuchiol"), 0.0) / BATCH_SIZE * 100.0
    if bakuchiol_pct >= 1.5:
        flags.append("Irritation_stack (Bakuchiol≥1.5%)")
    return flags


def build_scorecard() -> Dict[str, object]:
    formula = build_formula()
    total_mass = mass_balance(formula)
    mass_ok = abs(total_mass - BATCH_SIZE) < 0.1
    nlc_fraction = compute_nlc_fraction(formula)
    required_hlb, blend_hlb = compute_hlb(formula)
    hlb_delta = blend_hlb - required_hlb
    preservation = preservation_metrics(formula)
    sensory = sensory_score(formula)
    cost = cost_score()
    ee = ee_proxy_scores()
    color = color_stability_proxy()
    rheology = electrolyte_equivalent_risk(formula)
    net = net_score(preservation, sensory, cost, ee, color, rheology, mass_ok, hlb_delta, nlc_fraction)
    weight_lookup = {item.name: item.grams for item in formula}
    return {
        "mass_balance_ok": mass_ok,
        "total_mass_g": round(total_mass, 3),
        "pH_target": 5.80,
        "pH_trim_plan_g": {
            "10% citrate": "0.3–0.5 g increments",
            "10% NaHCO3": "0.3–0.5 g increments",
        },
        "rheology_risk": rheology,
        "nlc_solid_fraction_est": round(nlc_fraction, 3),
        "hlb": {
            "required": round(required_hlb, 2),
            "blend": {
                "Olivem 1000_g": round(weight_lookup.get("Olivem 1000", 0.0), 2),
                "Montanov 202_g": round(weight_lookup.get("Montanov 202", 0.0), 2),
                "Glyceryl Stearate_g": round(weight_lookup.get("Glyceryl Stearate", 0.0), 2),
                "blend_hlb": round(blend_hlb, 2),
            },
            "delta": round(hlb_delta, 2),
            "ASSUMPTION": "Oil-phase required HLB values approximate; needs vendor confirmation.",
        },
        "preservation": preservation,
        "compat_flags": compat_flags(formula),
        "sensory_score": round(sensory, 1),
        "cost_score": round(cost, 1),
        "ee_proxy": {k: round(v, 1) for k, v in ee.items()},
        "color_stability_proxy": round(color, 1),
        "net_score": net,
        "phase_a_nonvolatile_pct": round(
            sum(item.grams for item in formula if item.phase == "A" and item.name != "Distilled Water") / BATCH_SIZE * 100.0,
            3
        ),
    }


if __name__ == "__main__":
    humectant_names = {
        "Saccharide Isomerate",
        "Honeyquat PF",
        "DL-Panthenol",
        "Pentylene Glycol",
        "1,3-Propanediol",
        "Niacinamide",
        "N-Acetyl Glucosamine",
        "Sodium PCA",
        "Sodium Polyglutamate",
        "Ectoin",
        "Fructo-Oligosaccharides",
        "Inulin",
        "Beta-Glucan",
        "Sodium Hyaluronate 3-5 kDa",
        "Sodium Hyaluronate ~300 kDa",
        "Galactomyces Ferment Filtrate (A)",
        "Galactomyces Ferment Filtrate (C)",
        "Galactomyces Ferment Filtrate (D)",
        "Lactobacillus Ferment Lysate",
        "Bifidobacterium Longum Lysate",
    }

    print("=== Baseline ===")
    baseline = build_scorecard()
    print(json.dumps(baseline, indent=2))
    baseline_formula = build_formula()
    humectant_baseline = sum(item.grams for item in baseline_formula if item.name in humectant_names)
    print(f"Humectant load baseline: {humectant_baseline / BATCH_SIZE * 100:.1f}%")

    print("\n=== Modified ===")
    iter1_formula = build_iteration1_formula()
    total_mass_iter1 = mass_balance(iter1_formula)
    print(f"Iteration 1 mass: {total_mass_iter1:.2f} g (target 515 g)")
    original_builder = build_formula
    try:
        globals()["build_formula"] = lambda: iter1_formula
        iter1_card = build_scorecard()
    finally:
        globals()["build_formula"] = original_builder
    print(json.dumps(iter1_card, indent=2))
    humectant_iter1 = sum(item.grams for item in iter1_formula if item.name in humectant_names)
    print(f"Humectant load iteration 1: {humectant_iter1 / BATCH_SIZE * 100:.1f}%")

    print("\n=== Iteration 2 Candidate ===")
    iter2_formula = build_iteration2_formula()
    total_mass_iter2 = mass_balance(iter2_formula)
    print(f"Iteration 2 mass: {total_mass_iter2:.2f} g (target 515 g)")
    try:
        globals()["build_formula"] = lambda: iter2_formula
        iter2_card = build_scorecard()
    finally:
        globals()["build_formula"] = original_builder
    print(json.dumps(iter2_card, indent=2))
    humectant_iter2 = sum(item.grams for item in iter2_formula if item.name in humectant_names)
    print(f"Humectant load iteration 2: {humectant_iter2 / BATCH_SIZE * 100:.1f}%")

    print("\n=== Iteration 3 Candidate ===")
    iter3_formula = build_iteration3_formula()
    total_mass_iter3 = mass_balance(iter3_formula)
    print(f"Iteration 3 mass: {total_mass_iter3:.2f} g (target 515 g)")
    try:
        globals()["build_formula"] = lambda: iter3_formula
        iter3_card = build_scorecard()
    finally:
        globals()["build_formula"] = original_builder
    print(json.dumps(iter3_card, indent=2))
    humectant_iter3 = sum(item.grams for item in iter3_formula if item.name in humectant_names)
    print(f"Humectant load iteration 3: {humectant_iter3 / BATCH_SIZE * 100:.1f}%")
