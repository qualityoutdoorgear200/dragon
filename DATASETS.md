
# DATASETS for 300M pretraining @ Chinchilla scale (≈20–26 tokens per parameter)

This document proposes **high-quality, reasoning‑centric** corpora available on Hugging Face. The goal is ~**7–8B tokens** for a 0.3B (300M) parameter model.

## Rule of thumb
- **Chinchilla** (Hoffmann et al., 2022): compute‑optimal tokens are ≈ **20× parameters (in B)** → **~6B tokens** for 0.3B; Replications report ≈**25.6×** → **~7.7B tokens**. Target **7–8B** to be safe.

## Recommended pretraining mixture (knowledge / reasoning biased)
| Bucket | HF dataset IDs (examples) | Rationale | Weight (tokens) |
|---|---|---|---|
| **Educational web** | `HuggingFaceFW/fineweb-edu` | QA/explanatory tone, higher signal‑to‑noise vs general web | **50%** |
| **Scientific** | `armanc/scientific_papers` (`arxiv`,`pubmed`) | Formal reasoning, technical vocabulary | **15%** |
| **Math** | `open-web-math/open-web-math` | Symbolic/math reasoning | **10%** |
| **Encyclopedic** | `wikipedia` (en snapshot) | Factual backbone, clean prose | **5%** |
| **Legal** | `pile-of-law/pile-of-law` | Formal argumentative structure, statutory logic | **5%** |
| **Code** | `bigcode/the-stack-v2-dedup` (or `train-*-ids`) | Algorithmic structure, error‑intolerant patterns | **10%** |
| **Q&A (tech)** | `common-pile/stackexchange` (filtered to math/cs) | Structured explanations & solutions | **5%** |

> Adjust % to fit availability and storage. The above avoids social media/news; still *human‑authored*, but with more neutral, expository and technical style.

### Notes & licensing
- **FineWeb‑Edu / FineWeb‑2**: ODC‑BY; includes quality filtering and MinHash dedup.  
- **Dolma** (AI2) is a strong alternative/augment; huge (3T tokens).  
- **The Stack v2**: provenance & licenses matter; use `*-dedup` and respect attribution.  
- **Avoid copyrighted book corpora** (e.g., Books3). Stick to open/ODC‑BY sources.

## Dedup & filtering
Use **DataTrove** for scalable processing and MinHash‑LSH dedup. For small‑scale runs `text-dedup` works well. Deduplicate both **within** and **across** sources to avoid leakage and reduce overfitting.

- Near‑dup: MinHash (5‑grams), Jaccard ≈ 0.8; shard by source, then cross‑source pass.
- Language/format filters: drop boilerplate, lists, repeated lines; retain doc length 200–20k chars (tune per source).
- PII: anonymize emails/IPs.

## Token target and steps
Let `T_target` be 7.5e9 tokens (≈25×0.3e9). If your effective global tokens/step is `G = seq_len * global_batch_size` then required steps:  
`steps ≈ T_target / G`

Examples (seq_len=2048):
- G=262,144 (batch=128) → steps ≈ 28,640
- G=524,288 (batch=256) → steps ≈ 14,320
Use gradient accumulation to reach the desired `G` on limited GPUs.

## Evaluation suites
Keep held‑out splits for: **MMLU**, **HellaSwag**, **ARC‑Challenge**, **GSM8K**, **MATH**; plus long‑context perplexity curves.
