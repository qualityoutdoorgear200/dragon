import random
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
TRAIN_PATH = OUTPUT_DIR / "mix_full_train.txt"
VALID_PATH = OUTPUT_DIR / "mix_full_valid.txt"

TARGET_TOKENS = 520_000_000
VALID_RATIO = 0.02
MIN_TOKENS = 20

SOURCES = [
    {"name": "HuggingFaceFW/fineweb", "config": None, "split": "train", "quota": 0.45, "label": "fineweb"},
    {"name": "HuggingFaceFW/fineweb-edu", "config": None, "split": "train", "quota": 0.10, "label": "fineweb-edu"},
    {"name": "open-web-math/open-web-math", "config": None, "split": "train", "quota": 0.10, "label": "open-web-math"},
    {"name": "common-pile/stackexchange", "config": None, "split": "train", "quota": 0.15, "label": "stackexchange"},
    {"name": "codeparrot/codeparrot-clean-train", "config": None, "split": "train", "quota": 0.20, "label": "code"},
]

PRIORITY_COLS = ["text", "content", "body", "article", "document"]

def pick_text(row):
    for key in PRIORITY_COLS:
        val = row.get(key)
        if isinstance(val, str) and len(val.strip()) > 0:
            return val
    # fallback: first str field
    for val in row.values():
        if isinstance(val, str) and len(val.strip()) > 0:
            return val
    return None

random.seed(42)

with TRAIN_PATH.open("w", encoding="utf-8") as f_train, VALID_PATH.open("w", encoding="utf-8") as f_valid:
    total_tokens = 0
    stats = []
    for src in SOURCES:
        quota_tokens = int(TARGET_TOKENS * src["quota"])
        collected = 0
        kept_docs = 0
        rejected = 0
        print(f"[mix] sampling {src['label']} (target tokens {quota_tokens:,})")
        try:
            ds = load_dataset(src["name"], src.get("config"), split=src.get("split", "train"), streaming=True)
        except Exception as e:
            print(f"[mix] SKIP {src['label']}: {e}")
            continue
        for row in ds:
            text = pick_text(row)
            if not text:
                rejected += 1
                continue
            text = text.strip()
            if len(text) < 200:
                rejected += 1
                continue
            tokens = len(text.split())
            if tokens < MIN_TOKENS:
                rejected += 1
                continue
            target_file = f_valid if random.random() < VALID_RATIO else f_train
            target_file.write(text + "\n\n")
            collected += tokens
            total_tokens += tokens
            kept_docs += 1
            if collected >= quota_tokens:
                break
        stats.append((src['label'], collected, kept_docs, rejected))
        print(f"[mix] {src['label']} collected tokens {collected:,} from {kept_docs} docs (rejected {rejected})")
        if total_tokens >= TARGET_TOKENS:
            print("[mix] reached target token budget")
            break

print("[mix] total tokens:", total_tokens)
print("[mix] stats:")
for label, tokens, docs, rej in stats:
    print(f"  - {label}: tokens={tokens:,} docs={docs} rejected={rej}")
