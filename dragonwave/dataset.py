
import random, torch, os
from .synthetic_tasks import build_synthetic

def build_char_dataset_from_text(text: str, seq_len: int = 128, num_samples: int = 2048):
    vocab = sorted(list(set(text)))
    stoi = {c:i for i,c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    samples = []
    n = len(ids)
    for _ in range(num_samples):
        i = random.randrange(0, n-seq_len-1)
        samples.append((ids[i:i+seq_len], ids[i+1:i+seq_len+1]))
    return samples, stoi

def load_text(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

def build_dataset(kind: str, text_path: str, seq_len: int, num_samples: int):
    if kind == "text":
        if text_path and os.path.exists(text_path):
            return build_char_dataset_from_text(load_text(text_path), seq_len=seq_len, num_samples=num_samples)
        else:
            toy = "to hatch a dragon, feed it facts and questions. " * 256
            return build_char_dataset_from_text(toy, seq_len=seq_len, num_samples=num_samples)
    else:
        return build_synthetic(kind, seq_len=min(seq_len, 96), num_samples=num_samples)
