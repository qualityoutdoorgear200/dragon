
import random, os, hashlib, json, time
import torch
import numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, value, beta: float):
        self.value = value
        self.beta = beta
    def update(self, new):
        self.value = (1-self.beta)*self.value + self.beta*new
        return self.value

def batch_entropy(logits):
    p = logits.softmax(dim=-1)
    return -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)  # [B,T]

def config_fingerprint(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]

def now_ts():
    return time.strftime("%Y-%m-%d_%H-%M-%S")
