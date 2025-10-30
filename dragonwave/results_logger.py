
import os, json, csv
from .utils import config_fingerprint, now_ts

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def dump_config(cfg, save_dir):
    ensure_dir(save_dir)
    data = cfg.__dict__.copy()
    data = {k: getattr(cfg, k).__dict__ if hasattr(getattr(cfg,k), "__dict__") else getattr(cfg,k) for k in cfg.__dict__}
    fid = config_fingerprint(data)
    path = os.path.join(save_dir, f"config_{fid}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return fid, path

def log_metrics(metrics: dict, save_dir: str, fid: str, mode: str = "train"):
    ensure_dir(save_dir)
    jpath = os.path.join(save_dir, f"metrics_{mode}_{fid}.jsonl")
    with open(jpath, "a") as f:
        f.write(json.dumps({"ts": now_ts(), **metrics}) + "\n")
    return jpath
