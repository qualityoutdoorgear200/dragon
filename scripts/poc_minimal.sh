
#!/usr/bin/env bash
set -euo pipefail

# Minimal proof-of-concept runs
python -m dragonwave.train --config configs/base_bdh.yaml
python -m dragonwave.train --config configs/a_only.yaml

# Quick synthetic checks
python - <<'PY'
import yaml, subprocess
for task in ["synthetic_copy","synthetic_paren","synthetic_count"]:
    with open("configs/a_only.yaml") as f: cfg=yaml.safe_load(f)
    cfg["train"]["dataset"]=task
    cfg["train"]["save_dir"]=f"runs/poc_{task}"
    path=f"configs/poc_{task}.yaml"
    with open(path,"w") as g: yaml.safe_dump(cfg,g)
    print("Running", path)
    subprocess.run(["python","-m","dragonwave.train","--config",path], check=True)
PY
