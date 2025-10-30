
#!/usr/bin/env bash
set -e
python -m dragonwave.train --config configs/base_bdh.yaml
python -m dragonwave.train --config configs/a_only.yaml
python -m dragonwave.train --config configs/a_b.yaml
python -m dragonwave.train --config configs/a_c.yaml
python -m dragonwave.train --config configs/a_b_c_ei.yaml
python -m dragonwave.train --config configs/full_all.yaml
