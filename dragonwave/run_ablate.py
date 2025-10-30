import argparse, yaml, csv, os, subprocess, itertools


def load_existing(path, keys):
    existing = set()
    if not os.path.exists(path):
        return existing
    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if header is None:
            return existing
        key_count = len(keys)
        for row in reader:
            if len(row) >= key_count:
                existing.add(tuple(row[:key_count]))
    return existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing output and start a fresh sweep")
    args = ap.parse_args()
    with open(args.grid, "r") as f:
        grid = yaml.safe_load(f)
    keys = sorted(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    completed = set()
    if not args.fresh:
        completed = load_existing(args.out, keys)

    mode = "w" if args.fresh or not os.path.exists(args.out) else "a"
    with open(args.out, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(keys + ["status"])
        for vals in combos:
            key = tuple(str(v) for v in vals)
            if key in completed:
                continue
            env = os.environ.copy()
            for k, v in zip(keys, vals):
                env[k] = str(v)
            cfg = env.get("CFG", "configs/a_only.yaml")
            try:
                subprocess.run(
                    ["python", "-m", "dragonwave.train", "--config", cfg],
                    check=True,
                    env=env,
                )
                status = "ok"
            except subprocess.CalledProcessError as err:
                status = f"fail:{err.returncode}"
            writer.writerow(list(vals) + [status])
            csvfile.flush()
            try:
                os.fsync(csvfile.fileno())
            except OSError:
                pass


if __name__ == "__main__":
    main()
