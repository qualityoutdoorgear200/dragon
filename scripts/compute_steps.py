
#!/usr/bin/env python3
import argparse, math, yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_tokens', type=float, required=True, help='e.g., 7.5e9')
    ap.add_argument('--seq_len', type=int, default=2048)
    ap.add_argument('--global_batch', type=int, default=128, help='effective per-step sequences (after grad accumulation across GPUs)')
    args = ap.parse_args()
    G = args.seq_len * args.global_batch
    steps = math.ceil(args.target_tokens / G)
    print(f"Tokens/step: {G:,} | Required steps: {steps:,}")
    print("Examples:")
    for gb in [64,128,256,512]:
        G2 = args.seq_len * gb
        s2 = math.ceil(args.target_tokens / G2)
        print(f'  global_batch={gb:>4} -> tokens/step={G2:>9,} -> steps={s2:>9,}')

if __name__ == '__main__':
    main()
