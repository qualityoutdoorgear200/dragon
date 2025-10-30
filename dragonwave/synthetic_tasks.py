
import random, torch

def vocab_for(task):
    if task == "synthetic_paren":
        chars = "()"
    elif task == "synthetic_count":
        chars = "0123456789"
    else:
        chars = "abcdefghijklmnopqrstuvwxyz "
    return sorted(list(set(chars)))

def build_synthetic(task: str, seq_len: int = 64, num_samples: int = 4096):
    if task == "synthetic_copy":
        return build_copy_task(seq_len, num_samples)
    if task == "synthetic_paren":
        return build_paren_task(seq_len, num_samples)
    if task == "synthetic_count":
        return build_count_task(seq_len, num_samples)
    raise ValueError(f"Unknown synthetic task: {task}")

def build_copy_task(seq_len, num_samples):
    # Input: random lowercase letters; target is shifted by 1 (copy/recall)
    chars = "abcdefghijklmnopqrstuvwxyz "
    stoi = {c:i for i,c in enumerate(sorted(set(chars)))}
    xs, ys = [], []
    for _ in range(num_samples):
        s = "".join(random.choice(chars) for _ in range(seq_len+1))
        idx = torch.tensor([stoi[c] for c in s], dtype=torch.long)
        xs.append(idx[:-1]); ys.append(idx[1:])
    return list(zip(xs, ys)), stoi

def build_paren_task(seq_len, num_samples):
    # Balanced parentheses generation; target next char
    chars = "()"
    stoi = {c:i for i,c in enumerate(chars)}
    xs, ys = [], []
    for _ in range(num_samples):
        s = []
        bal = 0
        for t in range(seq_len+1):
            if bal == 0:
                c = "("
            else:
                c = "(" if random.random() < 0.5 else ")"
            s.append(c)
            bal += 1 if c=="(" else -1
            if bal < 0: bal = 0
        idx = torch.tensor([stoi[c] for c in s], dtype=torch.long)
        xs.append(idx[:-1]); ys.append(idx[1:])
    return list(zip(xs, ys)), stoi

def build_count_task(seq_len, num_samples):
    # Running digit-sum mod 10; target is next digit of running sum
    digits = "0123456789"
    stoi = {c:i for i,c in enumerate(digits)}
    xs, ys = [], []
    for _ in range(num_samples):
        s = [random.choice(digits) for _ in range(seq_len)]
        run = 0; target = []
        for ch in s:
            d = int(ch)
            run = (run + d) % 10
            target.append(str(run))
        # For next-token shift
        s2 = s + [random.choice(digits)]
        idx_x = torch.tensor([stoi[c] for c in s2[:-1]], dtype=torch.long)
        idx_y = torch.tensor([stoi[c] for c in target], dtype=torch.long)
        xs.append(idx_x); ys.append(idx_y)
    return list(zip(xs, ys)), stoi
