import os
import random
from array import array
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset

DEFAULT_CACHE_SUFFIX = ".tok"
BIN_SUFFIX = ".bin"
IDX_SUFFIX = ".idx"


def _cache_paths(text_path: str, cache_path: Optional[str]) -> Tuple[Path, Path]:
    if cache_path:
        prefix = Path(cache_path)
    else:
        prefix = Path(text_path).with_suffix(DEFAULT_CACHE_SUFFIX)
    bin_path = prefix.with_suffix(BIN_SUFFIX)
    idx_path = prefix.with_suffix(IDX_SUFFIX)
    return bin_path, idx_path


def _write_ids_to_file(ids, fout):
    if not ids:
        return 0
    arr = array("I", ids)
    arr.tofile(fout)
    return len(ids)


def ensure_token_cache(
    text_path: str,
    tokenizer,
    cache_path: Optional[str] = None,
    batch_lines: int = 2048,
) -> Tuple[Path, int]:
    """
    Tokenize `text_path` into a memory-mapped binary cache.
    Returns (bin_path, total_token_count).
    """
    bin_path, idx_path = _cache_paths(text_path, cache_path)
    if bin_path.exists() and idx_path.exists():
        total = int(idx_path.read_text().strip())
        return bin_path, total

    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    last_report = -1
    buffer = []
    with open(text_path, "r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        for line in fin:
            buffer.append(line.rstrip("\n") + "\n")
            if len(buffer) >= batch_lines:
                encodings = tokenizer.encode_batch(buffer)
                for enc in encodings:
                    total_tokens += _write_ids_to_file(enc.ids, fout)
                buffer.clear()
                current_report = total_tokens // 50000000
                if current_report > last_report:
                    print(f"[tokenize] {total_tokens:,} tokens", flush=True)
                    last_report = current_report
        if buffer:
            encodings = tokenizer.encode_batch(buffer)
            for enc in encodings:
                total_tokens += _write_ids_to_file(enc.ids, fout)
            current_report = total_tokens // 50000000
            if current_report > last_report:
                print(f"[tokenize] {total_tokens:,} tokens", flush=True)
                last_report = current_report

    idx_path.write_text(str(total_tokens))
    print(f"[tokenize] cached {total_tokens:,} tokens -> {bin_path}", flush=True)
    return bin_path, total_tokens


class TokenMemmapStreamingDataset(IterableDataset):
    def __init__(self, bin_path: Path, total_tokens: int, seq_len: int, seed: int = 0):
        self.bin_path = bin_path
        self.total_tokens = total_tokens
        self.seq_len = seq_len
        self.seed = seed
        self.is_streaming = True

    def __iter__(self):
        tokens = np.memmap(self.bin_path, dtype=np.uint32, mode="r")
        length = self.total_tokens
        seq_len = self.seq_len
        worker_info = torch.utils.data.get_worker_info()
        rng_seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(rng_seed)
        max_start = max(1, length - seq_len - 1)
        while True:
            start = rng.randrange(0, max_start)
            segment = np.asarray(tokens[start : start + seq_len + 1], dtype=np.uint32)
            x = torch.from_numpy(segment[:-1].astype(np.int64))
            y = torch.from_numpy(segment[1:].astype(np.int64))
            yield x, y


class TokenMemmapEvalDataset(Dataset):
    def __init__(
        self,
        bin_path: Path,
        total_tokens: int,
        seq_len: int,
        max_sequences: Optional[int] = None,
    ):
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        stride = seq_len
        available = (total_tokens - (seq_len + 1)) // stride
        if available <= 0:
            available = 1
        if max_sequences is None or max_sequences <= 0:
            self.length = available
        else:
            self.length = min(available, max_sequences)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        stride = self.seq_len
        start = idx * stride
        segment = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.uint32)
        x = torch.from_numpy(segment[:-1].astype(np.int64))
        y = torch.from_numpy(segment[1:].astype(np.int64))
        return x, y


def build_token_dataset(
    path: str,
    tokenizer,
    seq_len: int,
    *,
    cache_path: Optional[str] = None,
    seed: int = 0,
    mode: str = "train",
    num_samples: Optional[int] = None,
):
    """
    Build a token dataset backed by a memory-mapped cache.

    mode: "train" -> random streaming dataset
          "eval"  -> sequential dataset limited by num_samples
    """
    bin_path, total_tokens = ensure_token_cache(path, tokenizer, cache_path)
    if total_tokens <= seq_len + 1:
        raise RuntimeError(
            f"Tokenized corpus {path} is too small ({total_tokens} tokens) for seq_len {seq_len}."
        )

    if mode == "eval":
        dataset = TokenMemmapEvalDataset(bin_path, total_tokens, seq_len, max_sequences=num_samples)
    else:
        dataset = TokenMemmapStreamingDataset(bin_path, total_tokens, seq_len, seed=seed)

    vocab_size = tokenizer.get_vocab_size()
    return dataset, vocab_size
