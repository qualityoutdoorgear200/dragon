
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab_size", type=int, default=50000)
    ap.add_argument("--byte_level", action="store_true")
    args = ap.parse_args()
    if args.byte_level:
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.normalizer = normalizers.Sequence([])
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=["[UNK]"])
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
    else:
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=["[UNK]"])
    tok.train([args.input], trainer=trainer)
    tok.save(args.out)
    print(f"Saved tokenizer to {args.out} (vocab_size={tok.get_vocab_size()})")
if __name__ == "__main__":
    main()
