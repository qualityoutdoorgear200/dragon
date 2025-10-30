
from typing import Optional
def load_tokenizer(name_or_path: str):
    """
    Returns a tokenizer object exposing .encode(text).ids and .get_vocab_size().
    Tries tokenizers.Tokenizer (JSON) first; then transformers.AutoTokenizer.
    """
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(name_or_path)
        class Wrap:
            def encode(self, text):
                return type("Enc", (), {"ids": tok.encode(text).ids})

            def encode_batch(self, texts):
                encs = tok.encode_batch(texts)
                return [type("Enc", (), {"ids": enc.ids}) for enc in encs]

            def get_vocab_size(self):
                return tok.get_vocab_size()
        return Wrap()
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        class WrapTF:
            def encode(self, text):
                ids = tok(text, add_special_tokens=False)["input_ids"]
                return type("Enc", (), {"ids": ids})

            def encode_batch(self, texts):
                outputs = tok(texts, add_special_tokens=False)
                input_ids = outputs["input_ids"]
                return [type("Enc", (), {"ids": ids}) for ids in input_ids]

            def get_vocab_size(self):
                return int(tok.vocab_size)
        return WrapTF()
    except Exception as e:
        raise RuntimeError(f"Could not load tokenizer from '{name_or_path}': {e}")
