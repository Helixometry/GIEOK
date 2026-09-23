"""Build a tiny local Hugging Face Qwen2 checkpoint + word-level tokenizer for offline tests."""
from __future__ import annotations

import os
import re


def make_tiny_qwen2(directory: str, texts) -> tuple[str, str]:
    """Returns (model_dir, tokenizer_dir)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t)})
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    Qwen2ForCausalLM(config).save_pretrained(directory)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_dir = os.path.join(directory, "tokenizer")     # kept apart so it is not read as a Qwen2 tokenizer
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(tok_dir)
    return directory, tok_dir
