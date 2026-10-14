#!/usr/bin/env python3
"""Write a tiny GLM-style tokenizer.json (byte-level BPE, cl100k pre-tokenizer) for the GLM
GGUF tokenizer test (gguf_hardening.md case 5).

256 byte symbols (GPT-2 byte -> unicode mapping, as tok.h's tk_build_bytemap), 58 merges and
six control tokens (`<|endoftext|>`, `[MASK]`, `[gMASK]`, `<sop>`, `<|user|>`, `<|assistant|>`,
`<|observation|>` -- the last is dropped when the vocab would exceed --vocab) so that the id
space is exactly --vocab wide (default 320, the GLM fixture's --vocab). The pre_tokenizer is a
Split on the cl100k pattern + ByteLevel(add_prefix_space=false), which tok.h recognises as the
GLM family (no \\p{Lu} classes -> not o200k; a Split present -> not GPT-2).

    python3 07_Tests/IntegrationTest/make_tiny_glm_tok.py <tokenizer.json> [--vocab N]

Standard library only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_expert_streaming import byte_symbols  # noqa: E402

CL100K = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
          r"|\s*[\r\n]+|\s+(?!\S)|\s+")
CONTROL = ["<|endoftext|>", "[MASK]", "[gMASK]", "<sop>", "<|user|>", "<|assistant|>", "<|observation|>"]


def write(path, vocab_size=320):
    syms = byte_symbols()
    vocab = {s: i for i, s in enumerate(syms)}
    G = syms[32]
    pairs = ["t h", "th e", "i n", "e r", "a n", "r e", "o n", "a t", "e n", "o r", "e s", "i s", "i t", "a l", "a r", "s t",
             "o u", "n d", "in g", "l e", "o f", "a s", "h e", "t o", "e d", "i c", "l l", "r o", "m e", "d e", "c o", "n e",
             f"{G} t", f"{G} a", f"{G} th", f"{G} s", f"{G} o", f"{G} w", f"{G} i", f"{G} c", f"{G} b", f"{G} f", f"{G} m",
             f"{G} d", f"{G} p", f"{G} h", f"{G} l", f"{G} n", f"{G} e", f"{G} r", f"{G} g", f"{G} u", f"{G} y", f"{G} k",
             f"{G}th e", f"{G}a n", f"{G}o f", f"{G}t o", f"{G}i n", f"{G}i s", f"{G}i t", f"{G}a nd", f"{G}th at"]
    n_ctrl = min(len(CONTROL), max(1, vocab_size - 256 - 1))
    merges = []
    for p in pairs:
        a, b = p.split(" ")
        if a not in vocab or b not in vocab or (a + b) in vocab:
            continue
        if len(vocab) >= vocab_size - n_ctrl:
            break
        vocab[a + b] = len(vocab); merges.append(p)
    added = []
    for name in CONTROL[:n_ctrl]:
        if len(vocab) + len(added) >= vocab_size:
            break
        added.append({"id": len(vocab) + len(added), "content": name, "special": True})
    assert len(vocab) + len(added) == vocab_size, (len(vocab), len(added), vocab_size)
    tok = {"model": {"type": "BPE", "vocab": vocab, "merges": merges, "ignore_merges": True},
           "added_tokens": added,
           "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
               {"type": "Split", "pattern": {"Regex": CL100K}, "behavior": "Isolated", "invert": False},
               {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
           "decoder": {"type": "ByteLevel"}}
    Path(path).write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    return len(vocab), len(merges), len(added)


def main(argv):
    if not argv:
        sys.exit(__doc__)
    size = 320
    for i, a in enumerate(argv):
        if a == "--vocab":
            size = int(argv[i + 1])
    nv, nm, na = write(argv[0], size)
    print(f"{argv[0]}: {nv} pieces ({nm} merges) + {na} control tokens = {nv + na} ids, pre-tokenizer cl100k")


if __name__ == "__main__":
    main(sys.argv[1:])
