#!/usr/bin/env python3
"""Reference log-probs of the torch-built tiny Qwen3.6 fixture, in the sylph
`ppl-dump v1` format (lossless_oracle.md).

    python3 07_Tests/SystemTest/make_tiny_ref_logprobs.py <model_dir> <ref_full.json> [--out <dump.tsv>] [--topk 5]

Teacher-forced f32 forward over `full_ids` (the prompt plus the greedy
continuation make_qwen36_tiny.py recorded); for every position from
len(prompt_ids) on, the log-prob of the actual next token and the top-k, exactly
what the engine's PPL=1 PPL_DUMP writes, so compare_logprobs.py reads both.
Needs torch + transformers (tools/oracle-requirements.txt); fp32 on the CPU.
"""
import json
import math
import sys
from pathlib import Path


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    model_dir, ref_path = argv[0], argv[1]
    out = None; topk = 5
    for i, a in enumerate(argv):
        if a == "--out": out = argv[i + 1]
        if a == "--topk": topk = int(argv[i + 1])
    out = out or str(Path(model_dir) / "ref_logprobs.tsv")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        sys.exit(f"missing deps: {exc}. pip install -r c/tools/oracle-requirements.txt")
    ref = json.loads(Path(ref_path).read_text())
    prompt, full = ref["prompt_ids"], ref["full_ids"]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    ids = torch.tensor([full], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids).logits[0].float()           # [T, V]
    logp = torch.log_softmax(logits, dim=-1)
    V = logp.shape[-1]
    lines = [f"# sylph ppl-dump v1 model={model_dir} vocab={V} scored={len(full) - len(prompt)} topk={topk}"]
    for pos in range(len(prompt), len(full)):
        row = logp[pos - 1]                              # logits after position pos-1 predict full[pos]
        tgt = full[pos]
        lp = row[tgt].item()
        top = torch.topk(row, topk)
        tail = f" {lp:.7g}" + "".join(f" {int(i)}:{float(v):.7g}" for v, i in zip(top.values, top.indices))
        lines.append(f"{pos}\t{tgt}\t{lp:.7g}\t{tail}")
    Path(out).write_text("\n".join(lines) + "\n")
    nll = -sum(float(l.split("\t")[2]) for l in lines[1:]) / (len(lines) - 1)
    print(f"{out}: {len(lines) - 1} positions, TF-NLL {nll:.6f} (ppl {math.exp(nll):.4f})")


if __name__ == "__main__":
    main(sys.argv[1:])
