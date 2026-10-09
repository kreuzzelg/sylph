#!/usr/bin/env python3
"""Reference log-probs of the torch-built tiny Qwen3.6 fixture, in the sylph
`ppl-dump v1` format (lossless_oracle.md).

    python3 07_Tests/SystemTest/make_tiny_ref_logprobs.py <model_dir> <ref_full.json> [--out <dump.tsv>] [--topk 5]
                                                        [--windows N] [--window-len 512] [--seed 7] [--full K]

With --windows N, also writes <model_dir>/windows/win_<k>.ids (one id per line, seeded
random ids of the model's vocabulary) and win_<k>.tsv (teacher-forced torch log-probs of
positions 257…511, llama.cpp's protocol: first half context), plus win_<k>.full
(`full-logprob v1`) for the first K windows — the E2 / exact-KL inputs of the CI subset
(equivalence.md case 0).

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
    out = None; topk = 5; windows = 0; wlen = 512; seed = 7; full = 1
    for i, a in enumerate(argv):
        if a == "--out": out = argv[i + 1]
        if a == "--topk": topk = int(argv[i + 1])
        if a == "--windows": windows = int(argv[i + 1])
        if a == "--window-len": wlen = int(argv[i + 1])
        if a == "--seed": seed = int(argv[i + 1])
        if a == "--full": full = int(argv[i + 1])
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
    def dump(seq, first_scored, path, full_path=None):
        """Teacher-forced log-probs of seq[first_scored:] in the engine's ppl-dump v1
        format (tail ` <lp> <k> <id> <lp> …`, as decode_batch.h prints it); optionally the
        whole log-softmax rows as `full-logprob v1`."""
        ids = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            logits = model(ids).logits[0].float()        # [T, V]
        logp = torch.log_softmax(logits, dim=-1)
        V = logp.shape[-1]
        lines = [f"# sylph ppl-dump v1 model={model_dir} vocab={V} scored={len(seq) - first_scored} topk={topk}"]
        fh = open(full_path, "wb") if full_path else None
        if fh: fh.write(f"# sylph full-logprob v1 vocab={V} positions={len(seq) - first_scored}\n".encode())
        for pos in range(first_scored, len(seq)):
            row = logp[pos - 1]                          # logits after position pos-1 predict seq[pos]
            tgt = seq[pos]; lp = row[tgt].item()
            top = torch.topk(row, topk)
            tail = f" {lp:.6f} {topk}" + "".join(f" {int(i)} {float(v):.6f}" for v, i in zip(top.values, top.indices))
            lines.append(f"{pos}\t{tgt}\t{lp:.7g}\t{tail}")
            if fh: fh.write(row.cpu().numpy().astype("<f4").tobytes())
        if fh: fh.close()
        Path(path).write_text("\n".join(lines) + "\n")
        nll = -sum(float(l.split("\t")[2]) for l in lines[1:]) / (len(lines) - 1)
        print(f"{path}: {len(lines) - 1} positions, TF-NLL {nll:.6f} (ppl {math.exp(nll):.4f})")
        return V

    V = dump(full, len(prompt), out)
    if windows > 0:
        import random
        rng = random.Random(seed)
        wdir = Path(model_dir) / "windows"; wdir.mkdir(exist_ok=True)
        for k in range(windows):
            seq = [rng.randrange(V) for _ in range(wlen)]
            (wdir / f"win_{k}.ids").write_text("\n".join(map(str, seq)) + "\n")
            dump(seq, wlen // 2 + 1, wdir / f"win_{k}.tsv", wdir / f"win_{k}.full" if k < full else None)


if __name__ == "__main__":
    main(sys.argv[1:])
