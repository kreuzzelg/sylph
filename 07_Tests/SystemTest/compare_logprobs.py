#!/usr/bin/env python3
"""Compare two logprob dumps in the sylph `ppl-dump v1` format (lossless_oracle.md, E1 subset).

    python3 07_Tests/SystemTest/compare_logprobs.py <reference.tsv> <candidate.tsv> [--mean 1e-4] [--max 1e-3] [--tf-nll <printed value>]

Format (one header line, then one line per scored position):
    # sylph ppl-dump v1 model=<path> vocab=<V> scored=<N> topk=<k>
    <pos>\\t<target_id>\\t<logprob_target>\\t<tail>        tail = " <lp> <id>:<lp> <id>:<lp> ..."

Checks: same positions and targets; top-1 identical at every position; mean and
max |ΔNLL| within the thresholds; the dump's own -Σlogprob/N equals the engine's
printed TF-NLL (--tf-nll) within 1e-6. Standard library only. Exit 0 iff all pass.
"""
import math
import sys


def read_dump(path):
    header, rows = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                for tok in line.split()[3:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1); header[k] = v
                continue
            parts = line.split("\t")
            pos, tgt, lp = int(parts[0]), int(parts[1]), float(parts[2])
            tail = parts[3].split() if len(parts) > 3 else []
            top = []
            for t in tail[1:]:
                if ":" in t:
                    i, v = t.split(":", 1); top.append((int(i), float(v)))
            rows[pos] = (tgt, lp, top)
    return header, rows


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    ref_path, cand_path = argv[0], argv[1]
    mean_thr, max_thr, tf_nll = 1e-4, 1e-3, None
    for i, a in enumerate(argv):
        if a == "--mean": mean_thr = float(argv[i + 1])
        if a == "--max": max_thr = float(argv[i + 1])
        if a == "--tf-nll": tf_nll = float(argv[i + 1])
    hr, ref = read_dump(ref_path); hc, cand = read_dump(cand_path)
    fails = []
    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond: fails.append(msg)
    check(sorted(ref) == sorted(cand) and len(ref) > 0, f"same {len(ref)} scored positions (ref {len(ref)}, candidate {len(cand)})")
    common = sorted(set(ref) & set(cand))
    check(all(ref[p][0] == cand[p][0] for p in common), "same target ids")
    top1_bad = [p for p in common if ref[p][2] and cand[p][2] and ref[p][2][0][0] != cand[p][2][0][0]]
    check(not top1_bad, f"top-1 identical at every position ({len(top1_bad)} differ: {top1_bad[:8]})")
    deltas = [abs(ref[p][1] - cand[p][1]) for p in common]
    mean_d = sum(deltas) / len(deltas) if deltas else float("nan"); max_d = max(deltas) if deltas else float("nan")
    check(mean_d <= mean_thr, f"mean |ΔNLL| {mean_d:.3e} ≤ {mean_thr:g}")
    check(max_d <= max_thr, f"max |ΔNLL| {max_d:.3e} ≤ {max_thr:g}")
    nll_ref = -sum(ref[p][1] for p in common) / len(common); nll_c = -sum(cand[p][1] for p in common) / len(common)
    print(f"       TF-NLL reference {nll_ref:.6f} · candidate {nll_c:.6f} · ppl {math.exp(nll_ref):.4f} / {math.exp(nll_c):.4f}")
    if tf_nll is not None:
        check(abs(nll_c - tf_nll) <= 1e-6 + 1e-6 * abs(tf_nll), f"candidate dump reproduces the printed TF-NLL {tf_nll:.6f} (dump {nll_c:.6f})")
    print("RESULT:", "FAIL" if fails else "ok", f"({len(fails)} failures)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
