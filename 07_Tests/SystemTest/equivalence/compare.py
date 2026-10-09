#!/usr/bin/env python3
"""Comparer of the equivalence harness (FR-33). Standard library only.

Reads the interchange formats of formats.md, computes E1 (ΔNLL, top-1/top-5
agreement, coarsened and exact KL), E2 (PPL pair, chunk-wise sign test), E3 (prefix
lengths to the first near-tie), calibrates thresholds from a noise floor, and writes
report.md + report.json.

    python3 compare.py --check report.json        # validate a report file
"""
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_logprobs import parse_tail, read_dump  # noqa: E402

PRINT_UNIT = 1e-6   # one unit of the dumps' 7-significant-digit format around -5 nat


# ---- E1 -------------------------------------------------------------------------------
def e1_compare(ref_rows, cand_rows, top5=True):
    """ref_rows/cand_rows: {pos: (tgt, lp, top)} merged over all windows (pos unique)."""
    common = sorted(set(ref_rows) & set(cand_rows))
    if not common:
        return {"n": 0, "error": "no common positions"}
    bad_tgt = [p for p in common if ref_rows[p][0] != cand_rows[p][0]]
    deltas = [(abs(ref_rows[p][1] - cand_rows[p][1]), p) for p in common]
    top1_ok = top5_ok = n_tail = 0
    for p in common:
        rt, ct = ref_rows[p][2], cand_rows[p][2]
        if rt and ct:
            n_tail += 1
            top1_ok += rt[0][0] == ct[0][0]
            top5_ok += len({i for i, _ in rt[:5]} & {i for i, _ in ct[:5]}) == min(5, len(rt), len(ct))
    worst = sorted(deltas, reverse=True)[:8]
    return {"n": len(common), "bad_targets": len(bad_tgt), "mean": sum(d for d, _ in deltas) / len(deltas),
            "max": worst[0][0], "worst": [{"pos": p, "delta": d, "ref": ref_rows[p][1], "cand": cand_rows[p][1]} for d, p in worst],
            "top1": top1_ok / n_tail if n_tail else None, "top5": top5_ok / n_tail if n_tail else None, "n_tail": n_tail,
            "nll_ref": -sum(ref_rows[p][1] for p in common) / len(common), "nll_cand": -sum(cand_rows[p][1] for p in common) / len(common)}


def kl_coarse(top_ref, top_cand):
    """KL of the two distributions coarsened to {ids in both top lists} ∪ {rest}: a lower
    bound of the true KL that needs no full distribution (data-processing inequality)."""
    if not top_ref or not top_cand:
        return None
    r = dict(top_ref); c = dict(top_cand)
    both = [i for i in r if i in c]
    pr = [math.exp(r[i]) for i in both]; qc = [math.exp(c[i]) for i in both]
    p_rest, q_rest = max(1e-300, 1.0 - sum(pr)), max(1e-300, 1.0 - sum(qc))
    kl = sum(p * (math.log(p) - math.log(q)) for p, q in zip(pr, qc) if p > 0) + p_rest * (math.log(p_rest) - math.log(q_rest))
    return max(0.0, kl)


def read_full(path):
    """Yields log-softmax rows (lists of floats) of a full-logprob v1 file."""
    with open(path, "rb") as f:
        hdr = f.readline().decode().strip()
        if not hdr.startswith("# sylph full-logprob v1"):
            raise ValueError(f"{path}: not a full-logprob v1 file")
        kv = dict(t.split("=", 1) for t in hdr.split()[3:] if "=" in t)
        V, N = int(kv["vocab"]), int(kv["positions"])
        for _ in range(N):
            raw = f.read(4 * V)
            if len(raw) < 4 * V:
                raise ValueError(f"{path}: truncated")
            yield struct.unpack(f"<{V}f", raw)


def kl_exact(ref_full, cand_full):
    """Mean and max KL(ref || cand) over the positions of two full-logprob v1 files."""
    kls = []
    for pr, qc in zip(read_full(ref_full), read_full(cand_full)):
        kls.append(max(0.0, sum(math.exp(a) * (a - b) for a, b in zip(pr, qc))))
    if not kls:
        return None
    return {"n": len(kls), "mean": sum(kls) / len(kls), "max": max(kls)}


# ---- E2 -------------------------------------------------------------------------------
def binom_two_sided(k, n):
    """Two-sided exact binomial test p-value for k successes of n at p = 1/2."""
    if n == 0:
        return 1.0
    lo = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * lo)


def e2_compare(chunks_ref, chunks_cand):
    """chunks: [(n_scored, sum_nll)] per window, same windows in the same order."""
    n = min(len(chunks_ref), len(chunks_cand))
    if n == 0:
        return {"n": 0}
    tr = sum(x for x, _ in chunks_ref[:n]); tc = sum(x for x, _ in chunks_cand[:n])
    nll_r = sum(s for _, s in chunks_ref[:n]) / tr; nll_c = sum(s for _, s in chunks_cand[:n]) / tc
    deltas = [chunks_cand[i][1] / chunks_cand[i][0] - chunks_ref[i][1] / chunks_ref[i][0] for i in range(n)]
    pos = sum(1 for d in deltas if d > 0); nz = sum(1 for d in deltas if d != 0)
    return {"n": n, "ppl_ref": math.exp(nll_r), "ppl_cand": math.exp(nll_c), "rel": abs(math.exp(nll_c) - math.exp(nll_r)) / math.exp(nll_r),
            "nll_ref": nll_r, "nll_cand": nll_c, "sign_pos": pos, "sign_n": nz, "sign_p": binom_two_sided(pos, nz),
            "chunks": [{"chunk": i, "n": chunks_ref[i][0], "nll_ref": chunks_ref[i][1] / chunks_ref[i][0], "nll_cand": chunks_cand[i][1] / chunks_cand[i][0]} for i in range(n)]}


# ---- E3 -------------------------------------------------------------------------------
def e3_compare(ref_gens, cand_gens, margin_thr=0.05):
    """gens: list of e3-gen v1 dicts (same prompt order). Prefix = first index where ids
    (or, lacking ids, texts) differ. ok if no divergence, or the reference's top-2
    margin at the divergence is below margin_thr (a near-tie)."""
    out = []
    for r, c in zip(ref_gens, cand_gens):
        by_ids = bool(r.get("ids")) and bool(c.get("ids"))
        a, b = (r["ids"], c["ids"]) if by_ids else (r.get("text", ""), c.get("text", ""))
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        diverged = k < min(len(a), len(b))
        margin = (r.get("margin") or [None] * (k + 1))[k] if diverged and by_ids and k < len(r.get("margin") or []) else None
        if not diverged:
            verdict = "identical"
        elif margin is not None and margin < margin_thr:
            verdict = "near-tie"
        elif margin is None:
            verdict = "diverged (no margin known)"
        else:
            verdict = "diverged"
        out.append({"prompt": r.get("prompt"), "by": "ids" if by_ids else "text", "prefix": k, "len_ref": len(a), "len_cand": len(b),
                    "margin_at_divergence": margin, "verdict": verdict, "ok": verdict in ("identical", "near-tie")})
    return {"prompts": out, "n_ok": sum(1 for o in out if o["ok"]), "n": len(out)}


# ---- thresholds ------------------------------------------------------------------------
def thresholds_from_floor(floor, proposal=None):
    """3 × the measured noise floor, never below one print unit (spec §4.3); the
    proposal values (§4.1) apply where no floor sample exists."""
    prop = {"e1_mean": 1e-4, "e1_max": 1e-3, "top1": 0.995, "kl": 1e-3, "e2_rel": 0.003}
    if proposal:
        prop.update(proposal)
    if not floor:
        return dict(prop, calibrated=False)
    return {"e1_mean": max(3 * floor.get("e1_mean", 0.0), PRINT_UNIT), "e1_max": max(3 * floor.get("e1_max", 0.0), 10 * PRINT_UNIT),
            "top1": min(prop["top1"], 1.0 - 3 * floor.get("e1_top1_disagree", 0.0)) if floor.get("e1_top1_disagree") is not None else prop["top1"],
            "kl": prop["kl"], "e2_rel": max(3 * floor.get("e2_rel", 0.0), 1e-6), "calibrated": True}


# ---- report ----------------------------------------------------------------------------
def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}e}" if (abs(x) < 1e-3 or abs(x) >= 1e4) and x != 0 else f"{x:.{nd + 1}g}"
    return str(x)


def write_report(out_dir, R):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(R, indent=1, default=str) + "\n")
    L = [f"# Equivalence report — {R.get('title', '')}", ""]
    m = R.get("model", {})
    L += ["## Setup", "", f"- model: `{m.get('path')}` sha256 `{m.get('sha256', '?')}`", f"- {m.get('inspect', '')}"]
    for k, v in (R.get("versions") or {}).items():
        L.append(f"- {k}: {v}")
    for k, v in (R.get("host") or {}).items():
        L.append(f"- {k}: {v}")
    L += [f"- sylph cap: {R.get('cap')} · threads: {R.get('threads')} · mode: {R.get('mode')}", ""]
    tg = R.get("tokenizer_gate")
    if tg:
        L += ["## Tokenizer gate", "", f"- {'pass' if tg.get('pass') else 'FAIL'}: {tg.get('n_ids')} ids compared" + (f", first mismatch at {tg.get('first_mismatch')}" if tg.get("first_mismatch") is not None else ""), ""]
    fl = R.get("floor")
    th = R.get("thresholds", {})
    L += ["## Noise floor and thresholds", ""]
    if fl:
        L += [f"- floor E1: mean |ΔNLL| {fmt(fl.get('e1_mean'))}, max {fmt(fl.get('e1_max'))}, top-1 disagreement {fmt(fl.get('e1_top1_disagree'))}; E2 |ΔPPL|/PPL {fmt(fl.get('e2_rel'))} ({', '.join(fl.get('samples', []))})"]
    else:
        L += ["- no floor measured (proposal values of spec §4.1 used)"]
    L += [f"- thresholds ({'calibrated' if th.get('calibrated') else 'proposal'}): E1 mean ≤ {fmt(th.get('e1_mean'))}, max ≤ {fmt(th.get('e1_max'))}, top-1 ≥ {fmt(th.get('top1'))}, KL ≤ {fmt(th.get('kl'))}; E2 |ΔPPL|/PPL ≤ {fmt(th.get('e2_rel'))}", ""]
    e1 = R.get("E1")
    if e1:
        L += ["## E1 — teacher-forced log-probs", "", "| positions | mean \\|ΔNLL\\| | max \\|ΔNLL\\| | top-1 | top-5 | KL exact (mean / max, n) | KL coarsened mean | verdict |", "|---|---|---|---|---|---|---|---|"]
        ke = e1.get("kl_exact") or {}
        L.append(f"| {e1.get('n')} | {fmt(e1.get('mean'))} | {fmt(e1.get('max'))} | {fmt(e1.get('top1'))} | {fmt(e1.get('top5'))} | {fmt(ke.get('mean'))} / {fmt(ke.get('max'))} ({ke.get('n', '—')}) | {fmt(e1.get('kl_coarse_mean'))} | **{'pass' if e1.get('pass') else 'FAIL'}** |")
        if e1.get("worst"):
            L += ["", "Worst positions: " + ", ".join(f"pos {w['pos']} Δ {fmt(w['delta'])} (ref {w['ref']:.4f}, cand {w['cand']:.4f})" for w in e1["worst"][:5])]
        L.append("")
    e2 = R.get("E2")
    if e2 and e2.get("n"):
        L += ["## E2 — perplexity", "", f"| windows | PPL reference | PPL sylph | \\|ΔPPL\\|/PPL | chunks sylph worse | sign test p | verdict |", "|---|---|---|---|---|---|---|",
              f"| {e2['n']} | {e2['ppl_ref']:.4f} | {e2['ppl_cand']:.4f} | {fmt(e2['rel'])} | {e2['sign_pos']} of {e2['sign_n']} | {e2['sign_p']:.3f} | **{'pass' if e2.get('pass') else 'FAIL'}** |", ""]
    e3 = R.get("E3")
    if e3:
        L += ["## E3 — greedy generation", ""]
        for name, res in e3.items():
            L += [f"### vs {name}: {res.get('n_ok')} of {res.get('n')} prompts identical or near-tie — **{'pass' if res.get('pass') else 'FAIL'}**", "",
                  "| prompt | compared by | prefix | ref len | sylph len | margin at divergence | verdict |", "|---|---|---|---|---|---|---|"]
            for o in res.get("prompts", []):
                L.append(f"| {o['prompt']} | {o['by']} | {o['prefix']} | {o['len_ref']} | {o['len_cand']} | {fmt(o['margin_at_divergence'])} | {o['verdict']} |")
            L.append("")
    dv = R.get("deviations")
    if dv:
        L += ["## Deliberate differences (FR-35)", ""]
        for name, d in dv.items():
            L.append(f"- `{name}`: E1 mean {fmt(d.get('e1_mean'))} / max {fmt(d.get('e1_max'))}, E2 PPL {fmt(d.get('ppl'))} ({fmt(d.get('e2_rel'))} rel)")
        L.append("")
    L += ["## Verdict", ""]
    for lvl in ("E1", "E2", "E3"):
        v = R.get("verdict", {}).get(lvl)
        if v is not None:
            L.append(f"- {lvl}: **{'pass' if v else 'FAIL'}**")
    L += ["", "## Files", ""] + [f"- `{f['path']}` {f.get('bytes', '?')} bytes" + (f" sha256 `{f['sha256'][:16]}…`" if f.get("sha256") else "") for f in R.get("files", [])] + [""]
    (out_dir / "report.md").write_text("\n".join(L))
    return out_dir / "report.md"


def check_report(path):
    R = json.loads(Path(path).read_text())
    need = ["model", "thresholds", "verdict", "mode"]
    missing = [k for k in need if k not in R]
    for lvl in ("E1", "E2", "E3"):
        if lvl in R and R[lvl] is not None and R.get("verdict", {}).get(lvl) is None:
            missing.append(f"verdict.{lvl}")
    if missing:
        print("report.json: missing", missing); return 1
    print(f"report.json ok: mode {R['mode']}, verdict {R['verdict']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        sys.exit(check_report(sys.argv[2]))
    print(__doc__)
