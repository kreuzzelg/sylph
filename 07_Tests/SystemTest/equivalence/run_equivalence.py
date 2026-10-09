#!/usr/bin/env python3
"""Equivalence harness driver (07_Tests/SystemTest/equivalence.md). Standard library only.

Owner's machine (real model, real references):
    python3 run_equivalence.py --model <gguf> --llama <llama.cpp bin dir> [--ollama http://host:11434 --tag <tag>]
        [--text wiki.test.raw] [--chunks 16] [--cap N] [--levels E1,E2,E3] [--out <dir>] [--kl-chunks 1]
        [--prompts 32] [--threads N] [--llama-cuda <bin dir>] [--deviation COLI_DENSE_I8=1] [--force]

CI subset (FR-34; tiny torch-built model, the transformers forward pass as the reference):
    python3 run_equivalence.py --ci --model qwen36_tiny_f32.gguf --torch-ref qwen36_tiny --out equiv_ci

Every step caches under --out; a re-run continues. Exit 0 iff every level that ran passed.
"""
import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import compare  # noqa: E402
import sylph_runner as sr  # noqa: E402
from compare_logprobs import read_dump  # noqa: E402

ROOT = HERE.parents[2]
C = ROOT / "06_Code" / "c"
PROMPTS = HERE.parent / "fixtures" / "e3_prompts.txt"
N_CTX = 512
FIRST_SCORED = N_CTX // 2 + 1        # llama.cpp scores targets 257…511 of a 512 window

CHAT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"   # Qwen3 ChatML, rendered identically for every engine


def log(msg):
    print(f"[equiv {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_prompts(path, limit):
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        chat = ln.startswith("chat:")
        out.append({"text": ln[5:].strip() if chat else ln, "chat": chat})
    return out[:limit]


def merge_dumps(paths, offset_by_chunk=True):
    """{(chunk, pos): (tgt, lp, top)} over several ppl-dump files."""
    rows = {}
    for k, p in enumerate(paths):
        _, r = read_dump(p)
        for pos, v in r.items():
            rows[(k, pos)] = v
    return rows


def chunk_stats(paths):
    out = []
    for p in paths:
        _, r = read_dump(p)
        out.append((len(r), -sum(v[1] for v in r.values())))
    return out


def inspect_summary(model):
    try:
        sys.path.insert(0, str(C)); import ggufinfo  # noqa: E402
        s = ggufinfo.summarize(ggufinfo.open_set(str(model)))
        return f"{s.get('architecture')} · {s.get('trunk_layers')} blocks · {s.get('parts')} part(s) · {s.get('tensors')} tensors · {s.get('file_bytes', 0) / 1e9:.2f} GB · {s.get('type_mix')}"
    except Exception as ex:  # noqa: BLE001
        return f"(inspect failed: {ex})"


def host_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(), "python": platform.python_version()}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    return info


# ---- CI subset -------------------------------------------------------------------------
def run_ci(a):
    ref_dir = Path(a.torch_ref)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = {"mode": "ci", "title": f"CI subset — {Path(a.model).name} vs transformers", "model": {"path": str(a.model), "inspect": inspect_summary(a.model)},
         "versions": {"reference": "transformers forward pass (torch), make_tiny_ref_logprobs.py"}, "host": host_info(), "cap": a.cap, "threads": a.threads}
    # thresholds: the lossless gate of lossless_oracle.md
    R["thresholds"] = {"e1_mean": 1e-6, "e1_max": 1e-5, "top1": 1.0, "kl": 1e-9, "e2_rel": 1e-6, "calibrated": True, "source": "lossless_oracle.md criterion 3"}
    th = R["thresholds"]; verdict = {}
    ref_full = json.loads((ref_dir / "ref_full.json").read_text())
    # E1 on ref_full
    s1 = sr.ppl_window(a.model, ref_full["full_ids"], len(ref_full["prompt_ids"]), out / "sylph", "ref_full", cap=a.cap, threads=a.threads, force=a.force)
    _, rr = read_dump(ref_dir / "ref_logprobs.tsv"); _, cr = read_dump(s1["dump"])
    e1 = compare.e1_compare(rr, cr)
    e1["kl_coarse_mean"] = _coarse_mean(rr, cr)
    # E2 + exact KL on the windows
    wins = sorted((ref_dir / "windows").glob("win_*.ids"), key=lambda p: int(p.stem.split("_")[1])) if (ref_dir / "windows").exists() else []
    chunks_ref, chunks_cand, kl_ex = [], [], None
    for k, w in enumerate(wins):
        ids = [int(x) for x in w.read_text().split()]
        full = (ref_dir / "windows" / f"win_{k}.full").exists()
        s = sr.ppl_window(a.model, ids, FIRST_SCORED, out / "sylph", f"win_{k}", cap=a.cap, threads=a.threads, full=full, force=a.force)
        _, r = read_dump(ref_dir / "windows" / f"win_{k}.tsv"); _, c = read_dump(s["dump"])
        chunks_ref.append((len(r), -sum(v[1] for v in r.values()))); chunks_cand.append((len(c), -sum(v[1] for v in c.values())))
        e1w = compare.e1_compare(r, c)
        e1["mean"] = max(e1["mean"], e1w["mean"]); e1["max"] = max(e1["max"], e1w["max"]); e1["n"] += e1w["n"]
        if e1w.get("top1") is not None:
            e1["top1"] = min(e1["top1"], e1w["top1"]) if e1.get("top1") is not None else e1w["top1"]
        if full:
            kl_ex = compare.kl_exact(ref_dir / "windows" / f"win_{k}.full", s["full"])
    e1["kl_exact"] = kl_ex
    e1["pass"] = e1["mean"] <= th["e1_mean"] and e1["max"] <= th["e1_max"] and (e1.get("top1") or 0) >= th["top1"] and (kl_ex is None or kl_ex["mean"] <= th["kl"]) and e1["bad_targets"] == 0
    R["E1"] = e1; verdict["E1"] = e1["pass"]
    if chunks_ref:
        e2 = compare.e2_compare(chunks_ref, chunks_cand); e2["pass"] = e2["rel"] <= th["e2_rel"]
        R["E2"] = e2; verdict["E2"] = e2["pass"]
    # E3: greedy ids of the reference-id mode vs torch's full_ids, margins from the torch tails
    gen = sr.ref_mode_ids(a.model, ref_dir / "ref_full.json", cap=a.cap, threads=a.threads)
    np_ = len(ref_full["prompt_ids"]); tgt = ref_full["full_ids"][np_:]
    margins = [(rr[p][2][0][1] - rr[p][2][1][1]) if p in rr and len(rr[p][2]) > 1 else None for p in range(np_, len(ref_full["full_ids"]))]
    e3 = compare.e3_compare([{"prompt": 0, "ids": tgt, "margin": margins}], [{"prompt": 0, "ids": gen[:len(tgt)]}])
    e3["pass"] = e3["n_ok"] == e3["n"]
    R["E3"] = {"transformers (greedy ids of ref_full.json)": e3}; verdict["E3"] = e3["pass"]
    R["verdict"] = verdict
    R["files"] = [{"path": str(p), "bytes": p.stat().st_size} for p in sorted((out / "sylph").glob("*")) if p.is_file()]
    rep = compare.write_report(out, R)
    log(f"report: {rep}")
    for lvl, ok in verdict.items():
        log(f"{lvl}: {'pass' if ok else 'FAIL'}")
    log(f"E1 mean {e1['mean']:.3e} max {e1['max']:.3e} top-1 {e1.get('top1')} KL exact {kl_ex}")
    return 0 if all(verdict.values()) else 1


def _coarse_mean(rr, cr):
    vals = [compare.kl_coarse(rr[p][2], cr[p][2]) for p in rr if p in cr]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---- owner's machine -------------------------------------------------------------------
def run_real(a):
    import ref_llama as rl
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    levels = set(a.levels.upper().split(","))
    model = Path(a.model)
    R = {"mode": "real", "title": f"{model.name} — sylph vs llama.cpp" + (" vs Ollama" if a.ollama else ""), "model": {"path": str(model), "inspect": inspect_summary(model)},
         "versions": {"llama.cpp": rl.version(a.llama)}, "host": host_info(), "cap": a.cap, "threads": a.threads}
    sha = out / "model.sha256"
    if not sha.exists() or a.force:
        log("sha256 of the model (once)…"); sha.write_text(sr.sha256_file(model))
    R["model"]["sha256"] = sha.read_text().strip()
    if a.ollama:
        import ref_ollama as ro
        R["versions"]["ollama"] = ro.version(a.ollama)
        try:
            R["versions"]["ollama_show"] = ro.show(a.ollama, a.tag)
        except Exception as ex:  # noqa: BLE001
            R["versions"]["ollama_show"] = f"(show failed: {ex})"
    verdict = {}
    prompts = load_prompts(a.prompts_file, a.prompts)
    # -- tokenizer gate (always) --
    gate_text = out / "gate_text.txt"
    if a.text:
        gate_text = Path(a.text)
    else:
        gate_text.write_text("\n".join(p["text"] for p in prompts) + "\n", encoding="utf-8")
    log("tokenizer gate…")
    ids_llama = rl.tokenize(a.llama, model, gate_text, add_bos=False)
    ids_sylph = sr.encode(model, gate_text)
    first_bad = next((i for i, (x, y) in enumerate(zip(ids_sylph, ids_llama)) if x != y), None)
    gate = {"pass": ids_sylph == ids_llama, "n_ids": len(ids_llama), "n_sylph": len(ids_sylph), "first_mismatch": first_bad}
    R["tokenizer_gate"] = gate
    if not gate["pass"]:
        R["verdict"] = {"gate": False}; compare.write_report(out, R)
        log(f"tokenizer gate FAILED: sylph {len(ids_sylph)} ids, llama {len(ids_llama)}, first mismatch at {first_bad} — E1–E3 would compare different inputs; stopping")
        return 1
    log(f"tokenizer gate pass: {len(ids_llama)} ids identical")
    floor = None; th = compare.thresholds_from_floor(None)
    if levels & {"E1", "E2"}:
        if not a.text:
            sys.exit("--text <wiki.test.raw> is needed for E1/E2")
        # -- noise floor: llama.cpp vs itself at two thread counts (+ CUDA build) --
        log("noise floor: llama-perplexity -t 1 vs -t <cores>…")
        base = rl.perplexity(a.llama, model, a.text, a.chunks, out / "llama" / "t1", threads=1, force=a.force)
        many = rl.perplexity(a.llama, model, a.text, a.chunks, out / "llama" / "tN", threads=a.threads or os.cpu_count(), force=a.force)
        samples = [f"-t 1 vs -t {a.threads or os.cpu_count()}"]
        e1f = compare.e1_compare(merge_dumps(base["dumps"]), merge_dumps(many["dumps"]))
        floor = {"e1_mean": e1f["mean"], "e1_max": e1f["max"], "e1_top1_disagree": 1 - (e1f.get("top1") or 1), "e2_rel": abs(many["ppl"] - base["ppl"]) / base["ppl"]}
        if a.llama_cuda:
            cuda = rl.perplexity(a.llama_cuda, model, a.text, a.chunks, out / "llama" / "cuda", threads=a.threads, extra=["-ngl", "99"], force=a.force)
            e1c = compare.e1_compare(merge_dumps(base["dumps"]), merge_dumps(cuda["dumps"]))
            floor = {"e1_mean": max(floor["e1_mean"], e1c["mean"]), "e1_max": max(floor["e1_max"], e1c["max"]),
                     "e1_top1_disagree": max(floor["e1_top1_disagree"], 1 - (e1c.get("top1") or 1)), "e2_rel": max(floor["e2_rel"], abs(cuda["ppl"] - base["ppl"]) / base["ppl"])}
            samples.append("CPU vs CUDA")
        floor["samples"] = samples
        th = compare.thresholds_from_floor(floor)
        log(f"floor: {floor}; thresholds: {th}")
        # -- sylph windows on llama's token stream --
        ref = many
        dumps_s, fulls_s = [], []
        for k, toks in enumerate(ref["tokens"]):
            s = sr.ppl_window(model, toks, FIRST_SCORED, out / "sylph" / "default", f"win_{k}", cap=a.cap, threads=a.threads, full=k < a.kl_chunks, force=a.force)
            dumps_s.append(s["dump"]); fulls_s.append(s["full"])
            log(f"sylph window {k + 1}/{len(ref['tokens'])}: TF-NLL {s['tf_nll']:.4f} {s.get('reads') or ''}")
        rr, cr = merge_dumps(ref["dumps"]), merge_dumps(dumps_s)
        if "E1" in levels:
            e1 = compare.e1_compare(rr, cr); e1["kl_coarse_mean"] = _coarse_mean(rr, cr)
            e1["kl_exact"] = compare.kl_exact(ref["full0"], fulls_s[0]) if ref.get("full0") and fulls_s and fulls_s[0] else None
            e1["pass"] = e1["mean"] <= th["e1_mean"] and e1["max"] <= th["e1_max"] and (e1.get("top1") or 0) >= th["top1"] and e1["bad_targets"] == 0 and (e1["kl_exact"] is None or e1["kl_exact"]["mean"] <= th["kl"])
            R["E1"] = e1; verdict["E1"] = e1["pass"]
        if "E2" in levels:
            e2 = compare.e2_compare(ref["chunks"], chunk_stats(dumps_s)); e2["ppl_ref_printed"] = ref["ppl"]
            e2["pass"] = e2["rel"] <= th["e2_rel"] and e2["sign_p"] > 0.05
            R["E2"] = e2; verdict["E2"] = e2["pass"]
        # -- deliberate differences (FR-35) --
        for dev in a.deviation or []:
            k_, v_ = dev.split("=", 1)
            dd = []
            for k, toks in enumerate(ref["tokens"]):
                s = sr.ppl_window(model, toks, FIRST_SCORED, out / "sylph" / dev.replace("=", "_"), f"win_{k}", cap=a.cap, threads=a.threads, env={k_: v_}, force=a.force)
                dd.append(s["dump"])
            e1d = compare.e1_compare(rr, merge_dumps(dd)); e2d = compare.e2_compare(ref["chunks"], chunk_stats(dd))
            R.setdefault("deviations", {})[dev] = {"e1_mean": e1d["mean"], "e1_max": e1d["max"], "ppl": e2d["ppl_cand"], "e2_rel": e2d["rel"]}
    R["floor"] = floor; R["thresholds"] = th
    if "E3" in levels:
        gens_s = []; srv = sr.Serve(model, cap=a.cap, threads=a.threads)
        try:
            for i, p in enumerate(prompts):
                text = CHAT_TEMPLATE.format(p["text"]) if p["chat"] else p["text"]
                g = srv.generate(text, a.n_predict); g["prompt"] = i; g["engine"] = "sylph"; gens_s.append(g)
                log(f"sylph E3 prompt {i}: {len(g['ids'])} ids")
        finally:
            srv.close()
        (out / "e3_sylph.jsonl").write_text("\n".join(json.dumps(g) for g in gens_s) + "\n")
        R["E3"] = {}
        # llama-server (ids + margins), fallback llama-cli (text)
        gens_l = []
        try:
            server = rl.Server(a.llama, model, port=a.port, threads=a.threads)
            try:
                for i, p in enumerate(prompts):
                    text = CHAT_TEMPLATE.format(p["text"]) if p["chat"] else p["text"]
                    g = server.generate(text, a.n_predict); g["prompt"] = i; g["engine"] = "llama-server"; gens_l.append(g)
                    log(f"llama-server E3 prompt {i}: {len(g['ids'])} ids")
            finally:
                server.close()
        except SystemExit as ex:
            log(f"llama-server unavailable ({ex}); falling back to llama-cli text")
            for i, p in enumerate(prompts):
                text = CHAT_TEMPLATE.format(p["text"]) if p["chat"] else p["text"]
                g = rl.generate_cli(a.llama, model, text, a.n_predict, threads=a.threads); g["prompt"] = i; g["engine"] = "llama-cli"; gens_l.append(g)
        (out / "e3_llama.jsonl").write_text("\n".join(json.dumps(g) for g in gens_l) + "\n")
        e3l = compare.e3_compare(gens_l, gens_s); e3l["pass"] = e3l["n_ok"] >= math.ceil(0.95 * e3l["n"])
        R["E3"][gens_l[0]["engine"] if gens_l else "llama.cpp"] = e3l; verdict["E3"] = e3l["pass"]
        if a.ollama:
            import ref_ollama as ro
            gens_o = []
            for i, p in enumerate(prompts):
                g = ro.generate(a.ollama, a.tag, p["text"] if p["chat"] else p["text"], a.n_predict, chat=p["chat"]); g["prompt"] = i; g["engine"] = "ollama"; gens_o.append(g)
                log(f"ollama E3 prompt {i}: {len(g['text'])} chars, {g['timing']['tok_s']:.1f} tok/s")
            (out / "e3_ollama.jsonl").write_text("\n".join(json.dumps(g) for g in gens_o) + "\n")
            # text-level comparison; margins from llama-server's generation where the texts share a prefix
            e3o = compare.e3_compare([dict(g, ids=[]) for g in gens_o], [dict(g, ids=[]) for g in gens_s])
            e3o["pass"] = e3o["n_ok"] >= math.ceil(0.95 * e3o["n"]); e3o["note"] = "compared by decoded text; Ollama returns no ids"
            R["E3"]["ollama"] = e3o; verdict["E3"] = verdict.get("E3", True) and e3o["pass"]
    R["verdict"] = verdict
    R["files"] = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file() and p.suffix in (".tsv", ".full", ".kld", ".jsonl", ".log")]
    rep = compare.write_report(out, R)
    log(f"report: {rep} — copy it to 08_Documents/equivalence/<date>-{model.stem}-<host>.md")
    for lvl, ok in verdict.items():
        log(f"{lvl}: {'pass' if ok else 'FAIL'}")
    return 0 if verdict and all(verdict.values()) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--llama"); ap.add_argument("--llama-cuda")
    ap.add_argument("--ollama"); ap.add_argument("--tag"); ap.add_argument("--text"); ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--cap", type=int, default=int(os.environ.get("CAP", "8"))); ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--levels", default="E1,E2,E3"); ap.add_argument("--out", default="equivalence_out"); ap.add_argument("--kl-chunks", type=int, default=1)
    ap.add_argument("--prompts", type=int, default=32); ap.add_argument("--prompts-file", default=str(PROMPTS)); ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--port", type=int, default=8089); ap.add_argument("--deviation", action="append"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--ci", action="store_true"); ap.add_argument("--torch-ref")
    a = ap.parse_args(argv)
    if a.ci:
        if not a.torch_ref:
            sys.exit("--ci needs --torch-ref <dir with ref_full.json, ref_logprobs.tsv, windows/>")
        return run_ci(a)
    if not a.llama:
        sys.exit("--llama <llama.cpp bin dir> is required (or --ci)")
    return run_real(a)


if __name__ == "__main__":
    sys.exit(main())
