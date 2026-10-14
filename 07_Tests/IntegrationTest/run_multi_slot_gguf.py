#!/usr/bin/env python3
"""Integration test for several conversations at once on a GGUF source (phase 7, multi_slot_gguf.md).

Standard library only. Builds the tiny Qwen3.6 fixture and its F32 GGUF, then drives upstream's
tests/serve_mux_check.py (arrives with the v2.0.0 sync) with the GGUF as its snapshot, checks the
GGUF reads accounting, the KV_SLOTS refusals, reference mode and the Vulkan knobs.

    python3 07_Tests/IntegrationTest/run_multi_slot_gguf.py [--keep] [--no-build]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
C = ROOT / "06_Code" / "c"
sys.path.insert(0, str(C)); sys.path.insert(0, str(HERE))
import run_expert_streaming as es  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""
QWEN36 = C / f"qwen36{EXE}"
MUX = C / "tests" / "serve_mux_check.py"
ST2GGUF = C / "tools" / "st2gguf.py"
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def run(args, env=None, timeout=600):
    e = dict(os.environ)
    for k in ("SNAP", "SERVE", "KV_SLOTS", "TOK", "PPL", "PPL_DUMP", "COLI_VK_TIER", "COLI_VK_CHAIN", "DRAFT"):
        e.pop(k, None)
    e.setdefault("OMP_NUM_THREADS", "2")
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, errors="replace", env=e, cwd=str(C), timeout=timeout)


def case0(tmp):
    print("case 0: fixtures (snapshot, tokenizer, F32 GGUF, upstream's serve_mux_check.py)")
    fx = {}
    hf = tmp / "tiny_hf"; r = run([sys.executable, HERE / "make_tiny_qwen36_hf.py", hf])
    check(r.returncode == 0, "torch-free snapshot")
    tok = tmp / "tiny_tok.json"; es.write_tiny_tok(tok); fx["tok"] = tok
    g = tmp / "tiny_f32.gguf"; r = run([sys.executable, ST2GGUF, hf, "--out", g, "--type", "f32", "--tokenizer", tok])
    if check(r.returncode == 0 and g.exists(), "F32 GGUF"):
        fx["gguf"] = g
    check(MUX.exists(), f"{MUX.relative_to(ROOT)} present (upstream v2.0.0, #1955)")
    ref = tmp / "ref.json"; ref.write_text(json.dumps({"prompt_ids": es.PROMPT, "full_ids": es.PROMPT + list(range(32, 32 + es.NGEN))}))
    fx["ref"] = ref
    c = C / "qwen36_tiny_c"      # the CI job's int8 container of the torch-built tiny model, when present
    if c.is_dir():
        fx["container"] = c
    return fx


def mux(snap, slots, tok, extra=None):
    args = [sys.executable, MUX, QWEN36, snap, str(slots), f"TOK={tok}", "COLI_DENSE_I8=0", "OMP_NUM_THREADS=2"]
    for k, v in (extra or {}).items():
        args.append(f"{k}={v}")
    return run(args, timeout=900)


def case1(fx):
    print("case 1: frames multiplexed == alone (upstream's byte-for-byte check on the GGUF)")
    if "gguf" not in fx or not MUX.exists():
        return
    r = mux(fx["gguf"], 3, fx["tok"])
    out = r.stdout + r.stderr
    check(r.returncode == 0, f"serve_mux_check.py exits 0 ({(out.strip().splitlines() or [''])[-1][:160]})")
    m = re.search(r"KV_SLOTS=3: (\d+) decode steps, (\d+) rows \(([0-9.]+) a step\)", out)
    check(m is not None and float(m.group(3)) > 1.0, f"decode steps of more than one row ({m.group(0) if m else 'no report'})")
    return out


def case2(fx, mux_out):
    print("case 2: GGUF reads accounting with slots")
    if "gguf" not in fx or not mux_out:
        return
    reads = [es.gguf_reads(ln) for ln in mux_out.splitlines() if "GGUF reads:" in ln]
    reads = [g for g in reads if g]
    check(bool(reads), f"GGUF reads: lines present in the multiplexed session ({len(reads)})")
    if reads:
        check(all(g[0] % 3 == 0 for g in reads), "slices = 3 × misses on every turn")
    # three single sessions, same first-wave prompts (the check's own waves are byte prompts; the
    # comparison here is coarse: the multiplexed total must not exceed the sum of the single ones)
    mux_total = sum(g[0] for g in reads)
    single = 0
    for i in range(3):
        ok, res, log = es.serve_session(fx["gguf"], ["the cat sat on the mat", "once upon a time", "hello hello"][i:i + 1], tok=fx["tok"])
        g = es.gguf_reads(log.decode("utf-8", "replace"))
        if g:
            single += g[0]
    check(mux_total <= single or single == 0, f"multiplexed slices ({mux_total}) <= three single sessions ({single})")


def case3(fx):
    print("case 3: KV_SLOTS refusals")
    if "gguf" not in fx:
        return
    for v in ("17", "0"):
        try:
            p = subprocess.run([str(QWEN36), "8", "8"], input=b"", capture_output=True, timeout=60, cwd=str(C),
                               env=dict(os.environ, SNAP=str(fx["gguf"]), SERVE="1", KV_SLOTS=v, TOK=str(fx["tok"]), OMP_NUM_THREADS="2"))
            out = (p.stdout + p.stderr).decode("utf-8", "replace")
            check(p.returncode == 2 and "KV_SLOTS must be between 1 and 16" in out, f"KV_SLOTS={v}: refused with upstream's message (exit {p.returncode})")
        except subprocess.TimeoutExpired:
            check(False, f"KV_SLOTS={v}: the engine did not refuse (timeout)")


def case4(fx):
    print("case 4: reference mode ignores KV_SLOTS")
    if "gguf" not in fx:
        return
    a = run([QWEN36, 8, 8, fx["ref"]], env={"SNAP": fx["gguf"], "COLI_DENSE_I8": "0"})
    b = run([QWEN36, 8, 8, fx["ref"]], env={"SNAP": fx["gguf"], "COLI_DENSE_I8": "0", "KV_SLOTS": "3"})
    ia, ib = es.parse_ids(a.stdout + a.stderr), es.parse_ids(b.stdout + b.stderr)
    check(ia is not None and ia == ib, "ids identical with KV_SLOTS=3")
    return ia


def case5(fx, ids):
    print("case 5: Vulkan knobs on a GGUF source say so by name")
    if "gguf" not in fx or ids is None:
        return
    for knob in ("COLI_VK_TIER", "COLI_VK_CHAIN"):
        r = run([QWEN36, 8, 8, fx["ref"]], env={"SNAP": fx["gguf"], "COLI_DENSE_I8": "0", knob: "1"})
        out = r.stdout + r.stderr
        check(es.parse_ids(out) == ids, f"{knob}=1: ids unchanged")
        check(knob in out and ("ignored" in out or "without Vulkan" in out), f"{knob}=1: one message names the knob")


def case6(fx):
    print("case 6: container unchanged (serve_mux_check on the int8 container)")
    if "container" not in fx or not MUX.exists():
        print("  note: no int8 container here (torch); the CI job runs this half")
        return
    r = mux(fx["container"], 3, fx["tok"])
    check(r.returncode == 0, "serve_mux_check.py on the container exits 0")


def main(argv):
    keep = "--keep" in argv
    if "--no-build" not in argv and not es.build_engine():
        print("RESULT: FAIL (qwen36 does not build)"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="sylph_mux_"))
    try:
        fx = case0(tmp)
        out = case1(fx)
        case2(fx, out)
        case3(fx)
        ids = case4(fx)
        case5(fx, ids)
        case6(fx)
    finally:
        if keep:
            print(f"fixtures kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
