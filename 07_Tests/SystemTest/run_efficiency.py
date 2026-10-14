#!/usr/bin/env python3
"""The efficiency verdict tool, checked on the tiny models (phase 7, efficiency_verdict.md).

    python3 07_Tests/SystemTest/run_efficiency.py [--keep] [--no-build]
    python3 07_Tests/SystemTest/run_efficiency.py --owner MODEL=<gguf> CONTAINER=<dir> [OLLAMA=<url> TAG=<tag>]

Standard library only. In --owner mode the tool runs the full protocol on the real file and the
runner checks the report's structure; the ratios go to the state table by hand.
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
IT = ROOT / "07_Tests" / "IntegrationTest"
C = ROOT / "06_Code" / "c"
sys.path.insert(0, str(IT)); sys.path.insert(0, str(C))
import run_expert_streaming as es  # noqa: E402

TOOL = C / "tools" / "efficiency_report.py"
ST2GGUF = C / "tools" / "st2gguf.py"
PROMPTS = HERE / "fixtures" / "e3_prompts.txt"
SECTIONS = ("## Host", "## Systems", "## Results", "## Ratios", "## Verdict")
FIELDS = ("tok_s", "ttft_s", "prefill_tok_s", "bytes_per_token", "rss_gb", "hit_rate", "nll", "ids_stable", "reads")
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def run(args, timeout=1800):
    e = dict(os.environ)
    for k in ("SNAP", "SERVE", "TOK", "PPL", "PPL_DUMP", "KV_SLOTS"):
        e.pop(k, None)
    e.setdefault("OMP_NUM_THREADS", "2")
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, errors="replace", env=e, cwd=str(C), timeout=timeout)


def case0(tmp):
    print("case 0: fixtures")
    fx = {}
    check(PROMPTS.exists(), "fixtures/e3_prompts.txt present")
    hf = tmp / "tiny_hf"; r = run([sys.executable, IT / "make_tiny_qwen36_hf.py", hf])
    check(r.returncode == 0, "tiny snapshot")
    tok = tmp / "tok.json"; es.write_tiny_tok(tok); fx["tok"] = tok
    g = tmp / "tiny_f32.gguf"; r = run([sys.executable, ST2GGUF, hf, "--out", g, "--type", "f32", "--tokenizer", tok])
    if check(r.returncode == 0 and g.exists(), "tiny F32 GGUF"):
        fx["gguf"] = g
    c = C / "qwen36_tiny_c"
    if c.is_dir():
        fx["container"] = c
    else:
        print("  note: no int8 container here (torch); the container arm runs in the CI job")
    check(TOOL.exists(), f"{TOOL.relative_to(ROOT)} exists")
    return fx


def structure(md, js, label):
    for s in SECTIONS:
        check(s in md, f"{label}: section {s}")
    check(re.search(r"\b[0-9a-f]{40}\b|\bcommit\b", md) is not None, f"{label}: commit named")
    check(re.search(r"\b[0-9a-f]{64}\b", md) is not None, f"{label}: sha256 of the model")
    check("cold" in md.lower() or "warm" in md.lower(), f"{label}: cache state named")
    for sysname, obj in (js.get("systems") or {}).items():
        missing = [f for f in FIELDS if f not in obj]
        check(not missing, f"{label}: systems.{sysname} has every field ({missing or 'all'})")
    check(isinstance(js.get("ratios"), dict) and isinstance(js.get("verdict"), str) and js.get("host") and js.get("commit"), f"{label}: ratios, verdict, host, commit in the JSON")
    v = js.get("verdict") or ""
    check(len([s for s in re.split(r"(?<=[.!?])\s+", v.strip()) if s]) >= 3, f"{label}: the verdict has three sentences")


def case1to5(fx, tmp):
    print("case 1: the tool runs in --ci mode")
    if "gguf" not in fx or not TOOL.exists():
        return
    md, js = tmp / "report.md", tmp / "report.json"
    args = [sys.executable, TOOL, "--sylph", fx["gguf"], "--cap", "8", "--new", "16", "--runs", "2", "--cold", "none", "--prompts", PROMPTS, "--ci", "--out", md, "--json", js]
    if "container" in fx:
        args += ["--container", fx["container"]]
    r = run(args)
    if not check(r.returncode == 0 and md.exists() and js.exists(), f"tool exits 0 and writes the report ({(r.stderr.strip().splitlines() or [''])[-1][:160]})"):
        return
    text = md.read_text(encoding="utf-8"); data = json.loads(js.read_text(encoding="utf-8"))
    structure(text, data, "ci report")
    s = (data.get("systems") or {}).get("sylph") or {}
    print("case 2: accounting identity (S)")
    reads = s.get("reads") or {}
    toks = s.get("tokens") or 0
    bpt = s.get("bytes_per_token") or 0
    total = reads.get("bytes") or 0
    check(toks > 0 and total > 0 and abs(bpt * toks - total) <= 0.01 * total + 1, f"bytes_per_token × tokens == Σ GGUF reads bytes ({bpt:.0f} × {toks} vs {total})")
    check(reads.get("slices", 0) > 0 and reads.get("slices", 0) % 3 == 0, f"slices = 3 × misses ({reads.get('slices')})")
    check(reads.get("parts_touched") == 1, f"parts touched 1 ({reads.get('parts_touched')})")
    print("case 3: accounting identity (C)")
    c = (data.get("systems") or {}).get("container")
    if c:
        check(abs((c.get("bytes_per_token") or 0) - (c.get("misses") or 0) * (c.get("expert_bytes") or 0) / max(1, c.get("tokens") or 1)) <= 0.01 * max(1, c.get("bytes_per_token") or 1),
              "container bytes/token == misses × expert bytes / tokens")
    else:
        print("  note: no container arm in this run")
    print("case 4: the control")
    check(s.get("ids_stable") is True, "S: ids identical across the two passes")
    if c:
        print(f"  note: S vs C ids identical: {data.get('ratios', {}).get('ids_identical')}")
    print("case 5: report hygiene")
    check("--cap" in text or "cap " in text, "the report names the cache capacity")
    check("buffered" in text.lower() or "direct" in text.lower(), "the report names buffered vs direct I/O")
    check(re.search(r"\d+\s*tok/s", text) is not None and re.search(r"\d+(\.\d+)?\s*(MB|GB)", text) is not None, "numbers carry their units")


def owner_mode(argv):
    kv = dict(a.split("=", 1) for a in argv[argv.index("--owner") + 1:] if "=" in a)
    model = kv.get("MODEL"); cont = kv.get("CONTAINER")
    if not model:
        print("usage: --owner MODEL=<gguf> [CONTAINER=<dir>] [OLLAMA=<url> TAG=<tag>]"); return 2
    import datetime, socket
    stamp = datetime.date.today().isoformat(); host = socket.gethostname()
    out = ROOT / "08_Documents" / "benchmarks"; out.mkdir(parents=True, exist_ok=True)
    md = out / f"{stamp}-{host}-efficiency.md"; js = out / f"{stamp}-{host}-efficiency.json"
    args = [sys.executable, TOOL, "--sylph", model, "--out", md, "--json", js, "--prompts", PROMPTS, "--new", kv.get("NEW", "128"), "--runs", kv.get("RUNS", "3"), "--cold", kv.get("COLD", "drop")]
    if cont:
        args += ["--container", cont]
    if kv.get("OLLAMA"):
        args += ["--ollama", kv["OLLAMA"], "--tag", kv.get("TAG", "")]
    if kv.get("CAP"):
        args += ["--cap", kv["CAP"]]
    if kv.get("GPU"):
        args += ["--gpu", kv["GPU"]]
    print("case 6: owner mode:", " ".join(str(a) for a in args))
    r = subprocess.run([str(a) for a in args], cwd=str(C))
    if not check(r.returncode == 0 and md.exists() and js.exists(), "tool exits 0 and writes 08_Documents/benchmarks/<date>-<host>-efficiency.{md,json}"):
        return 1
    structure(md.read_text(encoding="utf-8"), json.loads(js.read_text(encoding="utf-8")), "owner report")
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)"); print(f"report: {md}")
    return 1 if failures else 0


def main(argv):
    if "--owner" in argv:
        return owner_mode(argv)
    keep = "--keep" in argv
    if "--no-build" not in argv and not es.build_engine():
        print("RESULT: FAIL (qwen36 does not build)"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="sylph_eff_"))
    try:
        fx = case0(tmp)
        case1to5(fx, tmp)
    finally:
        if keep:
            print(f"fixtures kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
