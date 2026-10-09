#!/usr/bin/env python3
"""Integration test for the NextN (MTP) block in qwen36 (phase 6). See qwen36_mtp.md.

    python3 07_Tests/IntegrationTest/run_qwen36_mtp.py [--keep] [--no-build]
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
import ggufinfo  # noqa: E402
import run_expert_streaming as es  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""
QWEN36 = C / f"qwen36{EXE}"
ST2GGUF = C / "tools" / "st2gguf.py"
COLI = C / "coli"
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def ref_run(snap, cap, ref, env=None):
    e = {"SNAP": str(snap), "COLI_DENSE_I8": "0"}
    if env:
        e.update(env)
    r = es.run([QWEN36, cap, 8, ref], env=e, cwd=C)
    out = r.stdout + r.stderr
    return ("Matching tokens:" in out and es.parse_ids(out) is not None), es.parse_ids(out), out


def ppl_body(snap, cap, ref, dump, env=None):
    e = {"SNAP": str(snap), "COLI_DENSE_I8": "0", "PPL": "1", "PPL_DUMP": str(dump)}
    if env:
        e.update(env)
    es.run([QWEN36, cap, 8, ref], env=e, cwd=C)
    if not Path(dump).exists():
        return None
    b = Path(dump).read_bytes()
    return b.split(b"\n", 1)[1] if b.startswith(b"#") else b


def mtp_clause(text):
    m = re.search(r"nextn ([^·]+)", es.startup_line(text))
    return m.group(1).strip() if m else ""


def case0(tmp):
    print("case 0: fixtures")
    fx = {}
    tok = tmp / "tok.json"; es.write_tiny_tok(tok)
    hf = tmp / "tiny_mtp_hf"; r = es.run([sys.executable, HERE / "make_tiny_qwen36_hf.py", hf, "--mtp"])
    check(r.returncode == 0, "snapshot with the MTP block")
    hfw = tmp / "tiny_mtp_wide_hf"; r = es.run([sys.executable, HERE / "make_tiny_qwen36_hf.py", hfw, "--mtp", "--hidden", "256", "--inter", "256"])
    check(r.returncode == 0, "wide snapshot with the MTP block (K-quant eh_proj)")
    for tag, src, extra in (("mtp_f32", hf, ["--type", "f32"]), ("nomtp", hf, ["--type", "f32", "--no-mtp"]), ("mtp_q4k", hfw, ["--type", "f32", "--mtp-type", "q4_k"])):
        p = tmp / f"tiny_{tag}.gguf"
        r = es.run([sys.executable, ST2GGUF, src, "--out", p, "--tokenizer", tok] + extra)
        if check(r.returncode == 0 and p.exists(), f"st2gguf {tag} ({(r.stderr.strip().splitlines() or [''])[-1][:120]})"):
            fx[tag] = p
    if "mtp_f32" in fx:
        try:
            parts = ggufinfo.open_set(fx["mtp_f32"]); s = ggufinfo.summarize(parts); names = {t.name for t in ggufinfo.all_tensors(parts)}
            check(s["block_count"] == 9 and s["trunk_layers"] == 8, f"block_count 9 = 8 + 1 nextn ({s['block_count']}, {s['trunk_layers']})")
            check(s["mtp"] and s["mtp"]["layer"] == 8 and s["mtp"]["eh_proj_type"] == "F32", f"mtp summary {s['mtp']}")
            check("blk.8.nextn.eh_proj.weight" in names and "blk.8.nextn.enorm.weight" in names and "blk.8.nextn.hnorm.weight" in names and "blk.8.nextn.shared_head_norm.weight" in names, "nextn.* tensors present")
            check("blk.8.attn_q.weight" in names and "blk.8.ffn_gate_exps.weight" in names, "the MTP block's own attention + MoE tensors under blk.8.*")
        except Exception as ex:  # noqa: BLE001
            check(False, f"ggufinfo: {ex}")
    ref = tmp / "ref.json"; ref.write_text(json.dumps({"prompt_ids": es.PROMPT, "full_ids": es.PROMPT + list(range(32, 32 + es.NGEN))}))
    fx["ref"] = ref; fx["tok"] = tok
    return fx


def case1(fx):
    print("case 1: detection and reporting")
    if "mtp_f32" not in fx:
        return
    r = es.run([sys.executable, COLI, "gguf", "inspect", fx["mtp_f32"]], cwd=C)
    check(r.returncode == 0 and re.search(r"mtp\s+blk\.8 nextn · eh_proj F32", r.stdout) is not None, "coli gguf inspect names the head")
    r = es.run([sys.executable, COLI, "doctor", "--model", fx["mtp_f32"], "--gpu", "none"], cwd=C)
    check("model.gguf.mtp_precision" in r.stdout and "[fail] model.gguf" not in r.stdout, "coli doctor: mtp_precision reported, no GGUF failure")


def case2(fx):
    print("case 2: lossless draft/verify")
    if "mtp_f32" not in fx or "nomtp" not in fx:
        return None
    ran_a, ids_a, out_a = ref_run(fx["mtp_f32"], 8, fx["ref"])
    ran_b, ids_b, out_b = ref_run(fx["mtp_f32"], 8, fx["ref"], {"MTP": "0"})
    ran_c, ids_c, out_c = ref_run(fx["nomtp"], 8, fx["ref"])
    check(ran_a and ran_b and ran_c, "three runs")
    check(re.search(r"eh_proj F32 \([0-9.]+ bpw\) loaded", mtp_clause(out_a)) is not None, f"startup line: head loaded ({mtp_clause(out_a)!r})")
    check("skipped (MTP=0)" in mtp_clause(out_b), f"MTP=0: skipped ({mtp_clause(out_b)!r})")
    check(ids_a is not None and ids_a == ids_b == ids_c, "greedy ids identical with the head, with MTP=0 and from the trunk-only file")
    m = re.search(r"\[MTP\] proposed (\d+) · accepted (\d+)", out_a)
    check(m is not None and int(m.group(1)) >= 1, "[MTP] statistics printed, at least one proposal")
    check("[MTP]" not in out_b, "no [MTP] line when skipped")
    return ids_a


def case3(fx, ids):
    print("case 3: precision guard")
    if "mtp_q4k" not in fx:
        return
    ran, ids0, out0 = ref_run(fx["mtp_q4k"], 8, fx["ref"])
    check(ran and re.search(r"skipped \(4\.50 bpw < 8", mtp_clause(out0)) is not None, f"Q4_K eh_proj skipped by default ({mtp_clause(out0)!r})")
    ran, ids1, out1 = ref_run(fx["mtp_q4k"], 8, fx["ref"], {"MTP": "1"})
    check(ran and "loaded" in mtp_clause(out1) and ids1 == ids0, "MTP=1 forces loading; ids unchanged (lossless)")


def case4(fx, tmp):
    print("case 4: PPL mode ignores the head")
    if "mtp_f32" not in fx:
        return
    a = ppl_body(fx["mtp_f32"], 8, fx["ref"], tmp / "a.tsv"); b = ppl_body(fx["mtp_f32"], 8, fx["ref"], tmp / "b.tsv", {"MTP": "0"})
    check(a is not None and a == b, "dumps byte-identical with and without MTP=0")


def case5(fx):
    print("case 5: serve frames identical")
    if "mtp_f32" not in fx:
        return
    prompts = ["the cat sat on the mat and then", "once upon a time"]
    ok1, a, _ = es.serve_session(fx["mtp_f32"], prompts, tok=fx["tok"])
    ok2, b, _ = es.serve_session(fx["mtp_f32"], prompts, tok=fx["tok"], env={"MTP": "0"})
    check(ok1 and ok2, "both sessions reach READY")
    if ok1 and ok2:
        strip = lambda rs: [(r["accept"], r["frames"]) for r in rs]   # DONE carries timings; ids and tails are the contract
        check(strip(a) == strip(b), "frames (ids, tails) identical to the MTP=0 session")


def case6(fx, ids):
    print("case 6: streaming with the head")
    if "mtp_f32" not in fx or ids is None:
        return
    ran, ids1, out1 = ref_run(fx["mtp_f32"], 1, fx["ref"])
    check(ran and ids1 == ids, "cap 1: ids identical to cap 8")
    g = es.gguf_reads(out1)
    check(g is not None and g[0] > 0 and g[0] % 3 == 0, f"GGUF reads: slices = 3 × misses incl. the MTP layer ({g})")


def main(argv):
    keep = "--keep" in argv
    if "--no-build" not in argv and not es.build_engine():
        print("RESULT: FAIL (qwen36 does not build)"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="sylph_mtp_"))
    try:
        fx = case0(tmp)
        case1(fx)
        ids = case2(fx)
        case3(fx, ids)
        case4(fx, tmp)
        case5(fx)
        case6(fx, ids)
    finally:
        if keep:
            print(f"fixtures kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
