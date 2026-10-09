#!/usr/bin/env python3
"""Integration test for the CUDA expert tier on raw ggml blocks (phase 5). See cuda_tier_kquant.md.

Standard library only. Part A runs without a GPU: the host-side C tests, the engine with
the fake backend linked (tests/test_qwen36_tier_kq_engine) against the real qwen36 on the
same GGUFs, the planner. Part B (`--cuda`) runs `make cuda-test-gq` when nvcc and a device
are present (the owner's card). Exit 0 iff every case passes.

    python3 07_Tests/IntegrationTest/run_cuda_tier_kquant.py [--keep] [--no-build] [--cuda]
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
import run_expert_streaming as es  # noqa: E402  (helpers: run, write_tiny_tok, parse_ids, startup_line, gguf_reads)

EXE = ".exe" if sys.platform == "win32" else ""
QWEN36 = C / f"qwen36{EXE}"
FAKE_ENGINE = C / "tests" / f"test_qwen36_tier_kq_engine{EXE}"
ST2GGUF = C / "tools" / "st2gguf.py"
GB = 1024 ** 3
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def make(target):
    r = es.run(["make", "-C", C, target])
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-4:])
        print("       " + tail.replace("\n", "\n       ")[:600])
    return r.returncode == 0


# ---- engine drivers -----------------------------------------------------------------------
TIER_ENV = {"COLI_CUDA": "1", "COLI_GPUS": "0", "QT_UPLOAD_SYNC": "1", "COLI_PLACE": "off",
            "CUDA_EXPERT_GB": "1", "HEAT_FILE": "", "QT_NO_WARMSTART": "0"}


def ref_run(binary, snap, cap, ref, env=None):
    e = {"SNAP": snap, "COLI_DENSE_I8": "0"}
    if env:
        e.update(env)
    r = es.run([binary, cap, 8, ref], env=e, cwd=C)
    out = r.stdout + r.stderr
    ran = "Matching tokens:" in out and es.parse_ids(out) is not None
    return ran, es.parse_ids(out), out


def ppl_run(binary, snap, cap, ref, dump, env=None):
    e = {"SNAP": snap, "COLI_DENSE_I8": "0", "PPL": "1", "PPL_DUMP": dump}
    if env:
        e.update(env)
    r = es.run([binary, cap, 8, ref], env=e, cwd=C)
    body = None
    if Path(dump).exists():
        b = Path(dump).read_bytes(); body = b.split(b"\n", 1)[1] if b.startswith(b"#") else b
    return r.returncode == 0 and body is not None, body, r.stdout + r.stderr


def parse_dump(body):
    """ppl-dump v1 lines: pos \t target \t lp \t tail -> [(pos, target, lp)]"""
    rows = []
    for ln in body.decode("utf-8", "replace").splitlines():
        f = ln.split("\t")
        if len(f) >= 3:
            rows.append((int(f[0]), int(f[1]), float(f[2])))
    return rows


def max_dlp(a, b):
    ra, rb = parse_dump(a), parse_dump(b)
    if len(ra) != len(rb) or any(x[:2] != y[:2] for x, y in zip(ra, rb)):
        return float("inf")
    return max((abs(x[2] - y[2]) for x, y in zip(ra, rb)), default=0.0)


def fake_line(text):
    m = re.search(r"\[fake-cuda\] uploads (\d+) · block fmts (\d+)/(\d+)/(\d+)/(\d+) · issues (\d+) · expert rows (\d+) · matmuls (\d+)", text)
    if not m:
        return None
    v = [int(x) for x in m.groups()]
    return {"uploads": v[0], "n24": v[1], "n28": v[2], "n29": v[3], "n30": v[4], "issues": v[5], "rows": v[6], "matmuls": v[7]}


def resident(text):
    m = re.search(r"\[qtier\] resident (\d+)/(\d+) experts \| uploads (\d+) \| miss\(CPU\) (\d+)", text)
    return tuple(int(x) for x in m.groups()) if m else None


def footprint(nbytes):
    """dev_alloc_footprint (qwen36_tier.c): cudaMalloc granularity, measured table."""
    KiB, MiB = 1024, 1048576
    if nbytes > MiB:
        return (nbytes + 2 * MiB - 1) // (2 * MiB) * (2 * MiB)
    if nbytes > 512 * KiB:
        return MiB
    b = max(nbytes + nbytes // 16, 8 * KiB)
    return (b + 8 * KiB - 1) // (8 * KiB) * (8 * KiB)


# ---- cases ----------------------------------------------------------------------------------
def case0(tmp):
    print("case 0: fixtures (wide snapshot for K-quant experts, preset snapshot, tokenizer, GGUFs)")
    fx = {}
    tok = tmp / "tiny_tok.json"; es.write_tiny_tok(tok); fx["tok"] = tok
    kq = tmp / "tiny_kq_hf"
    r = es.run([sys.executable, HERE / "make_tiny_qwen36_hf.py", kq, "--hidden", "256", "--inter", "256"])
    check(r.returncode == 0, "snapshot with hidden 256 / inter 256 (K-quant blocks need multiples of 256)")
    small = tmp / "tiny_hf"
    r = es.run([sys.executable, HERE / "make_tiny_qwen36_hf.py", small])
    check(r.returncode == 0, "preset snapshot (hidden 64)")
    cfg = json.loads((kq / "config.json").read_text())
    fx["geom"] = {"L": cfg["num_hidden_layers"], "E": cfg["num_experts"], "K": cfg["num_experts_per_tok"], "H": cfg["hidden_size"], "I": cfg["moe_intermediate_size"]}
    p = tmp / "tiny_q4k.gguf"
    r = es.run([sys.executable, ST2GGUF, kq, "--out", p, "--type", "q8_0", "--expert-type", "q4_k", "--down-type", "q6_k", "--tokenizer", tok])
    if check(r.returncode == 0 and p.exists(), f"st2gguf --expert-type q4_k --down-type q6_k ({(r.stderr.strip().splitlines() or [''])[-1][:120]})"):
        fx["q4k"] = p
        try:
            mix = ggufinfo.summarize(ggufinfo.open_set(p))["type_mix"]
            names = set(mix) if isinstance(mix, dict) else {m[0] if isinstance(m, (list, tuple)) else m for m in mix}
            check({"Q4_K", "Q6_K", "Q8_0"} <= set(str(n) for n in names), f"type mix lists Q4_K, Q6_K, Q8_0 ({sorted(str(n) for n in names)})")
        except Exception as ex:  # noqa: BLE001
            check(False, f"ggufinfo on tiny_q4k.gguf: {ex}")
    for tag, src, extra in (("xq8_0", small, ["--type", "f32", "--expert-type", "q8_0"]), ("q8_0", small, ["--type", "q8_0", "--expert-type", "q8_0"])):
        p = tmp / f"tiny_{tag}.gguf"
        r = es.run([sys.executable, ST2GGUF, src, "--out", p] + extra + ["--tokenizer", tok])
        if check(r.returncode == 0 and p.exists(), f"st2gguf {tag}"):
            fx[tag] = p
    ref = tmp / "ref.json"
    ref.write_text(json.dumps({"prompt_ids": es.PROMPT, "full_ids": es.PROMPT + list(range(32, 32 + es.NGEN))}))
    fx["ref"] = ref
    return fx


def case1():
    print("case 1: block-format truth table (tests/test_cuda_block_fmt_guard)")
    if not check(make("tests/test_cuda_block_fmt_guard"), "builds"):
        return
    r = es.run([C / "tests" / f"test_cuda_block_fmt_guard{EXE}"])
    check(r.returncode == 0 and "all passed" in r.stdout, f"passes ({(r.stdout.strip().splitlines() or [''])[-1][:100]})")


def case2():
    print("case 2: tier on the fake backend, GGUF flavour (tests/test_qwen36_tier_kq)")
    if not check(make("tests/test_qwen36_tier_kq"), "builds"):
        return
    r = es.run([C / "tests" / f"test_qwen36_tier_kq{EXE}"])
    for ln in r.stdout.splitlines():
        if ln.startswith("  FAIL") or ln.startswith("    "):
            print("      " + ln.strip())
    check(r.returncode == 0 and "all passed" in r.stdout, "passes")
    check("has no CUDA kernel" in r.stderr, "the by-type refusal note was printed")


def engine_pair(fx, key, env, label, exact):
    """Run qwen36 and the fake-tier engine on fixture `key`; compare ids and dumps."""
    snap, cap = str(fx[key]), fx["geom"]["E"] if key == "q4k" else 8
    ran_a, ids_a, out_a = ref_run(QWEN36, snap, cap, fx["ref"])
    ran_b, ids_b, out_b = ref_run(FAKE_ENGINE, snap, cap, fx["ref"], env)
    check(ran_a and ran_b, f"{label}: both engines ran")
    check(ids_a is not None and ids_a == ids_b, f"{label}: ids identical")
    d_a, d_b = fx["tmp"] / f"{key}_cpu.tsv", fx["tmp"] / f"{key}_{label.split()[0]}.tsv"
    ok_a, body_a, _ = ppl_run(QWEN36, snap, cap, fx["ref"], d_a)
    ok_b, body_b, out_p = ppl_run(FAKE_ENGINE, snap, cap, fx["ref"], d_b, env)
    if check(ok_a and ok_b, f"{label}: both dumps written"):
        if exact:
            same = body_a == body_b
            if not same:
                print(f"       max |Δlp| = {max_dlp(body_a, body_b):.3g}")
            check(same, f"{label}: PPL dump bodies byte-identical")
        else:
            m = max_dlp(body_a, body_b)
            check(m <= 1e-5, f"{label}: dumps within the lossless gate (max |Δlp| {m:.3g} ≤ 1e-5)")
    return out_b, out_p


def case3(fx):
    print("case 3: engine on the fake, full residency, trunk on the CPU")
    if not check(make("tests/test_qwen36_tier_kq_engine"), "the engine with the fake backend builds"):
        return False
    g = fx["geom"]
    for key, fmts in (("q4k", (0, 2, 0, 1)), ("xq8_0", (3, 0, 0, 0))):
        if key not in fx:
            continue
        L, E = (g["L"], g["E"]) if key == "q4k" else (8, 8)
        out, out_p = engine_pair(fx, key, TIER_ENV, f"{key} full", exact=True)
        f = fake_line(out)
        if check(f is not None, f"{key}: [fake-cuda] line present"):
            want = tuple(n * L * E for n in fmts)
            got = (f["n24"], f["n28"], f["n29"], f["n30"])
            check(got == want, f"{key}: block fmt uploads {got} == {want} (every expert resident)")
            check(f["matmuls"] == 0, f"{key}: no dense matmul on the device (COLI_PLACE=off)")
            check(f["issues"] > 0 and f["rows"] >= f["issues"], f"{key}: {f['issues']} group issues, {f['rows']} expert rows")
        check(out.count("[gpu] MoE experts -> CUDA VRAM tier") == 1, f"{key}: tier banner once")
        check("experts on CUDA tier" in es.startup_line(out), f"{key}: startup line names the CUDA tier ({es.startup_line(out)[-60:]})")
        rs = resident(out)
        check(rs is not None and rs[0] == rs[1] == L * E and rs[3] == 0, f"{key}: [qtier] resident {rs}")
    return True


def case4(fx):
    print("case 4: engine on the fake, dense trunk placed (COLI_PLACE=auto)")
    if "q8_0" not in fx:
        return
    env = dict(TIER_ENV, COLI_PLACE="auto", COLI_TRUNK_PROBE="0")
    out, _ = engine_pair(fx, "q8_0", env, "q8_0 placed", exact=False)
    f = fake_line(out)
    if check(f is not None, "[fake-cuda] line present"):
        check(f["matmuls"] > 0, f"dense GEMVs answered by the device ({f['matmuls']})")
        check(f["n24"] > 3 * 8 * 8, f"fmt 24 uploads = experts + placed Q8_0 dense matrices ({f['n24']} > {3 * 8 * 8})")
    check(re.search(r"\[place\].*lmhead", out) is not None, "[place] names lmhead")


def case5(fx):
    print("case 5: engine on the fake, partial residency")
    if "q4k" not in fx:
        return
    g = fx["geom"]; L, E = g["L"], g["E"]
    kb = [gq_row_bytes_q4k(g["H"]) * g["I"]] * 2 + [gq_row_bytes_q6k(g["I"]) * g["H"]]
    exp = sum(footprint(b) for b in kb)
    half = (L * E) // 2
    env = dict(TIER_ENV, CUDA_EXPERT_GB=f"{(half * exp + exp // 2) / GB:.17g}")
    out, _ = engine_pair(fx, "q4k", env, "q4k half", exact=False)
    rs = resident(out)
    if check(rs is not None, "[qtier] resident line present"):
        check(0 < rs[0] < L * E, f"resident {rs[0]}/{rs[1]} (planned for {half})")
        check(rs[3] > 0, f"CPU misses counted ({rs[3]})")
    check(es.gguf_reads(out) is not None, "GGUF reads: line still accounts the slices")


def gq_row_bytes_q4k(I): return I // 256 * 144
def gq_row_bytes_q6k(I): return I // 256 * 210


def case6(fx):
    print("case 6: planner — trunk bytes and expert capacity for a GGUF on an 8 GB card")
    key = "q4k" if "q4k" in fx else ("q8_0" if "q8_0" in fx else None)   # any GGUF does; q4k preferred
    if key is None:
        return
    gguf = fx[key]
    import resource_plan
    gpu = {"index": 0, "name": "NVIDIA GeForce RTX 3070", "total_bytes": 8 * GB, "free_bytes": int(7.5 * GB), "unified_memory": False}
    try:
        plan = resource_plan.build_plan(str(gguf), gpus=[gpu], available_memory=16 * GB, available_disk=100 * GB)
    except Exception as ex:  # noqa: BLE001
        check(False, f"build_plan with an injected GPU: {ex}"); return
    vram = plan["tiers"]["vram"]
    parts = ggufinfo.open_set(gguf)
    trunk_names = re.compile(r"^(output\.weight|blk\.\d+\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight)$")
    want_trunk = sum(t.nbytes for t in ggufinfo.all_tensors(parts) if trunk_names.match(t.name))
    info = resource_plan.analyze_model(str(gguf))
    usable = int(7.5 * GB) - 2 * GB
    check(vram.get("trunk_bytes") == want_trunk, f"tiers.vram.trunk_bytes {vram.get('trunk_bytes')} == stored bytes of the trunk components {want_trunk}")
    check(vram["budget_bytes"] == min(usable - want_trunk, info["expert_bytes"]), f"budget {vram['budget_bytes']} == min(usable − trunk, experts)")
    check(vram["expert_capacity"] == vram["budget_bytes"] // info["typical_expert_bytes"], f"expert capacity {vram['expert_capacity']}")
    text = "\n".join(resource_plan.format_plan(plan)) if isinstance(resource_plan.format_plan(plan), list) else str(resource_plan.format_plan(plan))
    check(re.search(r"VRAM\s+\S+ [GM]B trunk \+ .*hot tier · ~\d+ experts", text) is not None, "rendered VRAM line has trunk, hot tier and expert count")
    plan0 = resource_plan.build_plan(str(gguf), gpus=[], available_memory=16 * GB, available_disk=100 * GB)
    check(plan0["tiers"]["vram"]["budget_bytes"] == 0 and not plan0["tiers"]["vram"]["devices"], "no GPU: CPU plan as in phase 4")


def case7(fx):
    print("case 7: FR-36 note on the CPU-only build")
    key = "q4k" if "q4k" in fx else ("q8_0" if "q8_0" in fx else None)
    if key is None:
        return
    ran0, ids0, _ = ref_run(QWEN36, str(fx[key]), 8, fx["ref"])
    ran, ids, out = ref_run(QWEN36, str(fx[key]), 8, fx["ref"], {"COLI_CUDA": "1"})
    check(ran0 and ran and ids == ids0, "ids unchanged with COLI_CUDA=1")
    check(out.count("COLI_CUDA=1 ignored: built without CUDA") == 1, "the build note, once")
    check("does not take GGUF K-quant experts yet" not in out, "the phase-4 'not yet' note is gone")


def case8():
    print("case 8: NFR-5 regression net (int8/int4 tier tests unchanged)")
    for t in ("tests/test_qwen36_tier_int8_engine", "tests/test_qwen36_tier_int8_decode", "tests/test_qwen36_tier_dense", "tests/test_cuda_fmt_guard"):
        if check(make(t), f"{t} builds"):
            r = es.run([C / (t + EXE)], cwd=C)
            check(r.returncode == 0, f"{t} passes")


def case_b():
    print("part B: kernel oracle on the device (make cuda-test-gq)")
    r = es.run(["make", "-C", C, "cuda-test-gq"])
    out = r.stdout + r.stderr
    for ln in out.splitlines():
        if ln.startswith("FAIL") or "max |delta|" in ln:
            print("      " + ln.strip())
    check(r.returncode == 0 and ("all passed" in out or "skipped" in out), f"cuda-test-gq ({(out.strip().splitlines() or [''])[-1][:100]})")


def main(argv):
    keep = "--keep" in argv
    if "--no-build" not in argv and not es.build_engine():
        print("RESULT: FAIL (qwen36 does not build)"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="sylph_kq_"))
    try:
        fx = case0(tmp); fx["tmp"] = tmp
        case1()
        case2()
        if case3(fx):
            case4(fx)
            case5(fx)
        case6(fx)
        case7(fx)
        case8()
        if "--cuda" in argv:
            case_b()
    finally:
        if keep:
            print(f"fixtures kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
