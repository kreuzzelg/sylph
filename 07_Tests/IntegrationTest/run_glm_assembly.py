#!/usr/bin/env python3
"""Integration test for the GLM-5.2 (glm-dsa) assembly in colibri.c (phase 6). See glm_assembly.md.

Standard library only. Builds the torch-free tiny GLM snapshot (make_tiny_glm_hf.py), the
GGUFs (tools/st2gguf.py --arch glm-dsa), runs tests/test_gguf_load_glm in both modes and
drives ./colibri in reference mode. Exit 0 iff every case passes.

    python3 07_Tests/IntegrationTest/run_glm_assembly.py [--keep] [--no-build]
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
sys.path.insert(0, str(C))
import ggufinfo  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""
COLIBRI = C / f"colibri{EXE}"
T_LOAD = C / "tests" / f"test_gguf_load_glm{EXE}"
ST2GGUF = C / "tools" / "st2gguf.py"
COLI = C / "coli"
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def run(args, env=None, **kw):
    e = dict(os.environ)
    for k in ("SNAP", "REF", "TF", "MTP", "DSA", "PPL", "PPL_DUMP"):
        e.pop(k, None)
    e.setdefault("OMP_NUM_THREADS", "2")
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, errors="replace", check=False, env=e, **kw)


def make(target):
    r = run(["make", "-C", C, target])
    if r.returncode != 0:
        print("       " + "\n       ".join((r.stderr or r.stdout).strip().splitlines()[-4:])[:600])
    return r.returncode == 0


def parse_ids(text):
    m = re.search(r"C engine\s*:\s*((?:\d+\s+)+)", text)   # colibri prints "GLM C engine      : <ids>"
    return [int(x) for x in m.group(1).split()] if m else None


def gguf_line(text):
    for ln in text.splitlines():
        if ln.startswith("[GGUF] glm-dsa"):
            return ln
    return ""


def ref_run(snap, cap, ref, env=None):
    e = {"SNAP": str(snap), "REF": str(ref), "COLI_TEMP": "0"}
    if env:
        e.update(env)
    r = run([COLIBRI, cap, 16, 16], env=e, cwd=C)
    out = r.stdout + r.stderr
    ran = parse_ids(out) is not None
    return ran, parse_ids(out), out


def case0(tmp):
    print("case 0: fixtures (tiny GLM snapshot, GGUFs)")
    fx = {}
    hf = tmp / "tiny_glm_hf"; r = run([sys.executable, HERE / "make_tiny_glm_hf.py", hf])
    if check(r.returncode == 0, "torch-free tiny GLM snapshot"):
        fx["hf"] = hf
    hfm = tmp / "tiny_glm_mtp_hf"; r = run([sys.executable, HERE / "make_tiny_glm_hf.py", hfm, "--mtp"])
    if check(r.returncode == 0, "snapshot with the NextN block"):
        fx["hf_mtp"] = hfm
    for tag, src, extra in (("f32", "hf", ["--type", "f32"]), ("q8_0", "hf", ["--type", "q8_0", "--expert-type", "q8_0"]),
                            ("mtp_f32", "hf_mtp", ["--type", "f32"]), ("mtp_q4k", "hf_mtp", ["--type", "f32", "--mtp-type", "q4_k"]),
                            ("fused", "hf", ["--type", "f32", "--kv-b", "fused"])):
        if src not in fx:
            continue
        p = tmp / f"tiny_glm_{tag}.gguf"
        r = run([sys.executable, ST2GGUF, fx[src], "--arch", "glm-dsa", "--out", p] + extra)
        if check(r.returncode == 0 and p.exists(), f"st2gguf --arch glm-dsa {tag} ({(r.stderr.strip().splitlines() or [''])[-1][:120]})"):
            fx[tag] = p
    if "f32" in fx:
        try:
            parts = ggufinfo.open_set(fx["f32"]); s = ggufinfo.summarize(parts)
            names = {t.name for t in ggufinfo.all_tensors(parts)}
            check(s["architecture"] == "glm-dsa" and s["engine"] == "glm", f"architecture glm-dsa -> engine glm ({s['architecture']}, {s['engine']})")
            check(s["trunk_layers"] == 5 and not s["mtp"], "5 trunk blocks, no nextn")
            check("blk.0.attn_k_b.weight" in names and "blk.0.attn_v_b.weight" in names and "blk.0.attn_kv_b.weight" not in names, "absorbed MLA split written (attn_k_b/attn_v_b, no attn_kv_b)")
            check(all(f"blk.{i}.indexer.attn_k.weight" in names for i in range(5)), "indexer tensors on every trunk block")
            check("blk.3.exp_probs_b.bias" in names and "blk.0.ffn_gate.weight" in names, "exp_probs_b on MoE blocks, dense MLP on the leading blocks")
        except Exception as ex:  # noqa: BLE001
            check(False, f"ggufinfo on the GLM GGUF: {ex}")
    if "mtp_f32" in fx:
        try:
            s = ggufinfo.summarize(ggufinfo.open_set(fx["mtp_f32"]))
            check(s["trunk_layers"] == 5 and s["mtp"] and s["mtp"]["layer"] == 5, f"nextn block counted inside block_count ({s['block_count']} = 5 + 1), eh_proj at blk.5")
        except Exception as ex:  # noqa: BLE001
            check(False, f"ggufinfo on the MTP GGUF: {ex}")
    ref = tmp / "ref.json"
    ref.write_text(json.dumps({"prompt_ids": [1, 2, 3, 4, 5], "full_ids": [1, 2, 3, 4, 5] + list(range(32, 48)), "tf_pred": list(range(32, 48)) + [1, 2, 3, 4]}))
    fx["ref"] = ref
    return fx


def case1():
    print("case 1: name table, MLA reconciliation, indexer derivation, NextN predicate (tests/test_gguf_load_glm)")
    if not check(make("tests/test_gguf_load_glm"), "builds"):
        return False
    r = run([T_LOAD], cwd=C)
    for ln in r.stdout.splitlines():
        if ln.startswith("FAIL"):
            print("      " + ln)
    check(r.returncode == 0 and "all passed" in r.stdout, "self-contained suite passes")
    return True


def case2(fx):
    print("case 2: cross-check GGUF vs snapshot (Cfg, dense tensors, kv_b from the split, expert slices)")
    for tag, tol in (("f32", None), ("q8_0", "q8")):
        if tag not in fx:
            continue
        r = run([T_LOAD, fx[tag], fx["hf"]] + (["--tol", tol] if tol else []), cwd=C)
        for ln in r.stdout.splitlines():
            if ln.startswith("FAIL"):
                print("      " + ln)
        check(r.returncode == 0, f"{tag}: cross-check passes")


def case3(fx):
    print("case 3: the engine runs from the GGUF (reference mode, startup line)")
    if "f32" not in fx:
        return None
    ran, ids, out = ref_run(fx["f32"], 8, fx["ref"])
    check(ran and ids is not None and len(ids) >= 16, "16 tokens generated from the F32 GGUF")
    ln = gguf_line(out)
    check(ln != "", "[GGUF] glm-dsa line present")
    check("kv_b from attn_k_b/attn_v_b" in ln, "startup line: kv_b rebuilt from the split")
    check("indexer 5/5" in ln, "startup line: indexer on 5/5 blocks")
    check("nextn absent" in ln, "startup line: nextn absent")
    check("[DSA] indexer active" in out, "[DSA] indexer active")
    return ids


def case4(fx, ids_f32):
    print("case 4: MLA policy 1 (attn_kv_b written fused)")
    if "fused" not in fx or ids_f32 is None:
        return
    ran, ids, out = ref_run(fx["fused"], 8, fx["ref"])
    check(ran and ids == ids_f32, "ids identical to the split file")
    check("kv_b from attn_kv_b" in gguf_line(out), "startup line: kv_b from attn_kv_b")


def case5(fx):
    print("case 5: NextN block (loaded, MTP=0, precision guard, lossless draft)")
    if "mtp_f32" not in fx:
        return
    ran0, ids0, out0 = ref_run(fx["mtp_f32"], 8, fx["ref"], {"MTP": "0"})
    check(ran0 and "skipped (MTP=0)" in gguf_line(out0), "MTP=0: nextn skipped (MTP=0)")
    ran1, ids1, out1 = ref_run(fx["mtp_f32"], 8, fx["ref"])
    check(ran1 and re.search(r"nextn blk\.5 eh_proj F32 \([0-9.]+ bpw\) loaded", gguf_line(out1)) is not None, "nextn blk.5 eh_proj F32 loaded")
    check(ids0 is not None and ids0 == ids1, "greedy ids identical with and without the MTP head (draft verified, never trusted)")
    check(re.search(r"\[MTP\] proposed \d+ accepted \d+", out1) is not None, "[MTP] proposed/accepted statistics printed")
    if "mtp_q4k" in fx:
        ran2, _, out2 = ref_run(fx["mtp_q4k"], 8, fx["ref"])
        check(ran2 and re.search(r"skipped \(4\.50 bpw < 8", gguf_line(out2)) is not None, "Q4_K eh_proj: skipped (4.50 bpw < 8)")
        ran3, _, out3 = ref_run(fx["mtp_q4k"], 8, fx["ref"], {"MTP": "1"})
        check(ran3 and "loaded" in gguf_line(out3), "MTP=1 overrides the precision guard")


def case6(fx, tmp):
    print("case 6: refusals by name")
    if "hf" not in fx:
        return
    for tag, extra, needle in (("gate1", ["--gating-func", "1"], "expert_gating_func"),
                               ("group2", ["--expert-groups", "2"], "expert_group_count"),
                               ("nokvb", ["--kv-b", "none"], "attn_k_b"),
                               ("q2k", ["--expert-type", "q2_k"], "Q2_K")):
        p = tmp / f"tiny_glm_{tag}.gguf"
        r = run([sys.executable, ST2GGUF, fx["hf"], "--arch", "glm-dsa", "--out", p, "--type", "f32"] + extra)
        if tag == "q2k":
            check(r.returncode != 0 and "Q2_K" in (r.stderr + r.stdout), "st2gguf refuses to write a Q2_K expert tensor (outside the supported set)")
            continue
        if not check(r.returncode == 0 and p.exists(), f"fixture {tag} written"):
            continue
        ran, ids, out = ref_run(p, 8, fx["ref"])
        check(not ran and needle in out, f"{tag}: refused naming {needle}")


def case7(fx):
    print("case 7: registry and CLI")
    if "f32" not in fx:
        return
    sys.path.insert(0, str(C))
    try:
        import family_registry
        res = family_registry.resolve_model(str(fx["f32"]))
        cfg = res.config
        check(res.descriptor.id == "glm", f"family {res.descriptor.id}")
        want = {"hidden_size": 128, "num_hidden_layers": 5, "n_routed_experts": 8, "kv_lora_rank": 32, "qk_rope_head_dim": 8, "v_head_dim": 32,
                "first_k_dense_replace": 3, "index_topk": 4096, "q_lora_rank": 64, "qk_nope_head_dim": 24, "num_attention_heads": 4}
        bad = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
        check(not bad, f"config fields from the glm-dsa keys ({bad or 'all as expected'})")
    except Exception as ex:  # noqa: BLE001
        check(False, f"resolve_model on the GLM GGUF: {ex}")
    r = run([sys.executable, COLI, "info", "--model", fx["f32"]], cwd=C)
    check(r.returncode == 0 and "colibri" in r.stdout and ".coli-tiny_glm_f32" in r.stdout, "coli info names the colibri engine and the sidecar path")
    r = run([sys.executable, COLI, "plan", "--model", fx["f32"]], cwd=C)
    check(r.returncode == 0 and "CPU" in r.stdout, "coli plan runs on the GLM GGUF")
    r = run([sys.executable, COLI, "doctor", "--model", fx["f32"], "--deep", "--gpu", "none"], cwd=C)
    check("[fail] model.gguf" not in r.stdout, "coli doctor: every model.gguf.* check passes")


def case8(fx, ids_f32):
    print("case 8: streaming parity (cap, threads), reads accounted, nothing written beside the model")
    if "f32" not in fx or ids_f32 is None:
        return
    before = sorted(p.name for p in fx["f32"].parent.iterdir())
    for cap in (1, 2):
        ran, ids, out = ref_run(fx["f32"], cap, fx["ref"])
        check(ran and ids == ids_f32, f"cap {cap}: ids identical to cap 8")
    ran, ids, out = ref_run(fx["f32"], 8, fx["ref"], {"OMP_NUM_THREADS": "4"})
    check(ran and ids == ids_f32, "4 threads: ids identical to 2 threads")
    check(re.search(r"GGUF reads: (\d+) slices", out) is not None, "GGUF reads: line present")
    after = sorted(p.name for p in fx["f32"].parent.iterdir())
    check(before == after, "nothing written beside the model")


def main(argv):
    keep = "--keep" in argv
    if "--no-build" not in argv and not make("colibri"):
        print("RESULT: FAIL (colibri does not build)"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="sylph_glm_"))
    try:
        fx = case0(tmp)
        if case1():
            case2(fx)
        ids = case3(fx)
        case4(fx, ids)
        case5(fx)
        case6(fx, tmp)
        case7(fx)
        case8(fx, ids)
    finally:
        if keep:
            print(f"fixtures kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
