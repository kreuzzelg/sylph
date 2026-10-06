#!/usr/bin/env python3
"""Integration test for the tensor-source façade (phase 3). See src_facade.md.

Standard library only. Regenerates the torch-free tiny HF snapshot, derives GGUFs
with tools/st2gguf.py, runs tests/test_gguf_load and tests/test_tok_gguf (built here
if missing), exercises the Python arms and the refusal paths. Exit 0 iff every
case passes (case 5 may skip offline).

    python3 07_Tests/IntegrationTest/run_src_facade.py [--no-net] [--keep] [--no-build]
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
C = ROOT / "06_Code" / "c"
sys.path.insert(0, str(C))
import ggufinfo  # noqa: E402
from tools.make_gguf_fixture import GgufWriter, ARR, STR  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""
T_LOAD = C / "tests" / f"test_gguf_load{EXE}"
T_TOK = C / "tests" / f"test_tok_gguf{EXE}"
T_GGUF = C / "tests" / f"test_gguf{EXE}"
ST2GGUF = C / "tools" / "st2gguf.py"
CORPUS = HERE / "fixtures" / "tok_corpus.txt"
UNSLOTH = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TOKJSON = ["https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/tokenizer.json",
           "https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64/resolve/main/tokenizer.json"]
CACHE = Path(os.environ.get("SYLPH_TEST_CACHE", Path(tempfile.gettempdir()) / "sylph_test_cache"))

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, check=False, **kw)


def last_line(r):
    return (r.stdout.strip().splitlines() or [""])[-1]


def build(target):
    if target.exists():
        return True
    b = run(["make", "-C", C, f"tests/{target.name}"])
    return b.returncode == 0


# ---- case 0: the torch-free snapshot ------------------------------------------------
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def case0(tmp):
    print("case 0: torch-free tiny Qwen3.6 snapshot")
    hf = tmp / "tiny_hf"
    r = run([sys.executable, HERE / "make_tiny_qwen36_hf.py", hf])
    check(r.returncode == 0, f"generator: {r.stdout.strip() or r.stderr.strip()}")
    hdr, data0 = st_header(hf / "model.safetensors")
    names = [k for k in hdr if k != "__metadata__"]
    size = (hf / "model.safetensors").stat().st_size
    ends = sorted(v["data_offsets"] for k, v in hdr.items() if k != "__metadata__")
    contiguous = all(ends[i][1] == ends[i + 1][0] for i in range(len(ends) - 1)) and ends[0][0] == 0 and data0 + ends[-1][1] == size
    check(len(names) == 317 and contiguous, f"{len(names)} F32 tensors, offsets contiguous and the file ends with the last one")
    cfg = json.loads((hf / "config.json").read_text())
    check(cfg["hidden_size"] == 64 and cfg["num_hidden_layers"] == 8 and cfg["num_experts"] == 8 and cfg["layer_types"].count("full_attention") == 2,
          "config.json carries the geometry (hidden 64, 8 blocks, 8 experts, 2 attention blocks)")
    r2 = run([sys.executable, HERE / "make_tiny_qwen36_hf.py", tmp / "tiny_hf2"])
    same = hashlib.sha256((hf / "model.safetensors").read_bytes()).hexdigest() == hashlib.sha256((tmp / "tiny_hf2" / "model.safetensors").read_bytes()).hexdigest()
    check(r2.returncode == 0 and same, "deterministic for the default seed")
    return hf


# ---- case 1: st2gguf -----------------------------------------------------------------
def case1(tmp, hf):
    print("case 1: tools/st2gguf.py writes qwen35moe GGUFs the phase-1 reader accepts")
    out = {}
    for tag, extra in (("f32", ["--type", "f32"]), ("f16", ["--type", "f16"]), ("q8_0", ["--type", "q8_0", "--expert-type", "q8_0"])):
        p = tmp / f"tiny_{tag}.gguf"
        r = run([sys.executable, ST2GGUF, hf, "--out", p] + extra)
        if not check(r.returncode == 0 and p.exists(), f"st2gguf --type {tag}: {last_line(r) or r.stderr.strip()[:200]}"):
            continue
        out[tag] = p
        parts = ggufinfo.open_set(p)
        s = ggufinfo.summarize(parts)
        kv = parts[0].kv
        need = ["qwen35moe.block_count", "qwen35moe.embedding_length", "qwen35moe.attention.head_count", "qwen35moe.attention.head_count_kv",
                "qwen35moe.attention.key_length", "qwen35moe.attention.value_length", "qwen35moe.attention.layer_norm_rms_epsilon",
                "qwen35moe.rope.freq_base", "qwen35moe.rope.dimension_count", "qwen35moe.expert_count", "qwen35moe.expert_used_count",
                "qwen35moe.expert_feed_forward_length", "qwen35moe.expert_shared_feed_forward_length", "qwen35moe.full_attention_interval",
                "qwen35moe.ssm.conv_kernel", "qwen35moe.ssm.state_size", "qwen35moe.ssm.group_count", "qwen35moe.ssm.time_step_rank",
                "qwen35moe.ssm.inner_size", "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.eos_token_id"]
        missing = [k for k in need if k not in kv]
        check(s["engine"] == "qwen36" and s["trunk_layers"] == 8 and not missing, f"{tag}: engine qwen36, 8 trunk blocks, KV set complete (missing: {missing})")
        tn = {t.name for t in ggufinfo.all_tensors(parts)}
        check("output.weight" in tn and "blk.0.ffn_gate_exps.weight" in tn and "blk.3.attn_q.weight" in tn and "blk.0.attn_qkv.weight" in tn and "blk.3.attn_qkv.weight" not in tn,
              f"{tag}: names per the table (output, 3-D experts, attn_q at block 3, attn_qkv at block 0)")
        if T_GGUF.exists() or build(T_GGUF):
            rg = run([T_GGUF, p])
            check(rg.returncode == 0, f"{tag}: phase-1 reader accepts the file")
    return out


# ---- cases 2-4: the façade binary ------------------------------------------------------
def case2_4(hf, ggufs):
    print("case 2: façade cross-check F32 (Cfg equal, every tensor and expert slice bit-exact)")
    if "f32" in ggufs:
        r = run([T_LOAD, ggufs["f32"], hf], cwd=C)
        check(r.returncode == 0 and last_line(r) == "all passed", f"test_gguf_load f32: {last_line(r) or r.stderr.strip()[:200]}")
    print("case 3: façade cross-check F16 (one f16 ulp)")
    if "f16" in ggufs:
        r = run([T_LOAD, ggufs["f16"], hf, "--tol", "f16"], cwd=C)
        check(r.returncode == 0 and last_line(r) == "all passed", f"test_gguf_load f16: {last_line(r) or r.stderr.strip()[:200]}")
    print("case 4: self-contained suite (name table, transforms, refusals)")
    r = run([T_LOAD], cwd=C)
    check(r.returncode == 0 and last_line(r) == "all passed", f"test_gguf_load: {last_line(r) or r.stderr.strip()[:200]}")


# ---- case 5: tokenizer equality on the real metadata (network) ---------------------------
def fetch(url, start=None, length=None, name=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (name or hashlib.md5(f"{url}:{start}:{length}".encode()).hexdigest())
    if key.exists():
        return key
    req = urllib.request.Request(url, headers={"User-Agent": "sylph-src-facade"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{start + length - 1}")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
        total = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
    key.write_bytes(data)
    if start is not None and total.isdigit():
        os.truncate(key, int(total))      # sparse: the reader checks sizes, never reads tensor data here
    return key


def case5(no_net):
    print("case 5: tokenizer from the real GGUF metadata == tokenizer.json (network)")
    if no_net:
        print("  skip  --no-net")
        return
    try:
        gguf = fetch(UNSLOTH, 0, 11 << 20, "unsloth_qwen36_header.gguf")   # header + tokenizer arrays (10.99 MB)
        tok = None
        for u in TOKJSON:
            try:
                tok = fetch(u, name="qwen36_tokenizer.json"); break
            except Exception as ex:      # noqa: BLE001
                print(f"  note  {u.split('/')[3]}: {ex}")
        if tok is None:
            raise RuntimeError("no tokenizer.json source reachable")
    except Exception as ex:              # noqa: BLE001
        print(f"  skip  network: {ex}")
        return
    if not (T_TOK.exists() or build(T_TOK)):
        check(False, "tests/test_tok_gguf not built")
        return
    r = run([T_TOK, gguf, tok, CORPUS], cwd=C)
    check(r.returncode == 0 and last_line(r) == "all passed", f"test_tok_gguf on {CORPUS.name}: {last_line(r) or r.stderr.strip()[:200]}")


# ---- case 6: Python arms -----------------------------------------------------------------
def case6(tmp, ggufs):
    print("case 6: family registry and resource plan on a GGUF source")
    if "f32" not in ggufs:
        return
    import family_registry, resource_plan  # noqa: E402
    try:
        res = family_registry.resolve_model(ggufs["f32"])
        fc = res.family_config
        check(res.descriptor.id == "qwen36" and fc.get("hidden_size") == 64 and fc.get("num_hidden_layers") == 8 and fc.get("num_experts") == 8,
              f"resolve_model: family {res.descriptor.id}, hidden {fc.get('hidden_size')}, layers {fc.get('num_hidden_layers')}, experts {fc.get('num_experts')}")
    except Exception as ex:   # noqa: BLE001
        check(False, f"resolve_model on a GGUF: {type(ex).__name__}: {ex}")
    try:
        a = resource_plan.analyze_model(ggufs["f32"])
        s = ggufinfo.summarize(ggufinfo.open_set(ggufs["f32"]))
        check(a.get("expert_bytes") == s["expert_bytes"] and a.get("dense_bytes") == s["dense_bytes"] and a.get("expert_layers") == 8,
              f"analyze_model expert/dense bytes {a.get('expert_bytes')}/{a.get('dense_bytes')} equal the summary's {s['expert_bytes']}/{s['dense_bytes']}, 8 expert layers")
    except Exception as ex:   # noqa: BLE001
        check(False, f"analyze_model on a GGUF: {type(ex).__name__}: {ex}")
    for args in (["doctor", "--model", ggufs["f32"], "--gpu", "none"], ["plan", "--model", ggufs["f32"]]):
        r = run([sys.executable, C / "coli"] + args, cwd=C)
        check(r.returncode == 0, f"coli {args[0]} on the GGUF exits 0 ({(r.stderr or r.stdout).strip().splitlines()[-1][:120] if (r.stderr or r.stdout).strip() else ''})")
    # unknown architecture is refused by name
    bad = rewrite(ggufs["f32"], tmp / "llama.gguf", kv_override={"general.architecture": "llama"})
    try:
        family_registry.resolve_model(bad)
        check(False, "general.architecture = llama was accepted")
    except Exception as ex:   # noqa: BLE001
        check("llama" in str(ex), f"unknown architecture refused by name: {type(ex).__name__}: {str(ex)[:100]}")


# ---- case 7: refusals (derived files) -----------------------------------------------------
def rewrite(src, dst, kv_override=None, drop=(), retype=None, add_tensors=()):
    """Copy a single-file GGUF through the phase-1 writer with modifications."""
    part = ggufinfo.read_part(src)
    w = GgufWriter(alignment=part.alignment)
    for key, (vtype, atype, _n) in part.kv_types.items():
        val = part.kv.get(key)
        if kv_override and key in kv_override:
            val = kv_override[key]
        if val is None:
            continue
        if vtype == ARR:
            w.add_arr(key, atype, val)
        else:
            w.add(key, vtype, val)
    with open(src, "rb") as f:
        for t in part.tensors:
            if t.name in drop:
                continue
            f.seek(t.off); payload = f.read(t.nbytes)
            tid = t.type
            if retype and t.name in retype:
                tid = ggufinfo.TYPE_IDS[retype[t.name]]
                rs = ggufinfo.row_size(tid, t.ne[0]); n = rs
                for e in t.ne[1:]: n *= e
                payload = (payload * (n // len(payload) + 1))[:n]
            ne = list(t.ne)
            while len(ne) > 1 and ne[-1] == 1: ne.pop()      # the reader pads to 4 dims; llama.cpp writes none
            w.add_tensor(t.name, tid, ne, payload=payload)
    for name, tid, ne in add_tensors:
        w.add_tensor(name, tid, ne)
    w.write(dst)
    return dst


def case7(tmp, hf, ggufs):
    print("case 7: refusals name tensor, type and rule; NextN block skipped")
    if "f32" not in ggufs:
        return
    src = ggufs["f32"]
    cases = [
        ("unsupported", rewrite(src, tmp / "bad_type.gguf", retype={"blk.0.ffn_gate_exps.weight": "Q4_1"}), ["unsupported", "Q4_1", "blk.0.ffn_gate_exps.weight"]),
        ("missing", rewrite(src, tmp / "bad_missing.gguf", drop=("blk.3.attn_q.weight",)), ["missing", "blk.3.attn_q.weight"]),
        ("layer kind", rewrite(src, tmp / "bad_kind.gguf", kv_override={"qwen35moe.full_attention_interval": 2}), ["layer kind"]),
    ]
    for tag, path, words in cases:
        r = run([T_LOAD, path, hf], cwd=C)
        msg = (r.stderr + r.stdout)
        check(r.returncode != 0 and all(wd in msg for wd in words), f"{tag}: refused, message names {words}")
    nextn = rewrite(src, tmp / "nextn.gguf",
                    kv_override={"qwen35moe.block_count": 9},
                    add_tensors=[("blk.8.nextn.eh_proj.weight", "F32", [128, 64]), ("blk.8.nextn.enorm.weight", "F32", [64]),
                                 ("blk.8.nextn.hnorm.weight", "F32", [64]), ("blk.8.nextn.shared_head_norm.weight", "F32", [64])])
    # nextn_predict_layers is a new key: append it through a second rewrite
    part = ggufinfo.read_part(nextn)
    w = GgufWriter(alignment=part.alignment)
    for key, (vtype, atype, _n) in part.kv_types.items():
        val = part.kv.get(key)
        if val is None: continue
        w.add_arr(key, atype, val) if vtype == ARR else w.add(key, vtype, val)
    w.add("qwen35moe.nextn_predict_layers", 4, 1)   # U32
    with open(nextn, "rb") as f:
        for t in part.tensors:
            ne = list(t.ne)
            while len(ne) > 1 and ne[-1] == 1: ne.pop()
            f.seek(t.off); w.add_tensor(t.name, t.type, ne, payload=f.read(t.nbytes))
    w.write(tmp / "nextn2.gguf")
    r = run([T_LOAD, tmp / "nextn2.gguf", hf], cwd=C)
    check(r.returncode == 0 and "nextn" in (r.stdout + r.stderr).lower(), f"NextN block: loads with 8 trunk blocks and reports the skipped block ({last_line(r)[:100]})")


def main(argv):
    no_net = "--no-net" in argv; keep = "--keep" in argv; no_build = "--no-build" in argv
    tmp = Path(tempfile.mkdtemp(prefix="sylph_facade_"))
    try:
        hf = case0(tmp)
        if not ST2GGUF.exists():
            check(False, f"{ST2GGUF.relative_to(ROOT)} does not exist")
        for t in (T_LOAD, T_TOK):
            if not no_build and not t.exists() and not build(t):
                check(False, f"build tests/{t.name}")
        if failures:
            print("RESULT: FAIL (module not built — expected until phase 3 lands st2gguf.py, src.h, test_gguf_load, test_tok_gguf)")
            return 1
        ggufs = case1(tmp, hf)
        case2_4(hf, ggufs)
        case5(no_net)
        case6(tmp, ggufs)
        case7(tmp, hf, ggufs)
        print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
        return 1 if failures else 0
    finally:
        if keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
