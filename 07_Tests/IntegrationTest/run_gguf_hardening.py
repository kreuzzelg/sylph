#!/usr/bin/env python3
"""Hardening of the GGUF path (phase 7, gguf_hardening.md): mutation fuzz of the readers and the
engines, truncated payloads, split-set rules, the two stdlib readers against each other, the GLM
tokenizer from the GGUF.

Standard library only. Exit 0 iff every case passes.

    python3 07_Tests/IntegrationTest/run_gguf_hardening.py [--keep] [--no-build] [--seed N] [--mutations N]
"""
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
C = ROOT / "06_Code" / "c"
sys.path.insert(0, str(C)); sys.path.insert(0, str(C / "tools")); sys.path.insert(0, str(HERE))
import ggufinfo  # noqa: E402
import run_expert_streaming as es  # noqa: E402
from make_gguf_fixture import GgufWriter, U16, U32, STR, ARR, write_split_set  # noqa: E402

EXE = ".exe" if sys.platform == "win32" else ""
QWEN36 = C / f"qwen36{EXE}"
COLIBRI = C / f"colibri{EXE}"
T_GGUF = C / "tests" / f"test_gguf{EXE}"
T_TOK = C / "tests" / f"test_tok_gguf_glm{EXE}"
ST2GGUF = C / "tools" / "st2gguf.py"
COLI = C / "coli"
CORPUS = HERE / "fixtures" / "tok_corpus.txt"
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def run(args, env=None, timeout=120, cwd=None):
    e = dict(os.environ)
    for k in ("SNAP", "REF", "SERVE", "TOK", "PPL", "PPL_DUMP", "KV_SLOTS", "MTP", "DSA"):
        e.pop(k, None)
    e.setdefault("OMP_NUM_THREADS", "2")
    if env:
        e.update({k: str(v) for k, v in env.items()})
    try:
        r = subprocess.run([str(a) for a in args], capture_output=True, text=True, errors="replace", env=e, cwd=str(cwd or C), timeout=timeout)
        return r.returncode, r.stdout + r.stderr, False
    except subprocess.TimeoutExpired:
        return None, "", True


def make(target):
    r = subprocess.run(["make", "-C", str(C), target], capture_output=True, text=True)
    if r.returncode != 0:
        print("       " + "\n       ".join((r.stderr or r.stdout).strip().splitlines()[-4:])[:600])
    return r.returncode == 0


# ---- verdicts -----------------------------------------------------------------------------------
def c_reader(path):
    """('ok'|'refused'|'crash'|'silent'|'hang', text)"""
    rc, out, hung = run([T_GGUF, path], timeout=20)
    if hung:
        return "hang", out
    if rc == 0:
        return "ok", out
    if rc is not None and rc < 0:
        return "crash", f"signal {-rc}"
    return ("refused" if "gguf:" in out else "silent"), out


def py_reader(path):
    try:
        parts = ggufinfo.open_set(str(path))
        return "ok", f"{len(ggufinfo.all_tensors(parts))} tensors"
    except ggufinfo.GgufError as ex:
        return "refused", str(ex)
    except Exception as ex:  # noqa: BLE001 -- the contract: nothing but GgufError escapes
        return "foreign", f"{type(ex).__name__}: {ex}"


def engine(binary, path, ref, extra=None, timeout=60):
    env = {"SNAP": str(path), "REF": str(ref), "COLI_DENSE_I8": "0", "COLI_TEMP": "0"}
    if extra:
        env.update(extra)
    args = [binary, 8, 8, ref] if binary == QWEN36 else [binary, 8, 16, 16]
    rc, out, hung = run(args, env=env, timeout=timeout)
    if hung:
        return "hang", out
    if rc is not None and rc < 0:
        return "crash", f"signal {-rc}"
    if "Matching tokens" in out:        # the run reached the end (the exit code grades the oracle match, not the load)
        return "ok", out
    named = any(tag in out for tag in ("[GGUF]", "gguf:", "config:", "refus", "missing", "short read", "invalid", "out of range", "unsupported"))
    return ("refused" if named else "silent"), out


# ---- mutation generators ------------------------------------------------------------------------
def boundaries(path):
    """structural offsets of a valid single-part file: (data_off, [tensor info starts], kv region end)"""
    parts = ggufinfo.open_set(str(path))
    p = parts[0]
    return getattr(p, "data_off", None) or getattr(p, "data_offset", None), len(p.tensors), p


def mutations(blob, data_off, rng, n):
    """n (label, bytes) pairs: bit flips / fills in the metadata region, truncations at boundaries"""
    out = []
    meta = max(1, min(data_off or len(blob), len(blob)))
    kinds = ["flip0", "flip7", "fill00", "fillff", "trunc", "trunc_payload", "swap"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        b = bytearray(blob)
        if kind.startswith("flip"):
            pos = rng.randrange(meta); b[pos] ^= 1 << (0 if kind == "flip0" else 7); out.append((f"{kind}@{pos}", bytes(b)))
        elif kind.startswith("fill"):
            pos = rng.randrange(meta); ln = rng.randrange(1, 9); v = 0x00 if kind == "fill00" else 0xFF
            b[pos:pos + ln] = bytes([v]) * len(b[pos:pos + ln]); out.append((f"{kind}@{pos}+{ln}", bytes(b)))
        elif kind == "trunc":
            pos = rng.randrange(8, meta); out.append((f"trunc@{pos}", bytes(b[:pos])))
        elif kind == "trunc_payload":
            if data_off and len(blob) > data_off + 64:
                pos = rng.randrange(data_off + 1, len(blob)); out.append((f"trunc_payload@{pos}", bytes(b[:pos])))
            else:
                out.append((f"trunc@{meta // 2}", bytes(b[:meta // 2])))
        else:
            a, c = rng.randrange(meta), rng.randrange(meta); b[a], b[c] = b[c], b[a]; out.append((f"swap@{a},{c}", bytes(b)))
    return out


def structural_edits(out_dir):
    """deliberately malformed synthetic files, each a named rule the readers must refuse"""
    cases = {}

    def base(**kw):
        w = GgufWriter(**kw)
        w.add_str("general.architecture", "demo").add_u32("demo.block_count", 2)
        return w

    w = base(); w.declared_tensor_count = 1 << 40; w.add_tensor("a", "F32", [8]); cases["tensor_count_inflated"] = w.build()
    w = base(); w.declared_kv_count = 1 << 40; w.add_tensor("a", "F32", [8]); cases["kv_count_inflated"] = w.build()
    w = base(); w.add_tensor("a", "F32", [8]); blob = bytearray(w.build())
    i = blob.find(b"general.architecture") - 8; blob[i:i + 8] = struct.pack("<Q", 1 << 40); cases["string_len_2^40"] = bytes(blob)
    w = base(); w.add_arr("demo.arr", U32, [1, 2, 3]); w.add_tensor("a", "F32", [8]); blob = bytearray(w.build())
    i = blob.find(b"demo.arr") + len(b"demo.arr") + 4 + 4; blob[i:i + 8] = struct.pack("<Q", 1 << 31); cases["array_len_2^31"] = bytes(blob)
    w = base(); w.add_tensor("a", "F32", [8]); blob = bytearray(w.build())
    i = blob.find(b"demo.arr"); nested = struct.pack("<Q", 7) + b"demo.nst" + struct.pack("<I", ARR) + struct.pack("<I", ARR) + struct.pack("<Q", 1) + struct.pack("<I", U32) + struct.pack("<Q", 1) + struct.pack("<I", 5)
    j = blob.find(b"demo.block_count") - 8; kvn = struct.unpack("<Q", blob[16:24])[0]; blob[16:24] = struct.pack("<Q", kvn + 1); blob[j:j] = nested; cases["nested_array"] = bytes(blob)
    for al in (0, 3, 1 << 31):
        w = GgufWriter(); w.add("general.alignment", U32, al); w.add_str("general.architecture", "demo"); w.add_tensor("a", "F32", [8]); cases[f"alignment_{al}"] = w.build()
    w = base(); w.add_tensor("a", "F32", [8]); blob = bytearray(w.build())
    i = blob.find(struct.pack("<Q", 1) + b"a", 24); i = i + 8 + 1 + 4 + 8 + 4   # name, n_dims, ne[0], type -> offset
    blob[i:i + 8] = struct.pack("<Q", 1 << 40); cases["offset_beyond_data"] = bytes(blob)
    w = base(); w.add_tensor("a", "F32", [8]); w.add_tensor("a", "F32", [8]); cases["duplicate_name"] = w.build()
    w = base(); w.add_tensor("a", "F32", [1 << 40, 1 << 40], payload=b"\0" * 32); cases["ne_overflow"] = w.build()
    w = base(); w.add_tensor("a", "F32", [2, 2, 2, 2, 2], payload=b"\0" * 128); cases["n_dims_5"] = w.build()
    w = base(); w.add_tensor("a", 200, [8], payload=b"\0" * 32); cases["unknown_type_200"] = w.build()
    for v in (2, 4):
        w = GgufWriter(version=v); w.add_str("general.architecture", "demo"); w.add_tensor("a", "F32", [8]); cases[f"version_{v}"] = w.build()
    w = base(); w.add_tensor("a", "F32", [64]); blob = w.build(); cases["payload_short"] = blob[:-100]
    w = GgufWriter(magic=b"GGUX"); w.add_str("general.architecture", "demo"); w.add_tensor("a", "F32", [8]); cases["bad_magic"] = w.build()
    w = base(); w.add_tensor("a", "Q4_K", [100], payload=b"\0" * 144); cases["q4k_not_whole_blocks"] = w.build()
    paths = {}
    for name, blob in cases.items():
        p = out_dir / f"edit_{name}.gguf"; p.write_bytes(blob); paths[name] = p
    return paths


# ---- cases --------------------------------------------------------------------------------------
def case0(tmp):
    print("case 0: fixtures and test binaries")
    fx = {}
    hf = tmp / "q_hf"; rc, out, _ = run([sys.executable, HERE / "make_tiny_qwen36_hf.py", hf])
    check(rc == 0, "tiny Qwen snapshot")
    g = tmp / "q_f32.gguf"; rc, out, _ = run([sys.executable, ST2GGUF, hf, "--out", g, "--type", "f32"])
    if check(rc == 0 and g.exists(), "tiny Qwen F32 GGUF"):
        fx["qwen"] = g
    sd = tmp / "q_split"; sd.mkdir(); rc, out, _ = run([sys.executable, ST2GGUF, hf, "--out", sd / "q.gguf", "--type", "f32", "--split", "3"])
    if check(rc == 0 and len(list(sd.glob("*.gguf"))) == 3, "tiny Qwen split set (3 parts)"):
        fx["split"] = sd
    gh = tmp / "g_hf"; rc, out, _ = run([sys.executable, HERE / "make_tiny_glm_hf.py", gh, "--vocab", "320"])
    check(rc == 0, "tiny GLM snapshot (vocab 320)")
    gt = tmp / "glm_tok.json"; rc, out, _ = run([sys.executable, HERE / "make_tiny_glm_tok.py", gt, "--vocab", "320"])
    if check(rc == 0, "GLM-style tokenizer.json (cl100k pre-tokenizer, 320 ids)"):
        fx["glm_tok"] = gt
    gg = tmp / "glm_f32.gguf"; rc, out, _ = run([sys.executable, ST2GGUF, gh, "--arch", "glm-dsa", "--out", gg, "--type", "f32", "--tokenizer", gt])
    if check(rc == 0 and gg.exists(), f"tiny GLM F32 GGUF with the tokenizer ({(out.strip().splitlines() or [''])[-1][:100]})"):
        fx["glm"] = gg
    qref = tmp / "qref.json"; qref.write_text(json.dumps({"prompt_ids": es.PROMPT, "full_ids": es.PROMPT + list(range(32, 32 + es.NGEN))})); fx["qref"] = qref
    gref = tmp / "gref.json"; gref.write_text(json.dumps({"prompt_ids": [1, 2, 3, 4, 5], "full_ids": [1, 2, 3, 4, 5] + list(range(32, 48)), "tf_pred": list(range(32, 48)) + [1, 2, 3, 4]})); fx["gref"] = gref
    check(make("tests/test_gguf"), "tests/test_gguf builds")
    fx["tok_built"] = check(make("tests/test_tok_gguf_glm"), "tests/test_tok_gguf_glm builds")
    return fx


def case1(fx, tmp, rng, n):
    print(f"case 1: mutation sweep over the readers ({n} mutations + structural edits)")
    if "qwen" not in fx:
        return {}
    blob = fx["qwen"].read_bytes()
    try:
        data_off, _, _ = boundaries(fx["qwen"])
    except Exception:  # noqa: BLE001
        data_off = None
    muts = mutations(blob, data_off, rng, n)
    mdir = tmp / "mut"; mdir.mkdir(exist_ok=True)
    verdicts = {"c": {}, "py": {}}; bad = []
    files = []
    for label, b in muts:
        p = mdir / (label.replace("@", "_").replace(",", "_").replace("+", "_") + ".gguf"); p.write_bytes(b); files.append((label, p))
    edits = structural_edits(mdir)
    for name, p in edits.items():
        files.append((f"edit:{name}", p))
    for label, p in files:
        vc, tc = c_reader(p); vp, tp = py_reader(p)
        verdicts["c"][label] = vc; verdicts["py"][label] = vp
        if vc in ("crash", "silent", "hang"):
            bad.append(f"C reader {vc} on {label}: {tc[-120:]}")
        if vp == "foreign":
            bad.append(f"ggufinfo raised a foreign exception on {label}: {tp[:160]}")
    for b in bad[:12]:
        print("      " + b)
    check(not bad, f"readers: no crash, hang, silent refusal or foreign exception ({len(bad)} bad of {len(files)})")
    acc_c = sum(v == "ok" for v in verdicts["c"].values()); acc_p = sum(v == "ok" for v in verdicts["py"].values())
    print(f"       C reader: {acc_c} accepted / {len(files) - acc_c} refused · ggufinfo: {acc_p} accepted / {len(files) - acc_p} refused")
    check(acc_c >= 1 and acc_p >= 1, "each reader accepts at least one mutation (the sweep is not trivial)")
    disagree = [n for n in edits if verdicts["c"][f"edit:{n}"] != verdicts["py"][f"edit:{n}"] and "hang" not in (verdicts["c"][f"edit:{n}"], verdicts["py"][f"edit:{n}"])]
    check(not disagree, f"structural edits: C reader and ggufinfo agree ({disagree or 'all'})")
    refused_edits = [n for n in edits if verdicts["c"][f"edit:{n}"] == "refused"]
    must = {"tensor_count_inflated", "kv_count_inflated", "string_len_2^40", "array_len_2^31", "offset_beyond_data", "duplicate_name", "ne_overflow", "version_4", "payload_short", "bad_magic", "q4k_not_whole_blocks"}
    missing = sorted(must - set(refused_edits))
    check(not missing, f"every structural defect is refused by the C reader ({missing or 'all'})")
    return {"files": files, "edits": edits, "verdicts": verdicts}


def case2(fx, sweep):
    print("case 2: mutation sweep over the engines (every 8th mutation + every structural edit)")
    if "qwen" not in fx or not sweep:
        return
    v0, t0 = engine(QWEN36, fx["qwen"], fx["qref"])
    check(v0 == "ok" and "Matching tokens" in t0, "control: the pristine file runs")
    picks = [f for i, f in enumerate(sweep["files"]) if i % 8 == 0 or f[0].startswith("edit:")]
    bad = []; oks = 0; p = picks[0][1]
    for label, p in picks:
        v, t = engine(QWEN36, p, fx["qref"])
        if v == "ok":
            oks += 1
            if label.startswith("edit:"):
                bad.append(f"qwen36 ran a structurally broken file: {label}")
        elif v in ("crash", "hang", "silent"):
            bad.append(f"qwen36 {v} on {label}: {t[-160:]}")
    for b in bad[:10]:
        print("      " + b)
    check(not bad, f"qwen36: exit 0 or a named refusal on every pick ({len(picks)} picks, {oks} ran, {len(bad)} bad)")
    if "glm" in fx:
        blob = fx["glm"].read_bytes()
        try:
            data_off, _, _ = boundaries(fx["glm"])
        except Exception:  # noqa: BLE001
            data_off = None
        rng = random.Random(7); gm = mutations(blob, data_off, rng, 24); gdir = p.parent / "glm_mut"; gdir.mkdir(exist_ok=True)
        gbad = []
        for label, b in gm:
            q = gdir / (label.replace("@", "_").replace(",", "_").replace("+", "_") + ".gguf"); q.write_bytes(b)
            v, t = engine(COLIBRI, q, fx["gref"])
            if v in ("crash", "hang", "silent"):
                gbad.append(f"colibri {v} on {label}: {t[-160:]}")
        for b in gbad[:10]:
            print("      " + b)
        check(not gbad, f"colibri: exit 0 or a named refusal on every GLM mutation ({len(gm)}, {len(gbad)} bad)")


def case3(fx, tmp):
    print("case 3: truncated payload refuses at the first read past EOF")
    if "qwen" not in fx:
        return
    parts = ggufinfo.open_set(str(fx["qwen"])); ts = ggufinfo.all_tensors(parts)
    exps = [t for t in ts if "ffn_gate_exps" in t.name]; dense = [t for t in ts if t.name == "output.weight"]
    blob = fx["qwen"].read_bytes()
    if exps:
        last = max(exps, key=lambda t: t.off); cut = last.off + (last.nbytes or 0) // 2
        p = tmp / "q_cut_expert.gguf"; p.write_bytes(blob[:cut])
        v, out = engine(QWEN36, p, fx["qref"])
        check(v == "refused" and (("short read" in out) or ("past the end" in out)) and last.name in out and "Matching tokens" not in out,
              f"qwen36: cut inside {last.name}: refused naming the tensor (at open or at the first read), no output ({v})")
        vp, tp = py_reader(p)
        check(vp in ("refused", "ok"), f"ggufinfo on the cut file: {vp} ({tp[:80]})")
        rc, out, _ = run([sys.executable, COLI, "doctor", "--model", p, "--deep", "--gpu", "none"])
        check("Traceback" not in out and "[fail] model.gguf" in out and last.name in out,
              f"coli doctor --deep: the damage is a model.gguf.* failure naming the tensor, not config.arguments ({[l for l in out.splitlines() if '[fail]' in l][:1]})")
    if dense:
        d = dense[0]; cut = d.off + (d.nbytes or 0) // 2
        p = tmp / "q_cut_dense.gguf"; p.write_bytes(blob[:cut])
        v, out = engine(QWEN36, p, fx["qref"])
        check(v == "refused" and "Matching tokens" not in out, f"qwen36: cut inside output.weight: refused at load ({v})")
    if "glm" in fx:
        gb = fx["glm"].read_bytes(); gparts = ggufinfo.open_set(str(fx["glm"])); gts = ggufinfo.all_tensors(gparts)
        gexp = [t for t in gts if "ffn_gate_exps" in t.name]
        if gexp:
            last = max(gexp, key=lambda t: t.off); cut = last.off + (last.nbytes or 0) // 2
            p = tmp / "g_cut_expert.gguf"; p.write_bytes(gb[:cut])
            v, out = engine(COLIBRI, p, fx["gref"])
            check(v == "refused" and (("short read" in out) or ("past the end" in out)) and last.name in out, f"colibri: cut inside {last.name}: refused naming the tensor ({v})")


def patch_u16_value(path, key, value):
    b = bytearray(path.read_bytes()); i = b.find(key.encode())
    assert i >= 0, key
    j = i + len(key) + 4           # the 4-byte value type follows the key
    b[j:j + 2] = struct.pack("<H", value); path.write_bytes(bytes(b))


def case4(fx, tmp):
    print("case 4: split-set rules")
    if "split" not in fx:
        return
    parts = sorted(fx["split"].glob("*.gguf"))

    def verdicts(d):
        vp, tp = py_reader(d); vc, tc = c_reader(d)
        rc, dout, _ = run([sys.executable, COLI, "doctor", "--model", d, "--gpu", "none"])
        return vp, tp, vc, tc, dout

    # missing part
    d = tmp / "split_missing"; shutil.copytree(fx["split"], d); os.remove(sorted(d.glob("*.gguf"))[1])
    vp, tp, vc, tc, dout = verdicts(d)
    check(vp == "refused" and "split" in tp, f"missing part: ggufinfo refuses naming the split rule ({tp[:100]})")
    check(vc == "refused" and "split" in tc.lower(), f"missing part: C reader refuses naming the split rule ({tc.strip()[-100:]})")
    check("Traceback" not in dout and "[fail] model.gguf" in dout and "part 2" in dout,
          f"missing part: coli doctor reports it under model.gguf.* naming the part, not config.arguments ({[l[:60] for l in dout.splitlines() if '[fail]' in l][:1]})")
    # a part replaced by a copy of another (split.no disagrees)
    d = tmp / "split_dup"; shutil.copytree(fx["split"], d); ps = sorted(d.glob("*.gguf")); shutil.copyfile(ps[0], ps[1])
    vp, tp, vc, tc, dout = verdicts(d)
    check(vp == "refused" and "split" in tp, f"duplicated part: ggufinfo refuses ({tp[:100]})")
    check(vc == "refused", f"duplicated part: C reader refuses ({tc.strip()[-100:]})")
    # split.count disagrees in one part
    d = tmp / "split_count"; shutil.copytree(fx["split"], d); patch_u16_value(sorted(d.glob("*.gguf"))[2], "split.count", 5)
    vp, tp, vc, tc, dout = verdicts(d)
    check(vp == "refused" and "split" in tp, f"split.count disagrees: ggufinfo refuses ({tp[:100]})")
    check(vc == "refused", f"split.count disagrees: C reader refuses ({tc.strip()[-100:]})")
    # two stems in one directory
    d = tmp / "split_two"; shutil.copytree(fx["split"], d)
    for p in sorted(d.glob("*.gguf")):
        shutil.copyfile(p, d / p.name.replace("q-", "other-"))
    vp, tp, vc, tc, dout = verdicts(d)
    check(vp == "refused" and ("stem" in tp or "two" in tp or "sets" in tp), f"two split sets in one directory: ggufinfo refuses naming both ({tp[:120]})")
    check(vc == "refused", f"two split sets in one directory: C reader refuses ({tc.strip()[-100:]})")


def case5(fx, sweep):
    print("case 5: one reader -- ggufinfo vs tools/gguf_reader.py agree; the parser is shared")
    import gguf_reader
    files = [p for p in (fx.get("qwen"), fx.get("glm")) if p]
    if "split" in fx:
        files += sorted(fx["split"].glob("*.gguf"))
    if sweep:
        files += [p for label, p in sweep["files"] if sweep["verdicts"]["py"][label] == "ok"][:10]
    diffs = []
    for p in files:
        try:
            part = ggufinfo.read_part(str(p))
            with gguf_reader.GGUFReader(str(p)) as r:
                def trim(ne):
                    ne = list(ne)
                    while len(ne) > 1 and ne[-1] == 1:
                        ne.pop()
                    return ne
                mine = {t.name: (t.type, trim(t.ne), t.rel_off) for t in part.tensors}
                theirs = {t["name"]: (t["ggml_type"], trim(t["dims"]), t["offset"]) for t in r.tensors}
                if mine != theirs:
                    diffs.append(f"{p.name}: tensor tables differ")
                mk = {k for k, v in part.kv.items() if v is not None}; tk = set(r.metadata)
                if mk - tk:
                    diffs.append(f"{p.name}: keys only in ggufinfo: {sorted(mk - tk)[:3]}")
                for k in mk & tk:
                    a, b = part.kv[k], r.metadata[k]
                    if isinstance(a, (int, str, bool)) and a != b and not isinstance(a, float):
                        diffs.append(f"{p.name}: {k}: {a!r} vs {b!r}")
        except (ggufinfo.GgufError, gguf_reader.GGUFError) as ex:
            if p.parent.name != "mut":      # a mutated file: a stricter rule on one side is allowed; a fixture is not
                diffs.append(f"{p.name}: one reader refused what the other read: {ex}")
    for d in diffs[:8]:
        print("      " + d)
    check(not diffs, f"differential: {len(files)} files, tensor tables and metadata equal")
    impl = getattr(ggufinfo, "GGUF_READER_IMPL", None)
    check(isinstance(impl, str) and impl.endswith("gguf_reader.py"), f"ggufinfo.GGUF_READER_IMPL names tools/gguf_reader.py as the parser ({impl!r})")


def case6(fx):
    print("case 6: the GLM tokenizer from the GGUF equals its tokenizer.json twin")
    if not fx.get("tok_built") or "glm" not in fx:
        return
    rc, out, _ = run([T_TOK, fx["glm"], fx["glm_tok"], CORPUS])
    for ln in out.splitlines():
        if ln.startswith("FAIL"):
            print("      " + ln)
    check(rc == 0 and "all passed" in out, "test_tok_gguf_glm: identical ids and pieces on the corpus, control tokens by name")
    rc, out, _ = run([T_TOK, fx["glm"], "--encode", CORPUS])
    n_lines = len([ln for ln in CORPUS.read_text(encoding="utf-8").split("\n") if True]) - (1 if CORPUS.read_text(encoding="utf-8").endswith("\n") else 0)
    check(rc == 0 and len([ln for ln in out.splitlines() if not ln.startswith("[")]) >= n_lines - 1, "--encode prints one id line per corpus line")
    if "qwen" in fx:
        rc, out, _ = run([T_TOK, fx["qwen"], "--expect-refusal"])
        check(rc == 0 and "refused as expected" in out, "a qwen35 pre-tokenizer is refused by the GLM constructor, not guessed")


def case7(fx, tmp):
    print("case 7: coli doctor on a truncated file")
    p = tmp / "q_cut_expert.gguf"
    if not p.exists():
        return
    rc, out, _ = run([sys.executable, COLI, "doctor", "--model", p, "--deep", "--gpu", "none"])
    check("Traceback" not in out and ("[fail] model.gguf.payload" in out or "[fail] model.gguf.header" in out), f"model.gguf.payload (or header) fails, no traceback ({[l[:60] for l in out.splitlines() if '[fail]' in l][:1]})")


def main(argv):
    keep = "--keep" in argv
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 20261014
    n = int(argv[argv.index("--mutations") + 1]) if "--mutations" in argv else 160
    if "--no-build" not in argv and not (make("qwen36") and make("colibri")):
        print("RESULT: FAIL (engines do not build)"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="sylph_hard_"))
    rng = random.Random(seed)
    try:
        fx = case0(tmp)
        sweep = case1(fx, tmp, rng, n)
        case2(fx, sweep)
        case3(fx, tmp)
        case4(fx, tmp)
        case5(fx, sweep)
        case6(fx)
        case7(fx, tmp)
    finally:
        if keep:
            print(f"fixtures kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
