#!/usr/bin/env python3
"""Integration test for GGUF expert streaming (phase 4). See expert_streaming.md.

Standard library only. Regenerates the torch-free tiny snapshot, a 320-token
byte-level tokenizer, the int8 container (tools/convert_qwen36.py) and the GGUFs
(tools/st2gguf.py, incl. a 3-part split), then drives the qwen36 engine in
reference-id mode, PPL-dump mode and serve mode. Exit 0 iff every case passes.

    python3 07_Tests/IntegrationTest/run_expert_streaming.py [--keep] [--no-build] [--baseline <qwen36 binary>]
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
QWEN36 = C / f"qwen36{EXE}"
ST2GGUF = C / "tools" / "st2gguf.py"
CONVERT = C / "tools" / "convert_qwen36.py"
COLI = C / "coli"
PROMPT = [1, 2, 3, 4, 5]
NGEN = 16

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def run(args, env=None, **kw):
    e = dict(os.environ)
    e.pop("SNAP", None); e.pop("TOK", None); e.pop("PPL", None); e.pop("PPL_DUMP", None)
    e.setdefault("OMP_NUM_THREADS", "2")
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, errors="replace", check=False, env=e, **kw)


def build_engine():
    if QWEN36.exists():
        return True
    return run(["make", "-C", C, "qwen36"]).returncode == 0


# ---- the 320-token byte-level tokenizer -----------------------------------------------
def byte_symbols():
    """GPT-2 byte -> unicode mapping, as build_byte_sym in qwen36.c."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    syms, n = [], 0
    for b in range(256):
        if b in keep:
            syms.append(chr(b))
        else:
            syms.append(chr(256 + n)); n += 1
    return syms


def write_tiny_tok(path):
    syms = byte_symbols()
    vocab = {s: i for i, s in enumerate(syms)}
    G = syms[32]  # the symbol of the space byte ('Ġ')
    pairs = ["t h", "th e", "i n", "e r", "a n", "r e", "o n", "a t", "e n", "o r", "e s", "i s", "i t", "a l", "a r", "s t",
             "o u", "n d", "in g", "l e", "o f", "a s", "h e", "t o", "e d", "i c", "l l", "r o", "m e", "d e", "c o", "n e",
             f"{G} t", f"{G} a", f"{G} th", f"{G} s", f"{G} o", f"{G} w", f"{G} i", f"{G} c", f"{G} b", f"{G} f", f"{G} m",
             f"{G} d", f"{G} p", f"{G} h", f"{G} l", f"{G} n", f"{G} e", f"{G} r", f"{G} g", f"{G} u", f"{G} y", f"{G} k",
             f"{G}th e", f"{G}a n", f"{G}o f", f"{G}t o", f"{G}i n", f"{G}i s", f"{G}i t", f"{G}a nd", f"{G}th at"]
    merges = []
    for p in pairs:
        a, b = p.split(" ")
        if a not in vocab or b not in vocab or (a + b) in vocab:
            continue
        vocab[a + b] = len(vocab); merges.append(p)
        if len(vocab) == 319:
            break
    assert len(vocab) == 319, len(vocab)
    tok = {"model": {"type": "BPE", "vocab": vocab, "merges": merges},
           "added_tokens": [{"id": 319, "content": "<|im_end|>", "special": True}],
           "pre_tokenizer": {"type": "Sequence", "pretokenizers": []}}
    Path(path).write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")


# ---- engine drivers -------------------------------------------------------------------
def parse_ids(text):
    m = re.search(r"C engine :\s*((?:\d+\s+)+)", text)
    return [int(x) for x in m.group(1).split()] if m else None


def parse_hit(text):
    m = re.search(r"Expert cache hit rate: ([0-9.]+)% \(hit=(\d+) miss=(\d+)\)", text)
    return (int(m.group(2)), int(m.group(3))) if m else None


def ref_run(snap, cap, ref, env=None, binary=None):
    e = {"SNAP": snap, "COLI_DENSE_I8": "0"}
    if env:
        e.update(env)
    r = run([binary or QWEN36, cap, 8, ref], env=e, cwd=C)
    out = r.stdout + r.stderr
    # reference mode exits 1 when the ids differ from ref.json's (ours are arbitrary):
    # "ran" means the summary was printed, not that the exit code was 0
    r.ran = "Matching tokens:" in out and parse_ids(out) is not None
    return r, out, parse_ids(out), parse_hit(out)


def ppl_run(snap, cap, ref, dump, env=None):
    """Returns (result, dump body without the header line, output). The header names
    the SNAP path, which differs between the single file and the parts."""
    e = {"SNAP": snap, "COLI_DENSE_I8": "0", "PPL": "1", "PPL_DUMP": dump}
    if env:
        e.update(env)
    r = run([QWEN36, cap, 8, ref], env=e, cwd=C)
    body = None
    if Path(dump).exists():
        b = Path(dump).read_bytes(); body = b.split(b"\n", 1)[1] if b.startswith(b"#") else b
    return r, body, r.stdout + r.stderr


def serve_session(snap, prompts, tok=None, env=None):
    """Drive SERVE=1: returns (ok, frames per prompt, log). Frames = list of (header, bytes)."""
    e = dict(os.environ); e.pop("TOK", None); e.pop("PPL", None); e.pop("PPL_DUMP", None)
    e.update({"SNAP": str(snap), "SERVE": "1", "COLI_DENSE_I8": "0", "OMP_NUM_THREADS": "2"})
    if tok:
        e["TOK"] = str(tok)
    if env:
        e.update(env)
    p = subprocess.Popen([str(QWEN36), "8", "8"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e, cwd=str(C))
    log = []
    try:
        line = p.stdout.readline()
        while line and b"READY" not in line:
            log.append(line); line = p.stdout.readline()
        if not line:
            return False, [], b"".join(log) + p.stderr.read()
        results = []
        for i, pr in enumerate(prompts):
            pb = pr.encode("utf-8")
            p.stdin.write(f"SUBMIT r{i} 0 {len(pb)} 24 0 1 logprobs=5\n".encode()); p.stdin.write(pb + b"\n"); p.stdin.flush()
            frames = []; accept = None; done = None
            while True:
                hdr = p.stdout.readline()
                if not hdr:
                    break
                log.append(hdr)
                if hdr.startswith(b"ACCEPT"):
                    accept = hdr.decode().split()
                elif hdr.startswith(b"DATA"):
                    n = int(hdr.split()[2])
                    body = p.stdout.read(n); p.stdout.read(1)
                    frames.append((hdr.decode("utf-8", "replace").strip(), body))
                elif hdr.startswith(b"DONE"):
                    done = hdr.decode().strip(); break
                elif hdr.startswith(b"ERROR"):
                    done = hdr.decode().strip(); break
            results.append({"accept": accept, "frames": frames, "done": done})
        p.stdin.close()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        return True, results, b"".join(log) + p.stderr.read()
    finally:
        if p.poll() is None:
            p.kill()


def startup_line(text):
    for ln in text.splitlines():
        if ln.startswith("[GGUF] ") and "blocks" in ln:
            return ln
    return ""


def gguf_reads(text):
    m = re.search(r"GGUF reads: (\d+) slices · ([0-9.]+) MB · ([0-9.]+) MB/token · (\d+) part", text)
    return (int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))) if m else None


# ---- cases ----------------------------------------------------------------------------
def case0(tmp):
    print("case 0: fixtures (snapshot, 320-token tokenizer, container, GGUFs, split set)")
    hf = tmp / "tiny_hf"
    r = run([sys.executable, HERE / "make_tiny_qwen36_hf.py", hf])
    check(r.returncode == 0, "torch-free snapshot")
    tok = tmp / "tiny_tok.json"; write_tiny_tok(tok)
    tj = json.loads(tok.read_text(encoding="utf-8"))
    check(len(tj["model"]["vocab"]) == 319 and tj["added_tokens"][0]["id"] == 319, "320-token byte-level tokenizer (256 bytes + 63 merges + <|im_end|>)")
    cont = tmp / "tiny_c"
    fx = {"hf": hf, "tok": tok}
    r = run([sys.executable, CONVERT, "--model", hf, "--out", cont, "--ebits", "8", "--no-readme"])
    if r.returncode == 0 and (cont / "qwen36_meta.json").exists():
        shutil.copy(tok, cont / "tokenizer.json"); fx["c"] = cont
        check(True, "container (convert_qwen36 --ebits 8)")
    elif "torch" in (r.stderr + r.stdout):
        print("  skip container: tools/convert_qwen36.py needs torch (not installed here); the gguf-oracle CI job carries the container comparisons")
    else:
        check(False, f"container (convert_qwen36 --ebits 8): {(r.stderr.strip().splitlines() or [''])[-1][:160]}")
    for tag, extra in (("f32", ["--type", "f32"]), ("q8_0", ["--type", "q8_0", "--expert-type", "q8_0"]), ("xq8_0", ["--type", "f32", "--expert-type", "q8_0"])):
        p = tmp / f"tiny_{tag}.gguf"
        r = run([sys.executable, ST2GGUF, hf, "--out", p] + extra + ["--tokenizer", tok])
        if check(r.returncode == 0 and p.exists(), f"st2gguf {tag} with the tokenizer"):
            fx[tag] = p
    split_dir = tmp / "split"; split_dir.mkdir()
    r = run([sys.executable, ST2GGUF, hf, "--out", split_dir / "tiny_split.gguf", "--type", "f32", "--tokenizer", tok, "--split", "3"])
    parts = sorted(split_dir.glob("tiny_split-*-of-00003.gguf"))
    if check(r.returncode == 0 and len(parts) == 3, f"st2gguf --split 3 writes three parts ({[q.name for q in parts]}; {(r.stderr.strip().splitlines() or [''])[-1][:160]})"):
        fx["split_dir"] = split_dir; fx["parts"] = parts
        try:
            ps = ggufinfo.open_set(parts[0]); s = ggufinfo.summarize(ps)
            single = ggufinfo.summarize(ggufinfo.open_set(fx["f32"]))
            check(len(ps) == 3 and s["tensors"] == single["tensors"], f"ggufinfo: 3 parts, {s['tensors']} tensors as the single file")
        except Exception as ex:  # noqa: BLE001
            check(False, f"ggufinfo on the split set: {ex}")
    (tmp / "empty_mirror").mkdir(exist_ok=True)
    ref = tmp / "ref.json"
    ref.write_text(json.dumps({"prompt_ids": PROMPT, "full_ids": PROMPT + list(range(32, 32 + NGEN))}))
    fx["ref"] = ref
    fx["created"] = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*"))
    return fx


def case1(fx):
    print("case 1: capacity parity (FR-27): ids identical across caps; hit/miss GGUF vs container")
    res = {}
    for src in ("f32", "c"):
        if src not in fx:
            continue
        ids_by_cap, hit_by_cap = {}, {}
        for cap in (1, 2, 8):
            r, out, ids, hit = ref_run(fx[src], cap, fx["ref"])
            check(r.ran and len(ids) == NGEN, f"{src} cap {cap}: runs, {NGEN} generated ids")
            ids_by_cap[cap] = ids; hit_by_cap[cap] = hit
        same = len({tuple(v) for v in ids_by_cap.values() if v}) == 1
        check(same, f"{src}: ids identical at cap 1, 2, 8")
        res[src] = (ids_by_cap, hit_by_cap)
    if "f32" in res and "c" in res:
        for cap in (1, 2, 8):
            gi, gh = res["f32"][0][cap], res["f32"][1][cap]
            ci, ch = res["c"][0][cap], res["c"][1][cap]
            if gi == ci:
                check(gh == ch, f"cap {cap}: identical ids -> identical hit/miss (GGUF {gh}, container {ch})")
            else:
                print(f"  note cap {cap}: ids differ (int8 vs f32 experts); hit/miss GGUF {gh}, container {ch} (reported)")
    return res.get("f32", ({}, {}))[0].get(8), res.get("c", ({}, {}))[0].get(8)


def case2(fx, tmp):
    print("case 2: thread determinism (PPL dumps OMP 1 vs 4)")
    for tag in ("f32", "q8_0"):
        if tag not in fx:
            continue
        d1, d4 = tmp / f"d_{tag}_t1.tsv", tmp / f"d_{tag}_t4.tsv"
        r1, b1, _ = ppl_run(fx[tag], 8, fx["ref"], d1, {"OMP_NUM_THREADS": "1"})
        r4, b4, _ = ppl_run(fx[tag], 8, fx["ref"], d4, {"OMP_NUM_THREADS": "4"})
        check(r1.returncode == 0 and r4.returncode == 0 and b1 is not None and b1 == b4, f"{tag}: dumps byte-identical with 1 and 4 threads")


def case3(fx, ids_f32, ids_c):
    print("case 3: pilot and pins produce the same ids")
    for src, base in (("f32", ids_f32), ("c", ids_c)):
        if src not in fx or base is None:
            continue
        for env in ({"PILOT": "1"}, {"PILOT": "1", "PILOT_REAL": "1"}, {"HOT": "2"}):
            r, out, ids, _ = ref_run(fx[src], 8, fx["ref"], env)
            tag = " ".join(f"{k}={v}" for k, v in env.items())
            ok = r.ran and ids == base
            if "HOT" in env:
                ok = ok and "[HOT] Pinned" in out
            check(ok, f"{src} {tag}: exit 0, ids as the plain run" + (" , [HOT] line" if "HOT" in env else ""))


def case4(fx, tmp, ids_f32):
    print("case 4: split set loads from the directory and from any part")
    if "parts" not in fx or ids_f32 is None:
        check(False, "split fixture missing (case 0)"); return
    _, base, _ = ppl_run(fx["f32"], 8, fx["ref"], tmp / "d_single.tsv")
    for snap in (fx["split_dir"], fx["parts"][0], fx["parts"][1]):
        r, out, ids, _ = ref_run(snap, 1, fx["ref"])
        sl = startup_line(out)
        check(r.ran and ids == ids_f32, f"SNAP={snap.name}: ids as the single file")
        check("3 part" in sl, f"SNAP={snap.name}: startup line counts 3 parts ({sl[:90]})")
        rd = gguf_reads(out)
        if snap == fx["split_dir"]:
            check(rd is not None and rd[3] >= 2, f"stats: parts touched >= 2 at cap 1 ({rd})")
        d = tmp / f"d_{snap.name}.tsv"; _, b, _ = ppl_run(snap, 8, fx["ref"], d)
        check(base is not None and b == base, f"SNAP={snap.name}: PPL dump identical to the single file")
    hidden = fx["parts"][2].with_suffix(".hidden")
    os.rename(fx["parts"][2], hidden)
    try:
        r, out, _, _ = ref_run(fx["parts"][0], 1, fx["ref"])
        check(not r.ran and "00003" in out, "missing part 3 refused by name")
    finally:
        os.rename(hidden, fx["parts"][2])


LABELS = ["blocks (", "attention)", "part", "experts ", "MB each", "dense ", "token_embd ", "output ", "experts on CPU", "sidecars "]


def case5(fx, tmp):
    print("case 5: startup line (FR-30)")
    for tag in ("f32", "q8_0"):
        if tag not in fx:
            continue
        r, out, _, _ = ref_run(fx[tag], 8, fx["ref"])
        sl = startup_line(out)
        missing = [l for l in LABELS if l not in sl]
        check(not missing, f"{tag}: all labels present (missing {missing}) :: {sl[:120]}")
        if tag == "q8_0":
            check("token_embd Q8_0 on demand" in sl, "q8_0: token_embd Q8_0 on demand")
        else:
            check("output F32 matmul_d" in sl, "f32: output F32 matmul_d")
        sd = sidecar_dir_py(fx[tag])
        check(sd is not None and sd in sl, f"{tag}: sidecar path in the startup line equals family_registry.sidecar_dir ({sd})")


def case6(fx):
    print("case 6: reads accounting (slices = 3 x miss, MB = miss x expert bytes)")
    if "f32" not in fx:
        return
    s = ggufinfo.summarize(ggufinfo.open_set(fx["f32"]))
    exp_bytes = s.get("typical_expert_bytes")  # bytes per expert (three slices), as ggufinfo reports
    for cap in (1, 8):
        r, out, _, hit = ref_run(fx["f32"], cap, fx["ref"])
        rd = gguf_reads(out)
        if not check(rd is not None and hit is not None, f"cap {cap}: 'GGUF reads:' line present ({rd})"):
            continue
        slices, mb, mbtok, _ = rd
        check(slices == 3 * hit[1], f"cap {cap}: {slices} slices = 3 x {hit[1]} misses")
        if exp_bytes:
            want = hit[1] * exp_bytes / 1e6
            check(abs(mb - want) <= 0.01 * want + 0.005, f"cap {cap}: {mb} MB vs misses x expert bytes {want:.3f} MB")
        check(abs(mbtok - mb / NGEN) <= 0.01 * mb / NGEN + 0.005, f"cap {cap}: MB/token = MB / {NGEN}")


def sidecar_dir_py(path):
    try:
        import family_registry  # noqa: E402
        return str(family_registry.sidecar_dir(str(path)))
    except Exception as ex:  # noqa: BLE001
        print(f"  note sidecar_dir: {type(ex).__name__}: {ex}")
        return None


def case7(fx, tmp):
    print("case 7: sidecar hygiene and API (FR-29)")
    now = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if not p.name.startswith("d_") and p.suffix != ".tsv")
    extra = sorted(set(now) - set(fx["created"]))
    check(not extra, f"nothing written beside the model or in the scratch dir (extra: {extra[:6]})")
    stray = [p for p in C.glob("qwen36_logits.f32")]
    check(not stray, "no stray dump in the engine directory")
    if "f32" in fx:
        want = str(tmp / ".coli-tiny_f32") 
        got = sidecar_dir_py(fx["f32"])
        check(got is not None and got.rstrip("/\\") == want, f"sidecar_dir(file) = {got}")
    if "parts" in fx:
        want = str(fx["split_dir"] / ".coli-tiny_split")
        check((sidecar_dir_py(fx["split_dir"]) or "").rstrip("/\\") == want and (sidecar_dir_py(fx["parts"][1]) or "").rstrip("/\\") == want,
              "sidecar_dir(parts dir) = sidecar_dir(part 2) = <dir>/.coli-tiny_split")
    if "c" in fx:
        check((sidecar_dir_py(fx["c"]) or "").rstrip("/\\") == str(fx["c"]), "sidecar_dir(container) = the model directory")
    if "f32" in fx and COLI.exists():
        r = run([sys.executable, COLI, "info", "--model", fx["f32"]])
        check(r.returncode == 0 and "sidecars" in r.stdout and "none yet" in r.stdout, "coli info prints the sidecar path with (none yet)")


def case8(fx, tmp):
    print("case 8: embedding on demand == f32 at load")
    if "q8_0" not in fx:
        return
    d1, d2 = tmp / "d_embd_demand.tsv", tmp / "d_embd_load.tsv"
    r1, b1, o1 = ppl_run(fx["q8_0"], 8, fx["ref"], d1)
    r2, b2, o2 = ppl_run(fx["q8_0"], 8, fx["ref"], d2, {"COLI_GGUF_EMBED": "0"})
    check(r1.returncode == 0 and r2.returncode == 0 and b1 is not None and b1 == b2, "dumps byte-identical")
    check("on demand" in startup_line(o1) and "f32 at load" in startup_line(o2), "startup line names the mode")


def case9(fx, ids_f32, ids_c, tmp):
    print("case 9: FR-28 knobs ignored identically (qwen36 has no DIRECT/URING/mirror/split-dir paths)")
    empty = tmp / "empty_mirror"
    env = {"DIRECT": "1", "URING": "1", "PIPE": "1", "COLI_MODEL_MIRROR": str(empty), "COLI_MODEL_DIRS": str(empty)}
    for src, base in (("f32", ids_f32), ("c", ids_c)):
        if src not in fx or base is None:
            continue
        r, out, ids, _ = ref_run(fx[src], 8, fx["ref"], env)
        check(r.ran and ids == base and "[MIRROR]" not in out and "URING" not in out, f"{src}: runs, ids unchanged, no mirror/uring lines")


def case10(fx, ids_f32):
    print("case 10: CUDA tier note (FR-36)")
    if "f32" not in fx or ids_f32 is None:
        return
    r, out, ids, _ = ref_run(fx["f32"], 8, fx["ref"], {"COLI_CUDA": "1"})
    note = "the VRAM expert tier does not take GGUF K-quant experts yet"
    check(r.ran and ids == ids_f32 and out.count(note) == 1, "note printed once, ids unchanged")


def case11(fx):
    print("case 11: serve smoke (tokenizer from metadata == TOK=json)")
    if "f32" not in fx:
        return
    prompts = ["the cat sat on the mat and then", "Grüße ä€🐦 end"]
    ok1, a, log1 = serve_session(fx["f32"], prompts)
    ok2, b, log2 = serve_session(fx["f32"], prompts, tok=fx["tok"])
    check(ok1 and ok2, "both sessions reach READY")
    check(b"tokenizer.json required" not in log1, "GGUF without TOK does not demand tokenizer.json")
    if ok1 and ok2 and len(a) == len(b) == 2:
        for i in range(2):
            check(a[i]["accept"] == b[i]["accept"] and a[i]["accept"] and a[i]["accept"][0] == "ACCEPT", f"prompt {i}: ACCEPT equal ({a[i]['accept']})")
            check(a[i]["frames"] == b[i]["frames"] and len(a[i]["frames"]) > 0, f"prompt {i}: {len(a[i]['frames'])} DATA frames identical (bytes and logprob tails)")
            check(a[i]["done"] and a[i]["done"].startswith("DONE") and "STAT" in a[i]["done"], f"prompt {i}: DONE … STAT ({a[i]['done']})")


def case12(fx, ids_c, baseline):
    print("case 12: container untouched (NFR-5)")
    if not baseline or "c" not in fx:
        print("  note: no --baseline binary given; the CI job's container run is the record"); return
    base = Path(baseline)
    for cap in (1, 2, 8):
        r0, _, i0, h0 = ref_run(fx["c"], cap, fx["ref"], binary=base)
        r1, _, i1, h1 = ref_run(fx["c"], cap, fx["ref"])
        check(r0.ran and r1.ran and i0 == i1 and h0 == h1, f"cap {cap}: baseline and current binary agree on ids and hit/miss")


def main(argv):
    keep = "--keep" in argv
    baseline = argv[argv.index("--baseline") + 1] if "--baseline" in argv else None
    if "--no-build" not in argv and not build_engine():
        print("RESULT: FAIL (qwen36 does not build)"); return 1
    tmp = Path(tempfile.mkdtemp(prefix="sylph_stream_"))
    try:
        fx = case0(tmp)
        ids_f32, ids_c = case1(fx)
        case2(fx, tmp)
        case3(fx, ids_f32, ids_c)
        case4(fx, tmp, ids_f32)
        case5(fx, tmp)
        case6(fx)
        case8(fx, tmp)
        case9(fx, ids_f32, ids_c, tmp)
        case10(fx, ids_f32)
        case11(fx)
        case7(fx, tmp)
        case12(fx, ids_c, baseline)
    finally:
        if keep:
            print(f"fixtures kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
