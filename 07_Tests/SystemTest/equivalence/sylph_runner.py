#!/usr/bin/env python3
"""sylph side of the equivalence harness: tokenizer ids, teacher-forced windows, greedy
generation over the serve protocol. Standard library only.

Every engine call caches its outputs under <out>/sylph/<config>/ keyed by window or prompt,
so a re-run after a crash continues where it stopped (--force redoes).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C = ROOT / "06_Code" / "c"
EXE = ".exe" if sys.platform == "win32" else ""
QWEN36 = C / f"qwen36{EXE}"
T_TOK = C / "tests" / f"test_tok_gguf{EXE}"


def build():
    for target in ("qwen36", "tests/test_tok_gguf"):
        p = C / (target + EXE)
        if not p.exists():
            r = subprocess.run(["make", "-C", str(C), target], capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"build of {target} failed:\n{r.stderr[-2000:]}")


def base_env(extra=None, threads=None):
    e = dict(os.environ)
    for k in ("SNAP", "TOK", "PPL", "PPL_DUMP", "PPL_DUMP_FULL", "SERVE"):
        e.pop(k, None)
    e.setdefault("COLI_DENSE_I8", "0")
    if threads:
        e["OMP_NUM_THREADS"] = str(threads)
    if extra:
        e.update({k: str(v) for k, v in extra.items()})
    return e


def encode(gguf, text_path):
    """ids of a text file with the GGUF's tokenizer (tests/test_tok_gguf --encode)."""
    build()
    r = subprocess.run([str(T_TOK), str(gguf), "--encode", str(text_path)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"test_tok_gguf --encode failed: {r.stderr[-500:]}")
    return [int(x) for x in r.stdout.split()]


def ppl_window(gguf, ids, first_scored, out_dir, tag, cap=8, env=None, threads=None, full=False, force=False):
    """Teacher-forced run over `ids`, scoring positions first_scored…len-1.
    Returns dict(dump=path, full=path|None, tf_nll=float, n=int)."""
    build()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dump = out_dir / f"{tag}.tsv"; fullp = out_dir / f"{tag}.full"; meta = out_dir / f"{tag}.json"
    if meta.exists() and dump.exists() and (not full or fullp.exists()) and not force:
        return json.loads(meta.read_text())
    ref = out_dir / f"{tag}.ref.json"
    ref.write_text(json.dumps({"prompt_ids": ids[:first_scored], "full_ids": ids}))
    e = base_env(env, threads); e.update({"SNAP": str(gguf), "PPL": "1", "PPL_DUMP": str(dump)})
    if full:
        e["PPL_DUMP_FULL"] = str(fullp)
    r = subprocess.run([str(QWEN36), str(cap), "8", str(ref)], capture_output=True, text=True, errors="replace", env=e, cwd=str(C))
    m = re.search(r"TF-NLL: ([0-9.]+) nats/token over (\d+) tokens", r.stdout + r.stderr)
    if r.returncode != 0 or not m or not dump.exists():
        sys.exit(f"qwen36 PPL run failed for {tag}:\n{(r.stdout + r.stderr)[-1500:]}")
    res = {"dump": str(dump), "full": str(fullp) if full else None, "tf_nll": float(m.group(1)), "n": int(m.group(2)),
           "reads": re.search(r"GGUF reads: .*", r.stdout + r.stderr).group(0) if "GGUF reads:" in r.stdout + r.stderr else None}
    meta.write_text(json.dumps(res))
    return res


def ref_mode_ids(gguf, ref_json, cap=8, env=None, threads=None):
    """Greedy ids of the reference-id mode (`C engine :` line); the engine exits 1 when
    they differ from ref.json's, so the exit code is not an error here."""
    build()
    e = base_env(env, threads); e["SNAP"] = str(gguf)
    r = subprocess.run([str(QWEN36), str(cap), "8", str(ref_json)], capture_output=True, text=True, errors="replace", env=e, cwd=str(C))
    out = r.stdout + r.stderr
    m = re.search(r"C engine :\s*((?:\d+\s+)+)", out)
    if not m:
        sys.exit(f"qwen36 reference run failed:\n{out[-1500:]}")
    return [int(x) for x in m.group(1).split()]


class Serve:
    """One SERVE=1 engine process; greedy generation with logprobs=k per request."""

    def __init__(self, gguf, cap=8, env=None, threads=None, topk=10):
        build()
        e = base_env(env, threads); e.update({"SNAP": str(gguf), "SERVE": "1"})
        self.topk = topk
        self.p = subprocess.Popen([str(QWEN36), str(cap), "8"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=e, cwd=str(C))
        line = self.p.stdout.readline()
        while line and b"READY" not in line:
            line = self.p.stdout.readline()
        if not line:
            sys.exit("qwen36 SERVE did not reach READY")
        self.n = 0

    def generate(self, prompt, n_predict=128):
        pb = prompt.encode("utf-8"); self.n += 1; rid = f"g{self.n}"
        self.p.stdin.write(f"SUBMIT {rid} 0 {len(pb)} {n_predict} 0 1 logprobs={self.topk}\n".encode()); self.p.stdin.write(pb + b"\n"); self.p.stdin.flush()
        ids, text, top, margin, np_ = [], b"", [], [], None
        while True:
            hdr = self.p.stdout.readline()
            if not hdr:
                raise RuntimeError("engine died")
            if hdr.startswith(b"ACCEPT"):
                np_ = int(hdr.split()[2])
            elif hdr.startswith(b"DATA"):
                parts = hdr.decode("utf-8", "replace").split()
                n = int(parts[2]); body = self.p.stdout.read(n); self.p.stdout.read(1)
                text += body
                pairs = sorted_pairs(parts[3:])
                if pairs:
                    ids.append(pairs[0][0]); top.append(pairs[:self.topk])
                    margin.append(pairs[0][1] - pairs[1][1] if len(pairs) > 1 else None)
            elif hdr.startswith(b"DONE") or hdr.startswith(b"ERROR"):
                break
        return {"ids": ids, "text": text.decode("utf-8", "replace"), "top": top, "margin": margin, "n_prompt": np_,
                "stop": "length" if len(ids) >= n_predict else "eos"}

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()


def sorted_pairs(tail_tokens):
    """` <lp> <k> <id> <lp> …` already split; returns [(id, lp)] sorted by lp descending.
    Greedy sampling picks the argmax, so the first pair is the emitted token."""
    if len(tail_tokens) < 2 or not tail_tokens[1].isdigit():
        return []
    k = int(tail_tokens[1]); pairs = []
    for i in range(k):
        if 3 + 2 * i < len(tail_tokens):
            pairs.append((int(tail_tokens[2 + 2 * i]), float(tail_tokens[3 + 2 * i])))
    return sorted(pairs, key=lambda iv: -iv[1])


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
