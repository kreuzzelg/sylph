#!/usr/bin/env python3
"""llama.cpp as the reference (FR-32): tokenizer ids, perplexity with per-position
log-probs from `--kl-divergence-base`, greedy generation through llama-server's
/completion (token ids + n_probs) or llama-cli (text only). Standard library only.

The kld file layout is read as `tools/perplexity/perplexity.cpp` writes it (magic
"_logits_", n_ctx, n_vocab, n_chunk, the tokens, then per chunk one row per scored
position: two f32 (scale, min_log_prob) and n_vocab uint16 steps, padded to an even
count). The parser does not trust that reading: it recomputes the perplexity from the
parsed target log-probs and refuses unless it matches the PPL llama-perplexity printed.
"""
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXE = ".exe" if sys.platform == "win32" else ""


def binary(llama_dir, name):
    p = Path(llama_dir) / (name + EXE)
    if not p.exists():
        for alt in (name.replace("llama-", ""), "llama-" + name):
            q = Path(llama_dir) / (alt + EXE)
            if q.exists():
                return q
        sys.exit(f"{name} not found in {llama_dir}")
    return p


def version(llama_dir):
    r = subprocess.run([str(binary(llama_dir, "llama-cli")), "--version"], capture_output=True, text=True)
    return (r.stderr or r.stdout).strip().splitlines()[0] if (r.stderr or r.stdout).strip() else "unknown"


def tokenize(llama_dir, model, text_path, add_bos=False):
    cmd = [str(binary(llama_dir, "llama-tokenize")), "-m", str(model), "-f", str(text_path), "--ids", "--log-disable"]
    if not add_bos:
        cmd.append("--no-bos")
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        sys.exit(f"llama-tokenize failed: {r.stderr[-800:]}")
    m = re.search(r"\[([-0-9,\s]*)\]", r.stdout)
    if not m:
        sys.exit(f"llama-tokenize: no id list in output:\n{r.stdout[-800:]}")
    return [int(x) for x in m.group(1).replace(",", " ").split()]


def perplexity(llama_dir, model, text_path, chunks, out_prefix, threads=None, n_ctx=512, extra=(), force=False):
    """Runs llama-perplexity with --kl-divergence-base, returns
    {ppl, dumps: [per-chunk ppl-dump paths], chunks: [(n, sum_nll)], full0: path|None, log}."""
    out_prefix = Path(out_prefix); out_prefix.parent.mkdir(parents=True, exist_ok=True)
    meta = Path(str(out_prefix) + ".json")
    if meta.exists() and not force:
        return json.loads(meta.read_text())
    kld = Path(str(out_prefix) + ".kld")
    cmd = [str(binary(llama_dir, "llama-perplexity")), "-m", str(model), "-f", str(text_path), "-c", str(n_ctx), "-b", str(n_ctx),
           "--chunks", str(chunks), "--kl-divergence-base", str(kld)] + list(extra)
    if threads:
        cmd += ["-t", str(threads)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    log = r.stdout + r.stderr
    Path(str(out_prefix) + ".log").write_text(log)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", log)
    if r.returncode != 0 or not m or not kld.exists():
        sys.exit(f"llama-perplexity failed:\n{log[-2000:]}")
    ppl_printed = float(m.group(1))
    res = parse_kld(kld, out_prefix, topk=10)
    recomputed = math.exp(sum(s for _, s in res["chunks"]) / sum(n for n, _ in res["chunks"]))
    if abs(recomputed - ppl_printed) > 1e-3 * ppl_printed + 1e-3:
        sys.exit(f"kld parse self-check failed: recomputed PPL {recomputed:.4f} vs printed {ppl_printed:.4f} — the file layout differs from what this parser expects")
    res.update({"ppl": ppl_printed, "ppl_recomputed": recomputed, "seconds": time.time() - t0, "cmd": cmd, "log": str(out_prefix) + ".log"})
    meta.write_text(json.dumps(res))
    return res


def parse_kld(path, out_prefix, topk=10, full_chunks=1):
    """Converts the kld base file to ppl-dump v1 (one per chunk) and full-logprob v1
    for the first `full_chunks` chunks."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"_logits_":
            sys.exit(f"{path}: bad magic {magic!r}")
        n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
        tokens = struct.unpack(f"<{n_chunk * n_ctx}i", f.read(4 * n_chunk * n_ctx))
        first = n_ctx // 2
        n_tok = n_ctx - 1 - first
        nv = 2 * ((n_vocab + 1) // 2) + 4
        dumps, chunks, full0 = [], [], None
        for c in range(n_chunk):
            rows = f.read(n_tok * nv * 2)
            if len(rows) < n_tok * nv * 2:
                sys.exit(f"{path}: truncated at chunk {c}")
            dump = Path(str(out_prefix) + f".chunk{c}.tsv")
            lines = [f"# sylph ppl-dump v1 model={path} vocab={n_vocab} scored={n_tok} topk={topk}"]
            fh = None
            if c < full_chunks:
                full0 = Path(str(out_prefix) + f".chunk{c}.full")
                fh = open(full0, "wb"); fh.write(f"# sylph full-logprob v1 vocab={n_vocab} positions={n_tok}\n".encode())
            sum_nll = 0.0
            for i in range(n_tok):
                off = i * nv * 2
                scale, min_lp = struct.unpack("<ff", rows[off:off + 8])
                q = struct.unpack(f"<{n_vocab}H", rows[off + 8:off + 8 + 2 * n_vocab])
                lp = [min_lp + scale * x for x in q]
                tgt = tokens[c * n_ctx + first + i + 1]
                pos = first + i + 1
                sum_nll -= lp[tgt]
                top = sorted(range(n_vocab), key=lambda j: -lp[j])[:topk]
                tail = f" {lp[tgt]:.6f} {topk}" + "".join(f" {j} {lp[j]:.6f}" for j in top)
                lines.append(f"{pos}\t{tgt}\t{lp[tgt]:.7g}\t{tail}")
                if fh:
                    fh.write(struct.pack(f"<{n_vocab}f", *lp))
            if fh:
                fh.close()
            dump.write_text("\n".join(lines) + "\n")
            dumps.append(str(dump)); chunks.append((n_tok, sum_nll))
    return {"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk, "dumps": dumps, "chunks": chunks, "full0": str(full0) if full0 else None,
            "tokens": [list(tokens[c * n_ctx:(c + 1) * n_ctx]) for c in range(n_chunk)]}


class Server:
    """llama-server for greedy generation with token ids and top probabilities."""

    def __init__(self, llama_dir, model, port=8089, n_ctx=4096, threads=None, extra=()):
        cmd = [str(binary(llama_dir, "llama-server")), "-m", str(model), "--port", str(port), "-c", str(n_ctx), "--log-disable"] + list(extra)
        if threads:
            cmd += ["-t", str(threads)]
        self.port = port
        self.p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        self.close(); sys.exit("llama-server did not become healthy")

    def generate(self, prompt, n_predict=128, n_probs=10, seed=1):
        body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": seed, "n_probs": n_probs,
                           "cache_prompt": False, "return_tokens": True, "samplers": ["temperature"]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            res = json.loads(r.read())
        ids = list(res.get("tokens") or [])
        top, margin = [], []
        for cp in res.get("completion_probabilities") or []:
            cands = cp.get("top_logprobs") or cp.get("probs") or []
            pairs = []
            for cnd in cands:
                lp = cnd.get("logprob")
                if lp is None and cnd.get("prob") is not None:
                    lp = math.log(max(cnd["prob"], 1e-300))
                pairs.append((cnd.get("id", cnd.get("tok_id", -1)), lp))
            pairs.sort(key=lambda iv: -iv[1])
            top.append(pairs); margin.append(pairs[0][1] - pairs[1][1] if len(pairs) > 1 else None)
            if not ids and cp.get("id") is not None:
                ids.append(cp["id"])
        return {"ids": ids, "text": res.get("content", ""), "top": top, "margin": margin,
                "stop": "eos" if res.get("stopped_eos") else "length" if res.get("stopped_limit") else "other"}

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()


def generate_cli(llama_dir, model, prompt, n_predict=128, threads=None, seed=1):
    """llama-cli greedy text (no ids, no margins): the second, weaker E3 reference."""
    cmd = [str(binary(llama_dir, "llama-cli")), "-m", str(model), "-p", prompt, "-n", str(n_predict), "--temp", "0", "--seed", str(seed),
           "-no-cnv", "--no-display-prompt", "--simple-io", "--log-disable"]
    if threads:
        cmd += ["-t", str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=3600)
    return {"ids": [], "text": r.stdout.strip("\n"), "top": [], "margin": [], "stop": "unknown"}
