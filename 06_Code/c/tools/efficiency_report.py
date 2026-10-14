#!/usr/bin/env python3
"""efficiency_report.py — the R2 verdict in one tool (sylph phase 7, efficiency_verdict.md).

Runs up to three systems under one protocol and reads the engines' own accounting:

  S  sylph-GGUF        a SERVE=1 session of qwen36 on the GGUF (COLI_TEMP=0, --cap)
  C  colibrì gs64      the same session on the container directory (SNAP=<dir>)
  O  Ollama            /api/generate, temperature 0, greedy, keep_alive

Every system gets the same prompts, the same number of new tokens and the same
cache-state protocol: pass 1 is cold (fresh process; page cache dropped when --cold drop
is permitted), passes 2… are warm in the same process. One long prompt measures prefill.
Numbers come from the engines' lines, never from a wall clock the tool cannot attribute:
`DONE … STAT <gen> <tok/s> <hit%> <rss>` (decode tok/s, RSS), `PROF` (disk / moe /
attention / head seconds), `GGUF reads:` (slices, MB, parts), `CACHE hits= misses=
expert_bytes=` (the container's bytes per token = misses × expert bytes), Ollama's
`prompt_eval_duration` / `eval_count` / `eval_duration`. TTFT is the tool's own clock
from SUBMIT to the first DATA frame (there is no engine line for it).

    python3 tools/efficiency_report.py --sylph <model.gguf> [--container <dir>] [--ollama <url> --tag <tag>]
            [--cap N] [--new 128] [--prompts <file>] [--prefill-tokens 2000] [--runs 3] [--cold drop|none]
            [--gpu none|cuda] [--text <wiki.test.raw>] [--ppl-ref <ref_logprobs.tsv>] [--threads N]
            [--equivalence <report.json>] [--ci] --out <report.md> [--json <report.json>]

Standard library only. The report follows docs/benchmarking.md's minimum viable report
(host, commit, sha256, exact commands, prompts, cache policy and state, request sizes,
buffered I/O) and ends with a three-sentence verdict: does it run, is it equivalent,
how efficient — and the single limiting factor the profile shows.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
ROOT = C.parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
QWEN36 = C / f"qwen36{EXE}"
T_TOK = C / "tests" / f"test_tok_gguf{EXE}"
EQUIV = ROOT / "07_Tests" / "SystemTest" / "equivalence"
FIRST_SCORED = 257
WINDOW = 512


def log(msg):
    print(f"[eff {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---- host and model facts -------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=str(ROOT), capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def mem_gb():
    tot = avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                tot = int(line.split()[1]) / 1048576
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 1048576
    except OSError:
        pass
    return tot, avail


def gpus():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return [x.strip() for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def storage_of(path):
    """Device and file system holding `path` (df), plus whether it is also the OS volume."""
    try:
        r = subprocess.run(["df", "-P", str(path)], capture_output=True, text=True)
        line = r.stdout.strip().splitlines()[-1].split()
        root = subprocess.run(["df", "-P", "/"], capture_output=True, text=True).stdout.strip().splitlines()[-1].split()
        return {"device": line[0], "mount": line[-1], "same_as_os": line[0] == root[0]}
    except (OSError, IndexError):
        return {"device": "?", "mount": "?", "same_as_os": None}


def host_info(model_path):
    tot, avail = mem_gb()
    return {"hostname": socket.gethostname(), "cpu": cpu_model(), "logical_cpus": os.cpu_count(), "threads_used": int(os.environ.get("OMP_NUM_THREADS", "0")) or None,
            "ram_total_gb": tot, "ram_available_gb": avail, "gpus": gpus(), "os": f"{platform.system()} {platform.release()}",
            "storage": storage_of(model_path), "python": platform.python_version()}


def drop_caches(mode):
    """Returns a sentence describing the cache state actually established."""
    if mode == "none":
        return "page cache not dropped (--cold none): 'cold' means a fresh process with an empty expert cache"
    if mode == "drop":
        try:
            subprocess.run(["sync"], check=False)
            r = subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return "page cache dropped (sync; echo 3 > /proc/sys/vm/drop_caches) before the cold pass"
            return "page cache NOT dropped (no passwordless sudo): 'cold' means a fresh process with an empty expert cache; the file may be in the page cache"
        except (OSError, subprocess.TimeoutExpired):
            return "page cache NOT dropped (sudo unavailable): 'cold' means a fresh process with an empty expert cache"
    return f"cache state '{mode}' not established by the tool (reboot by hand before the run)"


# ---- the engine session ---------------------------------------------------------------------
class Session:
    """One SERVE=1 process of qwen36 on a GGUF or a container; greedy generation with the
    engine's own accounting per turn (DONE/PROF on stdout, GGUF reads/CACHE on stderr)."""

    def __init__(self, snap, cap, threads=None, env=None, label="sylph"):
        e = dict(os.environ)
        for k in ("SNAP", "TOK", "PPL", "PPL_DUMP", "PPL_DUMP_FULL", "SERVE", "KV_SLOTS"):
            e.pop(k, None)
        e.update({"SNAP": str(snap), "SERVE": "1", "COLI_TEMP": "0"})
        if threads:
            e["OMP_NUM_THREADS"] = str(threads)
        if env:
            e.update({k: str(v) for k, v in env.items()})
        self.label = label
        self.cmd = [str(QWEN36), str(cap), "8"]
        self.env_note = {k: e[k] for k in ("SNAP", "SERVE", "COLI_TEMP", "OMP_NUM_THREADS", "COLI_CUDA", "COLI_DENSE_I8") if k in e}
        self.t_start = time.time()
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e, cwd=str(C))
        self.err_lines = []
        self.err_lock = threading.Lock()
        self.err_thread = threading.Thread(target=self._drain, daemon=True)
        self.err_thread.start()
        line = self.p.stdout.readline()
        while line and b"READY" not in line:
            line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"{label}: qwen36 SERVE did not reach READY:\n" + "\n".join(self.err_lines[-30:]))
        self.load_s = time.time() - self.t_start
        stat = self.p.stdout.readline().decode("utf-8", "replace").split()   # STAT 0 0.00 0.0 <rss>
        self.rss_after_load = float(stat[4]) if len(stat) >= 5 else None
        self.n = 0
        time.sleep(0.2)
        self.startup = [l for l in self.errs() if l.startswith("[GGUF]") or l.startswith("[meta]") or "resident weights" in l or "[qwen36]" in l]

    def _drain(self):
        for raw in self.p.stderr:
            with self.err_lock:
                self.err_lines.append(raw.decode("utf-8", "replace").rstrip("\n"))

    def errs(self):
        with self.err_lock:
            return list(self.err_lines)

    def generate(self, prompt, n_new):
        pb = prompt.encode("utf-8"); self.n += 1; rid = f"e{self.n}"
        n_err0 = len(self.errs())
        t0 = time.time()
        self.p.stdin.write(f"SUBMIT {rid} 0 {len(pb)} {n_new} 0 1 logprobs=1\n".encode()); self.p.stdin.write(pb + b"\n"); self.p.stdin.flush()
        ids, text, ttft, np_, prof, done = [], b"", None, None, None, None
        while True:
            hdr = self.p.stdout.readline()
            if not hdr:
                raise RuntimeError(f"{self.label}: engine died:\n" + "\n".join(self.errs()[-20:]))
            if hdr.startswith(b"ACCEPT"):
                np_ = int(hdr.split()[2])
            elif hdr.startswith(b"DATA"):
                if ttft is None:
                    ttft = time.time() - t0
                parts = hdr.decode("utf-8", "replace").split()
                n = int(parts[2]); body = self.p.stdout.read(n); self.p.stdout.read(1)
                text += body
                tail = parts[3:]
                if len(tail) >= 4 and tail[1].isdigit():
                    ids.append(int(tail[2]))
            elif hdr.startswith(b"PROF"):
                f = hdr.split()
                prof = {"wall_s": float(f[1]), "np": int(f[2]), "gen": int(f[3]), "disk_s": float(f[4]), "moe_s": float(f[6]), "attn_s": float(f[7]), "head_s": float(f[8]), "forwards": int(f[9])}
            elif hdr.startswith(b"DONE") or hdr.startswith(b"ERROR"):
                f = hdr.decode("utf-8", "replace").split()
                if hdr.startswith(b"DONE") and len(f) >= 8:
                    done = {"gen": int(f[3]), "tok_s": float(f[4]), "rss_gb": float(f[6]), "np": int(f[7]), "limited": int(f[8]) if len(f) > 8 else 0}
                break
        wall = time.time() - t0
        # this turn's accounting lines (stderr arrives a moment after DONE)
        reads = cache = None
        for _ in range(50):
            new = self.errs()[n_err0:]
            reads = next((l for l in new if l.startswith("GGUF reads:")), reads)
            cache = next((l for l in new if l.startswith("CACHE ")), cache)
            if cache:
                break
            time.sleep(0.02)
        rd = None
        if reads:
            m = re.search(r"GGUF reads: (\d+) slices · ([0-9.]+) MB · ([0-9.]+) MB/token · (\d+) parts? touched", reads)
            if m:
                rd = {"slices": int(m.group(1)), "bytes": float(m.group(2)) * 1e6, "mb_per_token": float(m.group(3)), "parts_touched": int(m.group(4))}
        ch = None
        if cache:
            m = re.search(r"hits=(\d+) misses=(\d+) expert_bytes=(\d+)", cache)
            if m:
                ch = {"hits": int(m.group(1)), "misses": int(m.group(2)), "expert_bytes": int(m.group(3))}
        return {"ids": ids, "text": text.decode("utf-8", "replace"), "np": np_, "ttft_s": ttft, "wall_s": wall, "done": done, "prof": prof, "reads": rd, "cache": ch}

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.p.kill()


def read_prompts(path, limit=None):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s[5:].strip() if s.startswith("chat:") else s)
    return out[:limit] if limit else out


def long_prompt(prompts, text_path, approx_tokens):
    """About `approx_tokens` tokens of text for the prefill measurement: the start of --text
    when given, else the prompt file repeated (4 characters per token as the estimate)."""
    chars = approx_tokens * 4
    if text_path and Path(text_path).exists():
        return Path(text_path).read_text(encoding="utf-8", errors="replace")[:chars]
    base = "\n".join(prompts) + "\n"
    return (base * (chars // max(1, len(base)) + 1))[:chars]


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize_system(label, passes, prefill, load_s, startup, rss_after_load, env_note, cmd, cold_note):
    """passes: list of lists of per-prompt turn dicts (pass 0 cold, the rest warm)."""
    def agg(turns):
        toks = sum((t["done"] or {}).get("gen", 0) for t in turns)
        tps = [(t["done"] or {}).get("tok_s") for t in turns]
        ttft = [t.get("ttft_s") for t in turns]
        rss = [(t["done"] or {}).get("rss_gb") for t in turns]
        rb = sum((t["reads"] or {}).get("bytes", 0.0) for t in turns)
        slices = sum((t["reads"] or {}).get("slices", 0) for t in turns)
        parts = max([(t["reads"] or {}).get("parts_touched", 0) for t in turns] or [0])
        hits = sum((t["cache"] or {}).get("hits", 0) for t in turns)
        miss = sum((t["cache"] or {}).get("misses", 0) for t in turns)
        eb = max([(t["cache"] or {}).get("expert_bytes", 0) for t in turns] or [0])
        prof = {k: sum((t["prof"] or {}).get(k, 0.0) for t in turns) for k in ("wall_s", "disk_s", "moe_s", "attn_s", "head_s")}
        has_reads = any(t["reads"] for t in turns)
        bpt = (rb / toks) if (has_reads and toks) else ((miss * eb / toks) if (toks and eb) else None)
        return {"tokens": toks, "tok_s": median(tps), "tok_s_min": min([x for x in tps if x is not None] or [None]), "tok_s_max": max([x for x in tps if x is not None] or [None]),
                "ttft_s": median(ttft), "rss_gb": max([x for x in rss if x is not None] or [None]),
                "bytes_per_token": bpt, "bytes_source": "GGUF reads" if has_reads else ("misses × expert_bytes" if eb else "n/a"),
                "hit_rate": (hits / (hits + miss)) if (hits + miss) else None, "hits": hits, "misses": miss, "expert_bytes": eb,
                "reads": {"bytes": rb, "slices": slices, "parts_touched": parts}, "prof": prof}
    cold = agg(passes[0]); warm = agg([t for p in passes[1:] for t in p]) if len(passes) > 1 else None
    ids_stable = all(a["ids"] == b["ids"] for p in passes[1:] for a, b in zip(passes[0], p)) if len(passes) > 1 else None
    # top level: the accounting (tokens, bytes, reads, hits/misses) over every pass -- the
    # identities bytes/token × tokens = Σ reads and slices = 3 × misses hold over the whole
    # run (a fully resident tiny model reads nothing in its warm passes); the speed fields
    # (tok/s, TTFT, RSS) are the warm pass's medians, as the ratios use them
    whole = agg([t for p in passes for t in p]); speed = warm or cold
    out = dict(whole)
    out.update({k: speed.get(k) for k in ("tok_s", "tok_s_min", "tok_s_max", "ttft_s", "rss_gb")})
    out["scope"] = {"accounting": "all passes", "speed": "warm passes" if warm else "cold pass"}
    out.update({"label": label, "cold": cold, "warm": warm, "ids_stable": ids_stable, "load_s": load_s, "rss_after_load_gb": rss_after_load,
                "prefill_tok_s": (prefill["np"] / prefill["ttft_s"]) if prefill and prefill.get("ttft_s") else None,
                "prefill_tokens": prefill.get("np") if prefill else None, "prefill_ttft_s": prefill.get("ttft_s") if prefill else None,
                "startup": startup, "env": env_note, "cmd": cmd, "cold_note": cold_note, "nll": None, "first_ids": [t["ids"][:32] for t in passes[0]]})
    return out


def run_engine_system(label, snap, a, prompts, env=None):
    cold_note = drop_caches(a.cold)
    log(f"{label}: starting {snap} (cap {a.cap})")
    s = Session(snap, a.cap, threads=a.threads, env=env, label=label)
    passes = []
    try:
        for r in range(a.runs):
            turns = []
            for i, p in enumerate(prompts):
                t = s.generate(p, a.new); turns.append(t)
                d = t["done"] or {}
                log(f"{label} pass {r + 1}/{a.runs} prompt {i}: {d.get('gen')} tok · {d.get('tok_s', 0):.2f} tok/s · TTFT {t['ttft_s'] or 0:.2f} s" + (f" · {t['reads']['mb_per_token']:.0f} MB/token" if t.get("reads") else ""))
            passes.append(turns)
        lp = long_prompt(prompts, a.text, a.prefill_tokens)
        pre = s.generate(lp, 1)
        log(f"{label} prefill: {pre['np']} tokens in {pre['ttft_s'] or 0:.2f} s")
    finally:
        s.close()
    return summarize_system(label, passes, pre, s.load_s, s.startup, s.rss_after_load, s.env_note, s.cmd, cold_note)


# ---- quality: teacher-forced NLL on the first window ----------------------------------------
def nll_window(label, snap, gguf_for_ids, a, out_dir, env=None):
    """PPL=1 TF-NLL over the first 512-token window of --text (positions 257…511), the
    engine's own line; ids from the GGUF's tokenizer (identical for the container's)."""
    if not a.text or not Path(a.text).exists() or not T_TOK.exists():
        return None
    sys.path.insert(0, str(EQUIV))
    import sylph_runner as sr  # noqa: E402
    ids = sr.encode(gguf_for_ids, a.text)[:WINDOW]
    if len(ids) < WINDOW:
        return None
    e = dict(env or {}); e["SNAP"] = str(snap)
    res = sr.ppl_window(snap, ids, FIRST_SCORED, out_dir / label, "win_0", cap=a.cap, env=e, threads=a.threads, full=False)
    out = {"tf_nll": res["tf_nll"], "n": res["n"], "ppl": float(__import__("math").exp(res["tf_nll"])), "dump": res["dump"]}
    if a.ppl_ref and Path(a.ppl_ref).exists():
        import compare  # noqa: E402
        def rows(p):
            d = {}
            for line in open(p):
                if line[0] in "#p":
                    continue
                f = line.rstrip("\n").split("\t")
                d[int(f[0])] = (int(f[1]), float(f[2]), [])
            return d
        e1 = compare.e1_compare(rows(a.ppl_ref), rows(res["dump"]))
        out["vs_ref"] = {"mean_abs_dnll": e1["mean"], "max_abs_dnll": e1["max"], "n": e1["n"]}
    return out


# ---- Ollama ---------------------------------------------------------------------------------
def ollama_post(base, path, obj, timeout=3600):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_ps(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/ps", timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError):
        return None


def run_ollama(a, prompts):
    base, tag = a.ollama, a.tag
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/version", timeout=10) as r:
            version = json.loads(r.read()).get("version", "?")
    except (urllib.error.URLError, OSError) as ex:
        log(f"ollama unreachable: {ex}")
        return None
    subprocess.run(["ollama", "stop", tag], capture_output=True, text=True)
    opts = {"temperature": 0, "seed": 1, "num_predict": a.new, "num_ctx": 4096, "repeat_penalty": 1.0, "repeat_last_n": 0,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0}
    passes, ps_info = [], None
    for r in range(a.runs):
        turns = []
        for i, p in enumerate(prompts):
            t0 = time.time()
            res = ollama_post(base, "/api/generate", {"model": tag, "prompt": p, "raw": True, "stream": False, "options": opts, "keep_alive": "30m"})
            wall = time.time() - t0
            ev, evd, pd, ld = res.get("eval_count") or 0, res.get("eval_duration") or 0, res.get("prompt_eval_duration") or 0, res.get("load_duration") or 0
            turns.append({"gen": ev, "tok_s": ev / (evd / 1e9) if evd else None, "ttft_s": (pd + (ld if r == 0 and i == 0 else 0)) / 1e9, "prompt_eval_s": pd / 1e9,
                          "load_s": ld / 1e9, "wall_s": wall, "text": res.get("response", ""), "stop": res.get("done_reason")})
            log(f"ollama pass {r + 1}/{a.runs} prompt {i}: {ev} tok · {turns[-1]['tok_s'] or 0:.2f} tok/s · prompt eval {pd / 1e9:.2f} s")
            if ps_info is None:
                ps_info = ollama_ps(base)
        passes.append(turns)
    lp = long_prompt(prompts, a.text, a.prefill_tokens)
    res = ollama_post(base, "/api/generate", {"model": tag, "prompt": lp, "raw": True, "stream": False, "options": dict(opts, num_predict=1), "keep_alive": "30m"})
    pec, ped = res.get("prompt_eval_count") or 0, res.get("prompt_eval_duration") or 0
    def agg(turns):
        return {"tokens": sum(t["gen"] for t in turns), "tok_s": median([t["tok_s"] for t in turns]),
                "tok_s_min": min([t["tok_s"] for t in turns if t["tok_s"]] or [None]), "tok_s_max": max([t["tok_s"] for t in turns if t["tok_s"]] or [None]),
                "ttft_s": median([t["ttft_s"] for t in turns]), "prompt_eval_s": median([t["prompt_eval_s"] for t in turns])}
    cold = agg(passes[0]); warm = agg([t for p in passes[1:] for t in p]) if len(passes) > 1 else None
    texts_stable = all(x["text"] == y["text"] for p in passes[1:] for x, y in zip(passes[0], p)) if len(passes) > 1 else None
    vram_gb = rss_gb = None; processor = None
    for m in (ps_info or {}).get("models", []):
        if m.get("name", "").startswith(tag) or tag in m.get("name", "") or m.get("model", "").startswith(tag):
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            vram_gb = vram / 1e9; rss_gb = (size - vram) / 1e9; processor = f"{100 * vram / size:.0f}% GPU" if size else None
    out = dict(warm or cold)
    out.update({"label": "ollama", "cold": cold, "warm": warm, "ids_stable": texts_stable, "version": version, "tag": tag, "options": opts,
                "prefill_tok_s": pec / (ped / 1e9) if ped else None, "prefill_tokens": pec, "prefill_ttft_s": ped / 1e9 if ped else None,
                "bytes_per_token": None, "bytes_source": "n/a (Ollama exposes no read counter)", "rss_gb": rss_gb, "vram_gb": vram_gb, "processor": processor,
                "hit_rate": None, "nll": None, "reads": None, "ps": ps_info, "first_ids": None, "cold_note": "first request after `ollama stop <tag>` (model load included in its TTFT)"})
    return out


# ---- report ---------------------------------------------------------------------------------
def fmt(x, nd=2, unit=""):
    if x is None:
        return "—"
    if isinstance(x, float):
        s = f"{x:.{nd}f}" if abs(x) >= 0.01 or x == 0 else f"{x:.2e}"
    else:
        s = str(x)
    return s + (f" {unit}" if unit else "")


def ratio(a, b):
    return (a / b) if (a is not None and b not in (None, 0)) else None


def limiting_factor(sysd):
    p = (sysd or {}).get("prof") or {}
    parts = {"disk (expert reads)": p.get("disk_s", 0.0), "MoE compute": p.get("moe_s", 0.0), "attention / DeltaNet": p.get("attn_s", 0.0), "lm_head": p.get("head_s", 0.0)}
    if not any(parts.values()):
        return None, parts
    k = max(parts, key=parts.get)
    return k, parts


def write_report(a, R):
    S, Cn, O = R["systems"].get("sylph"), R["systems"].get("container"), R["systems"].get("ollama")
    h = R["host"]
    L = [f"# Efficiency report — {Path(a.sylph).name} on {h['hostname']} ({R['date']})", ""]
    L += ["## Host", "",
          f"- commit `{R['commit']}` (sylph); engine `{QWEN36.name}` built from this tree",
          f"- CPU {h['cpu']} · {h['logical_cpus']} logical CPUs · threads used: {a.threads or 'engine default (physical cores)'}",
          f"- RAM {fmt(h['ram_total_gb'], 1, 'GB')} total · {fmt(h['ram_available_gb'], 1, 'GB')} available at start",
          f"- GPU {', '.join(h['gpus']) or 'none'} · sylph/container GPU use: {a.gpu}",
          f"- model volume {h['storage']['device']} mounted at {h['storage']['mount']}" + (" (also the OS volume)" if h['storage']['same_as_os'] else "") + " · buffered I/O (pread, no O_DIRECT in this engine)",
          f"- OS {h['os']} · Python {h['python']}", ""]
    L += ["## Systems", "", "| system | source | sha256 / id | command / request |", "|---|---|---|---|"]
    L.append(f"| S sylph-GGUF | `{a.sylph}` | `{R['model']['sha256']}` | `{' '.join(S['cmd'])}` with {S['env']} |" if S else "| S sylph-GGUF | — | — | not run |")
    if Cn:
        L.append(f"| C colibrì gs64 container | `{a.container}` | `{R['model'].get('container_sha256', '—')}` (index) | `{' '.join(Cn['cmd'])}` with {Cn['env']} |")
    if O:
        L.append(f"| O Ollama {O['version']} | tag `{O['tag']}` | blob `{R['model'].get('ollama_from', '?')}` | `/api/generate` raw, options {O['options']}, keep_alive 30m |")
    for k, v in R["systems"].items():
        if k not in ("sylph", "container", "ollama"):
            L.append(f"| {k} | `{a.sylph if 'sylph' in k else a.container}` | — | `{' '.join(v['cmd'])}` with {v['env']} |")
    L += ["", f"Protocol: {len(R['prompts'])} prompts (`{a.prompts}`), {a.new} new tokens each, greedy (temperature 0); {a.runs} passes — pass 1 cold, passes 2… warm in the same process; "
          f"prefill from one {a.prefill_tokens}-token prompt; expert cache `--cap {a.cap}` slots per layer (sylph and container); request sizes: prompts of {min(len(p) for p in R['prompts'])}–{max(len(p) for p in R['prompts'])} characters.",
          f"Cache state: {S['cold_note'] if S else drop_caches('none')}", ""]
    L += ["## Results", "", "| system | state | decode tok/s (median, min–max) | TTFT (median) | prefill tok/s | bytes/token read | RSS | VRAM | expert hit rate | TF-NLL (PPL) first window | ids stable |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sd in R["systems"].items():
        for state in ("cold", "warm"):
            st = sd.get(state)
            if not st:
                continue
            nll = sd.get("nll") or {}
            L.append(f"| {name} | {state} | {fmt(st.get('tok_s'))} ({fmt(st.get('tok_s_min'))}–{fmt(st.get('tok_s_max'))}) tok/s | {fmt(st.get('ttft_s'), 2, 's')} | "
                     f"{fmt(sd.get('prefill_tok_s'), 1, 'tok/s') if state == 'warm' or not sd.get('warm') else '—'} | "
                     f"{fmt((st.get('bytes_per_token') or 0) / 1e6, 1, 'MB') if st.get('bytes_per_token') else '—'} | {fmt(st.get('rss_gb') if 'rss_gb' in st else sd.get('rss_gb'), 2, 'GB')} | "
                     f"{fmt(sd.get('vram_gb'), 2, 'GB') if sd.get('vram_gb') else '—'} | {fmt(100 * st['hit_rate'], 1, '%') if st.get('hit_rate') is not None else '—'} | "
                     f"{(fmt(nll.get('tf_nll'), 4) + ' (' + fmt(nll.get('ppl'), 3) + ')') if nll else '—'} | {sd.get('ids_stable')} |")
    L += ["", "bytes/token: S from the engine's `GGUF reads:` lines; C = misses × bytes per expert as stored; O not observable. "
          "tok/s and RSS are the engines' own `DONE … STAT` fields; TTFT is the tool's clock from request to first token (Ollama: `prompt_eval_duration`, plus the model load on its first cold request).", ""]
    L += ["## Ratios", ""]
    rows = []
    for other, key in (("container", "S/C"), ("ollama", "S/O")):
        od = R["systems"].get(other)
        if S and od:
            w = S.get("warm") or S.get("cold"); ow = od.get("warm") or od.get("cold")
            rows.append(f"| {key} | {fmt(ratio(w.get('tok_s'), ow.get('tok_s')))} | {fmt(ratio(w.get('ttft_s'), ow.get('ttft_s')))} | "
                        f"{fmt(ratio(S.get('prefill_tok_s'), od.get('prefill_tok_s')))} | {fmt(ratio(w.get('bytes_per_token'), ow.get('bytes_per_token')))} | {fmt(ratio(S.get('rss_gb'), od.get('rss_gb')))} |")
    for name, sd in R["systems"].items():
        if name not in ("sylph", "container", "ollama") and S:
            w = sd.get("warm") or sd.get("cold"); sw = S.get("warm") or S.get("cold")
            rows.append(f"| {name}/S | {fmt(ratio(w.get('tok_s'), sw.get('tok_s')))} | {fmt(ratio(w.get('ttft_s'), sw.get('ttft_s')))} | {fmt(ratio(sd.get('prefill_tok_s'), S.get('prefill_tok_s')))} | {fmt(ratio(w.get('bytes_per_token'), sw.get('bytes_per_token')))} | {fmt(ratio(sd.get('rss_gb'), S.get('rss_gb')))} |")
    if rows:
        L += ["| ratio (warm) | decode tok/s | TTFT | prefill tok/s | bytes/token | RSS |", "|---|---|---|---|---|---|"] + rows
    else:
        L.append("- one system only: no ratios")
    L += ["", "Deliberate differences of S (spec §4.2): no re-quantization (the file's blocks as stored), f32 activations on the expert kernels (upstream's container runs int8), "
          "exact float dense tensors (`COLI_DENSE_I8` off on a GGUF), `token_embd` decoded per token.", ""]
    if R.get("not_run"):
        L += ["Not run: " + "; ".join(f"{k} ({v.strip().splitlines()[-1][:160] if v.strip() else 'failed'})" for k, v in R["not_run"].items()), ""]
    L += ["## Verdict", "", R["verdict"], ""]
    if R.get("equivalence"):
        L += [f"Equivalence source: `{R['equivalence'].get('source')}` — {R['equivalence'].get('summary')}", ""]
    L += ["## Raw", "", "```"] + [json.dumps({k: v for k, v in R["systems"].items()}, default=str)[:4000]] + ["```", ""]
    return "\n".join(L)


def verdict_text(a, R):
    S, Cn, O = R["systems"].get("sylph"), R["systems"].get("container"), R["systems"].get("ollama")
    if not S or not S.get("tokens"):
        s1 = "It does not run: the sylph session produced no tokens on this file."
    else:
        s1 = (f"It runs: {Path(a.sylph).name} served {len(R['prompts'])} prompts × {a.runs} passes with {a.new} new tokens each, ids stable across passes = {S.get('ids_stable')}, "
              f"{fmt((S.get('warm') or S.get('cold')).get('tok_s'))} tok/s warm decode, TTFT {fmt((S.get('warm') or S.get('cold')).get('ttft_s'), 2, 's')}, RSS {fmt(S.get('rss_gb'), 2, 'GB')}.")
    eq = R.get("equivalence")
    s2 = f"Equivalence: {eq['summary']} (from {Path(eq['source']).name}; the reading of those gates is in 08_Documents/equivalence/)." if eq else "Equivalence: not measured by this tool (see the equivalence report under 08_Documents/equivalence/)."
    parts = []
    for other, key in (("container", "gs64 container"), ("ollama", "Ollama")):
        od = R["systems"].get(other)
        if S and od:
            w = S.get("warm") or S.get("cold"); ow = od.get("warm") or od.get("cold")
            parts.append(f"vs {key}: decode {fmt(ratio(w.get('tok_s'), ow.get('tok_s')))}×, TTFT {fmt(ratio(w.get('ttft_s'), ow.get('ttft_s')))}×, prefill {fmt(ratio(S.get('prefill_tok_s'), od.get('prefill_tok_s')))}×"
                         + (f", bytes/token {fmt(ratio(w.get('bytes_per_token'), ow.get('bytes_per_token')))}×" if ow.get("bytes_per_token") else "") + f", RSS {fmt(ratio(S.get('rss_gb'), od.get('rss_gb')))}×")
    lf, lp = limiting_factor(S.get("warm") or S.get("cold")) if S else (None, {})
    s3 = ("Efficiency (sylph relative to the others, warm): " + "; ".join(parts) + "." if parts else "Efficiency: no second system ran, so there are no ratios.") + \
         (f" The limiting factor in sylph's own profile is {lf} ({', '.join(f'{k} {v:.1f} s' for k, v in lp.items())})." if lf else " The profile did not attribute the time.")
    return " ".join([s1, s2, s3])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sylph", required=True); ap.add_argument("--container"); ap.add_argument("--ollama"); ap.add_argument("--tag", default="")
    ap.add_argument("--cap", type=int, default=64); ap.add_argument("--new", type=int, default=128); ap.add_argument("--prompts", default=str(ROOT / "07_Tests" / "SystemTest" / "fixtures" / "e3_prompts.txt"))
    ap.add_argument("--n-prompts", type=int, default=None); ap.add_argument("--prefill-tokens", type=int, default=2000); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cold", default="drop", choices=["drop", "reboot", "none"]); ap.add_argument("--gpu", default="none", choices=["none", "cuda"])
    ap.add_argument("--text"); ap.add_argument("--ppl-ref"); ap.add_argument("--threads", type=int); ap.add_argument("--equivalence")
    ap.add_argument("--ci", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--json")
    ap.add_argument("--from-json", help="rewrite --out from a previous run's --json (no engines run)")
    a = ap.parse_args()
    if a.from_json:
        R = json.loads(Path(a.from_json).read_text(encoding="utf-8"))
        R["verdict"] = verdict_text(a, R)
        Path(a.out).write_text(write_report(a, R), encoding="utf-8")
        if a.json:
            Path(a.json).write_text(json.dumps(R, indent=1, default=str) + "\n", encoding="utf-8")
        print(R["verdict"]); return 0
    if a.ci:
        a.cold = "none"; a.runs = max(2, min(a.runs, 2)); a.prefill_tokens = min(a.prefill_tokens, 256)
    if not QWEN36.exists():
        r = subprocess.run(["make", "-C", str(C), "qwen36"], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"qwen36 does not build:\n{r.stderr[-1500:]}")
    prompts = read_prompts(a.prompts, a.n_prompts or (8 if a.ci else None))
    out_md = Path(a.out); out_md.parent.mkdir(parents=True, exist_ok=True)
    work = out_md.with_suffix("").parent / (out_md.stem + "_work")
    R = {"date": datetime.date.today().isoformat(), "commit": git_commit(), "host": host_info(a.sylph), "prompts": prompts,
         "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "model": {"path": str(a.sylph), "sha256": sha256_file(a.sylph)}, "systems": {}}
    if a.container:
        idx = Path(a.container) / "model.safetensors.index.json"
        R["model"]["container_sha256"] = sha256_file(idx) if idx.exists() else None
    if a.ollama:
        try:
            show = ollama_post(a.ollama, "/api/show", {"model": a.tag}, timeout=60)
            m = re.search(r"^FROM\s+(\S+)", show.get("modelfile", ""), re.M); R["model"]["ollama_from"] = m.group(1) if m else None
        except Exception as ex:  # noqa: BLE001
            R["model"]["ollama_from"] = f"(show failed: {ex})"
    # S, C (and their CUDA twins)
    R["not_run"] = {}

    def arm(label, snap, env=None):
        """One engine system; a failure to start (e.g. a container directory without its
        shards) is recorded and the other systems still run."""
        try:
            d = run_engine_system(label, snap, a, prompts, env=env)
            d["nll"] = nll_window(label, snap, a.sylph, a, work, env=env)
            R["systems"][label] = d
        except (RuntimeError, OSError, SystemExit) as ex:
            log(f"{label}: not run ({str(ex).strip().splitlines()[-1][:200] if str(ex).strip() else ex})")
            R["not_run"][label] = str(ex)[-600:]
            if label == "sylph":
                raise
    arm("sylph", a.sylph)
    if a.container:
        arm("container", a.container)
    if a.gpu == "cuda":
        arm("sylph_cuda", a.sylph, env={"COLI_CUDA": "1"})
        if a.container:
            arm("container_cuda", a.container, env={"COLI_CUDA": "1"})
    if a.ollama:
        O = run_ollama(a, prompts)
        if O:
            R["systems"]["ollama"] = O
    # ratios and verdict
    S = R["systems"]["sylph"]
    R["ratios"] = {}
    for other in [k for k in R["systems"] if k != "sylph"]:
        od = R["systems"][other]; w = S.get("warm") or S.get("cold"); ow = od.get("warm") or od.get("cold")
        R["ratios"][f"sylph/{other}"] = {"tok_s": ratio(w.get("tok_s"), ow.get("tok_s")), "ttft_s": ratio(w.get("ttft_s"), ow.get("ttft_s")),
                                        "prefill_tok_s": ratio(S.get("prefill_tok_s"), od.get("prefill_tok_s")), "bytes_per_token": ratio(w.get("bytes_per_token"), ow.get("bytes_per_token")),
                                        "rss_gb": ratio(S.get("rss_gb"), od.get("rss_gb"))}
    if "container" in R["systems"]:
        R["ratios"]["ids_identical"] = S.get("first_ids") == R["systems"]["container"].get("first_ids")
    if a.equivalence and Path(a.equivalence).exists():
        try:
            eq = json.loads(Path(a.equivalence).read_text()); v = eq.get("verdict") or {}
            R["equivalence"] = {"source": a.equivalence, "summary": ", ".join(f"{k} {'pass' if ok else 'FAIL'}" for k, ok in v.items()) or "no verdict in the file"}
        except (OSError, ValueError) as ex:
            R["equivalence"] = {"source": a.equivalence, "summary": f"unreadable ({ex})"}
    R["verdict"] = verdict_text(a, R)
    out_md.write_text(write_report(a, R), encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(R, indent=1, default=str) + "\n", encoding="utf-8")
    log(f"report: {out_md}" + (f" · json: {a.json}" if a.json else ""))
    print(R["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
