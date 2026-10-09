#!/usr/bin/env python3
"""Ollama as the reference for E3 (FR-32): /api/version, /api/show (which blob the tag
runs), /api/generate with raw prompts and /api/chat for chat prompts, temperature 0,
fixed seed; `logprobs`/`top_logprobs` where the server accepts them (Ollama ≥ 0.12).
Standard library only."""
import json
import math
import re
import urllib.error
import urllib.request


def _post(base, path, obj, timeout=3600):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def version(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/version", timeout=10) as r:
            return json.loads(r.read()).get("version", "?")
    except (urllib.error.URLError, OSError) as ex:
        return f"unreachable ({ex})"


def show(base, tag):
    """The Modelfile's FROM line (the blob the tag runs) and the model details."""
    res = _post(base, "/api/show", {"model": tag, "verbose": False}, timeout=60)
    mf = res.get("modelfile", "")
    m = re.search(r"^FROM\s+(\S+)", mf, re.M)
    return {"from": m.group(1) if m else None, "details": res.get("details"), "parameters": res.get("parameters")}


def _logprobs(res):
    top, margin, toks = [], [], []
    for lp in res.get("logprobs") or []:
        toks.append(lp.get("token"))
        cands = [(c.get("token"), c.get("logprob")) for c in lp.get("top_logprobs") or [] if c.get("logprob") is not None]
        cands.sort(key=lambda x: -x[1])
        top.append(cands); margin.append(cands[0][1] - cands[1][1] if len(cands) > 1 else None)
    return top, margin, toks


def generate(base, tag, prompt, n_predict=128, seed=1, num_ctx=4096, want_logprobs=True, chat=False):
    opts = {"temperature": 0, "seed": seed, "num_predict": n_predict, "num_ctx": num_ctx}
    if chat:
        body = {"model": tag, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": opts}
    else:
        body = {"model": tag, "prompt": prompt, "raw": True, "stream": False, "options": opts}
    if want_logprobs:
        body.update({"logprobs": True, "top_logprobs": 10})
    try:
        res = _post(base, "/api/chat" if chat else "/api/generate", body)
    except urllib.error.HTTPError as ex:
        if want_logprobs and ex.code in (400, 404):      # older server: retry without logprobs
            return generate(base, tag, prompt, n_predict, seed, num_ctx, want_logprobs=False, chat=chat)
        raise
    text = res.get("message", {}).get("content", "") if chat else res.get("response", "")
    top, margin, toks = _logprobs(res)
    return {"ids": [], "text": text, "tokens": toks, "top": top, "margin": margin,
            "stop": res.get("done_reason", "unknown"),
            "timing": {"prompt_eval_ms": (res.get("prompt_eval_duration") or 0) / 1e6, "eval_ms": (res.get("eval_duration") or 0) / 1e6,
                       "eval_count": res.get("eval_count"), "tok_s": (res.get("eval_count") or 0) / max(1e-9, (res.get("eval_duration") or 1) / 1e9)}}
