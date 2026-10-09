# Equivalence harness — interchange formats

Fixed by `../equivalence.md`. Every reference and sylph itself are converted into these
before anything is compared, so the comparer never sees an engine-specific layout.

## `ppl-dump v1` (per-position log-probs; FR-31)

```
# sylph ppl-dump v1 model=<path> vocab=<V> scored=<N> topk=<k>
<pos>\t<target_id>\t<logprob_target>\t <lp> <id>:<lp> <id>:<lp> …
```

One line per scored position. `pos` is the 0-based index of the **target** token in the
window's id sequence; `logprob_target` in nats, 7 significant digits; the tail repeats the
target's log-prob and lists the top-k ids with their log-probs (sylph: the serve
protocol's `logprobs` channel text, so dump and server never disagree). Writers: the
engine (`PPL=1 PPL_DUMP=`), `ref_llama.py perplexity` (from the `--kl-divergence-base`
file), `make_tiny_ref_logprobs.py` (torch, CI).

## `full-logprob v1` (whole distribution; exact KL, FR-33)

```
# sylph full-logprob v1 vocab=<V> positions=<N>\n
<N × V little-endian f32: log-softmax rows, in the same position order as the ppl-dump>
```

Writers: the engine (`PPL_DUMP_FULL=`), `ref_llama.py` (dequantizing the kld file's
uint16 rows: `lp = min_log_prob + scale · q`), torch (CI). 254 MB per 256 positions on
Qwen3.6-35B-A3B, hence `KL_CHUNKS` (default 1).

## `e2-chunks v1` (per-window perplexity)

```
chunk\tn_scored\tsum_nll\tppl_running
0\t255\t1356.2\t…
```

`sum_nll` in nats over the chunk's scored positions; `ppl_running` = exp(Σ sum_nll / Σ n)
up to and including this chunk (what `llama-perplexity` prints as it goes).

## `e3-gen v1` (greedy generations)

JSON lines, one per prompt and engine:

```
{"prompt": 0, "engine": "sylph|llama-server|llama-cli|ollama", "ids": [..], "text": "…",
 "top": [[[id, lp], …k], …], "margin": [m0, m1, …], "stop": "eos|length|near-tie|error"}
```

`ids` empty and `top` empty where the reference gives text only (`llama-cli`, Ollama
without `top_logprobs`); `margin[i]` = top-1 minus top-2 log-prob at generated position
`i` where known (sylph: `logprobs=10` tails; llama-server: `n_probs`; Ollama:
`top_logprobs`).

## Windows for E1/E2 (llama.cpp's protocol)

`n_ctx = 512`, chunk `k` = ids `[512k, 512k+512)` of the text's token stream; the first
half is context, logits at indices `256 … 510` predict targets `257 … 511`: **255 scored
positions per chunk**. sylph: `prompt_ids = w[0:257]`, `full_ids = w[0:512]`
(`tf_nll` scores `pos = 257 … 511`). `add_bos` as the tokenizer gate found it (false for
Qwen3.6).

## `report.json`

```
{"model": {"path", "sha256", "inspect"}, "versions": {...}, "host": {...}, "cap": n,
 "tokenizer_gate": {"pass", "n_ids", "first_mismatch"},
 "floor": {"e1_mean", "e1_max", "e1_top1_disagree", "e2_rel", "samples": [...]},
 "thresholds": {"e1_mean", "e1_max", "top1", "kl", "e2_rel"},
 "E1": {"mean", "max", "top1", "top5", "kl_exact": {...}, "kl_trunc": [...], "worst": [...], "pass"},
 "E2": {"ppl_sylph", "ppl_ref", "rel", "sign_p", "chunks": [...], "pass"},
 "E3": {"prompts": [...], "n_ok", "pass"},
 "deviations": {"COLI_DENSE_I8=1": {...}}, "files": [{"path", "bytes", "sha256"}]}
```
