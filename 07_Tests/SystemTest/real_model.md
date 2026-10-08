# System test — the real model: `coli chat/serve/web` from a Qwen3.6-35B-A3B GGUF, and the A/B against Ollama and the gs64 container (phase 4, owner's machine)

Written 2026-10-08, **before** phase 4. Specification G1–G3, R1/R2 (`01_Requirements`:
"wie effizient ist es"), FR-27…FR-30, NFR-7 ("reported, not promised"), acceptance
§8.5–§8.6; `06_Code/docs/benchmarking.md` (the protocol: five rules, minimum viable
report). This is the test the whole project exists for: the owner's machine, the owner's
files, the owner's two references.

**Host (owner):** the machine with the RTX 3070 (8 GB). Record CPU model, cores/threads,
RAM total and available, OS and kernel, storage device holding the GGUF (NVMe/SATA, model
and whether it also holds the OS), Ollama version, colibrì upstream commit used for the
gs64 container runs, sylph commit. Phase 4 is **CPU-only** for sylph (FR-36); Ollama and
the container run as the owner normally runs them (GPU allowed and stated), because R2
asks how the owner's deployment compares, not how two CPU paths compare. A sylph-GPU
column arrives with phase 5.

## Inputs

| Input | Notes |
|---|---|
| `MODEL` | the same GGUF as `equivalence.md` (Ollama's blob or unsloth `UD-Q4_K_M`); 22.1 GB; sha256 in the report |
| gs64 container | the owner's `qwen36_i4_gs64` container (the #1370 setup), unchanged |
| Ollama | the tag whose blob is `MODEL`, same `num_ctx`, `temperature 0`, `seed 1` |
| prompts | `fixtures/e3_prompts.txt` (32) for correctness and TTFT; `tools/datapoint.py`'s rotating prompt set for throughput; one 2 000-token prompt (the first window of `wiki.test.raw`, decoded) for prefill |
| `CAP` | sylph's expert-cache slots per layer from `coli plan --model MODEL` (record the plan's RAM arithmetic: experts 1.90 MB × slots × 40 layers + dense 2.56 GB + KV) |

## Part A — it works as a product (functional)

| # | Step | Expected |
|---|---|---|
| A0 | `coli gguf inspect MODEL`, `coli doctor --model MODEL --deep --gpu none`, `coli plan --model MODEL`, `coli info --model MODEL` | as phase 1/3: `qwen35moe (engine: qwen36)`, every `model.gguf.*` and `model.tokenizer` check `pass`, `memory.ram` per the host; `plan` prints slots/layer and bytes per expert; `info` prints the sidecar path `(none yet)` |
| A1 | `coli chat --model MODEL` | startup `[GGUF] …` line (FR-30) and `[meta] from GGUF:`; first reply to "Hello, who are you?" within the TTFT recorded; three turns of conversation coherent; `:reset` works; `Ctrl-C` leaves no process (`coli stop`); **no file appears beside `MODEL`** (FR-29) — `ls -la` of the directory before and after |
| A2 | chat template and stop tokens | the reply ends at `<|im_end|>` without printing it; a `think`-style model preamble is handled as for the container (`--think/--no-think`); the rendered prompt equals the one the container path renders for the same conversation (`COLI_DEBUG=2`) |
| A3 | `coli serve --model MODEL --port 8080` + `curl` OpenAI `/v1/chat/completions` (stream and non-stream, `logprobs=5`) | valid responses, `usage` counts, token-level `logprobs` tails present; two concurrent requests complete |
| A4 | `coli web --model MODEL` | the dashboard shows the expert map (`EMAP`) moving, hit rate, RSS; a chat round trip in the browser |
| A5 | restart with `PIN=…`/`HOT=n` and `PILOT=1` | runs, prints `[HOT] Pinned …`/pilot statistics; replies identical to A1 for `temperature 0` |
| A6 | split set | `MODEL` re-split with llama.cpp's `gguf-split --split-max-size 8G` into a directory: `coli chat --model <dir>` identical replies, `[GGUF] … 3 parts` |
| A7 | tokenizer | `tests/test_tok_gguf MODEL <tokenizer.json of Qwen3.6> fixtures/tok_corpus.txt --llama-tokenize <llama.cpp>/llama-tokenize` → `all passed` (the owner-side half of case 5 of `src_facade.md`) |

Pass: A0–A7 as expected; A1's "no file beside MODEL" is a hard criterion.

## Part B — how efficient is it (A/B, `docs/benchmarking.md`)

Three systems, same prompts, same machine, interleaved runs (rule 4: alternate the
systems prompt by prompt, never three ordered sweeps), each number the median of 3
with the spread:

| System | Command | Notes |
|---|---|---|
| **S** sylph-GGUF (CPU) | `coli chat/serve --model MODEL --cap CAP` | `COLI_TEMP=0`; cache state per run: **cold** (`coli stop`, drop page cache or reboot, fresh process) and **warm** (second pass of the same prompt set in one process) |
| **O** Ollama | `/api/generate` with `options.temperature 0`, `seed 1`, `num_predict 128`, `keep_alive` so the model stays loaded between prompts; TTFT and tok/s from `prompt_eval_duration`, `eval_count/eval_duration` | GPU as configured by the owner (state `ollama ps` VRAM split); cold = first request after `ollama stop`, warm = subsequent |
| **C** colibrì gs64 container | `coli chat/serve --model qwen36_i4_gs64 --cap CAP_C` upstream build or the same sylph binary (NFR-5 says they are identical) | CPU, or `COLI_CUDA=1` tier as the owner runs it (state it); same cache-state protocol as S |

Measured per system and cache state (the "minimum viable report" list):

- **TTFT** (ms) on the 32 prompts and on the 2 000-token prompt; **decode tok/s** over
  128 generated tokens; **prefill tok/s** on the 2 000-token prompt.
- **bytes read per generated token** (S: `GGUF reads:` line; C: the container's expert
  statistics; O: not observable, state "n/a") and expert cache **hit rate**.
- **RSS** (S, C: `PEAK RSS`), **VRAM** (O: `ollama ps`; C with tier: `nvidia-smi`).
- **quality control** beside every number (rule 4): the E2 perplexity of
  `equivalence.md` for S and O (same file ⇒ same PPL within threshold), the #1370 PPL
  for C (7.325 gs64), and the E3 prefix lengths — a faster number with a different PPL
  is not a win.
- the independent control: a fixed prompt re-run at the end of every session must
  reproduce its first TTFT/tok/s within the spread; stop and discard the session if it
  does not (thermal drift, background load).

Record everything with `python3 06_Code/c/tools/datapoint.py` (machine, cold/warm,
rotating prompt record) and keep the raw logs. The report
`08_Documents/benchmarks/<date>-qwen36-35b-<host>.md` has one table per cache state,
the host description, the exact commands, the model sha256, the Ollama/colibrì/sylph
versions, excluded runs with reasons, and a short reading of the result that names what
sylph does differently (K-quant experts on CPU, no GPU yet) so that R2 is answered
honestly: *is the GGUF path a viable way, and how efficient is it, today, on this
machine*.

## Part C — no regression of the container (acceptance §8.6)

C's numbers with the sylph binary equal C's numbers with the upstream binary of the same
commit within the spread (two columns in the report). NFR-5 at system level.

## Pass criterion

Part A passes; Part B is **reported**, not judged, with the quality controls green (E2
PPL of S = O within threshold; E3 prefixes per `equivalence.md`); Part C within spread.
Negative results (sylph slower than Ollama on this CPU, as is likely before phase 5)
are published with the same detail (G3).

## State

| Date | Result |
|---|---|
| 2026-10-08 | written; needs phase 4 (`[GGUF]` line, `GGUF reads:`, sidecar rule, `coli info` field) and the owner's machine. The owner's open points (spec §9): the Ollama tag/blob, whether the gs64 container is the #1370 one. |
