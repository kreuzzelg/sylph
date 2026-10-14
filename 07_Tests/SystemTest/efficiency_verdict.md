# System test — the efficiency verdict: one tool answers R2 (sylph-GGUF vs colibrì gs64 vs Ollama) and CI keeps it honest (phase 7)

Written 2026-10-14, before phase 7. Owner requirement R2 ("ob es einen gangbaren Weg gibt.
Wenn ja, wie effizient ist es"), G3 (negative results are published), NFR-7 (performance is
reported, not promised), acceptance §8.5 (A/B published per `docs/benchmarking.md`: cache
state, request sizes, controls); `real_model.md` Part B (the protocol, owner-run, no tool
yet), `gpu_rtx3070.md` G4, `kernel_throughput.md` (kernel-level numbers, 2026-10-06).

The question the fork exists for has, after six phases, every ingredient but no single
answer: kernel ratios (`08_Documents/kernels/`), a lossless oracle, an equivalence harness,
a CUDA tier, two engines. The owner's A/B protocol in `real_model.md` is a table to fill by
hand across three systems and two cache states. This test turns it into **one tool** that
runs the systems, reads the engines' own accounting, writes the report in the form
`docs/benchmarking.md` demands, and states the verdict — and a CI subset that runs the same
tool on the tiny models so its accounting can never drift from the engines' lines.

## Contract (`06_Code/c/tools/efficiency_report.py`, stdlib)

```
python3 tools/efficiency_report.py --sylph <model.gguf> [--container <dir>] [--ollama <url> --tag <tag>]
        [--cap N] [--new 128] [--prompts <file>] [--prefill-tokens 2000] [--runs 3] [--cold drop|reboot|none]
        [--gpu none|cuda] [--ppl-ref <ref_logprobs.tsv>] [--ci] --out <report.md> [--json <report.json>]
```

| System | How it is run | What is read |
|---|---|---|
| **S** sylph-GGUF | `SERVE=1` session of `qwen36` on the GGUF (`COLI_TEMP=0`, `--cap`), one request per prompt, `--runs` passes: pass 1 **cold** (page cache dropped per `--cold`, fresh process), passes 2… **warm** (same process) | `STAT` (RSS), `DONE … STAT` (decode tok/s), the `PROF` line (disk / moe / attention / head seconds), `GGUF reads:` (slices, MB, MB/token, parts) per turn, `[GGUF]` startup line (types, bytes), TTFT measured by the tool (first `DATA` frame) |
| **C** colibrì gs64 container | the same session on the container (`SNAP=<dir>`, same `--cap`) | `STAT`, `DONE … STAT`, `PROF` (disk seconds), the cache line (`hits/misses`); bytes/token = misses × bytes per expert from `coli plan` (the container has no read counter) |
| **O** Ollama (optional) | `/api/generate` with `options.temperature 0`, `seed 1`, `num_predict --new`, `keep_alive` | `prompt_eval_duration` → TTFT, `eval_count/eval_duration` → tok/s; RSS/VRAM from `ollama ps`; bytes/token not available (stated as such) |

Every system gets the **same prompts** (`--prompts`, default `fixtures/e3_prompts.txt`), the
same `--new`, the same cache-state protocol; one 2 000-token prompt (`--prefill-tokens`, the
first window of `wiki.test.raw` when given, else a repetition of the prompt file) measures
prefill tok/s. Quality rides along: `PPL=1` teacher-forced NLL on the first window for S and
C (the engines' own line), and when `--ppl-ref` is given the E1 ΔNLL of S vs the reference
(`compare_logprobs.py`). An independent control per run: the token ids of the first prompt
under `COLI_TEMP=0` must be identical across passes of the same system (else the run is
marked unstable and the report says so, rule 4 of `docs/benchmarking.md`).

### The report (`--out`)

1. **Minimum viable report** (`docs/benchmarking.md`): commit (`git describe`), host (CPU,
   cores used, RAM total/available, GPU, storage device and model-volume location), model
   id + quantization + sha256 of the GGUF (and of the container's index), exact commands,
   prompts and `--new`, cache policy/capacity/state and how it was established, buffered I/O,
   request sizes.
2. **The table**, one row per system × cache state: decode tok/s (median, min, max over the
   prompts), TTFT (median), prefill tok/s, bytes/token read, RSS (and VRAM where used),
   expert hit rate, PPL/NLL, the control (ids stable: yes/no).
3. **Ratios** S/C and S/O for tok/s, TTFT, bytes/token, RSS, with the deliberate differences
   named (spec §4.2: no re-quantization, raw blocks, f32 activations vs upstream's int8).
4. **Verdict**: three sentences, one per question — does it run (yes/no, what failed), is it
   equivalent (the equivalence report's result or "not run"), how efficient (the ratios) — and
   the single limiting factor the profile shows (disk, MoE compute, attention, lm_head).
5. `--json`: the same facts as one object (`systems[*].{tok_s, ttft_s, prefill_tok_s,
   bytes_per_token, rss_gb, hit_rate, nll, ids_stable, reads}`, `ratios`, `verdict`,
   `host`, `commit`), the runner's input.

### `--ci` (the tiny models, no owner machine)

The same code path with `--runs 2 --cold none --new 16` on the tiny Qwen3.6 F32 GGUF and,
where torch built it, the int8 container of the same snapshot; Ollama absent. The subset
asserts what can be asserted without a host: the report has every section, the JSON has
every field, and the **accounting identity** holds — `bytes_per_token × tokens` of S equals
the sum of its `GGUF reads:` lines, `slices = 3 × misses`, and the container's bytes/token
equals `misses × bytes per expert` from `coli plan`. tok/s and RSS are **reported, not
asserted** (a tiny model on a shared runner measures nothing). The ids of S are stable across
the two passes (the control) and, for the F32 GGUF vs the F32-dense container, identical.

## Cases

Runner: `python3 07_Tests/SystemTest/run_efficiency.py [--keep] [--no-build] [--owner MODEL=<gguf> CONTAINER=<dir> [OLLAMA=<url> TAG=<tag>]]`.

| # | Case | Expected |
|---|---|---|
| 0 | fixtures | tiny Qwen F32 GGUF (+ the int8 container when torch exists), `fixtures/e3_prompts.txt` present |
| 1 | the tool runs in `--ci` mode | exit 0; `report.md` and `report.json` written; every section of the minimum viable report present (`## Host`, `## Systems`, `## Results`, `## Ratios`, `## Verdict`, the commit, the sha256) |
| 2 | accounting identity (S) | `systems.sylph.bytes_per_token × tokens == Σ GGUF reads bytes` (± 1 %); every turn `slices = 3 × misses`; `reads.parts_touched == 1` |
| 3 | accounting identity (C) | where the container exists: `bytes_per_token == misses × expert_bytes` from `coli plan` (± 1 %) |
| 4 | the control | `ids_stable == true` for S (two passes identical); S vs C ids identical where the container is lossless-equivalent (reported otherwise) |
| 5 | report hygiene | the report names the cache state of every row, the request size, `--cap`, buffered I/O; no number without its unit; the verdict has three sentences |
| 6 | owner mode (`--owner …`) | the full protocol on the real file: the tool exits 0 and writes `08_Documents/benchmarks/<date>-<host>-efficiency.md` + `.json`; the runner checks the same structure and records the ratios in the state table below |

**Pass criterion:** cases 0–5 `RESULT: ok (0 failures)` locally and in the `gguf-oracle` job
(where the container exists); case 6 is the owner's run, the number R2 asks for.

## State

| Date | Result |
|---|---|
| 2026-10-14 | document and runner written; `tools/efficiency_report.py` is the phase-7 deliverable. Pre-implementation run: see the row below. |
| 2026-10-14 | pre-implementation run (`run_efficiency.py --no-build`): `RESULT: FAIL (1 failures)` — `tools/efficiency_report.py` does not exist yet, as expected; the fixtures (prompts, tiny snapshot, F32 GGUF) are in place. |
