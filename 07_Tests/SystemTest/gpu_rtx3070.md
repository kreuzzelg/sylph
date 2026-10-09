# System test — the real model on the RTX 3070 (8 GB): placement, CPU vs CUDA, A/B vs Ollama

Written 2026-10-09, **before** phase 5 (process gate in `../../04_Tasks/tasks.md`).
Specification: FR-37, FR-38, FR-14, FR-35, NFR-7, §4.2, §4.3, §8 items 4–6.
Architecture §3 (tier), §12 row "5 — GPU", and `../IntegrationTest/cuda_tier_kquant.md`
(the contracts this test exercises on real silicon).

Owner's machine only: an NVIDIA RTX 3070 (8 GB), the Qwen3.6-35B-A3B `Q4_K_M` GGUF of
`equivalence.md` (Ollama's blob or unsloth's `UD-Q4_K_M`; sha256 in the report), a CUDA build
of llama.cpp, Ollama with the tag whose blob is `MODEL`. Everything the development
container can check without a card is in `cuda_tier_kquant.md` part A.

## Inputs

| Input | Notes |
|---|---|
| `MODEL` | as `equivalence.md`; dense trunk `Q8_0` 2.5 GB (unsloth) or 1.9 GB (bartowski, `output` `Q6_K`); experts 19.6–20.4 GB, ≈ 1.9 MB each, 40 × 256 |
| build | `make -C 06_Code/c qwen36 CUDA=1 CUDA_ARCH=native` (`docs/qwen36-cuda-tier.md` → Usage); the CPU build of phase 4 kept beside it for the CPU arm |
| llama.cpp | one CPU build and one CUDA build (`-DGGML_CUDA=ON`), the same commit; `LLAMA`, `LLAMA_CUDA` bin dirs |
| Ollama | `ollama ps` shows the CPU/GPU split it chose for this model on this card (record it: it is the reference's own placement, §8.5 of the spec's intent — Ollama is measured as the user gets it) |
| text, prompts | `wiki.test.raw` (E1/E2) and `fixtures/e3_prompts.txt` (E3), as `equivalence.md` |
| knobs | `COLI_CUDA=1 COLI_GPUS=0`, `CUDA_EXPERT_GB=auto` (free minus headroom), `COLI_PLACE=auto` (default) with the startup probe, `HEAT_FILE=heat.bin` for the warm-heat run, `OMP_NUM_THREADS=<physical cores>` |

## Expected arithmetic (sanity check for G1, not a pass criterion)

Free VRAM on a desktop card with a display attached is ≈ 7.0–7.5 GB; the planner keeps a
2 GB reserve, the tier 1 GB headroom. With the trunk placed first (≈ 1.9–2.5 GB) the hot
tier gets ≈ 3–4.5 GB, i.e. **≈ 1 600–2 400 resident experts (16–24 % of 10 240)** at
≈ 1.9 MB + allocation granularity each (`dev_alloc_footprint`: a 1.9 MB slice is charged
2 MiB, a 0.5 MB one 1 MiB — a K-quant expert costs ≈ 2.2–2.5 MB of VRAM, the planner's
`typical_expert_bytes` is the payload; the difference is recorded, not hidden).
Upstream's int4 container on the same card: 4 391 resident (43 %), 44 % / 95 % VRAM hit
rate cold / warm, 9.2 / 9.9 tok/s (`docs/qwen36-cuda-tier.md`, Measured). With a lower
residency the hit rate will be lower; the heat file is what makes the second run count.

## Cases

| # | Case | Command | Expected |
|---|---|---|---|
| G0 | kernel oracle on the card | `make -C 06_Code/c cuda-test-gq` | `cuda-test-gq: all passed`: every block GEMV bit-identical to `gq_dot_row_ref`, fused expert within `1e-5 + 1e-4·|ref|` (the printed max `|Δ|` goes into the report), refusals, determinism under load |
| G1 | plan and doctor | `coli plan --model MODEL --gpu 0`; `coli doctor --model MODEL --deep --gpu 0` | the `VRAM` line reads `<trunk> trunk + <budget> hot tier · ~N experts · NVIDIA GeForce RTX 3070`, N within the arithmetic above; `placement.plan` `pass`; `coli info` unchanged |
| G2 | startup, placement, stability | `COLI_CUDA=1 COLI_GPUS=0 coli chat --model MODEL` (and `N_NEW=200 ./qwen36 256 4 prompt.txt`) | `[GGUF] …` line ends `experts on CUDA tier (<n> planned)`; `[gpu] MoE experts -> CUDA VRAM tier`; `[place] probe:` reports GPU vs CPU GEMV times and the decision; `[place]` lines per component with GGUF byte sizes; `nvidia-smi` memory used ≤ the planned budget + 300 MB during the run; no `out of memory`, no `expert group result unavailable`; at exit `[qtier] resident n/10240 | uploads … | miss(CPU) …`; the `GGUF reads:` line still accounts every slice (misses only) |
| G3 | **E1–E3 CPU vs CUDA**, same binary family | `python3 07_Tests/SystemTest/equivalence/run_equivalence.py --model MODEL --llama LLAMA --llama-cuda LLAMA_CUDA --text wiki.test.raw --chunks 16 --deviation COLI_CUDA=1 --levels E1,E2,E3` | (a) sylph-CPU vs llama.cpp passes as in `equivalence.md` (unchanged by this phase); (b) the `COLI_CUDA=1` deviation row: **E1 mean ΔNLL ≤ 1e-5 and max ≤ 1e-4** against sylph-CPU (the device `expf` and the `Q8_0` dense split are the only sources; both are far below llama.cpp's own CPU-vs-CUDA floor, which the run prints beside it — the report quotes both), top-1 agreement 100 %, E2 `|ΔPPL|/PPL ≤ 1e-5`; (c) E3: the 32 prompts token-identical CPU vs CUDA to the first near-tie (margin 0.05 nat), ≥ 31/32 to 128 tokens — a deviation arm of E3 is a harness item of this phase (`run_equivalence.py` runs sylph twice when a deviation is given and `E3` is in the levels) |
| G4 | **A/B on the card** (`docs/benchmarking.md`, FR-38, §8 item 5) | the Part-B protocol of `real_model.md` with the GPU arms added: **S-GPU** sylph-GGUF `COLI_CUDA=1` (cold heat, then `HEAT_FILE` warm), **O** Ollama with its own split (`ollama ps`), **C-GPU** the gs64 container with the tier as upstream runs it; plus the phase-4 CPU arms for reference | per system and cache state: TTFT (32 prompts, 2 000-token prompt), decode tok/s (128 tokens), prefill tok/s, VRAM used (`nvidia-smi`), RSS, bytes/token (`GGUF reads:`), VRAM hit rate (`[qtier]`), PPL (E2 of G3). Interleaved runs, median of 3 with spread. **Reported, not promised** (NFR-7); the one comparison the owner asked for is S-GPU vs O on the same file and card |
| G5 | int8-activation twin (FR-14, §4.2) | `--deviation QWEN_EXPERT_ACT=i8` on the CPU arm of G3 (E1, E2 and the E3 deviation arm), plus decode tok/s with and without | the twin is **opt-in** (`QWEN_EXPERT_ACT` unset = f32 activations on the GGUF path; the container keeps upstream's default), ΔNLL / ΔPPL / E3 divergence recorded in the report with the speed gain; the default configuration's verdicts (G3 a, `equivalence.md`) are unchanged |
| G6 | no regression of the container tier (§8 item 6) | `COLI_CUDA=1` with the owner's `qwen36_i4_gs64` container, the sylph binary vs the upstream v1.12.1 binary, 200-token decode, same `HEAT_FILE` | tok/s within noise (± 5 %), `[qtier]` resident count identical, logits cosine vs CPU as upstream reports (`docs/qwen36-cuda-tier.md`) |

## Report

`08_Documents/benchmarks/<date>-qwen36-35b-rtx3070.md`: host (CPU, RAM, driver, CUDA
version, llama.cpp commit, Ollama version), `coli plan` output, the G3 deviation table
(sylph-CPU vs sylph-CUDA next to llama.cpp CPU vs CUDA), the G4 tables per cache state,
G5's twin deltas, G6, and the raw `nvidia-smi --query-gpu=memory.used --format=csv -l 1`
log summarised (peak). The G0 output goes into `cuda_tier_kquant.md`'s state table.

## Pass criterion

G0 green; G1–G2 as expected; G3 (a) unchanged and (b)/(c) within the stated bounds; G6
within noise. G4 and G5 are reports: they pass by being complete (every cell filled, cache
state and `ollama ps` split stated). A G3 (b) failure with G0 green points at the engine's
summation order or at a dense matrix taking a different kernel than the CPU (the comparer's
worst-position print and the `[place]` lines say which component).

## State

| Date | Result |
|---|---|
| 2026-10-09 | written; needs phase 5 (block formats, `qt_init_gguf`, GGUF dense placement, planner field, the harness's E3 deviation arm) and the owner's card. The owner's open points (spec §9): the Ollama tag/blob. |
| 2026-10-09 | phase 5 implemented: everything this test exercises exists (`make -C 06_Code/c qwen36 CUDA=1`, `make cuda-test-gq`, `coli plan --gpu 0`, `--deviation COLI_CUDA=1` incl. the E3 arm, `QWEN_EXPERT_ACT=i8`); verified on the fake backend only — G0–G6 are the owner's runs. |
