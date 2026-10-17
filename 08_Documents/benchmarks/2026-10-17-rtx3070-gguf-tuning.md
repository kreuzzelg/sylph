# sylph-GGUF on the RTX 3070 — the tier's knobs, 2026-10-17

Owner's host (Threadripper PRO 3945WX, 94 GB, RTX 3070 8 GB = CUDA device 0, Quadro RTX 4000 = device 1),
unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `qwen36` built with `CUDA=1` after the colibrì `dev` merge, greedy
(`COLI_TEMP=0`), 128 new tokens from one English prompt, cap 256 (every expert in RAM), 12 OpenMP threads
pinned to cores. The owner asked for the knob set upstream recommends; two of them did not work on a GGUF
until this session (below). Numbers are single runs of a short prompt: ±1 tok/s run to run.

| configuration (all with `COLI_CUDA=1 COLI_GPUS=0`) | decode tok/s | VRAM hit rate | note |
|---|---|---|---|
| tier as before (defaults, no heat file) | 18.4 | 16 % | the "16.4 tok/s" of the 64-token runs; the probe already kept the trunk on the GPU |
| + `COLI_PLACE=auto COLI_TRUNK_PROBE=0 CUDA_EXPERT_GB=auto OMP_NUM_THREADS=12 OMP_PLACES=cores OMP_PROC_BIND=close` | 18.9 | 16 % | the probe was not the problem here (0.138 ms GPU vs 0.246 ms CPU) |
| + `HEAT_FILE=heat.bin`, second run (warm heat) | 24.2 | 71 % | the heat file is the single biggest lever on this card |
| + `Q36_DN_GPU=1` (DeltaNet end to end on the device) | 27.3 | 72 % | **failed before** ("GPU step failed" on every layer): `coli_cuda_dn_step` accepted int8 rows only |
| + `COLI_CUDA_CHAIN=1` (every layer on the device) | **32.5** | 71 % | **failed before** ("unspecified launch failure, the device is lost"): the chain's GEMM had no block-format branch |
| chain, 256 new tokens | 35.0 | 80 % | the longer the run, the more the heat matches |
| chain, `COLI_LOOKUP=0` | 31.8 | 71 % | lookup decoding adds ~2 %; with the chain its timing gate makes the text run-dependent at near-ties |
| chain, `COLI_GPUS=0,1` (both cards) | 23.3 | 95 % | more experts in VRAM, slower: the cross-device synchronisation costs more than the misses it saves |
| chain, `QWEN_EXPERT_ACT=i8` (a deviation) | 28.4 | — | faster CPU misses, different text; not the reference configuration |

For scale: Ollama on the same file and card measured 43.9 tok/s (2026-10-15), the gs64 container on the CPU 14.8,
sylph-GGUF on the CPU 12.3–14.1.

## What was fixed

- `coli_cuda_tensor_gemm_async` (the entry `cuda_chain.cu` and the DeltaNet step reach): a branch for the raw
  ggml block formats (`blk_matmul` on the chain's stream).
- `coli_cuda_dn_step`: the projection/out tensors may be block tensors (`fmt 16 + type`); the format and the row
  bytes go to `quant_matmul_launch` instead of a hard-wired int8.
- `tests/test_gq_cuda.cu` B5 pins the chain GEMM against `gq_dot_row_ref` (bit-exact, S = 1 and 3); `make
  cuda-test-gq`: all passed on the RTX 3070.

The generated text is identical across tier-only, DeltaNet-on-GPU and chain for the same heat file (md5 of the
first 600 characters), so the three paths compute the same numbers up to the named device deviations.

## Where the time goes now

The chain reports, per 128-token decode: 1.7 s waiting for the device, 1.8 s of routed experts on the host.
The remaining lever is the expert hit rate in 4.7 GB of VRAM: the heat file, a longer context, or a card with
more memory. The dense trunk and the DeltaNet recurrence are on the device; the CPU's share is the 29 % of
expert reads that miss VRAM.

## Reproduce

```sh
make -C 06_Code/c qwen36 CUDA=1 NVCC=/usr/bin/nvcc
COLI_CUDA=1 COLI_GPUS=0 COLI_PLACE=auto COLI_TRUNK_PROBE=0 CUDA_EXPERT_GB=auto COLI_CUDA_CHAIN=1 \
HEAT_FILE=heat.bin OMP_NUM_THREADS=12 OMP_PLACES=cores OMP_PROC_BIND=close COLI_TEMP=0 N_NEW=128 \
SNAP=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf 06_Code/c/qwen36 256 4 prompt.txt   # run twice: the 2nd has the heat
```

## Why still slower than Ollama (measured the same afternoon)

Ollama on the same file, same prompt, 256 greedy tokens: **43.5 tok/s** (23 ms/token). Its log says where the
weights sit: `CUDA0 model buffer 4984 MiB`, `CUDA1 3213 MiB`, **`CUDA_Host model buffer 13247 MiB`** — 8.2 GB on
the two cards, 13.2 GB of experts on the host, 12 threads. While it decodes the cards are 36 % and 14 % busy:
most of its token is the CPU's expert work too. sylph's chain holds 6.4 GB on one card (1.65 GB trunk +
4.7 GB experts), 71 % of expert reads hit VRAM, and the card is 50 % busy.

One 128-token decode of sylph's chain, per token (`COLI_TIMERS=1`, the chain's own report):

| phase | ms/token | note |
|---|---|---|
| waiting for the device | 12.3 | trunk + resident experts + DeltaNet; **~1 100 kernel launches per token** (97 725 ops / 128 forwards), the traffic itself is ~5 ms at 448 GB/s — the rest is launch and sync latency; ggml records the decode step as one CUDA graph |
| routed experts on the host | 5.5–15 | 93 misses × 1.95 MB = 181 MB at 12–16 GB/s; the same path moves 29 GB/s when a layer has 8 experts to do (trunk-only run: 21.7 ms for 624 MB) — 2–3 misses a layer leave the OpenMP regions mostly idle |
| everything else on the host | ~6 | routing top-k, sampling, frame bookkeeping, embedding row |
| **total** | **30–34** | 29–34 tok/s; Ollama 23 ms |

The device wait and the host misses are **serialised** in the chain (12.3 + 11 ≈ the token); the tier-only path
overlaps them but keeps DeltaNet on the CPU (15 ms). Two cards do not help sylph (23 tok/s at 95 % hits): the
tier gathers per token across devices, while ggml splits whole layers between cards and never crosses them
inside a token. `QWEN_EXPERT_ACT=i8` changed nothing on the miss path (11.4 vs 11.8 ms): it is not compute-bound.

Done today on the miss path: the token's misses go to one `gq_moe_run` (two OpenMP regions instead of three per
expert, scratch reused) — tier-only 11.8 → 10.1 ms/token, the chain 32.5 → 33.9 tok/s in the better runs. The
three levers left, in order of size: a CUDA graph for the chain's frame (upstream's design), overlapping the host
misses with the device's resident experts inside a layer, and a planar/pinned expert layout for the CPU misses so
a 2-expert layer still saturates memory bandwidth.
