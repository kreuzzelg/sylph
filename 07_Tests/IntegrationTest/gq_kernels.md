# Integration test — ggml block kernels (`gq.h` ⟷ gguf-py golden ⟷ `gsgemv.h`/`expert_ffn.h` ⟷ OpenMP)

Written 2026-10-05, **before** phase 2 (process gate in `../../04_Tasks/tasks.md`).
Specification: `../../02_Specifications/gguf-specification.md` §4.1 (E0), FR-8…FR-13,
NFR-1/NFR-3; architecture: `../../03_Architecture/gguf-architecture.md` §5.

**Module under test:** `06_Code/c/gq.h` through its unit-test binary
`06_Code/c/tests/test_gq_kernels` (both are phase-2 deliverables; this document fixes
what the binary must offer).
**Neighbours exercised:** llama.cpp's reference decode (via committed golden
vectors), upstream's `tools/gguf_dequant.py` (second oracle), `gsgemv.h`'s
`matmul_q_gs` (consumer of the `Q8_0` split), `expert_ffn.h`'s layer-runner contract,
OpenMP (thread-count independence).
**Inputs:** `fixtures/e0/` (17 fixtures + `manifest.json`), the binary's CLI below.
**Outputs:** bit-exact f32 dequantization (E0), refusals by type name, a
thread-independent layer digest.

## What "E0" pins

ggml's `dequantize_row_<type>` is the reference (spec §4.1). Its **expression order
is part of the contract**: `Q8_0`: `d·q`; `Q4_0`: `d·(q−8)`; `Q4_K`/`Q5_K`:
`(d·sc)·q − (dmin·m)` per 32-element sub-block with the 6-bit scale/min unpacking of
`get_scale_min_k4`; `Q6_K`: `(d·sc)·(q−32)` per 16-element sub-block; `F16`/`BF16`:
widening. Bit-exact includes the sign of zero (a zero scale yields `±0` depending on
the operand signs) and excludes fused multiply-add: the C reference path must not
be contracted by the compiler (`-ffp-contract=off` for that translation unit or the
function; the default `-O3 -march=x86-64-v3` build **does** contract). Case 2 catches
both mistakes.

## Golden fixtures (`fixtures/e0/`)

Generated once by `make_e0_golden.py` with llama.cpp's own **gguf-py 0.19.0**
(`gguf.quants.dequantize`, numpy 2.4.6); regenerate only when the manifest's
provenance changes. Cross-checked on 2026-10-05 against upstream colibrì's
independent ggml port `06_Code/c/tools/gguf_dequant.py`: **16 of 16 shared fixtures
bit-exact** (upstream has no `Q4_0`). Two independent decoders agreeing on every bit
is the evidence the spec asks for ("cross-checked once against a llama.cpp dump").

| Fixture | Type | Source (HTTP range of the public file) | Elements |
|---|---|---|---|
| `real_unsloth_output_q6_k` | Q6_K | unsloth `UD-Q4_K_M`, `output.weight` rows 0–3 | 8192 |
| `real_unsloth_token_embd_q8_0` | Q8_0 | unsloth, `token_embd.weight` rows 0–3 | 8192 |
| `real_unsloth_attn_qkv_q8_0` | Q8_0 | unsloth, `blk.0.attn_qkv.weight` rows 4096–4099 (value third) | 8192 |
| `real_unsloth_gate_exps_q4_k_e0` / `_e200` | Q4_K | unsloth, `blk.0.ffn_gate_exps.weight` expert 0 / expert 200, rows 0–3 | 8192 each |
| `real_unsloth_down_exps_q5_k_e0` | Q5_K | unsloth, `blk.0.ffn_down_exps.weight` expert 0, rows 0–7 | 4096 |
| `real_bartowski_down_exps_q6_k_e0` | Q6_K | bartowski `Q4_K_M`, `blk.0.ffn_down_exps.weight` expert 0, rows 0–7 | 4096 |
| `real_bartowski_gate_inp_bf16` | BF16 | bartowski, `blk.40.ffn_gate_inp.weight` rows 0–1 (MTP router) | 4096 |
| `real_unsloth_attn_norm_f32` | F32 | unsloth, `blk.0.attn_norm.weight` (stored `1+w`) | 2048 |
| `synth_<type>` ×8 | F32 F16 BF16 Q4_0 Q8_0 Q4_K Q5_K Q6_K | seed 20261005; random blocks with **finite** f16 scale fields, plus 6 edge blocks each: scale `+0`, `−0`, `65504`, smallest subnormal, `±1` with all-zero / all-ones quants | 1032–2560 |

Byte ranges, sha256 of every file and generator versions are in `manifest.json`.
Total 337 KiB. The phase-2 unit test copies the eight `synth_*` pairs verbatim to
`06_Code/c/tests/fixtures/gq_e0/` so `make check` carries the CI subset of E0
(FR-34) without reaching outside the subtree; the real rows stay here.

## CLI contract of `tests/test_gq_kernels`

| Invocation | Behaviour |
|---|---|
| *(no arguments)* | Self-contained unit suite (below). Prints the ISA path (`path: avx2 …` / `scalar`), one `FAIL: …` line per failure, and ends with `all passed` and exit 0, or `N failure(s)` and exit 1 (the `test_expert_ffn.c` convention). |
| `deq <TYPE> <blocks.bin> <numel>` | Decodes `<numel>` elements of ggml blocks read from the file with the **scalar reference** `gq_deq_row_<TYPE>` and writes them to stdout as little-endian f32. `TYPE` is the ggml name (`Q4_K`, `Q8_0`, `BF16`, …). File size must equal `gq_row_size(TYPE, numel)`; otherwise `size mismatch` on stderr, exit 3. A type outside the v1 set (e.g. `Q2_K`, `IQ4_NL`, `MXFP4`) prints `unsupported type <TYPE>` on stderr and exits 2 **before** reading the file. |
| `moe-digest` | Runs one fixed `gq_moe_run` scenario (seeded LCG; `S=7, K=8, H=2048, F=512`, 12 distinct experts shared across rows, gate/up `Q4_K`, down alternating `Q5_K`/`Q6_K`, f32 activations) with the OpenMP thread count the environment provides and prints the 64-bit FNV-1a of the output bytes as 16 hex digits, exit 0. |

## Cases

| # | Case | Expected |
|---|---|---|
| 0 | Fixture integrity | every `.bin`/`.f32` matches the sha256 and sizes in `manifest.json` (guards against accidental edits of the golden). |
| 1 | Unit suite (no args) | exit 0, `all passed`. The suite must cover: type table equals ggml (block size, bytes/block for the 8 types; `gq_row_size`); `gq_supported` true for exactly the v1 set; every SIMD `gq_dot_row_*` **bit-identical** to its scalar reference (same per-lane order, `expert_ffn.h` rule) on random rows of `I ∈ {256, 512, 2048, 4096}`; scalar dot vs double-precision dot within `1e-6` of the summed magnitude; `gq_q8_0_split` is lossless (re-packing the int8 plane + scales yields the original `Q8_0` bytes) and `matmul_q_gs(gs=32)` on the split equals dequant·x within the same `1e-6` rule; `gq_matmul` (`Q6_K`, `Q8_0`) equals per-row dots; `gq_embed_row` equals `gq_deq_row`; `gq_moe_run` equals the per-token loop of GEMVs + silu + rank-ordered sum within `1e-5 + 1e-4·|ref|` (the `test_expert_ffn.c` layer-vs-loop tolerance) on `(S,K,H,F,experts) ∈ {(1,8,2048,512,16), (7,8,2048,512,12), (5,4,256,192,6)}`; an unrouted token (`idx = −1`) yields zeros; `gq_selftest()` returns 0. |
| 2 | **E0 vs llama.cpp golden** | for all 17 fixtures: `deq <TYPE> <bin> <numel>` output **byte-identical** to the `.f32` file (real Qwen3.6 rows of every type the engine will meet, synthetic blocks with edge scales). |
| 3 | Live cross-check (optional) | if `numpy` + `gguf` are importable: 16 fresh random blocks per block type (finite scales) decode identically in C and in `gguf.quants.dequantize`. Skips, never fails, without the packages. |
| 4 | Second oracle (optional) | upstream's `tools/gguf_dequant.py` reproduces the golden bit for bit on every type it implements (needs numpy). Recorded result 2026-10-05: 16/16. |
| 5 | Refusals | `deq Q2_K`, `Q3_K`, `IQ4_NL`, `MXFP4`, `TQ1_0`: non-zero exit, stderr contains `unsupported` and the type name (FR-10). |
| 6 | Thread independence | `moe-digest` prints the same digest under `OMP_NUM_THREADS=1`, `2`, `4` (rank-ordered reduction: the engine's determinism contract for `xf_moe_run` carries over to `gq_moe_run`). |

**Pass criterion:** cases 0, 1, 2, 5, 6 pass; cases 3 and 4 pass or skip.
Deviations from ggml's values are **never** tolerated at this level; a deliberate
numeric change (spec §4.2) lives in a separate kernel behind a knob and is measured at
E1–E3, not here.

## Runner

```sh
python3 07_Tests/IntegrationTest/run_gq_kernels.py              # builds tests/test_gq_kernels if missing
python3 07_Tests/IntegrationTest/run_gq_kernels.py --no-build   # reuse the binary
pip install "numpy>=1.26" "gguf>=0.19" && python3 07_Tests/IntegrationTest/make_e0_golden.py   # regenerate golden (maintainer only)
```

Standard library only; cases 3–4 light up when numpy/gguf are present. Until phase 2
lands, the runner reports `RESULT: FAIL (module not built …)` after case 0.

## State

| Date | Result |
|---|---|
| 2026-10-05 | document + runner + 17 golden fixtures written; oracles cross-checked (16/16 bit-exact); case 0 passes; build of `tests/test_gq_kernels` expected to fail until phase 2 |
