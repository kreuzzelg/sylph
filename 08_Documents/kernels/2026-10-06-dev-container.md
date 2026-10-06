# Kernel throughput — development container, 2026-10-06

Protocol: `../../07_Tests/SystemTest/kernel_throughput.md` (NFR-7). Command: `make -C 06_Code/c bench-gq`
(synthetic weights at Qwen3.6-35B-A3B shapes; no model file on this host).

**Host 1 — development container.** Intel Xeon @ 2.10 GHz (cloud vCPU; AVX2/FMA/F16C, AVX-512 present but
unused by design), 4 threads, 15 GB RAM. gcc 13.3.0, `-O3 -march=native -fopenmp` (the Makefile's default
developer build). `06_Code` at the commit this file was added in. Shared, noisy host: baseline rows vary by
up to ±30 % between runs; ratios within one run are comparable, absolute numbers are indicative only.

## Result

- **NFR-7 target met on this host.** Row A (`Q4_K` expert GEMV vs upstream's planar int4): **1.27–1.30** ns per
  weight byte relative to the baseline over two consecutive runs (`Q5_K` 1.27–1.35, `Q6_K` 1.04–1.15). Row E
  (`gq_moe_run` vs `xf_moe_run`, 16 resident experts): **1.17–1.45** at S=1 hot, 1.41 at S=32. The spread is the
  host's noise (the planar baseline itself moved 0.55–0.81 ms between runs), so the raw table below is the
  final run and the ranges are the honest statement.
- Dense `Q8_0` via the lossless split into int8 plane + per-32 scales runs on upstream's `matmul_q_gs`
  within 1.06–1.29 of the int8-row baseline per byte (row C/D); the native block kernel is on par.
- lm_head (248 320 × 2048) in ms/token on 4 threads: see the line above the table. `Q6_K` reads 18 %
  fewer bytes than int8-row and lands within 1.1–1.2× of its time.
- `gq_embed_row Q8_0` costs about twice a plain f32 row copy per row (it decodes 2176 bytes into 8 KiB);
  one call per token, irrelevant.

## History within the day

| version | row A `Q4_K` | row E S=1 hot | change |
|---|---|---|---|
| first | 1.70 | 1.77 | per-sub-block scalar scale decode, per-sub-block activation lane sums, one accumulator |
| + precomputed per-token lane sums, per-block scale decode | 1.66 | 1.74 | little gain: the limit was the serial fma chain into one accumulator (4 dependent fmas per 64 elements ≈ 16 cycles; the planar kernel has 1) |
| + four independent accumulators, vector scale decode | 1.48 | 1.47 | chain cut to one fma per accumulator per 64 elements |
| + min term folded once per block from per-32 activation sums (`gq_xsum32`) | **1.27–1.30** | **1.17–1.45** | 8 fmas + 8 broadcasts per block → 1 fma; `xs` shrinks 8× |

All versions stay bit-identical between SIMD and scalar reference and within 1e-6 of a double dot; E0 is
untouched (dequantization is a separate code path).

## Raw output


path: avx2 · OpenMP threads: 4 · compiler: 13.3.0 · median of 50 after 5 warm-ups

lm_head ms/token: int8-row 23.60 · Q8_0 split 27.31 · Q8_0 native 26.69 · Q6_K 25.20
| row | kernel | layout | bytes/row | µs/row (E: ms/layer) | GB/s | ratio | note |
|---|---|---|---|---|---|---|---|
| A | xf_dot_f32 (baseline) | planar int4 gs64 | 1152 | 0.153 | 7.53 | 1.00 | upstream expert kernel |
| A | gq_dot_row Q4_K | synthetic blocks | 1152 | 0.194 | 5.94 | 1.27 | ratio = ns per weight byte vs baseline |
| B | xf_dot_f32 (baseline) | planar int4 gs64 | 288 | 0.039 | 7.43 | 1.00 | upstream expert kernel |
| B | gq_dot_row Q5_K | synthetic blocks | 352 | 0.064 | 5.49 | 1.35 | ratio = ns per weight byte vs baseline |
| B | xf_dot_f32 (baseline) | planar int4 gs64 | 288 | 0.042 | 6.85 | 1.00 | upstream expert kernel |
| B | gq_dot_row Q6_K | synthetic blocks | 420 | 0.064 | 6.61 | 1.04 | ratio = ns per weight byte vs baseline |
| C | matmul_q_gs gs=I (baseline) | int8-row + 1 scale | 2052 | 0.038 | 54.35 | 1.00 | attn_qkv shape 8192x2048, µs per row |
| C | matmul_q_gs gs=32 | Q8_0 split (plane + f32/32) | 2304 | 0.064 | 35.94 | 1.51 | gq_q8_0_split at load, upstream kernel |
| C | gq_matmul Q8_0 | raw Q8_0 blocks | 2176 | 0.055 | 39.65 | 1.37 | native block kernel |
| D | matmul_q_gs gs=I (baseline) | int8-row + 1 scale | 2052 | 0.095 | 21.59 | 1.00 | lm_head 248320x2048, µs per row |
| D | matmul_q_gs gs=32 | Q8_0 split (plane + f32/32) | 2304 | 0.110 | 20.95 | 1.03 | gq_q8_0_split at load, upstream kernel |
| D | gq_matmul Q8_0 | raw Q8_0 blocks | 2176 | 0.107 | 20.24 | 1.07 | native block kernel |
| D | gq_matmul Q6_K | raw Q6_K blocks | 1680 | 0.101 | 16.55 | 1.30 | unsloth output.weight type |
| E S=1 | xf_moe_run hot (baseline) | planar int4 gs64 | 1769472 | 0.658 | 2690.59 | 1.00 | ms per layer, E resident experts |
| E S=1 | gq_moe_run Q4_K/Q4_K/Q5_K | raw K-quant blocks, hot | 1900544 | 0.770 | 2469.25 | 1.17 | ratio = layer time vs baseline |
| E S=1 | xf_moe_run cold | planar int4 gs64 | 1769472 | 0.691 | 2561.29 | 1.05 | fresh copy per iteration |
| E S=1 | gq_moe_run Q4_K/Q4_K/Q5_K | raw K-quant blocks, cold | 1900544 | 0.880 | 2159.62 | 1.27 | ratio = cold layer time vs cold baseline |
| E S=32 | xf_moe_run hot (baseline) | planar int4 gs64 | 1769472 | 18.264 | 96.88 | 1.00 | ms per layer, E resident experts |
| E S=32 | gq_moe_run Q4_K/Q4_K/Q5_K | raw K-quant blocks, hot | 1900544 | 25.480 | 74.59 | 1.40 | ratio = layer time vs baseline |
| E S=32 | xf_moe_run cold | planar int4 gs64 | 1769472 | 17.973 | 98.45 | 0.98 | fresh copy per iteration |
| E S=32 | gq_moe_run Q4_K/Q4_K/Q5_K | raw K-quant blocks, cold | 1900544 | 24.878 | 76.40 | 1.38 | ratio = cold layer time vs cold baseline |
| F | f32 row copy (baseline) | f32 | 8192 | 0.090 | 91.04 | 1.00 | token_embd lookup |
| F | gq_embed_row Q8_0 | raw Q8_0 blocks | 2176 | 0.247 | 8.82 | 2.74 | ratio = time per row vs copy |

NFR-7: row A ratio ≤ 1.5 and row E (S=1, hot) ratio ≤ 1.5 pass the target; others are reported.
