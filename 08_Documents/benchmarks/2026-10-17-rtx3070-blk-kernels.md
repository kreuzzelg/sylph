# Raw-block device kernels — bench_gq_cuda baseline, RTX 3070, 2026-10-17 (before phase 8)

Protocol: `make bench-gq-cuda` (median of 50 launches, device pointers for the dense GEMV, the tier's issue/take round trip for the group). Numbers before any kernel change; later rows are appended per step.

```
bench-gq-cuda: NVIDIA GeForce RTX 3070, 46 SMs, 448 GB/s theoretical (256-bit @ 7001 MHz), median of 50 launches
dense (blk_matmul through coli_cuda_tensor_gemm_async, device pointers):
  DeltaNet in_proj qkv++z            Q8_0    12288 x 2048     26.7 MB     153.4 us   174.3 GB/s
  attention q_proj                   Q8_0     8192 x 2048     17.8 MB     124.9 us   142.8 GB/s
  DeltaNet out_proj / attn o_proj    Q8_0     2048 x 4096      8.9 MB      84.0 us   106.3 GB/s
  shared expert gate/up              Q8_0      512 x 2048      1.1 MB      22.5 us    49.8 GB/s
  shared expert down                 Q8_0     2048 x 512       1.1 MB       9.2 us   121.1 GB/s
  shared expert gate/up (Q6_K)       Q6_K      512 x 2048      0.9 MB      14.3 us    60.6 GB/s
  expert gate (Q4_K)                 Q4_K      512 x 2048      0.6 MB      39.9 us    15.0 GB/s
  expert down (Q5_K)                 Q5_K     2048 x 512       0.7 MB      20.4 us    35.5 GB/s
  lm_head                            Q6_K   248320 x 2048    417.2 MB    1539.1 us   271.1 GB/s
expert group (grouped_hidden_blk_dual + grouped_down_blk through issue/take):
  routed experts, one token          Q4_K/Q5_K K=8  512x2048    15.3 MB     295.2 us    51.9 GB/s  (issue+take, host clock)
  routed experts, one token          Q4_K/Q6_K K=4  512x2048     8.2 MB     151.1 us    54.4 GB/s  (issue+take, host clock)
  routed experts, one token          Q4_K/Q5_K K=1  512x2048     1.9 MB      97.3 us    19.7 GB/s  (issue+take, host clock)
```

## Phase 8 steps (same card, same shapes; from step B on: median of 40 samples of 10 back-to-back launches)

GB/s per shape. "A+B": rows-per-block templates, 4-byte spans exchanged with `__shfl_sync`, the activation row
staged in shared memory for the K-quant lanes, `__half2float`, one barrier per row. "+bounds": `__launch_bounds__`
so 48 warps (Q8_0, 40 registers) or 32 warps (K-quants, 64 registers) are resident per SM — the compiler's own 44
registers had capped the Q8_0 trunk at 5 blocks an SM; this one attribute was the largest single step. "+table":
the Q4_K/Q5_K min-term sums (`blk_xsub32`) computed once per block into shared memory instead of inside the
sub-block loop on one lane at a time; rows per block 32 or 8 (16-thread blocks were never the better choice),
Q8_0 always 8.

| shape | baseline | A+B | +bounds | +table, R ∈ {32, 8} | target |
|---|---|---|---|---|---|
| in_proj 12288×2048 Q8_0 (26.7 MB) | 174 | 185 | 362 | **376** | ≥ 250 ✓ |
| q_proj 8192×2048 Q8_0 | 143 | 157 | 344 | **364** | |
| out_proj 2048×4096 Q8_0 | 106 | 120 | 288 | **289** | |
| shared gate/up 512×2048 Q8_0 (1.1 MB) | 50 | 78 | 129 | **126** | ≥ 120 ✓ |
| shared down 2048×512 Q8_0 | 121 | 202 | 222 | **222** | |
| shared gate/up 512×2048 Q6_K | 61 | 46 | 90 | **83** | ≥ 120 ✗ (512 rows: 64 blocks of 64 threads, latency-bound) |
| expert gate 512×2048 Q4_K | 15 | 23 | 23 | **68** | ✗ as above |
| expert down 2048×512 Q5_K | 36 | 56 | 55 | **97** | |
| lm_head 248320×2048 Q6_K (417 MB) | 271 | 341 | 253 (40 registers: spills) | **345** | |
| group K=8 Q4_K/Q5_K (15.3 MB, issue+take) | 52 | 106 | 98 | **156** | ≥ 150 ✓ |
| group K=4 Q4_K/Q6_K | 54 | 65 | 60 | **136** | |
| group K=1 Q4_K/Q5_K (1.9 MB) | 20 | 24 | 24 | **57** | ✗ (512 row pairs) |

`make cuda-test-gq` after every step: all passed (B1/B2/B5 bit-exact against `gq_dot_row_ref`). The numbers that
stay under the target are the single small K-quant matrices: 512 rows × 8 lanes = 128 warps on 46 SMs; only more
lanes per row (a different summation order, see the amendment's "rejected for now") or several matrices in one
launch (what the group kernel does — K = 8 reaches 156) change that.

Final output:

```
  DeltaNet in_proj qkv++z            Q8_0    12288 x 2048     26.7 MB      71.1 us   375.9 GB/s  
  attention q_proj                   Q8_0     8192 x 2048     17.8 MB      49.0 us   363.6 GB/s  
  DeltaNet out_proj / attn o_proj    Q8_0     2048 x 4096      8.9 MB      30.9 us   288.5 GB/s  
  shared expert gate/up              Q8_0      512 x 2048      1.1 MB       8.9 us   126.2 GB/s  
  shared expert down                 Q8_0     2048 x 512       1.1 MB       5.0 us   222.4 GB/s  
  shared expert gate/up (Q6_K)       Q6_K      512 x 2048      0.9 MB      10.4 us    83.2 GB/s  
  expert gate (Q4_K)                 Q4_K      512 x 2048      0.6 MB       8.8 us    68.1 GB/s  
  expert down (Q5_K)                 Q5_K     2048 x 512       0.7 MB       7.5 us    96.7 GB/s  
  lm_head                            Q6_K   248320 x 2048    417.2 MB    1210.2 us   344.7 GB/s  
  routed experts, one token          Q4_K/Q5_K K=8  512x2048    15.3 MB      98.5 us   155.7 GB/s  
  routed experts, one token          Q4_K/Q6_K K=4  512x2048     8.2 MB      60.6 us   135.8 GB/s  
  routed experts, one token          Q4_K/Q5_K K=1  512x2048     1.9 MB      33.5 us    57.1 GB/s
```

A/B of the policies with the final kernels (`COLI_BLK_ROWS=8|32`, `COLI_BLK_STAGE=0|1`): forcing 32 rows costs the
small shapes (out_proj 289 → 213, Q4_K 68 → 62), forcing 8 costs the group (156 → 126); staging Q8_0 costs it
(in_proj 376 → 287), not staging the K-quants costs the group (156 → 146). The defaults are the measured optimum
per class; the two knobs stay for A/Bs.

End to end (`2026-10-17-rtx3070-gguf-tuning.md`): the chain 33.8 → 53–54 tok/s with the same heat file (55 at
256 tokens), the device wait 12.3 → 6.4 ms a token, the text unchanged — after fixing an init race in the chain
that the faster frame exposed (that report's race section).
