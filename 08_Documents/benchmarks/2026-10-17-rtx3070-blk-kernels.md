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
