# System test — kernel throughput on CPU (phase 2, NFR-7 / G3)

Written 2026-10-05, before phase 2. Specification: NFR-7 ("a `Q4_K` expert GEMV on
CPU within 1.5× of upstream's planar-int4 kernel per byte, before the int8-activation
twin"; performance is **reported, not promised**), G3 (negative results are published
too). Architecture §5.2 (`Q6_K` lm_head measured against upstream's int8 lm_head).

This test does not decide correctness (that is `../IntegrationTest/gq_kernels.md`);
it produces the first efficiency numbers R2 asks for, at kernel level, before any
real-model run exists.

## Inputs

- Synthetic weights at **Qwen3.6-35B-A3B shapes**: `H = 2048`, `F = 512`, `K = 8`,
  routed experts gate/up `Q4_K [512×2048]`, down `Q5_K` or `Q6_K [2048×512]`; dense
  `Q8_0 [8192×2048]` (`attn_qkv`); lm_head `[248320×2048]` as `Q6_K`, as `Q8_0`-split
  and as upstream int8-row. Random finite blocks (seeded), so no model download.
- Optional `MODEL=<gguf>`: expert 0 of block 0 read from the real file, to confirm the
  synthetic numbers on real bytes (same shapes, same types).
- Host description: CPU model, cores/threads used, ISA path printed by the binary,
  compiler and `-march`, memory bandwidth if known.

## Command

```sh
make -C 06_Code/c bench-gq [MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf]
# builds tests/bench_gq.c; OMP_NUM_THREADS defaults to the physical core count
```

Protocol per row: 5 warm-up iterations, 50 timed, **median**; weights fully resident
("hot") and, for the expert rows, a cold variant that touches a fresh copy each
iteration (first-touch, the streaming case). Output is one Markdown table.

## Rows and baselines

| Row | Kernel | Baseline | Reported |
|---|---|---|---|
| A | `gq_dot_row_Q4_K` over 512 rows × 2048 | `xf_dot_f32` planar int4 gs64 (`expert_ffn.h`) | µs/row, bytes/row, **GB/s**, ratio of ns per weight byte vs baseline |
| B | `gq_dot_row_Q5_K`, `gq_dot_row_Q6_K` over 2048 rows × 512 | same planar kernel | same |
| C | `matmul_q_gs(gs=32)` on the `Q8_0` split, 8192 × 2048 | `matmul_q` int8-row (`load_tq` layout) | µs/call, GB/s |
| D | lm_head 248 320 × 2048: `gq_matmul Q6_K` and `Q8_0`-split | upstream int8-row lm_head | **ms/token** (upstream's reference point: 12.6 → 10.2 ms/token on their host) |
| E | `gq_moe_run` (gate/up `Q4_K`, down `Q5_K`) at `S=1` and `S=32`, 16 resident experts | `xf_moe_run` planar int4 | ms/layer hot and cold, ratio |
| F | `gq_embed_row Q8_0` single row | f32 row copy | ns/row |

## Pass criteria

1. The table is complete (rows A–F, hot and cold where defined) on **two hosts**: the
   development container and the owner's RTX 3070 host (CPU side), and committed as
   `08_Documents/kernels/<date>-<host>.md` with the host description and the exact
   commit of `06_Code`.
2. **NFR-7 target:** row A ratio ≤ 1.5 (ns per weight byte of `Q4_K` vs planar int4),
   hot. Row E at `S=1` hot: `gq_moe_run` ≤ 1.5× `xf_moe_run`.
3. Rows B–D, F: reported against their baselines; no threshold in v1 (the spec names
   the `Q6_K` lm_head as a reference point only).
4. A miss of criterion 2 does **not** fail the build. It is recorded as a finding in
   the report and becomes a task (`04_Tasks/tasks.md`: kernel optimisation or the
   opt-in int8-activation twin of spec §4.2, itself measured at E1–E3 before it may
   become a default).

## State

| Date | Result |
|---|---|
| 2026-10-05 | protocol written; `tests/bench_gq.c` and `make bench-gq` are phase-2 deliverables; to run after `gq_kernels.md` passes |
