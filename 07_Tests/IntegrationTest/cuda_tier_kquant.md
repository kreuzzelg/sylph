# Integration test — the CUDA expert tier on raw ggml blocks (`qwen36_tier` ⟷ `backend_cuda` ⟷ `Slot.kq` ⟷ placement ⟷ `resource_plan`)

Written 2026-10-09, **before** phase 5 (process gate in `../../04_Tasks/tasks.md`).
Specification: FR-36, FR-37, FR-38 (owner half in `../SystemTest/gpu_rtx3070.md`), FR-14,
FR-35, NFR-5, NFR-7; architecture §3 (tier component), §5.2, §8, §11 and the amendments
of 2026-10-06 and 2026-10-09.

**Modules under test:** the VRAM expert tier `qwen36_tier.c` fed with the GGUF slot
flavour (`Slot{kq, ktype[3], kbytes[3]}`), the CUDA backend's new block formats
(`backend_cuda.h/.cu`: raw `Q8_0`/`Q4_K`/`Q5_K`/`Q6_K` tensors and the GEMV kernels on
them), the engine's offer/issue/take path for GGUF experts (`tier_offer_slot`, `moe()`),
the dense-trunk placement of GGUF matrices (`qdw_bytes`/`qdw_place`, `COLI_PLACE`), the
planner's VRAM arithmetic for a GGUF (`resource_plan._analyze_gguf`, `coli plan`).
**Neighbours exercised:** `gq.h` (the scalar reference every kernel must reproduce),
the fake CUDA backend `tests/qwen36_fake_cuda.h` (extended to *compute* block formats so
the whole engine runs on it without a GPU), the PPL-dump path of phase 4, `tools/st2gguf.py`
(gains K-quant expert writers for the fixture), the int8 container tier (NFR-5: untouched).
**Inputs:** the torch-free tiny Qwen3.6 snapshot and 320-token tokenizer of
`run_expert_streaming.py`; GGUFs: experts `Q4_K` + `Q6_K` down with `Q8_0` dense,
experts `Q8_0` with `F32` dense, everything `Q8_0`; synthetic random blocks for the kernel
oracle.
**Outputs:** identical token ids and **byte-identical log-prob dumps** between the CPU path
and the tier path at full residency; kernel outputs bit-for-bit equal to `gq_dot_row_ref`;
a VRAM budget that accounts for every uploaded byte; one refusal note per unsupported
situation.

Two halves. Everything in **A** runs in `make check` and in CI without a GPU (fake backend,
CPU-only build). **B** is the kernel oracle on real silicon (`make cuda-test-gq`, nvcc), run by
the owner on the RTX 3070 and reported in `gpu_rtx3070.md`.

## What the contract is, and what it is not (FR-37 read precisely)

FR-37 says "bit-for-bit equal to the CPU path for the same inputs (as upstream's tier
contract)". Upstream's own number for that contract is a cosine of 1.0000001 (not bit
identity: the CUDA `expf` in the SiLU and the summation order of the shared expert differ
from the CPU). This test fixes the achievable, checkable statement:

1. **Every GEMV on a raw ggml block tensor is bit-for-bit `gq_dot_row_ref(type, row, x, I)`**,
   the scalar reference of phase 2: lane = element & 7, `fmaf`, the fixed 8→1 tree
   (`gq_hsum8_scalar`), the per-block scale applied where the reference applies it (`Q8_0`:
   `acc[l] = fmaf(lane[l], d, acc[l])` per 32-block; `Q4_K/Q5_K`: two 8-lane accumulators
   for low/high nibble sub-blocks plus one for the min terms with `xsub = gq_xsub32_ref`,
   combined once at the end; `Q6_K`: per-16 sub-block lane sums into `accb[k & 3]`, per
   block `acc = fmaf(accb, d, acc)`). The CPU expert path (`gq_moe_run` → `gq_dot_row_xs`)
   is bit-identical to that reference on AVX2/NEON/scalar (pinned by `gq_kernels.md`), so
   **expert GEMVs are bit-identical CPU vs GPU**.
2. **The engine sums in the same order on both paths**: routed experts in rank order `k`,
   `out[d] = fmaf(val[k], y_k[d], out[d])` (`qt_take` and `gq_moe_run`'s tail both written
   with `fmaf` so FP contraction cannot split them), **then** the shared expert. The tier
   path keeps computing the shared expert while the GPU works, but adds it after `qt_take`.
3. **Two named deviations, measured, not hidden:** (a) the device `expf` in the SiLU (≤ 2 ulp
   per element, CUDA libm vs glibc) — invisible on the fake backend (host `expf`), measured on
   the card by `gpu_rtx3070.md` G3 as sylph-CPU vs sylph-CUDA ΔNLL; (b) a dense `Q8_0`
   matrix runs on the CPU as the lossless split on `matmul_q_gs` (architecture decision A3)
   and on the GPU as raw `Q8_0` blocks through the same kernel as the experts — both within
   `1e-6` of the exact dot (`gq_kernels.md`), not bit-identical to each other. K-quant dense
   matrices (`output` as `Q6_K`) are `gq_matmul` on the CPU and therefore bit-identical.
4. Consequently the engine-level checks demand: **byte-identical dumps** when every expert
   is resident and the trunk stays on the CPU (fake backend); **the lossless gate**
   (`|Δlp| ≤ 1e-5`, ids identical) when the trunk is placed or residency is partial
   (misses are summed on the CPU before the hits — a different order, same values).

The specification's FR-37 wording is annotated with this reading (02_Specifications, note
dated 2026-10-09); the owner's acceptance remains §8 items 4–6.

## Contracts fixed by this test

### Block formats in the CUDA backend (`backend_cuda.h`, host-checkable; `backend_cuda.cu`)

```
fmt = COLI_FMT_GGML_BASE + ggml type id      COLI_FMT_GGML_BASE = 16
      Q8_0 → 24    Q4_K → 28    Q5_K → 29    Q6_K → 30
int coli_cuda_block_fmt_supported(int fmt);  /* exactly {24, 28, 29, 30} */
int coli_cuda_block_fmt_type(int fmt);       /* the ggml type id, -1 otherwise */
int coli_cuda_block_fmt_elems(int fmt);      /* 32 for Q8_0, 256 for the K-quants, 0 otherwise */
```

Plain C inline predicates next to `coli_cuda_weight_at_supported`, which stays as it is:
the block formats never route through `weight_at` (they have their own kernel branch, as
fmt 6 and 7 do), so `coli_cuda_weight_at_supported(24|28|29|30)` is **false** and the
absorb gates keep refusing them.

Upload: `coli_cuda_tensor_upload(&t, blocks, NULL, fmt, I, O, dev)` with `sc == NULL`;
the device copy is exactly `O × row_bytes` with `row_bytes = I / elems × block bytes`
(34, 144, 176, 210). Refused (returns 0, no allocation) when `I % elems != 0`, when
`sc != NULL`, or when `fmt` is not a supported block format. `coli_cuda_matmul(... fmt, S, I, O ...)`
and `coli_cuda_expert_group_issue/take` dispatch on `t->fmt`; one 256-thread block per
output row; the reduction order is a pure function of `(type, I)`, never of the launch
geometry, so S-invariance and run-to-run determinism are bit-exact.

### Tier API for GGUF experts (`qwen36_tier.h/.c`)

```
int  qt_init_gguf(int n_layers, int n_experts, int hidden, int inter, int cap, int topk,
                  const size_t slot_bytes[3], uint32_t types_present);
void qt_note_kq(int layer, int eid, const uint8_t *kq, const int ktype[3], const size_t kbytes[3]);
void qt_note_kq_planned(...same...);  void qt_note_kq_block(...same...);
```

- `slot_bytes[i]` are the largest gate/up/down slice sizes over all blocks (the slot size
  `cfg_from_gguf` already computes; `ffn_down_exps` alternates `Q5_K`/`Q6_K`), charged per
  expert as `Σ dev_alloc_footprint(slot_bytes[i])`. `types_present` is the bitmask of ggml
  type ids among the expert tensors; init prints `[qtier] GGUF expert type <name> has no
  CUDA kernel -> CPU path` and returns 0 when any bit is outside `{Q8_0, Q4_K, Q5_K, Q6_K}`
  (FR-36 refusal **by type**). `cap == n_experts` and the other `qt_init` rules hold.
- `qt_note_kq` stages a copy of `kbytes[0]+kbytes[1]+kbytes[2]` bytes (the tier never keeps
  a pointer into the slot; the LRU may evict it) and the uploader issues three uploads with
  `fmt = 16 + ktype[i]`, dims `(I=hidden, O=inter)`, `(hidden, inter)`, `(inter, hidden)`.
- `tier_offer_slot(layer, eid, s)` offers `kq` slots first: `if (s->kq) qt_note_kq(...)`;
  `tier_warmstart` does the same through `qt_note_kq_planned`. The int8/int4 branches are
  untouched (NFR-5; `test_qwen36_tier_int8_engine` and `test_qwen36_tier_int8_decode` keep
  passing unchanged).
- `qt_issue`, `qt_take`, residency, heat, LFRU, `QT_UPLOAD_SYNC`, `HEAT_FILE`, `COLI_GPUS`
  keep their meaning; `qt_stats` prints the same block.

### Dense placement for GGUF matrices (`qwen36.c`, `qwen36_tier.h`)

```
int qt_dense_init_kq(const uint8_t *blocks, int type, int I, int O, int device);   /* handle or -1 */
```

`qdw_bytes(QW)` returns the **bytes as stored** for a GGUF matrix: `gq_row_bytes(ktype, I) × O`
for a raw K-quant, `34/32 × I × O` for a `Q8_0` split (`gs == 32`); 0 for an f32-only
`QW` (nothing to offer, as today). `qdw_place` uploads a raw K-quant with
`qt_dense_init_kq`; a `Q8_0` split is re-joined with `gq_q8_0_join` into a temporary buffer
(lossless, phase 2) and uploaded as fmt 24, then the buffer is freed. The handles feed the
existing `qtd()` call sites; `output` goes the same way (a GGUF has no `lm_head.q`). The
trunk components, offer order and `COLI_PLACE` grammar are those of `docs/qwen36-cuda-tier.md`;
the probe (`[place] probe:`) runs as for containers.

### Engine messages (FR-36, FR-30)

| Situation | Line (stderr, once) |
|---|---|
| CPU-only build, `COLI_CUDA=1`, GGUF source | `[qwen36] COLI_CUDA=1 ignored: built without CUDA (make qwen36 CUDA=1)` |
| CUDA build, expert type without a kernel | `[qtier] GGUF expert type <name> has no CUDA kernel -> CPU path` |
| CUDA build, tier up | `[gpu] MoE experts -> CUDA VRAM tier` (as for containers) and the `[GGUF]` startup line ends with `experts on CUDA tier (<n> planned)` instead of `experts on CPU (gq_moe_run)` |

The phase-4 note `the VRAM expert tier does not take GGUF K-quant experts yet` disappears;
`run_expert_streaming.py` case 10 accepts either text during the transition.

### Planner (`resource_plan.py`, `coli plan`)

`_analyze_gguf` adds `trunk_gguf_bytes`: the stored bytes of `output.weight` and, per block,
`attn_qkv` + `attn_gate` (dnproj), `ssm_out` (dnout), `attn_q/k/v/output` (attnproj),
`ffn_{gate,up,down}_shexp` (shexp). `build_plan` treats it exactly as `trunk_int8_bytes`
(placed first on the first planning device if it fits, taken out of the expert budget);
`tiers.vram.trunk_bytes` carries it and the `VRAM` line reads
`<bytes> trunk + <bytes> hot tier · ~N experts · <device>` (the container keeps `int8 trunk`).

### Fake backend (`tests/qwen36_fake_cuda.h`)

`fake_block_compute = 1` makes the fake *compute*: uploads of block formats keep a copy of
the bytes; `coli_cuda_matmul` on such a tensor returns `gq_dot_row_ref` per row;
`coli_cuda_expert_group_issue` computes each expert as gate/up GEMVs (`gq_dot_row_ref`),
`h = silu(g) · u` with the host `expf`, down GEMV, and `take` returns the `[count][hidden]`
block. Counters: `fake_block_uploads[4]` (fmt 24/28/29/30), `fake_issues`, `fake_issue_rows`,
`fake_matmuls`, `last_sc` (the scale pointer of the last upload; `NULL` for a block tensor). The engine-level binary prints at exit
`[fake-cuda] uploads <n> · block fmts <n24>/<n28>/<n29>/<n30> · issues <n> · expert rows <n> · matmuls <n>`.

### Fixture (`tools/st2gguf.py`)

`--expert-type q4_k|q5_k|q6_k` writes valid K-quant expert tensors (`--down-type` optional,
default = expert type; the test uses `q6_k` down to exercise the alternating layout).
A faithful port of ggml's `quantize_row_*_ref` is preferred; a simpler encoder producing
valid blocks is acceptable for this test (E0 pins the decoder, not the encoder), as long
as `coli gguf inspect` reports the types and `test_gq_kernels deq` round-trips the blocks.

## Cases — part A (fake backend, `make check`, CI)

Runner: `python3 07_Tests/IntegrationTest/run_cuda_tier_kquant.py [--keep] [--no-build] [--cuda]`.
Exit 0 iff every case passes; `--cuda` also runs part B when nvcc and a device are present.

| # | Case | Expected |
|---|---|---|
| 0 | fixtures | snapshot, tokenizer, `tiny_q4k.gguf` (`--type q8_0 --expert-type q4_k --down-type q6_k`), `tiny_xq8_0.gguf` (experts `Q8_0`, dense F32), `tiny_q8_0.gguf`; `ggufinfo` type mix lists `Q4_K`, `Q6_K`, `Q8_0` for the first |
| 1 | format truth table — `tests/test_cuda_block_fmt_guard` | `coli_cuda_block_fmt_supported` true for exactly 24/28/29/30 over −8…64; `_type` returns 8/12/13/14 there and −1 elsewhere; `_elems` 32/256/256/256; `coli_cuda_weight_at_supported` unchanged (true 0,1,2,3,4,8; false 24,28,29,30) |
| 2 | tier on the fake, GGUF flavour — `tests/test_qwen36_tier_kq` | (a) `qt_init_gguf` with types `{Q4_K,Q6_K}` starts; with `{Q4_K,Q4_0}` returns 0 and the by-type note is printed, `qt_ready() == 0`; (b) `qt_note_kq` → after `qt_fill_wait` three uploads with fmts 28/28/30, `last_bytes` of each = the slice bytes, `sc == NULL`; (c) budget: with `CUDA_EXPERT_GB` = 5.5 × one expert's footprint, `qt_plan_fill` plans exactly 5 experts, `Σ footprint(slot_bytes)` is the charge (read `G.used`); (d) compute: `qt_issue` on K=2 resident experts + `qt_take` equals `gq_moe_run` on the same slots **bit for bit**, `out` pre-filled with the shared-expert contribution is *not* what the tier adds to (the test adds the shared part after `qt_take`, as the engine must); (e) an int8 container `qt_init` + `qt_note` still uploads fmt 1 (NFR-5) |
| 3 | engine on the fake, full residency — `tests/test_qwen36_tier_kq_engine` (the engine with the fake backend linked; same CLI as `qwen36`) | `COLI_CUDA=1 COLI_GPUS=0 QT_UPLOAD_SYNC=1 COLI_PLACE=off CUDA_EXPERT_GB=1 CAP=n_experts` on `tiny_q4k.gguf` vs plain `qwen36`: ids identical, **PPL dump bodies byte-identical**; `[fake-cuda]` line: block fmts `0/2·L·E/0/L·E` (L layers × E experts), issues = routed calls, matmuls 0; `[gpu] MoE experts -> CUDA VRAM tier` printed once; startup line says `experts on CUDA tier`; same on `tiny_xq8_0.gguf` (fmts `3·L·E/0/0/0`) |
| 4 | engine on the fake, trunk placed | as 3 with `COLI_PLACE=auto COLI_TRUNK_PROBE=0` on `tiny_q8_0.gguf`: ids identical, dumps within the lossless gate (`max |Δlp| ≤ 1e-5`); `[fake-cuda]` matmuls > 0 and block fmt 24 count = experts + placed dense matrices; `[place]` lines name `lmhead`, `dnproj` … with the GGUF byte sizes (`qdw_bytes`) |
| 5 | engine on the fake, partial residency | `CUDA_EXPERT_GB` sized for half the experts (computed from `dev_alloc_footprint` of the slot sizes, printed by the C test for the runner): ids identical, dumps within the gate; `[qtier] resident r/(L·E)` with `r` = planned count; `miss(CPU) > 0`; `GGUF reads:` line still accounts every slice |
| 6 | planner | `build_plan(tiny_q4k.gguf, gpus=[fake RTX 3070: total 8 GiB, free 7.5 GiB])`: `tiers.vram.trunk_bytes` equals the runner's own sum over the component tensors (`ggufinfo`), `budget_bytes == min(usable − trunk, expert_bytes)`, `expert_capacity == budget // typical_expert_bytes`; the rendered `VRAM` line matches `^VRAM\s+\S+ [GM]B trunk \+ .* hot tier · ~\d+ experts`; with `gpus=[]` the plan is unchanged from phase 4 |
| 7 | FR-36 messages | CPU build `qwen36` with `COLI_CUDA=1`: the `built without CUDA` note once, ids as without it; the phase-4 note is gone |
| 8 | NFR-5 regression net | `tests/test_qwen36_tier_int8_engine`, `test_qwen36_tier_int8_decode`, `test_qwen36_tier_dense`, `test_cuda_fmt_guard` pass unchanged (run by the runner) |

## Cases — part B (owner, `make cuda-test-gq`, RTX 3070)

`tests/test_gq_cuda.cu` + `tests/gq_ref.c` (the reference compiled as C and linked, the
`mxfp4_ref.c` pattern): random rows with finite f16 scales (`test_gq_kernels`'s generator,
narrow and wide), random activations in [−4, 4).

| # | Case | Expected |
|---|---|---|
| B1 | dense GEMV, each of `Q8_0 Q4_K Q5_K Q6_K`, shapes (I,O) ∈ {(2048,512), (512,2048), (2048,4096)}, S ∈ {1, 3} | every `y[s][o]` **bit-identical** to `gq_dot_row_ref`; row `s` of S=3 equals the S=1 result; two launches identical |
| B2 | expert group issue/take, K ∈ {1, 2, 8}, mixed types (gate/up `Q4_K`, down `Q6_K`; all `Q8_0`) | gate/up/down GEMV halves bit-identical (checked via B1 on the same tensors); fused output vs host reference (`gq_dot_row_ref` + host `expf`) within `1e-5 + 1e-4·|ref|` (the `gq_moe_run` layer tolerance); max observed `|Δ|` printed |
| B3 | refusals | `I = 100` for `Q8_0`, `I = 200` for `Q4_K`, `sc != NULL`, fmt 18 (`Q4_0`): upload returns 0, `coli_cuda_stats` unchanged |
| B4 | determinism under load | B1 repeated 50 × with other kernels in flight: bit-identical |

## Pass criterion

Part A: `RESULT: ok (0 failures)` locally and in CI (the `gguf-oracle` job runs the runner
after the streaming test). Part B: `cuda-test-gq: all passed` on the owner's card, recorded in
`../SystemTest/gpu_rtx3070.md`. A difference in case 3 (byte identity) with B1 green
points at the engine's summation order (contract 2) or at a type the fake computes
differently from the device — the runner prints the first differing position and the
layer kinds involved.

## State

| Date | Result |
|---|---|
| 2026-10-09 | document, runner, C/CUDA test sources and Makefile rules written; the block formats, `qt_init_gguf`/`qt_note_kq`, `qt_dense_init_kq`, the fake backend's compute mode, `st2gguf --expert-type q4_k/q5_k/q6_k` and the planner field are phase-5 deliverables. Pre-implementation run: see the row below. |
| 2026-10-09 | pre-implementation run on the phase-4 engine (`run_cuda_tier_kquant.py --no-build`): `RESULT: FAIL (8 failures)`, all of them the phase-5 contracts — `st2gguf --expert-type q4_k` (case 0; cases 3–5 therefore skipped), the three C tests do not build (cases 1–3: `coli_cuda_block_fmt_*`, `qt_init_gguf`/`fake_issue_rows`, `fake_block_compute` undefined), `tiers.vram.trunk_bytes` is 0 and the `VRAM` line has no trunk (case 6, on the Q8_0 fixture), the FR-36 note is still the phase-4 wording (case 7). Already passing and therefore the regression net: the wide snapshot (hidden 256 / inter 256), the Q8_0 fixtures, the planner's budget (`min(usable − trunk, experts)`) and expert capacity with an injected 8 GB device and the unchanged CPU plan without one, ids unchanged under `COLI_CUDA=1`, and the four NFR-5 tests (`test_qwen36_tier_int8_engine`, `_int8_decode`, `_dense`, `test_cuda_fmt_guard`). Syntax of the new C and CUDA sources checked against a stub of the phase-5 API (`gcc -fsyntax-only`, `g++ -x c++` for the `.cu`). |
| 2026-10-09 | **pass** — `RESULT: ok (0 failures)` on the phase-5 engine, all nine cases of part A. Case 3: PPL dump bodies **byte-identical** CPU vs fake tier at full residency on `tiny_q4k.gguf` (uploads 0/128/0/64 for 8 blocks × 8 experts) and `tiny_xq8_0.gguf` (192/0/0/0); case 4 (trunk placed, `COLI_DENSE_I8=1` so the `Q8_0` split exists to offer): ids identical, `max |Δlp| = 0`, dense GEMVs answered by the fake; case 5 (half the budget): 32/64 resident, 159 CPU misses, `max |Δlp| = 0`; case 6: trunk bytes 852 992 = the component tensors' stored bytes; case 7: `built without CUDA` once. Runner fix during implementation: case 4 runs both engines with the dense int8 path on (the phase-4 default `COLI_DENSE_I8=0` dequantizes `Q8_0` dense matrices to f32, leaving nothing to place). Implementation notes: the `[GGUF]` line is now printed after the tier decision; CPU misses on a GGUF take `gq_expert_cpu` (`gq_matmul` rows); the tier stages the slab's three slices and uploads them as fmt 16+type; `qt_dense_init_kq`/`qt_lmhead_init_kq`/`qt_dnproj_init_kq` keep a per-handle fmt. The int8-activation twin (`gq_i8.h`, `QWEN_EXPERT_ACT=i8`) is pinned by `test_gq_kernels` (row dots within 0.04–0.10 of the per-element bound, layer 0.6–0.9 % of max |out|); on the tiny fixture it moves the 16-token PPL from 375.82 to 357.92 (random weights: a deviation, not a quality statement). Part B (`make cuda-test-gq`) awaits the owner's card. |
| 2026-10-09 | **pass in CI** (`check` run 24, commit c45ad331): the `gguf-oracle` job runs part A after the streaming test (0 failures with the container present), and `make check` is green on Linux, macOS (arm64: `gq_i8.h` on the scalar twin, the tier tests on the fake backend) and Windows (MinGW). Process note: run 23 (the tests-first commit d518c5d5) was red on all three `make check` jobs because `TEST_RULES` auto-discovers every `tests/test_*` rule and the three new tests could not build before the implementation — the same "expected to fail until the phase lands" state phase 2 recorded, here visible in CI for 1 h 40 min; the engine-with-fake binary is now in `TEST_EXCLUDE` (driven by the runner, not a bare gate). |
