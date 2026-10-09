# Tasks — GGUF support in sylph

Single source of truth for implementation progress. Grouped by module as defined in
`../03_Architecture/gguf-architecture.md` (v2, Qwen3.6 first). Update after every
completed step. `[x]` done · `[ ]` open · `[~]` in progress.

Process gates: the owner **reviews the v2 architecture** before phase 2 starts;
integration and system tests for a phase are written under `../07_Tests/` **before**
that phase's implementation.

## Phase 0 — documents

- [x] v1 specification + architecture (GLM-5.2 target, 2026-10-04)
- [x] owner wrote `01_Requirements/` (2026-10-05)
- [x] v2 specification + architecture (Qwen3.6 target, Ollama/llama.cpp as reference, equivalence levels E0–E3)
- [x] **Owner review of the v2 architecture** (done 2026-10-05)
- [ ] Owner answers the open questions (spec §9): Ollama tag/blob, calibration approach, MTP, GLM-5.2 roadmap

## Phase 1 — reader (`gguf.h`, `ggufinfo.py`) — done

- [x] `c/gguf.h`, `c/ggufinfo.py`, `c/tools/make_gguf_fixture.py`, `coli gguf inspect`, `coli doctor` GGUF branch
- [x] unit tests (`test_gguf.c`, `test_ggufinfo.py`, doctor cases); `make check` green (Linux)
- [x] verified on GLM-5.2 `UD-Q4_K_XL` (11 parts) and on Qwen3.6-35B-A3B `UD-Q4_K_M` (unsloth) + `Q4_K_M` (bartowski)
- [x] `ENGINE_ARCHS`: `qwen35moe → qwen36`, `glm-dsa → glm`
- [x] upstream colibrì v1.12.1 merged into `06_Code/` (subtree); GGUF additions re-applied
- [x] CI green on Linux, macOS and Windows (run 7, commit c77e88b, 2026-10-05; Windows needed the split `mingw-w64-ucrt-x86_64-libgomp` package, `update: true` and `PYTHONUTF8=1` as upstream)
- [x] `make check` of the merged `06_Code` run end to end on Linux (upstream's full suite: all C suites pass, 1369 Python tests OK, 147 skipped without models/GPU)

## Phase 2 — kernels (`gq.h`)

Pre-conditions (met 2026-10-05): architecture reviewed; `07_Tests/IntegrationTest/gq_kernels.md` (+ runner, 17 golden fixtures) and `07_Tests/SystemTest/kernel_throughput.md` written.

- [x] type table, `gq_row_bytes`, `gq_supported` (F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) — `c/gq.h`, 2026-10-06
- [x] E0 oracle: gguf-py golden vectors in `07_Tests/IntegrationTest/fixtures/e0/` (real Qwen3.6 rows + synthetic edge blocks), cross-checked 16/16 bit-exact against upstream's `tools/gguf_dequant.py` — no new `gq_ref.py` needed
- [x] `gq_deq_row` reference path without FP contraction (GCC `optimize("fp-contract=off")` / `clang fp contract(off)` pragmas); ggml's expression order; E0 golden bit-exact on all 17 fixtures under gcc 13 (-march=native, x86-64-v3, scalar) and clang. Finding: the K-quant products (f16 × 6-bit × 4-bit) are exact in f32, so contraction could not have changed the values — the guard is insurance
- [x] `tests/test_gq_kernels` CLI: no-arg suite (`all passed`), `deq <TYPE> <bin> <numel>` (exit 2 `unsupported type`, exit 3 `size mismatch`), `moe-digest` (FNV-1a, thread-independent); fixtures found relative to the binary
- [x] eight `synth_*` golden pairs + manifest subset copied to `tests/fixtures/gq_e0/` (`make check` E0 subset, FR-34)
- [x] `gq_deq_row` for every type; `gq_dot_row` scalar reference; AVX2 (bit-identical, 8 lanes = element & 7); NEON (two float32x4, same lane order; verified by the macOS arm64 CI job, run 10 for commit 06211c2 — all three jobs green, Windows/MinGW included)
- [ ] AVX-512 f32 path deliberately absent (as `expert_ffn.h`: 16 lanes would change the fma order); VNNI only with the int8-activation twin (phase 5)
- [x] `gq_q8_0_split`/`gq_q8_0_join` (lossless, round-trip tested) and `matmul_q_gs(gs=32)` on the split within 1e-6 of the double dot and of `gq_dot_q8_0`
- [x] `gq_matmul` (OMP over rows; `Q6_K`/`Q8_0`/any supported type), `gq_embed_row`; unsupported types refused (-1)
- [x] `gq_moe_run` (`GqExpert{g,u,d,tg,tu,td}`, same work split and scratch discipline as `xf_moe_run`, rank-ordered sum); equals the per-token loop; digest identical under 1/2/4 threads
- [x] `gq_selftest`; `tests/test_gq_kernels.c` (auto-discovered by the Makefile's test rule scan)
- [x] `tests/bench_gq.c` + `make bench-gq` per `07_Tests/SystemTest/kernel_throughput.md`; **dev container: NFR-7 met** (`Q4_K` 1.27–1.30×, `Q5_K` 1.27–1.35×, `Q6_K` 1.04–1.15× per byte; layer 1.17–1.45× at S=1, two runs) → `08_Documents/kernels/2026-10-06-dev-container.md`
- [ ] **Owner runs** `make -C 06_Code/c bench-gq MODEL=<Qwen3.6 GGUF>` on the RTX 3070 host (CPU side) and commits the table as `08_Documents/kernels/<date>-<host>.md`
- [x] `07_Tests/IntegrationTest/run_gq_kernels.py` passes: 0 failures, all 6 cases incl. the live gguf-py and upstream-oracle checks (2026-10-06)

## Phase 3 — assembly (`src.h`, `qwen35_names.h`, `gguf_xform.h`, `ts_cfg`, tokenizer)

Pre-conditions (met 2026-10-06): `07_Tests/IntegrationTest/src_facade.md` (+ runner, torch-free fixture generator, tokenizer corpus) and `07_Tests/SystemTest/lossless_oracle.md` written.

- [x] `src.h` façade (`ts_init/ts_find/ts_read_f32/ts_read_q8_split/ts_read_raw_rows/ts_expert`); the safetensors arm delegates verbatim to `st.h` — 2026-10-06
- [x] `qwen35_names.h` (table of ARCH §6.1, both directions, per-expert slices)
- [x] `gguf_xform.h`: norm `1+w → w`, `A_log = log(−ssm_a)` (A1 resolved: the engine keeps `dn_alog`; the round trip costs ≤ 4·2⁻²⁴ absolute), v-head un-permutation for the 7 tensors on f32 rows/elements/columns and on raw block rows/column blocks, `Q8_0` split via `gq.h`; pinned by `tests/test_gguf_load` (no separate `test_gguf_xform.c`)
- [x] `cfg_from_gguf` in `qwen36.c` from `qwen35moe.*` keys + shapes (ARCH §6.2), layer kinds from the interval confirmed by tensor presence, NextN skipped; `validate_cfg` reused
- [x] `load_tokenizer_gguf` (arrays; `pre = qwen35`); `tests/test_tok_gguf`: identical ids and decodes on all 52 corpus lines against Qwen's `tokenizer.json` (GGUF vocab padded with `[PADn]` to 248320); `--llama-tokenize` comparison left for the owner
- [x] `model_init_range` / `load_t_n` / `load_tq` / `load_expert_merged` through the façade; `QW{kq,ktype,gs}` + `matmul_d` dispatch (Q8_0 split → `matmul_q_gs`, K-quants → `gq_matmul`); `Slot{kq,ktype,kbytes}` + `moe_gq_run`; xf_mode/tier guards (container path untouched)
- [x] logprob dump (`PPL_DUMP=<file>` on the `PPL=1` path; header line + `pos\ttarget\tlogprob\t<coli_logprob_tail>`)
- [x] `07_Tests/SystemTest/make_tiny_ref_logprobs.py` (torch) and `compare_logprobs.py` (stdlib); first execution in the `gguf-oracle` CI job
- [x] `tools/st2gguf.py` per the contract in `src_facade.md` (stdlib; F32/F16/BF16/Q8_0; converter transforms; placeholder or JSON tokenizer; `qwen36_tensor_kinds` as the name contract)
- [x] `family_registry.resolve_model` (config synthesized from the metadata), `resource_plan.analyze_model` (slices → expert bytes), `coli info/plan/doctor` and `require_model` on a GGUF path
- [x] `tests/test_gguf_load.c` and `tests/test_tok_gguf.c` per `src_facade.md` (both in `make check`)
- [x] `SNAP=<file.gguf>` (or a directory of parts) selects the GGUF source; startup `[GGUF] qwen35moe · N blocks · …` line
- [x] `gguf-oracle` job in `.github/workflows/check.yml` (upstream's tiny-oracle recipe + st2gguf + GGUF runs + E1 subset) — green on its first run (run 13, commit 2375ec1, 2026-10-06); ΔNLL dumps for F16/Q8_0/container added to the reported step afterwards
- [x] **System test `lossless_oracle.md`**: F32 GGUF token-exact 16/16 at cap 1/2/8 against upstream's torch-built `ref_full.json`; container run unchanged 16/16; façade cross-check 317 tensors bit-exact; E1 subset mean |ΔNLL| 6.25e-8, max 1.0e-6 (one print ulp), thresholds calibrated to 1e-6 / 1e-5 — CI run 13, 2026-10-06; summary `08_Documents/equivalence/2026-10-06-tiny-oracle-ci.md`
- [x] CI green on Linux, macOS and Windows for phase 4 (run 20: all three `make check` jobs with the new `src.h`/`qwen36.c` code and the `--encode` mode; run 21: `gguf-oracle` incl. the CI subset and the streaming test)
- [x] CI green on Linux, macOS and Windows for phase 3: run 13 was red on Windows only (`far` is an empty macro in the Windows headers; local variable in `tests/test_gguf_load.c` renamed); run 14 (commit 52fc566, 2026-10-06) green on all four jobs, Windows `make check` included; run 16 (commit e91e57f, 2026-10-08, repo public) green again with the `_xq8_0` measurement — criterion 4 of `lossless_oracle.md` holds (1.77e-4 vs the container's 2.35e-4)
- [x] `07_Tests/IntegrationTest/run_src_facade.py` passes: `RESULT: ok (0 failures)`, all 8 cases including the network case (2026-10-06)

## Phase 4 — streaming + equivalence harness

Pre-conditions (met 2026-10-08): `07_Tests/IntegrationTest/expert_streaming.md` (+ runner), `07_Tests/SystemTest/equivalence.md` (+ `fixtures/e3_prompts.txt`), `real_model.md` written. Finding while writing them: `qwen36.c` has **no** `DIRECT`/mirror/`URING`/split-dir machinery and writes no sidecar of its own (those live in `colibri.c`, `route_trace.h` etc.), so FR-28 is vacuous for v1 and FR-29 is a path rule (`sidecar_dir`), not a migration. `tools/convert_qwen36.py` needs torch; the container comparisons of the streaming test run in the `gguf-oracle` job.

- [x] `Slot.kq` + 3-slice loads; `moe()` dispatch by slot flavour; pilot/LRU/pin unchanged — landed in phase 3; pinned by `expert_streaming.md` cases 1–3 (pass on the phase-3 binary, 2026-10-08)
- [x] FR-28 knobs ignored identically on both sources (case 9 passes, 2026-10-09); finding recorded in the architecture — no O_DIRECT/mirror work in `qwen36` for v1
- [x] sidecar rule (FR-29): `ts_sidecar_dir` in `src.h`, `family_registry.sidecar_dir`, `coli info` field; nothing written beside a `.gguf` (case 7) — 2026-10-09
- [x] `tools/st2gguf.py --split N` (llama.cpp `gguf-split` layout, via `make_gguf_fixture.write_split_set`); engine loads from the directory or any part; missing part refused by name (cases 0, 4) — 2026-10-09
- [x] startup `[GGUF]` line per the FR-30 format of `expert_streaming.md`; `GGUF reads:` statistics line (slices = 3 × misses, MB, MB/token, parts touched; per turn in serve mode) (cases 5, 6); CUDA tier refusal note (FR-36, case 10) — 2026-10-09
- [x] `gq_embed_row` for `token_embd` (on demand, `ts_read_rows_any`), `COLI_GGUF_EMBED=0` A/B knob, documented in `docs/ENVIRONMENT.md`; dumps bit-identical (case 8) — 2026-10-09
- [x] harness `07_Tests/SystemTest/equivalence/` (`run_equivalence.py`, `ref_llama.py`, `ref_ollama.py`, `sylph_runner.py`, `compare.py`, `formats.md`): tokenizer gate, noise floor, E1 (ΔNLL, exact + coarsened KL, top-1/5), E2 (PPL pair, sign test), E3 (prefix to the first near-tie), deliberate-difference runs (FR-35), report `.md` + `.json`; `make -C 06_Code/c equivalence MODEL=… LLAMA=… [OLLAMA=… TAG=…]` — written 2026-10-09; validated locally sylph-vs-sylph (same file: all zeros; F16 candidate: E1 fails on KL as it should; Q8_0 candidate: E1/E2 fail); the llama.cpp/Ollama arms run first on the owner's machine (no binaries here)
- [x] engine: `PPL_DUMP_FULL=<file>` (full log-softmax per scored position, `full-logprob v1`; rows verified to sum to 1 and to match the dump at the target); `tests/test_tok_gguf <gguf> --encode <text>` — 2026-10-09
- [x] CI subset (FR-34): `run_equivalence.py --ci` in the `gguf-oracle` job — E1/E2/E3 on the tiny torch model incl. exact KL, report produced and parsed; plus `run_expert_streaming.py` in that job (container present there) — **green on run 21** (2026-10-09): E1 mean 1.8e-7 / max 1e-6, exact KL 6.2e-8, E2 PPL equal to 1.2e-8, E3 16/16; streaming test 0 failures with the container
- [ ] **Owner runs**: tokenizer equality, E1–E3 on the real GGUF (CPU), A/B vs Ollama and vs gs64 container; report in `08_Documents/equivalence/` — `07_Tests/SystemTest/equivalence.md` cases 1–8 and `real_model.md`; needs the Ollama tag/blob (spec §9 a)
- [x] harness fix found while building it: the comparer parsed the log-prob tail as `id:lp`, the engine prints ` <lp> <k> <id> <lp> …` (unordered); the top-1 check of runs 13–16 was vacuous (ΔNLL unaffected) — `compare_logprobs.py`, `make_tiny_ref_logprobs.py` and `lossless_oracle.md` corrected 2026-10-09
- [x] `docs/gguf.md` (user page), `docs/ENVIRONMENT.md` (`SNAP=<gguf>`, `COLI_GGUF_EMBED`, `PPL_DUMP`, `PPL_DUMP_FULL`), `CHANGELOG.md` — 2026-10-09

## Phase 5 — GPU (RTX 3070)

Pre-conditions (met 2026-10-09): `07_Tests/IntegrationTest/cuda_tier_kquant.md` (+ runner `run_cuda_tier_kquant.py`, C tests `tests/test_cuda_block_fmt_guard.c`, `tests/test_qwen36_tier_kq.c`, `tests/test_qwen36_tier_kq_engine.c`, device oracle `tests/test_gq_cuda.cu` + `tests/gq_ref.c`, Makefile rules `phase5-tests`, `cuda-test-gq`) and `07_Tests/SystemTest/gpu_rtx3070.md` written. Findings while writing them: (1) FR-37's "bit-for-bit" is fixed as **kernel-level identity with `gq_dot_row_ref`** plus the engine's summation order (routed experts in rank order with `fmaf`, shared expert afterwards); the device `expf` in the SiLU and the dense `Q8_0` split-vs-raw kernel are the two named, measured deviations (upstream's own "bit-identical" is a cosine of 1.0000001). (2) The tiny preset (hidden 64, inter 32) cannot hold 256-element K-quant blocks; `make_tiny_qwen36_hf.py` gained `--hidden/--inter/--layers/--experts`, the fixture for the fake-tier engine test is hidden 256 / inter 256. (3) `st2gguf.py` needs K-quant expert writers (`--expert-type q4_k|q5_k|q6_k`, `--down-type`). (4) The harness needs an E3 arm for `--deviation` (sylph vs sylph) for the CPU-vs-CUDA case.

- [x] `backend_cuda.h`: block formats `fmt = 16 + ggml type` (24/28/29/30), `coli_cuda_block_fmt_supported/_type/_elems/_bytes`; `backend_cuda.cu`: upload of raw block tensors (`sc == NULL`, size/shape refusals), `blk_*` kernels (32 rows × 8 lanes per 256-thread block, the reference's accumulators and tree) for the dense GEMV and the fused expert group (`grouped_hidden_blk_dual`, `grouped_down_blk`), dispatch in `quant_matmul_launch` and `expert_group_impl`; `weight_at` untouched — `test_cuda_block_fmt_guard` passes 2026-10-09; the device oracle `make cuda-test-gq` awaits the owner's card (no nvcc here; the `.cu` is syntax-checked as C++)
- [x] `qwen36_tier`: `qt_init_gguf` (slot footprints, refusal by type), `qt_note_kq[_planned|_block]` staging three raw slices, `qt_dense_init_kq`/`qt_lmhead_init_kq`/`qt_dnproj_init_kq` (per-handle fmt); `tier_offer_slot`/`tier_warmstart` offer `kq` slots; `moe()` adds the shared expert after `qt_take` on a GGUF and runs CPU misses through `gq_expert_cpu`; `gq_moe_run`'s tail uses `GQ_FMA` — `test_qwen36_tier_kq` passes (issue/take == `gq_moe_run` bit for bit), 2026-10-09
- [x] `qwen36.c` dense placement for GGUF: `qdw_bytes` (bytes as stored), `qdw_place` (raw K-quant upload; `Q8_0` split re-joined with `gq_q8_0_join`), `dnproj_bytes`/`dnproj_place_kq` (qkv ++ z when both halves share a type), `lmhead_place_kq`, the probe measures the matrix as it will be placed; the `[GGUF]` line is printed after the tier decision (`experts on CUDA tier (<n> planned)`); FR-36 messages — `run_cuda_tier_kquant.py` cases 3–5, 7 pass 2026-10-09 (full residency byte-identical, trunk placed and partial residency within the gate, `max |Δlp| = 0`)
- [x] fake backend compute mode (`fake_block_compute`, counters, `last_sc`, refusals as the real backend), `tests/test_qwen36_tier_kq_engine` (excluded from the bare test gate, driven by the runner); `run_cuda_tier_kquant.py` added to the `gguf-oracle` job — `RESULT: ok (0 failures)` locally and **in CI run 24** (2026-10-09; Linux, macOS, Windows `make check` green)
- [x] `tools/st2gguf.py --expert-type q4_k|q5_k|q6_k`, `--down-type` (direct min/max encoders in ggml's layouts, fixtures only); `resource_plan._analyze_gguf` `trunk_gguf_bytes` → `tiers.vram.trunk_bytes` + `trunk_label`, `VRAM … trunk + … hot tier` line — case 6 passes 2026-10-09
- [x] harness: E3 arm for `--deviation` (sylph vs sylph-deviation, in the report); `docs/qwen36-cuda-tier.md` (GGUF section), `docs/gguf.md` GPU row, `docs/ENVIRONMENT.md` (`QWEN_EXPERT_ACT` on GGUF), `CHANGELOG.md` — 2026-10-09
- [x] int8-activation twin (FR-14, §4.2): `gq_i8.h` — per-32 int8 activation blocks with f32 scales (finer than ggml's Q8_K per-256), integer dots for `Q8_0`/`Q4_K`/`Q5_K`/`Q6_K` (scalar + AVX2 `maddubs`/`madd`), `gq_moe_run_i8`; `QWEN_EXPERT_ACT=i8` opts in on the GGUF path (default f32; the container keeps upstream's default); `test_gq_kernels` pins it by tolerance and prints the deltas (row dots within 0.04–0.10 of the per-element error bound, layer 0.6–0.9 % of max |out|); speed and ΔPPL on the real model are the owner's G5 run — 2026-10-09
- [ ] **Owner runs** (`gpu_rtx3070.md`): G0 kernel oracle, G1 plan on 8 GB, G2 startup/placement, G3 E1–E3 CPU vs CUDA, G4 A/B vs Ollama on the card, G5 twin deltas, G6 container tier unchanged; report in `08_Documents/benchmarks/`

## Phase 6 — breadth

Pre-conditions (met 2026-10-09): `07_Tests/IntegrationTest/glm_assembly.md` (+ runner, `make_tiny_glm_hf.py`, `tests/test_gguf_load_glm.c`), `qwen36_mtp.md` (+ runner, `make_tiny_qwen36_hf.py --mtp`), `07_Tests/SystemTest/glm_lossless_oracle.md`, `glm_real_model.md`, `ud_survey.md` (+ runner, first report in `08_Documents/survey/`), `upstream_sync.md` (+ runner) written. Findings while writing them: (1) the GLM-5.2 engine is `colibri.c` (`glm-dsa → glm`), which already has MTP draft/verify and the DSA indexer for its container; the v1 design (commit 5184f3d8) is the GLM plan, phases 1–5 built every generic piece it needs. (2) `UD-Q4_K_XL` is 467 GB with a 21 GB `Q8_0` dense set: the real-model test is blocked on the owner's decision (spec §9 d) and on the survey's smaller quants. (3) unsloth and bartowski publish I-quant files (`IQ1_M`…`IQ4_XS`, `MXFP4_MOE`) beside the K-quants: the survey decides "more types". (4) `upstream/main` is 932 commits ahead (v2.0.0): the sync is a real merge, hence a tool with a dry run. (5) The Qwen MTP head is MAY; the test fixes the only acceptable contract (lossless under greedy) and the fallback wording if the head is not used.

- [x] GLM-5.2 (`glm-dsa`) assembly per the v1 design: `glm_names.h`, `st2gguf --arch glm-dsa` (+ `glm_tensor_kinds.py`, absorbed MLA split, `--kv-b fused|split|none`, `--mtp-type`, `--no-mtp`, `--gating-func`, `--expert-groups` for the refusal fixtures), `Cfg` from `glm-dsa.*` (shared `cfg_validate`), `glm_kv_b_from_split` (f32, lossless), `glm_idx_types`, `glm_mtp_bits_ok`, `ESlot` on raw blocks (`QT.fmt = 16 + type`, `qt_is_ggml` → `gq.h`), `[GGUF] glm-dsa` + `GGUF reads:` lines, tokenizer from the metadata (`pre = glm4`), sidecar dir for `.coli_usage`, registry/CLI arms, `tests/test_gguf_load_glm` (suite + cross-check), CI step with upstream's torch oracle — `run_glm_assembly.py`: **0 failures** (2026-10-09); `glm_lossless_oracle.md`: CI step added, first run pending (no torch in the dev container)
- [x] MTP/NextN for `qwen35moe`: names (`qwen35_names.h` NextN rows), `st2gguf` `mtp` arm (`--mtp-type`, `--no-mtp`), the engine detects and reports the head with type and bits — **decision recorded (2026-10-09, pending the owner's §9 c): the Qwen engine does not use the head** (`nextn … present, not used (MTP decoding not implemented)`; the fallback contract of `qwen36_mtp.md`) — `run_qwen36_mtp.py`: **0 failures**. The GLM engine loads and uses its NextN block behind the ≥ 8 bpw guard (upstream's verified draft)
- [x] more types: **none added** — the survey (`08_Documents/survey/2026-10-09-ud-quants.md`) shows every K-quant file at `Q4_K_M` and above runs; below Q4 needs `IQ4_XS` + `IQ3_S`/`IQ3_XXS` + `Q3_K` (13/8/9/9 files) — an owner decision, not a default (owner requirement: deliberate differences only)
- [x] upstream sync tooling `tools/upstream_sync.py` (`--check/--trial/--apply [--json] [--squash]`), procedure in `docs/gguf.md` — `run_upstream_sync.py --trial`: **0 failures** (base `ce370e87` v1.12.1, upstream `bf244291` v2.0.0, 932 behind, 15 overlap paths, 9 conflicted files in the dry merge); the apply itself is housekeeping below
- [x] `make check` green after phase 6 (2026-10-09): every C suite incl. `tests/test_gguf_load_glm`, 1405 Python tests OK (two tests updated for the new behaviour: `test_makefile_deps` lists `src.h`/`glm_names.h` as `colibri` prerequisites, `test_doctor` expects `placement.plan` to pass for a `glm-dsa` GGUF now that the registry prices it); runners `run_glm_assembly.py`, `run_qwen36_mtp.py`, `run_expert_streaming.py`, `run_cuda_tier_kquant.py`, `run_upstream_sync.py`: 0 failures each; CI run 27 (83441456): Linux + macOS `make check` green; the GLM torch oracle step produced 32/32 TF and 20/20 greedy on both the container and the F32 GGUF with identical ids, then failed on its own `diff` (banner line matched) — fixed with run 28; Windows failed on a two-argument `mkdir` (MinGW) — fixed with run 28
- [ ] **Owner decisions**: GLM-5.2 on the roadmap and on which host/quant (spec §9 d); whether the Qwen MTP head is used (§9 c — implemented as "reported, not used" until decided)

## Housekeeping

- [ ] upstream sync procedure documented and exercised once more (`git subtree pull`) — documented and dry-run (`--trial`: 9 conflicted files) 2026-10-09; the real `--apply` (v2.0.0, 932 commits, conflict resolution + the gates) is a session of its own
- [ ] decide whether the `coli` launcher keeps its name in sylph
- [ ] phase 4: `token_embd` lookup from the raw Q8_0 rows (`gq_embed_row`) instead of the f32 table the container path keeps (saves ~1.5 GB RSS on the 35B)
- [ ] consolidate the two stdlib GGUF readers (`c/ggufinfo.py` from phase 1, upstream's `c/tools/gguf_reader.py` since v1.12.1) and reuse upstream's `tools/gguf_dequant.py` in Python tooling
- [ ] Ollama blob inspection by the owner (`ollama show --modelfile`, `coli gguf inspect <blob>`)
