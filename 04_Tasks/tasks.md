# Tasks — GGUF support in sylph

Single source of truth for implementation progress. Grouped by module as
defined in `../03_Architecture/gguf-architecture.md`. Update after every
completed step. Status: `[x]` done · `[ ]` open · `[~]` in progress.

Rules from the project process: the architecture must be **reviewed by the
owner** before phase 2 starts; integration tests (one per module) and system
tests are written **before** implementing each phase, under `../07_Tests/`.

## Phase 0 — documents

- [x] Specification (`02_Specifications/gguf-specification.md`)
- [x] Architecture (`03_Architecture/gguf-architecture.md`)
- [ ] **Owner review of the architecture** (blocks phase 2)
- [ ] Owner writes `01_Requirements/` (the four decisions + goal)

## Phase 1 — reader (module `gguf.h`, module `ggufinfo.py`) — done 2026-10-05

- [x] `c/gguf.h`: GGUF v3 header, 13 KV types, nested arrays, tensor table, alignment
- [x] bounds / overflow validation per rule (ARCH §4.3), error-returning API + or-die wrapper
- [x] split sets (`-NNNNN-of-NNNNN`), extra search dirs, `split.*` consistency
- [x] FNV name index, duplicate detection
- [x] dual-SSD mirror acceptance (size + metadata region identical)
- [x] `gguf_describe`, `gguf_dump` (machine-readable, used by the cross-check)
- [x] `c/ggufinfo.py` stdlib reader with the same rules; `summarize()`; CLI
- [x] `c/tools/make_gguf_fixture.py` writer (demo, split sets, tiny `glm-dsa` layout)
- [x] `coli gguf inspect`, `coli doctor` GGUF branch (+ `--deep`), `SETTINGS.md`, CHANGELOG
- [x] unit tests: `tests/test_gguf.c`, `tests/test_ggufinfo.py`, GGUF cases in `tests/test_doctor.py`
- [x] `make check` green (Linux); CI on macOS/Windows pending the first push
- [x] verified on `unsloth/GLM-5.2-GGUF` `UD-Q4_K_XL` (report in `08_Documents/`)
- [x] NextN block convention (`block_count` includes MTP blocks) fixed in `ggufinfo`
- [ ] CI green on macOS and Windows (first run after the repository is pushed)

## Phase 2 — kernels (module `gq.h`)

Pre-conditions: architecture reviewed; `07_Tests/IntegrationTest/gq_kernels.md` written.

- [ ] type table + `gq_row_size` / `gq_supported` (F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)
- [ ] reference dequant per type (`gq_deq_row_T`) + Python reference `tools/gq_ref.py`
- [ ] scalar dot / axpy per type
- [ ] AVX2 paths; NEON paths; AVX-512 optional
- [ ] `gq_matmul`, `gq_matmul_pair` (fused gate+up), `gq_selftest` at startup
- [ ] `QT` fmt extension (`32 + type`), `qt_bytes`, `matmul_qt_ex` / `expert_gate_up` / `qt_addrow` dispatch (dead until phase 3)
- [ ] `tests/test_gq_kernels.c` (bit-exact vs reference, SIMD parity, pair == two matmuls)
- [ ] K-quant fixtures from `llama-quantize` on the F16 tiny GGUF, with provenance

## Phase 3 — assembly (modules `src.h`, `glm_names.h`, `ts_cfg`, `ts_tok`)

Pre-conditions: `07_Tests/IntegrationTest/src_facade.md`, `07_Tests/SystemTest/lossless_oracle.md` written.

- [ ] `src.h` façade; safetensors arm delegates verbatim to `st.h`
- [ ] `glm_names.h` HF⇄GGUF table (verified names: see inspection report)
- [ ] `ts_cfg` from KV (`n_layers = block_count − nextn_predict_layers`, `qk_nope = key_length_mla − rope.dimension_count`, …) + shared `cfg_validate`
- [ ] `ts_tok`: `tok_load_from_arrays` factored out of `tok_load`; `pre == glm4` → cl100k
- [ ] stop ids: eos/eot/eom keys ∪ by-name control tokens
- [ ] MLA reconciliation: `attn_v_b` as view, `attn_k_b` widened to f16 and transposed per head
- [ ] `model_init` through the façade; MTP (`blk.<n_layers>.nextn.*`) + precision guard; indexer (all layers when `types` absent)
- [ ] `coli` / `doctor` / `resource_plan` / gateway source detection (`model_arch` from `general.architecture`, refuse others)
- [ ] `tools/st2gguf.py` (HF snapshot → `glm-dsa` GGUF at F32/F16/Q8_0/Q4_0, stdlib + optional numpy)
- [ ] `tests/test_gguf_load.c`, `tests/test_tok_gguf.c`
- [ ] **System test: 32/32 TF + 20/20 greedy from an F16 GGUF of `glm_tiny`**; safetensors oracle unchanged

## Phase 4 — streaming

Pre-conditions: `07_Tests/IntegrationTest/expert_streaming.md`, `07_Tests/SystemTest/real_model.md` written.

- [ ] `ts_expert` → 3 slices; `expert_load_impl` per-slice reads, slab slack `+3·8192` for GGUF
- [ ] per-slice O_DIRECT windows; mmap views; uring per slice; prefetch/pilot per part
- [ ] mirror / split dirs for GGUF; sidecar dir `<dir>/.coli-<stem>/`
- [ ] `[GGUF]` startup line, `parts/expert` in the stats line, CPU-fallback notes for CUDA/Metal
- [ ] `docs/gguf.md` user page, ENVIRONMENT/SETTINGS/CHANGELOG
- [ ] real GLM-5.2 GGUF on Linux + macOS; A/B vs gs64 int4 container published (bytes/token, tok/s, TTFT, hit rate, RSS, quality probe)

## Phase 5 — GPU and speed

- [ ] CUDA `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` GEMV (+ HIP via compat header)
- [ ] int8-activation K-quant path behind `IDOT`, quality-measured
- [ ] transposed `attn_k_b` kernel (drop the widening)
- [ ] Metal `moe_gemv` for `Q4_K` if measured worthwhile
- [ ] `Q8_0` dense tensors resident on GPU for ≤25 GB hosts (dense set is 21 GB in `UD-Q4_K_XL`)

## Phase 6 — breadth

- [ ] `Q2_K`, `Q3_K`, `IQ4_NL`/`IQ4_XS`, `MXFP4`
- [ ] optional on-disk expert index (1 read per expert)
- [ ] inspect `UD-Q4_K_M` / `UD-Q4_K_S` dense-set sizes for small hosts

## Discovered / housekeeping

- [ ] upstream sync procedure tested once (`git subtree pull --prefix=06_Code upstream main`)
- [ ] decide whether the `coli` launcher keeps its name in sylph
