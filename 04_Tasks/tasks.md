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
- [ ] **Owner review of the v2 architecture** (blocks phase 2)
- [ ] Owner answers the open questions (spec §9): Ollama tag/blob, calibration approach, MTP, GLM-5.2 roadmap

## Phase 1 — reader (`gguf.h`, `ggufinfo.py`) — done

- [x] `c/gguf.h`, `c/ggufinfo.py`, `c/tools/make_gguf_fixture.py`, `coli gguf inspect`, `coli doctor` GGUF branch
- [x] unit tests (`test_gguf.c`, `test_ggufinfo.py`, doctor cases); `make check` green (Linux)
- [x] verified on GLM-5.2 `UD-Q4_K_XL` (11 parts) and on Qwen3.6-35B-A3B `UD-Q4_K_M` (unsloth) + `Q4_K_M` (bartowski)
- [x] `ENGINE_ARCHS`: `qwen35moe → qwen36`, `glm-dsa → glm`
- [x] upstream colibrì v1.12.1 merged into `06_Code/` (subtree); GGUF additions re-applied
- [ ] CI green on macOS and Windows (first Actions run)
- [ ] `make check` of the merged `06_Code` run end to end on Linux (upstream's full suite)

## Phase 2 — kernels (`gq.h`)

Pre-conditions: architecture reviewed; `07_Tests/IntegrationTest/gq_kernels.md` written.

- [ ] type table, `gq_row_size`, `gq_supported` (F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)
- [ ] `tools/gq_ref.py` pure-Python reference dequantizer (E0 oracle), cross-checked once against a llama.cpp dequant dump
- [ ] `gq_deq_row_T` for every type; `gq_dot_row_T` scalar; AVX2; AVX-512/VNNI; NEON
- [ ] `gq_q8_0_split` (Q8_0 → int8 plane + f32/32 scales) and its test against `gsgemv.h`'s `matmul_q_gs`
- [ ] `gq_matmul` (dense; `Q6_K`/`Q8_0` lm_head), `gq_embed_row`
- [ ] `gq_moe_run` (K-quant twin of `xf_moe_run`), fused gate+up, rank-ordered reduction
- [ ] `gq_selftest` at startup; `tests/test_gq_kernels.c`
- [ ] NFR-7 measurement: `Q4_K` expert GEMV vs planar int4 per byte; `Q6_K` lm_head vs int8 lm_head

## Phase 3 — assembly (`src.h`, `qwen35_names.h`, `gguf_xform.h`, `ts_cfg`, tokenizer)

Pre-conditions: `07_Tests/IntegrationTest/src_facade.md`, `07_Tests/SystemTest/lossless_oracle.md` written.

- [ ] `src.h` façade; safetensors arm delegates verbatim to `st.h`
- [ ] `qwen35_names.h` (table of ARCH §6.1)
- [ ] `gguf_xform.h`: norm `1+w → w`, `ssm_a` handling (A1), v-head un-permutation for the 7 affected tensors, `Q8_0` split; `tests/test_gguf_xform.c` seeded from the audit
- [ ] `ts_cfg` from `qwen35moe.*` keys + shapes (ARCH §6.2); `validate_cfg` reused
- [ ] tokenizer array constructor (`pre = qwen35`), equality test vs `tokenizer.json` path and vs `llama-tokenize`
- [ ] `model_init_range` / `load_tq` / `load_expert_merged` through the façade (container path byte-identical)
- [ ] logprob dump (`PPL_DUMP=<file>` on the `PPL=1` path, serve logprob-tail format)
- [ ] `tools/st2gguf.py` (tiny oracle → F32/F16/Q8_0 `qwen35moe` GGUF, applying the converter transforms)
- [ ] `coli`/`family_registry`: GGUF source → family; `resource_plan` byte accounting
- [ ] `tests/test_gguf_load.c`, `tests/test_tok_gguf.c`
- [ ] **System test: tiny Qwen3.6 oracle from an F32 GGUF reproduces `ref_qwen36.json`**; container oracle unchanged

## Phase 4 — streaming + equivalence harness

Pre-conditions: `07_Tests/IntegrationTest/expert_streaming.md`, `07_Tests/SystemTest/equivalence.md`, `real_model.md` written.

- [ ] `Slot.kq` + 3-slice loads; `moe()` dispatch by slot flavour; pilot/LRU/pin unchanged
- [ ] per-slice O_DIRECT, mirror, split dirs; sidecar dir `<dir>/.coli-<stem>/`
- [ ] startup `[GGUF]` line, stats fields, CUDA tier refusal note (FR-36)
- [ ] harness: reference runners (`llama-perplexity`, `llama-cli`, Ollama API), comparer, noise-floor calibration, `make equivalence`
- [ ] CI subset: E0 synthetic; E1/E2 on the tiny oracle vs the container path
- [ ] **Owner runs**: tokenizer equality, E1–E3 on the real GGUF (CPU), A/B vs Ollama and vs gs64 container; report in `08_Documents/equivalence/`
- [ ] `docs/gguf.md`, ENVIRONMENT/SETTINGS/CHANGELOG

## Phase 5 — GPU (RTX 3070)

- [ ] tier: K-quant expert uploads + CUDA `Q4_K`/`Q5_K`/`Q6_K` GEMV; `Q8_0`-group dense upload
- [ ] placement on 8 GB: dense set + expert budget; measured on the card
- [ ] E1–E3 CPU vs CUDA; A/B vs Ollama on the same card
- [ ] int8-activation twins (Q8_K-style) behind `IDOT`/`QWEN_EXPERT_ACT`, deltas recorded (spec §4.2)

## Phase 6 — breadth

- [ ] GLM-5.2 (`glm-dsa`) assembly per the v1 design (MLA split, indexer, NextN)
- [ ] MTP/NextN for `qwen35moe` (bartowski files)
- [ ] more types if a reference file needs them; `UD-*` dynamic quants survey

## Housekeeping

- [ ] upstream sync procedure documented and exercised once more (`git subtree pull`)
- [ ] decide whether the `coli` launcher keeps its name in sylph
- [ ] Ollama blob inspection by the owner (`ollama show --modelfile`, `coli gguf inspect <blob>`)
