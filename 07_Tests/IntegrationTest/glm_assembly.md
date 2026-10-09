# Integration test — GLM-5.2 (`glm-dsa`) assembly in `colibri.c` (`src.h` ⟷ `glm_names.h` ⟷ MLA split ⟷ indexer ⟷ NextN ⟷ expert slices ⟷ registry)

Written 2026-10-09, **before** phase 6 (process gate in `../../04_Tasks/tasks.md`).
Specification: FR-9, FR-18 (MLA reconciliation), FR-20/21 (tokenizer, stop ids), FR-23/24
(NextN), FR-26 (`glm-dsa → glm`), NFR-3 (never re-quantize), NFR-5 (container path
untouched); architecture v1 §6.1, §7.1–7.5, §8.1–8.3 (`git show 5184f3d8:03_Architecture/
gguf-architecture.md`), v2 §12 row "6 — breadth", the inspection
`08_Documents/inspection-glm52-ud-q4_k_xl-2026-10-05.md`.

**Modules under test:** the GGUF arm of the GLM-5.2 engine `colibri.c` — `Cfg` from the
`glm-dsa.*` keys (`cfg_from_gguf` twin), the name table `glm_names.h`, the MLA weight
reconciliation (`attn_k_b`/`attn_v_b` → the engine's `kv_b`, policy 2 of v1 §7.3), the
DSA indexer detection per block, the NextN (MTP) set and its precision guard, the
expert slices of `ffn_*_exps` into the `ESlot` slab as raw blocks (`QT.fmt = 16 + type`,
`gq.h` kernels), the dense leading blocks, the tokenizer (`pre = glm4` → cl100k family),
the stop ids; `family_registry._gguf_config` for `glm-dsa`; `coli doctor/info/plan`.
**Neighbours exercised:** the phase-1 reader and `ggufinfo`, the phase-2 kernels (`gq.h`,
unchanged), `tools/st2gguf.py` (gains a `glm-dsa` arm with `glm_tensor_kinds.py`), the
container path of `colibri.c` (unchanged: `tools/convert_fp8_to_int4.py` output).
**Inputs:** a **torch-free tiny GLM-5.2 snapshot** (`make_tiny_glm_hf.py`, this directory:
the geometry of upstream's `tools/make_glm_oracle.py` — 5 blocks, 3 dense + 2 MoE, 8 experts
top-2, 4 heads, `q_lora` 64, `kv_lora` 32, `qk_nope` 24, `qk_rope` 8, `v_head` 32, indexer
2 × 16, vocab 256 — with seeded bf16-representable weights and an optional NextN block),
GGUFs written from it by `st2gguf.py --arch glm-dsa` (F32; all-`Q8_0`; experts `Q4_K`/`Q6_K`
with hidden 256), and the phase-1 synthetic layout `make_gguf_fixture.tiny_glm_dsa`.
**Outputs:** a `Cfg` equal to the one `config.json` implies, every dense tensor and expert
slice bit-exact against the snapshot, a `kv_b` rebuilt from the absorbed split that equals
the HF `kv_b_proj` to the bit (F32) or to one f16 ulp, refusals by name, nothing re-quantized.

## What the v1 design fixed and what this test adds

The v1 architecture (2026-10-04) designed GLM-5.2 first; v2 moved Qwen3.6 ahead because its
numbers could be measured on the owner's machine. Phases 1–5 built every piece the GLM arm
needs (reader, kernels, façade pattern, streaming slot flavour, tier formats). This test
pins the GLM-specific plumbing: the name table, the MLA split that llama.cpp's converter
stores (`attn_k_b` per head **transposed**, `attn_v_b` in the engine's orientation, no
`attn_kv_b`), the indexer on every block without an `indexer.types` key, the NextN block
counted inside `block_count`, and the engine's `ESlot` views onto raw ggml blocks.

Contract for the MLA split (FR-18, v1 §7.3 policy 2):

```
attn_k_b  ne = {qk_nope, kv_lora, n_head}  ->  per head h: K_h[kv_lora][qk_nope]
attn_v_b  ne = {kv_lora, v_head, n_head}   ->  per head h: V_h[v_head][kv_lora]
engine kv_b [H*(qk_nope+v_head)][kv_lora]: row h*(qk_nope+v_head)+j = K_h[:, j]^T  (j < qk_nope)
                                            row h*(qk_nope+v_head)+qk_nope+j = V_h[j]
```

The k part is **widened** (f16/f32) and transposed at load (lossless, NFR-3); the v part is a
view/copy of the stored blocks. `attn_kv_b`, when a converter writes it, is used directly
(policy 1). A file carrying neither pair is refused by name.

## Contracts fixed by this test

### `tools/st2gguf.py --arch glm-dsa` (phase-6 deliverable, stdlib)

```
python3 tools/st2gguf.py <hf_dir> --arch glm-dsa --out <stem>.gguf [--type f32|f16|bf16|q8_0]
       [--expert-type q8_0|q4_k|q5_k|q6_k] [--down-type …] [--tokenizer tokenizer.json] [--split N]
```

`--arch` defaults to the snapshot's `model_type` (`glm_moe_dsa` → `glm-dsa`, Qwen → `qwen35moe`).
Names per v1 §6.1 (`glm_tensor_kinds.py` classifies every HF name; an unknown name stops the
conversion). The converter writes the **absorbed split**: `attn_k_b` `{qk_nope, kv_lora, H}`
from `kv_b_proj`'s k rows transposed per head, `attn_v_b` `{kv_lora, v_head, H}` from its v
rows — exactly llama.cpp's layout, so the engine's reconciliation is what the oracle
exercises. `exp_probs_b.bias` from `e_score_correction_bias`; `expert_gating_func = 2`,
`expert_weights_scale`, `expert_weights_norm` from the config; `leading_dense_block_count`;
the indexer keys from `index_topk/index_n_heads/index_head_dim`; `nextn_predict_layers` and
the `blk.<L>.nextn.*` + `blk.<L>.*` tensors when the snapshot carries `--mtp`'s MTP block;
`tokenizer.ggml.pre = glm4`.

### `Cfg` from metadata (`colibri.c`, GGUF arm)

The table of v1 §7.1 verbatim; additionally `qk_head = qk_nope + qk_rope`, `attn_scale` as
the container path derives it, `n_group`/`topk_group` 1 unless the keys say otherwise (≠ 1
refused as today), `expert_gating_func ≠ 2` refused, `vocab = len(tokens)` cross-checked
against `output.weight`. The same `CKR` range block validates both arms.

### Startup line (FR-30 twin)

```
[GGUF] glm-dsa · <B> blocks (<D> dense, <M> MoE) · <P> part(s) · experts <tg>/<tu>/<td> <x> MB each × <E> × <M> = <y> GB
       · dense <types> <z> GB · kv_b from attn_k_b/attn_v_b (k widened f16, <w> MB) · indexer <n>/<B> blocks
       · nextn <absent|blk.L eh_proj <type> (<bpw> bpw) loaded|skipped (<why>)> · experts on CPU (gq) · sidecars <dir>/.coli-<stem>/
```

### `ESlot` on raw blocks

`expert_load_impl`'s GGUF arm: three slices (`ts_expert`) into the slab, `QT{fmt = 16 + type,
q4 = slab + pos[k], s = NULL, gs = block, O/I as today}`; `matmul_qt_ex`/`expert_gate_up`
dispatch `qt_is_ggml(fmt)` to `gq_matmul`/`gq_moe_run`-class kernels; `qt_resolve_fmt` is
never called for a GGUF slice. The leading dense blocks' `ffn_gate/up/down` are dense `QW`s
as the Qwen arm holds them (raw K-quant → `gq_matmul`, `Q8_0` → the split).

### Reporting

`coli gguf inspect` and `coli doctor` already pass on the real file (phase 1). `coli info`
prints the engine `colibri (glm)`, the sidecar path, the kv_b widening cost; `coli plan`
prices the dense set as stored (21.0 GB on `UD-Q4_K_XL`) and the experts (22.81 MB each).

## Cases

Runner: `python3 07_Tests/IntegrationTest/run_glm_assembly.py [--keep] [--no-build]`.
Exit 0 iff every case passes. The engine binary is `06_Code/c/colibri` (`make colibri`).

| # | Case | Expected |
|---|---|---|
| 0 | fixtures | `make_tiny_glm_hf.py` writes `model.safetensors` + `config.json` (+ `--mtp`: `mtp` block as `model.layers.5.*` with `eh_proj`, `enorm`, `hnorm`, `shared_head.norm`); `st2gguf --arch glm-dsa` writes `tiny_glm_f32.gguf`, `tiny_glm_q8_0.gguf`, `tiny_glm_mtp_f32.gguf`; `ggufinfo.summarize`: `glm-dsa`, engine `glm`, 5 trunk + 1 nextn for the MTP file, `attn_k_b`/`attn_v_b` present and no `attn_kv_b`, indexer tensors on every block |
| 1 | name table — `tests/test_gguf_load_glm` (no args) | `all passed`: every v1 §6.1 row maps both ways; an HF name outside the table is refused by name; the MLA reconciliation on synthetic data (random `kv_b`, split as the converter does, rebuilt by the engine's routine) is bit-exact in f32 and one ulp in f16; `idx_type[]` derived from tensor presence equals the explicit `indexer.types` array when both exist; the NextN precision predicate: `Q8_0`/`F16`/`BF16`/`F32` ok, `Q6_K`/`Q5_K`/`Q4_K` → skipped unless `MTP=1` |
| 2 | cross-check — `tests/test_gguf_load_glm tiny_glm_f32.gguf tiny_hf` | `Cfg` from the GGUF == `Cfg` from `config.json` field by field (incl. `first_dense 3`, `qk_head 32`, indexer 2/16/4096, 3 stop ids); every dense tensor the engine loads (attention, norms, routers, `exp_probs_b`, shared expert, dense-block MLP, indexer, `token_embd`, `output`) bit-exact; `kv_b` rebuilt from the split bit-exact against `kv_b_proj`; every expert slice of both MoE blocks bit-exact; the `Q8_0` file with `--tol q8`: dense within the `Q8_0` round trip (exact dequant of the same bytes on both sides) |
| 3 | engine runs, reference mode | `SNAP=tiny_glm_f32.gguf REF=ref.json COLI_TEMP=0 ./colibri 8 16 16` (arbitrary ids, judged by the printed summary as the Qwen runners do): the `[GGUF] glm-dsa …` line with `kv_b from attn_k_b/attn_v_b`, `indexer 5/5 blocks`, `nextn absent`; `[DSA] indexer active`; 16 tokens generated; the same run from the container (`convert_fp8_to_int4.py` on the same snapshot, `--ebits 8`… see note) — **ids identical GGUF vs container at cap 1, 2, 8** when the container is lossless (f32 dense, int8 experts have no f32 twin → the comparison is F32 GGUF vs F32-dense container with experts-only `Q8_0` GGUF vs int8 container: reported, not required, as `lossless_oracle.md` does) |
| 4 | MLA policy 1 | a GGUF written with `--kv-b fused` (converter option writing `attn_kv_b` instead of the split): ids identical to case 3's F32 run; the startup line says `kv_b from attn_kv_b` |
| 5 | NextN | `tiny_glm_mtp_f32.gguf`: `nextn blk.5 eh_proj F32 (32.0 bpw) loaded`, `has_mtp = 1`, `MTP=0` → `skipped (MTP=0)`; a file whose `eh_proj` is `Q4_K` (`--mtp-type q4_k`) → `skipped (4.50 bpw < 8)` unless `MTP=1`; with the MTP head active, greedy ids identical to the run without it (the draft is verified, never trusted) and `[MTP] proposed N accepted M` printed |
| 6 | refusals | a `glm-dsa` GGUF with `expert_gating_func = 1` → refused naming the key; `expert_group_count = 2` → refused; a block without both `attn_k_b` and `attn_kv_b` → refused naming the tensor; a `Q2_K` expert tensor → refused by type name (FR-9), before any weight is read |
| 7 | registry and CLI | `family_registry.resolve_model(tiny_glm_f32.gguf)`: family `glm`, config `hidden_size 256`, `num_hidden_layers 5`, `n_routed_experts 8`, `kv_lora_rank 32`, `qk_rope_head_dim 8`, `v_head_dim 32`, `first_k_dense_replace 3`, `index_topk 4096`; `coli info --model tiny_glm_f32.gguf` names `colibri (glm)` and the sidecar path; `coli plan` prints bytes per expert = the three slice sizes; `coli doctor --deep` passes every `model.gguf.*` check |
| 8 | streaming parity (FR-27 twin) | ids identical across `cap` 1, 2, 8 and `OMP_NUM_THREADS` 1 vs 4 on the F32 file; `GGUF reads:` line (as the Qwen arm prints it) accounts 3 × misses; nothing written beside the model |
| 9 | tokenizer | `tests/test_tok_gguf tiny_glm_f32.gguf <tokenizer.json> fixtures/tok_corpus.txt` with a `glm4` pre-tokenizer fixture: identical ids both constructors (the GLM corpus half of FR-20); a GGUF with an unknown `tokenizer.ggml.pre` is refused, not guessed |

**Pass criterion:** `RESULT: ok (0 failures)` locally and in the `gguf-oracle` job (the
container cases run where `convert_fp8_to_int4.py`'s dependencies exist). The torch-built
reference (ids) is `../SystemTest/glm_lossless_oracle.md`.

## Pre-implementation note

`colibri.c` is 12 000 lines with its own `QT` format table (`fmt` 0–8, no `16 + type` block
formats yet), its own expert slab (`ESlot{g,u,d,slab}`), MTP draft/verify (`mtp_draft`,
`mtp_absorb`) and the DSA indexer. The phase-3 pattern (façade in `src.h`, names in a
header, un-transforms audited) carries over; the engine's `qt_*` consumers gain a
`qt_is_ggml(fmt)` branch to `gq.h`. Nothing of the container path may change
(`tests/test_glm_oracle.py` and upstream's GLM oracle recipe stay green).

## State

| Date | Result |
|---|---|
| 2026-10-09 | document, runner, torch-free fixture `make_tiny_glm_hf.py` and the C contract `tests/test_gguf_load_glm.c` written; `st2gguf --arch glm-dsa`, `glm_names.h`, the GGUF arm of `colibri.c` and the registry/CLI arms are phase-6 deliverables. Pre-implementation run: see the row below. |
| 2026-10-09 | pre-implementation run (`run_glm_assembly.py --no-build`, `colibri` built): `RESULT: FAIL (10 failures)`, all phase-6 contracts — `st2gguf` has no `--arch glm-dsa` (cases 0 and 6, hence 3–5, 7, 8 skipped), `tests/test_gguf_load_glm` does not build (case 1). Already passing: the torch-free snapshots with and without the NextN block (140 / 182 tensors). |
