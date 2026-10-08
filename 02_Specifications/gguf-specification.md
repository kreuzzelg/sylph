# GGUF support — specification

Status: **v2, draft for owner review** (2026-10-05). v1 (2026-10-04) targeted GLM-5.2;
v2 follows `../01_Requirements/README.md` and makes **Qwen3.6-35B-A3B** the primary
target with **Ollama/llama.cpp as the reference implementation**. This document says
**what** the GGUF support must do; **how** is in `../03_Architecture/gguf-architecture.md`;
progress is tracked in `../04_Tasks/tasks.md`.

Programming language (fixed by the owner): **C**, in the style of colibrì (header-only
modules, static functions, no external library); **Python 3 standard library** for tooling.

## 1. Inputs from the owner (`01_Requirements`)

Written by the owner on 2026-10-05 (German); restated here as the inputs this
specification serves:

| # | Requirement | Where it is served |
|---|---|---|
| R1 | **Motivation.** Testing colibrì showed the GGUF format more efficient than the gs64 container: on Qwen3.6-35B-A3B, Ollama's `Q4_K_M` reaches perplexity **7.147** against **7.325** for colibrì gs64 int4 (+2.5%), at 23.9 GB vs 22 GB; root causes named in [JustVugg/colibri#1370](https://github.com/JustVugg/colibri/issues/1370): K-quant super-blocks with per-32 min/scale, and higher precision (`Q6_K`) on `ffn_down_exps` and `output.weight`. | §2 goals, §5.2 formats, §8 acceptance |
| R2 | **Goal.** sylph, a fork named after a hummingbird, exists to find out **whether a viable path exists** for running GGUF in the colibrì engine and, if so, **how efficient** it is. | §2, §8 |
| R3 | **Style.** Keep colibrì's programming style: C, no reference to other libraries. | NFR-1, NFR-2 |
| R4 | **Reference.** Ollama's code may serve as the reference. **Automated tests must show equivalence**; where a difference remains it must be a **deliberate improvement**. | §4 reference semantics, §5.5 harness, §8 |
| R5 | **Test model.** Qwen 3.6 35B (A3B) is preferred: the owner has the most data for it and can compare against Ollama and colibrì on an **RTX 3070 (CUDA)**. | §3 scope, §5.3 assembly, §5.6 GPU |

## 2. Goals and how they are judged

| Goal | Judged by |
|---|---|
| G1 — A GGUF of Qwen3.6-35B-A3B runs in the engine without conversion. | `coli chat/serve/web` from `Qwen3.6-35B-A3B-*Q4_K_M*.gguf`; the lossless oracle (§8) proves the plumbing before any kernel question. |
| G2 — The result is **equivalent to Ollama / llama.cpp** on the same file. | The four equivalence levels E0–E3 of §4 pass with the tolerances of §4.3; any remaining difference is opt-in and documented (R4). |
| G3 — We learn **how efficient** it is. | The A/B of §8.5 on the owner's hardware: sylph vs Ollama (same GGUF) and vs colibrì gs64 container, reporting tok/s, TTFT, bytes/token, RSS/VRAM and perplexity together. Negative results are published too. |
| G4 — Nothing upstream breaks. | Upstream's own gates (`make check`, every engine's oracle) stay green; the subtree keeps merging. |

## 3. Scope

**In scope (v1, phases 1–4; GPU in phase 5):**

- The **`qwen36` engine** (`06_Code/c/qwen36.c`) reading **`qwen35moe` GGUF files**
  (Qwen3.6-35B-A3B and architecture-identical checkpoints), single file or split set,
  with the quant types found in the public `Q4_K_M`-class files: `Q4_K`, `Q5_K`,
  `Q6_K`, `Q8_0`, `F32`, `BF16` (plus `F16`, `Q4_0` from v1).
- Config, tokenizer and stop tokens from the GGUF metadata; `config.json`,
  `qwen36_meta.json` and `tokenizer.json` **not required**.
- Routed experts streamed from the GGUF with upstream's LRU/pin/pilot machinery.
- The **equivalence harness** (§5.5): automated, reproducible on the owner's machine,
  with a tiny-model subset that runs in CI.
- `coli doctor/info/plan/gguf inspect` on GGUF models (reader done in phase 1).
- **CUDA expert tier for the RTX 3070** (8 GB): phase 5.

**Out of scope (for now):**

- GGUF → container conversion, container → GGUF export.
- I-quants, `Q2_K`/`Q3_K`, `TQ*`, `MXFP4` (refused by name; none of the inspected
  `Q4_K_M` files use them), the vision tower (`mmproj-*.gguf`), MTP/NextN beyond
  detection (present only in bartowski's file; see §9).
- Other engines: GLM-5.2 (`glm-dsa`) keeps its phase-0 design and follows once the
  Qwen3.6 path is measured; the reader and kernels are architecture-agnostic.
- Parity with Ollama's *server* features (API, templates, sampling); only the
  **model function** is compared.

## 4. Reference semantics — what "equivalent to Ollama" means

Ollama runs Qwen3.6 through ggml's kernels (its Go engine builds the graph, ggml
computes it); llama.cpp uses the same kernels and is scriptable
(`llama-perplexity`, `llama-cli`), which is also how the owner measured R1. The
reference is therefore defined as **"ggml's computation of the same GGUF"**, observed
through llama.cpp where a number must be extracted, and through Ollama's API where the
owner's deployment is the thing to match.

### 4.1 Equivalence levels

| Level | Statement | Reference tool | Tolerance (proposal, §4.3) |
|---|---|---|---|
| **E0 — dequantization** | Every weight of every supported block type, decoded by sylph, equals ggml's `dequantize_row_*` value bit for bit. | Python reference implementation of the public block layouts, cross-checked once against a llama.cpp dump | 0 ulp |
| **E1 — logits (teacher-forced)** | On the same text and GGUF, sylph's per-position log-probabilities match llama.cpp's. | `llama-perplexity --kl-divergence-base` (per-token logits file) vs sylph's logprob dump | mean \|ΔNLL\| ≤ 3× the noise floor; top-1 agreement ≥ 99.5%; mean KL ≤ 1e-3 nat |
| **E2 — perplexity** | wikitext-2-raw, 16 chunks × 512 tokens, scored exactly as `llama-perplexity` does (second half of each chunk), the protocol of #1370. | `llama-perplexity` on the same GGUF | \|ΔPPL\| ≤ 0.3% and no systematic sign across chunks (binomial test, p > 0.05) |
| **E3 — generation** | Greedy decoding from the same prompt yields the same tokens until the first near-tie. | Ollama `/api/generate` (`temperature 0`, fixed seed, `raw: true`) and `llama-cli`; token-level `logprobs`/`top_logprobs` where the server offers them | identical prefix until the first position where the reference's top-2 margin < 0.05 nat; ≥ 95% of 32 prompts reach 128 tokens without divergence |

### 4.2 Deliberate differences (R4)

A deviation from the reference is allowed only if it is **opt-in**, **named**, and
**measured** against the same levels: e.g. int8 activations (`IDOT`-class), a
different accumulation order chosen for speed, placement policies. The default
configuration passes E0–E3 as-is. Every deviation ships with its ΔPPL/ΔNLL numbers in
`08_Documents/`.

### 4.3 Tolerances are calibrated, not assumed

ggml itself is not bit-stable across thread counts and backends (accumulation order).
Before thresholds are fixed, the harness measures the **noise floor**: llama.cpp vs
llama.cpp with different thread counts, and CPU vs CUDA, on the same file and text.
Thresholds are then set at 3× that floor and recorded in `08_Documents/`; the values
above are the starting proposal.

## 5. Functional requirements

Priority: **MUST** (v1), **SHOULD**, **MAY**. Phases refer to `../04_Tasks/tasks.md`.

### 5.1 Container reading (phase 1 — done)

FR-1…FR-7 of v1 hold and are implemented (`gguf.h`, `ggufinfo.py`): GGUF v3, all KV
types, split sets across drives, bounded/overflow-checked parsing, name index,
dual-SSD mirror, `coli gguf inspect`, `coli doctor` GGUF checks. Verified on
GLM-5.2 (11 parts, 1809 tensors) and on two Qwen3.6-35B-A3B files (733/753 tensors).

### 5.2 Quantized formats

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-8 | Compute **natively** on `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F32`, `F16`, `BF16` (and `Q4_0`): dequantize-on-the-fly kernels, f32 activations by default. No re-quantization at load. Lossless **layout changes** are allowed (e.g. splitting `Q8_0` blocks into an int8 plane + f32 per-32 scales so upstream's group-scaled int8 kernels run unchanged). | MUST | 2 |
| FR-9 | Per-tensor type dispatch (the inspected files mix 5–6 types; `ffn_down_exps` alternates `Q5_K`/`Q6_K`, bartowski's MTP block is `Q8_0`+`BF16`). | MUST | 2 |
| FR-10 | Unsupported types refused at load by tensor name and type name. | MUST | 2 |
| FR-11 | Scalar reference + AVX2 + AVX-512/VNNI + NEON paths; SIMD equals scalar within the project's existing tolerances; E0 bit-exact dequantization. | MUST | 2 |
| FR-12 | A K-quant **MoE layer runner** with the shape of upstream's `xf_moe_run` (gate+up fused over the touched experts, silu, down, rank-ordered sum) so a layer costs two OpenMP regions, not 3·K GEMVs. | MUST | 2 |
| FR-13 | **lm_head and embeddings in their GGUF types** (`output.weight` is `Q6_K` in unsloth's file, `Q8_0` in bartowski's; `token_embd` `Q8_0`): a dense `Q6_K`/`Q8_0` GEMV fast enough for one call per token (248 320 rows), and a single-row dequant for the embedding lookup. | MUST | 2 |
| FR-14 | Int8-activation twins (ggml's `Q8_K`-style) behind the existing `IDOT`/`QWEN_EXPERT_ACT` knobs, measured per §4.2. | MAY | 5 |

### 5.3 Model assembly — `qwen35moe` on the `qwen36` engine

Everything in this section was verified against real files on 2026-10-05
(`../08_Documents/inspection-qwen36-gguf-2026-10-05.md`).

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-15 | Fill the engine's `Cfg` from `qwen35moe.*` keys (`embedding_length`, `block_count`, `attention.head_count`/`head_count_kv`/`key_length`/`value_length`, `expert_count`/`expert_used_count`/`expert_feed_forward_length`/`expert_shared_feed_forward_length`, `full_attention_interval`, `rope.dimension_count`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`, `ssm.conv_kernel`/`group_count`/`inner_size`/`state_size`/`time_step_rank`) with the same `validate_cfg` checks as the container path; `qwen36_meta.json` is not needed. | MUST | 3 |
| FR-16 | One **name table** HF ⇄ GGUF for `qwen35moe` (dense: `attn_norm`, `post_attention_norm`, `attn_q/k/v/output`, `attn_q_norm/k_norm`; DeltaNet: `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_a`, `ssm_norm`, `ssm_out`; MoE: `ffn_gate_inp`, `ffn_*_exps`, `ffn_*_shexp`, `ffn_gate_inp_shexp`; `token_embd`, `output`, `output_norm`). | MUST | 3 |
| FR-17 | **Value transforms** of llama.cpp's converter undone at load, losslessly, so the engine's math is unchanged: RMSNorm weights stored as **1 + w** (`attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`; **not** `ssm_norm`); `ssm_a` = **−exp(A_log)**; the **32 DeltaNet value heads reordered** (GGUF head *i* = HF head 2*i* for *i* < 16, 2(*i*−16)+1 otherwise) consistently across `ssm_a`, `ssm_dt.bias`, `ssm_alpha`/`ssm_beta` rows, the value third of `attn_qkv` rows and of `ssm_conv1d` channels, `attn_gate` rows and `ssm_out` input columns; query/key heads unchanged; `ssm_conv1d` stored `{4, 8192}` = HF `[8192, 1, 4]` row-major. | MUST | 3 |
| FR-18 | Routed expert *e* of block *L* = slices of `ffn_gate_exps`/`ffn_up_exps` (`{2048, 512, 256}`) and `ffn_down_exps` (`{512, 2048, 256}`), ≈1.9–2.0 MB per expert. | MUST | 3–4 |
| FR-19 | Shared expert (`ffn_*_shexp`) and its sigmoid gate (`ffn_gate_inp_shexp`), router `ffn_gate_inp` (F32), no router bias (absent in these files; optional if present). | MUST | 3 |
| FR-20 | Tokenizer built from `tokenizer.ggml.{tokens, merges, token_type, bos/eos/padding_token_id, add_bos_token}` with `pre = qwen35` → the engine's Qwen pre-tokenizer (incl. upstream #1654 fixes); a `tokenizer.json` next to the file may be preferred, never required. Encode/decode must equal the `tokenizer.json` path on upstream's tokenizer corpus. | MUST | 3 |
| FR-21 | Stop ids: `eos_token_id` ∪ control tokens by name (`<|im_end|>`, `<|endoftext|>`), as the family registry defines them; chat template rendered by the gateway's registry (the GGUF's `tokenizer.chat_template` is logged, not used). | MUST | 3 |
| FR-22 | Rotary: `rope.dimension_count` = 64 of 256 dims (partial factor 0.25); `rope.dimension_sections` (mrope for vision) is ignored for text, exactly as the container path does. | MUST | 3 |
| FR-23 | Layer kinds from `full_attention_interval` (every 4th block is attention) **and** confirmed by tensor presence (`attn_q` vs `attn_qkv`); mismatch refuses. | MUST | 3 |
| FR-24 | MTP/NextN (`blk.<block_count−1>.nextn.*`, bartowski files only): detected and reported; loading is MAY (the engine has no MTP for Qwen3.6 today). | MAY | 6 |
| FR-25 | Lossless **oracle gate**: `tools/make_qwen36_oracle.py`'s tiny model written as an **F32 GGUF** reproduces `ref_qwen36.json` exactly as the container does. | MUST | 3 |
| FR-26 | `coli` resolves the family from `general.architecture` (`qwen35moe` → `qwen36`, `glm-dsa` → `glm`), refuses others by name; `coli doctor/info/plan` read GGUF sources. | MUST | 3 |

### 5.4 Streaming and placement

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-27 | Expert load = three slice reads into the slot; the slot keeps the **raw K-quant blocks** (no unpack); LRU, pin, pilot prefetch, `.coli_usage`, Brain/Atlas unchanged (identity is `(layer, eid)`). | MUST | 4 |
| FR-28 | `DIRECT`, mirror, split dirs, uring where upstream's qwen36 engine has them; per-slice 4 KiB windows for O_DIRECT. *Verified 2026-10-08: the `qwen36` engine has none of them (they are `colibri.c`/`glm53.c`/`kimi_k3.c` features), so this requirement is met vacuously in v1; the knobs must be ignored identically for both sources (`07_Tests/IntegrationTest/expert_streaming.md`).* | SHOULD | 4 |
| FR-29 | Sidecars (`.coli_usage`, KV/prefix state) in `<dir>/.coli-<stem>/` for GGUF models. | MUST | 4 |
| FR-30 | Startup line names the source, type mix, bytes per expert, dense bytes, and which tensors run on which backend. | MUST | 4 |

### 5.5 Equivalence harness (R4)

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-31 | **Logprob dump** in the engine: teacher-forced run over a token file writing per-position target log-prob, top-k ids and logits (reusing the existing `PPL=1`/`tf_nll` path and the serve protocol's `logprobs` channel format). | MUST | 3 |
| FR-32 | **Reference runners** (scripts, stdlib Python + the external binaries): `llama-perplexity` for E2 and its `--kl-divergence-base` logits for E1; Ollama REST (`/api/generate`, `/api/chat`) for E3; `llama-cli` as a second E3 reference. Inputs: the same GGUF path, the same token ids (sylph's tokenizer must match the reference's — a tokenizer-equality check runs first). | MUST | 4 |
| FR-33 | **Comparer** producing one report per run: ΔNLL per chunk, KL, top-1/top-5 agreement, PPL pair, generation prefix lengths, the noise floor it was calibrated against, pass/fail per level with thresholds (§4.3). Report lands in `08_Documents/equivalence/`. | MUST | 4 |
| FR-34 | **CI subset**: E0 on synthetic blocks; E1 on the tiny model where the reference is the **transformers forward pass** of upstream's torch-built fixture (per-position log-probs; torch in CI exactly as upstream's own tiny-oracle job) — *amended 2026-10-06: the container cannot serve as the reference, its experts are int8 at best*. No llama.cpp binary in CI. Full E1–E3 on the owner's machine via one `make equivalence MODEL=… LLAMA=… OLLAMA=…` entry point. | MUST | 2–4 |
| FR-35 | Every deliberate deviation (§4.2) has a harness switch and a recorded delta. | MUST | 4+ |

### 5.6 GPU (owner's RTX 3070, 8 GB)

| ID | Requirement | Prio | Phase |
|---|---|---|---|
| FR-36 | CPU-only is the v1 path; the CUDA tier (`qwen36_tier`) refuses K-quant experts explicitly and the run continues on CPU. | MUST | 4 |
| FR-37 | CUDA kernels for `Q4_K`/`Q5_K`/`Q6_K` experts and `Q8_0`/`Q6_K` dense matrices in the tier; placement by the existing auto-placer; bit-for-bit equal to the CPU path for the same inputs (as upstream's tier contract). | SHOULD | 5 |
| FR-38 | Measured against Ollama on the same card and file (Ollama's own CPU/GPU expert split) per §8.5. | SHOULD | 5 |

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **colibrì style.** Header-only C modules, static functions, SIMD guarded per ISA with a scalar reference, Italian/English comments as upstream; no external library at build or run time (R3). Reimplementing a public block layout is fine; copying ggml code is not (a verbatim table carries the MIT attribution upstream already uses). |
| NFR-2 | **Dependency-free gates.** `make check` needs a C compiler and the Python stdlib; the equivalence harness's external binaries (llama.cpp, Ollama) are only needed for the real-model runs and are never a CI dependency. |
| NFR-3 | **Precision invariant.** Weight bytes are used as the author quantized them; only lossless layout changes and lossless value un-transforms (FR-8, FR-17) at load. |
| NFR-4 | **Untrusted input.** As phase 1: bounded, overflow-checked, refusals name the rule. |
| NFR-5 | **Zero behaviour change** for container models: upstream's engines and their oracles stay byte-identical; GGUF code is reached only when the source is a GGUF. |
| NFR-6 | **Upstream sync.** `git subtree pull` from JustVugg/colibri stays routine; GGUF code lives in new files plus narrow seams. |
| NFR-7 | **Performance** is reported, not promised: tok/s, TTFT, bytes/token, RSS/VRAM alongside PPL (G3). The one engineering target: a `Q4_K` expert GEMV on CPU within 1.5× of upstream's planar-int4 kernel per byte, before the int8-activation twin. |
| NFR-8 | **Portability.** Linux, macOS, Windows (MinGW) as upstream; little-endian only. |
| NFR-9 | **Documentation.** `docs/ENVIRONMENT.md`/`SETTINGS.md` for every knob; a user page `docs/gguf.md`; equivalence reports in `08_Documents/`. |

## 7. Compatibility matrix (target after phase 4; phase 5 in italics)

| ggml type | bpw | where it occurs (Qwen3.6 Q4_K_M files) | CPU | CUDA tier |
|---|---|---|---|---|
| `F32` | 32 | norms, routers, `ssm_*` small tensors, `ffn_gate_inp_shexp` | ✓ | – |
| `BF16` | 16 | bartowski MTP routers | ✓ (widen) | – |
| `Q8_0` | 8.5 | attention/DeltaNet projections, shared experts, `token_embd`, bartowski `output` | ✓ via int8-plane + per-32 scales | *P5* |
| `Q4_K` | 4.5 | `ffn_gate_exps`, `ffn_up_exps` | ✓ | *P5* |
| `Q5_K` | 5.5 | `ffn_down_exps` (unsloth, most layers) | ✓ | *P5* |
| `Q6_K` | 6.56 | `ffn_down_exps` (bartowski, half the layers), unsloth `output.weight` | ✓ | *P5* |
| `F16`, `Q4_0` | 16 / 4.5 | not in these files; kept from v1 | ✓ | – |
| others | – | none in these files | refused | refused |

## 8. Acceptance criteria

1. **Gates green.** `make check` on Linux/macOS/Windows with all GGUF unit tests;
   upstream's oracles unchanged.
2. **Lossless oracle (FR-25).** The tiny Qwen3.6 oracle from an F32 GGUF reproduces
   `ref_qwen36.json` exactly; the container path still does.
3. **E0** bit-exact for every supported type (CI).
4. **E1–E3 on the real model**, owner's machine, same GGUF (`unsloth` `UD-Q4_K_M` or
   the Ollama blob the owner uses — see §9): thresholds of §4.3 after calibration,
   report committed under `08_Documents/equivalence/`.
5. **A/B published** (`docs/benchmarking.md` protocol: cache state, request sizes,
   controls): sylph-GGUF vs Ollama (same file) and vs colibrì gs64 container, on CPU
   and on the RTX 3070, with tok/s, TTFT, bytes/token, RSS/VRAM, PPL. This answers R2.
6. **No regression** of the container path on that host (within noise).

## 9. Risks and open points

| Risk | Mitigation |
|---|---|
| **Which GGUF does Ollama actually run?** `ollama.com` is not reachable from the development container; the Ollama blob may be Ollama's own quantization, not unsloth's. | The owner runs `ollama show qwen3.6:35b --modelfile`, locates the blob (`~/.ollama/models/blobs/sha256-…`) and runs `coli gguf inspect <blob>`; the harness always compares sylph and the reference on **that same file**. |
| Head permutation and norm offsets differ per architecture and could change with converter versions. | FR-17 is backed by a per-tensor audit script (`08_Documents/scripts/`) that re-verifies against an HF-named container; it runs again whenever a new converter version appears. |
| `Q8_0` dense trunk = 2.5 GB resident (unsloth) vs 1.9 GB (bartowski); plus 19.6–20.4 GB of experts. On the owner's RTX 3070 (8 GB) only part of the expert set fits in VRAM. | Phase 5 placement; CPU path first; measure. |
| K-quant kernels slower than upstream's planar int4 on CPU. | NFR-7 target; fused layer runner (FR-12); int8-activation twin later, measured per §4.2. |
| Tokenizer drift (`pre = qwen35`). | FR-20 equality test against the `tokenizer.json` path and against the reference runner's token ids (FR-32). |
| Scope creep (GLM-5.2, I-quants, MTP). | §3; GLM-5.2 after the Qwen3.6 numbers exist. |

Open questions for the owner: (a) the exact Ollama tag/blob to treat as the reference;
(b) agreement on the §4.3 calibration approach; (c) whether MTP (bartowski) matters for
the comparison (Ollama does not use it either); (d) whether GLM-5.2 stays on the roadmap.

## Sources

- Owner requirements `../01_Requirements/README.md`; [JustVugg/colibri#1370](https://github.com/JustVugg/colibri/issues/1370)
- Inspections: `../08_Documents/inspection-qwen36-gguf-2026-10-05.md`, `../08_Documents/inspection-glm52-ud-q4_k_xl-2026-10-05.md`
- Upstream engine: `06_Code/c/qwen36.c`, `expert_ffn.h`, `idot.h`, `gsgemv.h`, `qwen36_tier.h`, `docs/qwen36*.md`, `docs/FORMATS.md`, `docs/benchmarking.md`
- GGUF specification and ggml block layouts (see v1 sources in the architecture document)
