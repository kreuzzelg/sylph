# Integration test — tensor-source façade (`src.h` ⟷ `st.h` / `gguf.h` / `gq.h` ⟷ `qwen36.c` load path ⟷ `coli`)

Written 2026-10-06, **before** phase 3 (process gate in `../../04_Tasks/tasks.md`).
Specification: FR-9/FR-10, FR-15…FR-26, NFR-3, NFR-5; architecture §3, §6, §7, §8, §9
and the 2026-10-05 amendment.

**Modules under test:** `06_Code/c/src.h` (façade), `qwen35_names.h` (name table),
`gguf_xform.h` (load-time un-transforms), `ts_cfg` (config from `qwen35moe.*`
metadata), the GGUF tokenizer constructor in `qwen36.c`, `tools/st2gguf.py`, and the
GGUF arms of `family_registry.py` / `resource_plan.py`.
**Neighbours exercised:** `st.h` (the container arm must delegate verbatim),
`gguf.h` + `gq.h` (phases 1–2), the HF-named requests `model_init_range` issues,
`tools/qwen36_tensor_kinds.py` (the HF name contract), `coli doctor/plan`.
**Inputs:** a tiny Qwen3.6-shaped HF snapshot written **without torch** by
`make_tiny_qwen36_hf.py` (same geometry and names as upstream's
`tools/make_qwen36_tiny.py` default preset), GGUF files derived from it by
`tools/st2gguf.py`, deliberately broken derivatives, the real Qwen3.6 GGUF metadata
and Qwen's `tokenizer.json` (network, optional), `fixtures/tok_corpus.txt`.
**Outputs:** identical `Cfg` and identical tensor bytes from both sources, refusals
that name tensor, type and rule, identical token ids from both tokenizer
constructors.

## Why two generators

`make_tiny_qwen36_hf.py` (here, stdlib) produces the *same shapes and names*
transformers would save, with seeded random weights, so the façade can be tested in
`make check` and in this runner without torch: both sources are read by the same
engine code and must yield the same numbers. It has **no reference output**; the
mathematical gate (token-exact against transformers) is
`../SystemTest/lossless_oracle.md`, which uses upstream's torch-built fixture in CI.

## Contracts fixed by this test

### `tools/st2gguf.py` (phase-3 deliverable, stdlib)

```
python3 tools/st2gguf.py <hf_dir> --out <file.gguf> [--type f32|f16|bf16|q8_0] [--expert-type f32|f16|bf16|q8_0]
                         [--tokenizer tokenizer.json] [--name NAME] [--dry-run]
```

- Reads `config.json` (flat or `text_config`) and the safetensors shards (F32/F16/BF16)
  with the standard library; classifies **every** tensor name with
  `tools/qwen36_tensor_kinds.py` and stops on an unknown one (`mtp.*`/`visual.*` are
  skipped and counted, as the converter does).
- Writes a GGUF v3 with `general.architecture = qwen35moe` and the KV set of the real
  files: `general.{name,file_type,quantization_version}`,
  `qwen35moe.{block_count, context_length, embedding_length, feed_forward_length,
  attention.head_count, attention.head_count_kv, attention.key_length,
  attention.value_length, attention.layer_norm_rms_epsilon, rope.freq_base,
  rope.dimension_count, expert_count, expert_used_count, expert_feed_forward_length,
  expert_shared_feed_forward_length, full_attention_interval, ssm.conv_kernel,
  ssm.state_size, ssm.group_count, ssm.time_step_rank, ssm.inner_size}`;
  `rope.dimension_sections` only when the config has `mrope_section`.
- Tensor names per the table in the architecture §6.1 (`qwen35_names.h` is the C
  copy); routed experts stacked into the 3-D `ffn_{gate,up,down}_exps` tensors
  (`{ne0, ne1, n_expert}`, expert-major), `output.weight` always written (the engine
  never ties).
- **Applies llama.cpp's converter transforms** (architecture §7): `attn_norm`,
  `post_attention_norm`, `attn_q_norm`, `attn_k_norm` stored as `1 + w`; `ssm_a =
  −exp(A_log)`; the value-head permutation `h(j) = (j mod vk)·(vh/vk) + j div vk`
  (GGUF head `j` holds HF head `h(j)`; for vh = 32, vk = 16 this is the audited
  `[0,2,…,30,1,3,…,31]`) applied to `ssm_a`, `ssm_dt.bias`, `ssm_alpha`/`ssm_beta`
  rows, the value third of `attn_qkv` rows and of `ssm_conv1d` channels, `attn_gate`
  rows and `ssm_out` input columns; `ssm_conv1d` written `{conv_k, conv_dim}`;
  `ssm_norm` plain `w`.
- Types: `--type` for the dense matrices, `--expert-type` (default `--type`) for the
  three expert tensors; norms, routers, `ssm_*` vectors, `ffn_gate_inp_shexp` and
  `conv1d` always F32 (as llama.cpp writes them). `q8_0` is ggml's
  `quantize_row_q8_0_ref` (`d = amax/127`, `q = round(x/d)`) in pure Python: fit for
  the tiny fixture, not for a 35B model.
- Tokenizer: with `--tokenizer`, `tokenizer.ggml.{model=gpt2, pre=qwen35, tokens,
  token_type, merges, bos_token_id, eos_token_id, padding_token_id, add_bos_token=false}`
  from the JSON; without it, placeholder tokens `t<i>` for `vocab_size` ids, all of type
  normal, no merges, bos/eos/pad from `config.json` (enough for reference-id mode, which
  never encodes text).
- Prints one summary line per type (tensors, bytes) and the output path; `--dry-run`
  prints and writes nothing.

### `tests/test_gguf_load` (phase-3 deliverable)

| Invocation | Behaviour |
|---|---|
| *(no arguments)* | Self-contained suite: the HF ⇄ GGUF name table covers exactly the 24 layer kinds and 3 globals of `qwen36_tensor_kinds.py` (hard-coded copy, both directions); `gguf_xform`: norm offset exact, `A_log → −exp → log(−·)` within 2 f32 ulp, value-head permutation on audit-shaped synthetic tensors (vh 32, vk 16, vdim 128, hidden 2048) forward (as `st2gguf.py` does) then undone bit-exact for all seven tensors, `Q8_0` split lossless; refusal paths (unsupported type name, unknown HF name). Prints `all passed`, exit 0. |
| `<model.gguf> <hf_dir> [--tol f16]` | Cross-check through the façade: `ts_cfg(gguf)` equals the `Cfg` the test derives from `hf_dir/config.json` in **every** field of architecture §6.2 (`hidden, n_layers, vocab, q_heads, kv_heads, head_dim, k/v_head_dim, q_head_dim, o_in, rotary_dim, partial_rotary_factor, theta, eps, n_experts, topk, inter, shared_inter, norm_topk, n_group, topk_group, is_attn[], dn_vheads, dn_kheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim, attn_output_gate, has_qk_norm`) and `validate_cfg` passes; then for every HF-named tensor `model_init_range` loads (embed, final norm, lm_head; per block: norms, router, shared expert ×3 + its gate, attention `q/k/v/o` + `q_norm/k_norm` or the nine DeltaNet tensors) `ts_read_f32(gguf)` equals `st_read_f32(hf)` **bit for bit** (F32 GGUF) or within one f16 ulp per element (`--tol f16`), except `linear_attn.A_log` (within 4·2⁻²⁴ absolute: the exp/log round trip's error is about ulp(a)/|a|, independent of A_log's own magnitude); for every expert `ts_expert` returns the three slices whose exact dequantization (`gq_deq_row`) equals the HF rows under the same rule. Prints a per-kind count and `all passed`, exit 0. On a refusal: exit non-zero, the message names the tensor and the rule. |

### `tests/test_tok_gguf` (phase-3 deliverable)

`tests/test_tok_gguf <model.gguf> <tokenizer.json> [corpus.txt] [--llama-tokenize <bin>]`
builds the engine tokenizer twice, from `tokenizer.json` (`load_tokenizer`, unchanged)
and from the GGUF arrays (new constructor; `pre = qwen35` selects the engine's Qwen
pre-tokenizer); the GGUF's vocab must be the JSON's, padded at most with `[PADn]` placeholders to the embedding rows (llama.cpp writes 248320 ids for Qwen3.6's 248070); every line of the corpus
(default: the built-in strings of `test_qwen36_tokenizer.c`) must encode to identical
ids and decode to identical bytes. With `--llama-tokenize`, the ids are also compared
with `llama-tokenize --ids` (owner's machine; FR-32's tokenizer-equality gate). Prints
`all passed`, exit 0.

### Python side

- `family_registry.resolve_model(<file.gguf> | <dir with one .gguf> | <split part>)`
  returns the `qwen36` family with `family_config` built from the metadata
  (`model_type qwen3_5_moe`, `hidden_size`, `num_hidden_layers`, `num_experts`,
  `num_experts_per_tok`, `moe_intermediate_size`, `vocab_size`, `layer_types`, …);
  `general.architecture` outside `ENGINE_ARCHS` → `UnknownFamilyError` naming it.
- `resource_plan.analyze_model(<gguf>)` reports expert bytes = the three slice sizes
  × experts × blocks and dense bytes = everything else, equal to
  `ggufinfo.summarize`; `coli doctor --model <gguf>` (phase 1) and `coli plan --model
  <gguf>` run.

## Cases

| # | Case | Expected |
|---|---|---|
| 0 | `make_tiny_qwen36_hf.py` → `tiny_hf/` | 317 tensors in one F32 safetensors file (2.55 MB) (header parses, offsets contiguous), `config.json` with the geometry; deterministic for a seed (sha256 of the shard stable). |
| 1 | `st2gguf.py tiny_hf --type f32` (and `--type f16`, `--type q8_0 --expert-type q8_0`) | three files; `ggufinfo.summarize`: engine `qwen36`, 8 trunk blocks, 2 attention, expert bytes per the slice sizes, KV set complete; `tests/test_gguf <file>` (phase-1 reader) accepts each; the F32 file's tensor count = 7 globals/norms + per-block tensors as the real file's layout (`blk.N.*` names only from the §6.1 table). |
| 2 | **Façade cross-check, F32** | `tests/test_gguf_load tiny_f32.gguf tiny_hf` → `all passed`: `Cfg` equal in every field, every dense tensor bit-exact after the un-transforms, every expert slice bit-exact (NFR-3 for the lossless case). |
| 3 | Façade cross-check, F16 | `tests/test_gguf_load tiny_f16.gguf tiny_hf --tol f16` → `all passed` (one f16 ulp). |
| 4 | Self-contained suite | `tests/test_gguf_load` → `all passed` (name table, transforms incl. the audited permutation, refusals). |
| 5 | **Tokenizer equality** (network) | real unsloth GGUF metadata (header by range request, sparse-extended) + Qwen's `tokenizer.json`: `tests/test_tok_gguf` on `fixtures/tok_corpus.txt` → identical ids and decodes for all 50+ lines (ASCII, whitespace runs, 12 scripts, emoji with ZWJ, code, JSON, HTML, URLs, chat specials, zero-width and combining characters); skips offline. |
| 6 | Registry and plan | `resolve_model(tiny_f32.gguf).id == "qwen36"`, `family_config` fields as above; `analyze_model` expert/dense bytes equal `ggufinfo.summarize`; `coli doctor --model tiny_f32.gguf --gpu none` and `coli plan --model tiny_f32.gguf` exit 0; a GGUF with `general.architecture = llama` → `UnknownFamilyError` mentioning `llama`. |
| 7 | **Refusals** (derived from `tiny_f32.gguf` with the phase-1 writer) | (a) `blk.0.ffn_gate_exps.weight` retyped `Q4_1` → exit ≠ 0, message has `unsupported`, `Q4_1` and the tensor name (FR-10); (b) `blk.3.attn_q.weight` removed → `missing` + name; (c) `full_attention_interval = 2` while block 1 carries `attn_qkv` → `layer kind` (FR-23); (d) bartowski-style extra block `blk.8.nextn.*` with `nextn_predict_layers = 1` and `block_count = 9` → loads with 8 trunk blocks and reports the skipped NextN block (FR-24). |
| 8 | Container arm untouched (NFR-5) | with the façade compiled in, upstream's own gates are unchanged: `make check` green, and the tiny-oracle CI job's **container** run (see `lossless_oracle.md`) prints the same `Matching tokens` as before the change. |

**Pass criterion:** cases 0–4, 6, 7, 8 pass; case 5 passes or skips (offline).
Bit-exactness in cases 2 and 4 is the point: a tolerance anywhere but the f16 and
`A_log` exceptions would hide a wrong un-transform.

## Runner

```sh
python3 07_Tests/IntegrationTest/run_src_facade.py              # all cases; builds the binaries if missing
python3 07_Tests/IntegrationTest/run_src_facade.py --no-net     # skip case 5
python3 07_Tests/IntegrationTest/run_src_facade.py --keep       # leave tiny_hf/ and the GGUFs in the scratch dir
```

Standard library only; the tiny snapshot and GGUFs are regenerated into a temporary
directory on every run (seconds). Until phase 3 lands, the runner stops after case 0
with `RESULT: FAIL (module not built …)`.

## State

| Date | Result |
|---|---|
| 2026-10-06 | document, runner, torch-free fixture generator and tokenizer corpus written; case 0 passes; `tools/st2gguf.py`, `src.h`, `tests/test_gguf_load`, `tests/test_tok_gguf` and the Python arms are phase-3 deliverables |
| 2026-10-06 | **pass** — `RESULT: ok (0 failures)`, all 8 cases including case 5 on the real unsloth metadata and Qwen's `tokenizer.json` (52 lines identical). Harness fixes during implementation: the fixture's values are bf16-representable (so `1 + w` is exact), the `A_log` tolerance is 4·2⁻²⁴ absolute (the exp/log round trip's error does not scale with A_log), the GGUF vocab may be padded with `[PADn]`, the rewriter strips the reader's padded dims. |
