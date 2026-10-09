# Integration test — the NextN (MTP) block of a `qwen35moe` GGUF in `qwen36` (detection ⟷ precision guard ⟷ lossless draft/verify)

Written 2026-10-09, **before** phase 6 (process gate in `../../04_Tasks/tasks.md`).
Specification: FR-24 (MAY: "detected and reported; loading is MAY"), FR-9, FR-23, NFR-3,
NFR-5; architecture v1 §7.5, v2 §12 row "6 — breadth"; inspection
`08_Documents/inspection-qwen36-gguf-2026-10-05.md` (bartowski's `blk.40.*` + `nextn.*`).

**Module under test:** `qwen36.c`'s handling of `nextn_predict_layers ≥ 1`: today the block
is skipped with a note (`[GGUF] skipping 1 NextN (MTP) block`), FR-24 is met at the
"detected and reported" level. Phase 6 decides whether the head is **loaded and used** for
speculative decoding, the way `colibri.c` uses GLM-5.2's (`mtp_draft`/`mtp_absorb`). This
test fixes the contract either way: detection and reporting stay as they are; if the head
is loaded, greedy decoding must be **lossless** (the draft is verified by the trunk, never
trusted), the acceptance statistics are printed, and the precision guard of v1 §7.5
applies (`eh_proj` below 8 bpw is skipped unless `MTP=1`).
**Neighbours:** `tools/st2gguf.py` (gains the `mtp.*` → `blk.<L>.nextn.*` + `blk.<L>.*`
mapping instead of skipping the subtree), `make_tiny_qwen36_hf.py --mtp` (this directory),
`ggufinfo.summarize` (`mtp` field, done in phase 1), `coli doctor` (`model.gguf.mtp_precision`,
done), the serve path (a drafted token must go through the same `DATA` frames).
**Inputs:** the tiny Qwen3.6 snapshot with `--mtp` (one full-attention MTP block as
transformers saves it: `mtp.fc`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`,
`mtp.norm`, `mtp.layers.0.*`), written as F32 GGUF and as F32 with `--mtp-type q4_k`.
**Outputs:** identical greedy ids with and without the head; a `[MTP]` statistics line;
the `[GGUF]` line naming the head's type and bits.

**Layouts seen in the wild (survey 2026-10-09):** bartowski's main file carries the block
in-file (`block_count 41`, `nextn_predict_layers 1`, 20 `blk.40.*` tensors + `nextn.*`), and
the repository additionally publishes a 1.9 GB companion `mtp-<stem>.gguf` holding only that
block plus `token_embd`/`output`/`output_norm`. The in-file layout is the contract here; the
companion file is a MAY of its own (`MTP_FILE=<path>` or `mtp-<stem>.gguf` beside the model,
same keys), recorded so nobody mistakes it for a second converter convention.

## Why lossless is the only acceptable contract

Ollama and llama.cpp do not use the MTP block (spec §9 c); every equivalence level of §4 is
defined on the trunk. A speculative decoder that changed a token would therefore fail E3
by construction. Under greedy decoding (`COLI_TEMP=0`) the trunk verifies each drafted
token against its own argmax, so the sequence is exactly the plain-greedy sequence and the
head only changes **speed**. That is the property this test pins; tok/s is reported by
`real_model.md`/`gpu_rtx3070.md` when the owner runs bartowski's file.

## Contracts fixed by this test

### Names (`qwen35_names.h` gains the NextN rows)

| HF name (transformers Qwen3-Next) | GGUF name (`L = n_layers`) |
|---|---|
| `mtp.fc.weight` `[H, 2H]` | `blk.L.nextn.eh_proj.weight` |
| `mtp.pre_fc_norm_embedding.weight` | `blk.L.nextn.enorm.weight` |
| `mtp.pre_fc_norm_hidden.weight` | `blk.L.nextn.hnorm.weight` |
| `mtp.norm.weight` | `blk.L.nextn.shared_head_norm.weight` |
| `mtp.layers.0.<attention / MoE tensor>` | `blk.L.<the regular name>` (an attention block: `attn_q/k/v/output`, `attn_q_norm/k_norm`, `ffn_gate_inp`, `ffn_*_exps`, `ffn_*_shexp`, `ffn_gate_inp_shexp`) |

Norms carry the `1 + w` transform like every other norm (`gguf_xform.h`).

### Engine (`qwen36.c`)

```
[GGUF] … · nextn blk.L eh_proj <type> (<bpw> bpw) <loaded | skipped (MTP=0) | skipped (<bpw> bpw < 8; MTP=1 to force)> · …
[MTP] proposed <n> · accepted <m> (<p> %) · <extra tokens per step>          (at exit, when loaded)
```

- `MTP=0` always skips; `MTP=1` forces loading past the precision guard; unset = load when
  `bits_per_weight(eh_proj) ≥ 8` (bartowski: `Q8_0`, 8.5 bpw → loaded).
- The MTP block is a full-attention layer with its own KV; its experts are routed experts
  of layer `L` in the same LRU cache (`cap` slots, identity `(L, eid)`), streamed as raw
  blocks like every other layer; `GGUF reads:` counts them.
- Draft/verify: after the trunk emits token `t` at position `p` with hidden `h`, the head
  predicts `d = argmax(lm_head(norm(block(eh_proj[enorm(embed(t)) ; hnorm(h)]))))`; the
  next trunk step runs on `[t, d]` (two positions); if the trunk's argmax at `t` equals `d`
  the token is accepted and the step gained one token, else `d` is discarded and the
  trunk's token stands. Greedy ids are therefore identical to the run without the head.
- Serve mode (`SERVE=1`): accepted tokens stream as ordinary `DATA` frames with their
  log-prob tails from the trunk's logits (never the head's).
- Container path (`convert_qwen36.py` has no MTP): untouched (NFR-5).

### Fixture (`tools/st2gguf.py`)

`mtp.*` is converted (not skipped) when the config says `mtp_num_hidden_layers = 1`;
`--mtp-type <type>` sets the type of `eh_proj` (default: `--type`); `--no-mtp` skips it as
before. `block_count = n_layers + 1`, `nextn_predict_layers = 1`.

## Cases

Runner: `python3 07_Tests/IntegrationTest/run_qwen36_mtp.py [--keep] [--no-build]`.

| # | Case | Expected |
|---|---|---|
| 0 | fixtures | `make_tiny_qwen36_hf.py --mtp` → 8 + 1 blocks; `st2gguf` → `tiny_mtp_f32.gguf` (`block_count 9`, `nextn_predict_layers 1`, `blk.8.nextn.eh_proj F32`, `blk.8.attn_q` present), `tiny_mtp_q4k.gguf` (`--mtp-type q4_k`, hidden 256 fixture), `tiny_nomtp.gguf` (`--no-mtp`); `ggufinfo.summarize(...)["mtp"] == {"layer": 8, "eh_proj_type": "F32", …}` |
| 1 | detection, reporting | `coli gguf inspect`: `mtp  blk.8 nextn · eh_proj F32 (32.00 bpw)`; `coli doctor`: `model.gguf.mtp_precision pass`; the `[GGUF]` line names the head |
| 2 | **lossless** | reference mode, 16 tokens, cap 8: ids with the head (default) == ids with `MTP=0` == ids from `tiny_nomtp.gguf`; the `[MTP] proposed … accepted …` line present only when loaded; `proposed ≥ 1` |
| 3 | precision guard | `tiny_mtp_q4k.gguf`: `skipped (4.50 bpw < 8; MTP=1 to force)`; with `MTP=1` loaded and still lossless |
| 4 | PPL mode | `PPL=1` ignores the head (teacher forcing has nothing to draft): dumps byte-identical with and without `MTP=0` |
| 5 | serve | `SERVE=1`: two prompts, frames identical (ids, tails) to the `MTP=0` session; `DONE` present |
| 6 | streaming | `GGUF reads:` with the head counts the MTP layer's expert slices (slices = 3 × misses over 9 blocks); cap 1 and cap 8 ids identical |
| 7 | container untouched | the int8 container of the same snapshot (where torch exists) ignores `mtp.*` as before: ids identical to the GGUF trunk-only run |

**Pass criterion:** `RESULT: ok (0 failures)`. If phase 6 decides **not** to load the head
(owner's call, spec §9 c), cases 2–6 are replaced by one: the `[GGUF]` line says
`nextn blk.L eh_proj <type> (<bpw> bpw) present, not used (MTP decoding not implemented)`
and ids equal the `--no-mtp` file's — and this document records the decision.

## State

| Date | Result |
|---|---|
| 2026-10-09 | document, runner and the fixture option written; the name rows, `st2gguf`'s `mtp` arm and the engine's loader/draft are phase-6 deliverables (MAY). Pre-implementation run: see the row below. |
| 2026-10-09 | pre-implementation run (`run_qwen36_mtp.py --no-build`): `RESULT: FAIL (6 failures)`, all phase-6 contracts — `st2gguf` skips `mtp.*` (no `nextn.*` tensors, `mtp` summary `None`, `--no-mtp`/`--mtp-type` unknown), `coli gguf inspect` has no head to name. Already passing and therefore the regression net: the fixtures with the MTP block (preset and wide), `block_count 9 = 8 + 1` written from `mtp_num_hidden_layers`, `coli doctor` without a GGUF failure, PPL dumps byte-identical with `MTP=0`, serve frames (ids, tails) identical between sessions. Runner fix: the `DONE` line carries timings, so case 5 compares `ACCEPT` and `DATA` frames only. Survey fact: bartowski's main `Q4_K_M` carries the block in-file (20 `blk.40.*` tensors + `nextn.*`, `eh_proj Q8_0` 8.5 bpw); unsloth's files have none. |
| 2026-10-09 | **decision (pending the owner's §9 c): the Qwen engine reports the head and does not use it** — the fallback contract above. Implemented: `qwen35_names.h` NextN rows (`qn_nextn_layer` set from the Cfg), `st2gguf` converts `mtp.*` (`--mtp-type`, `--no-mtp`; `block_count`/`nextn_predict_layers` only when the block is written), `qwen36_tensor_kinds.py` classifies `mtp.*` as `("mtp", kind)`, the `[GGUF]` line says `nextn blk.L eh_proj <type> (<bpw> bpw) present, not used (MTP decoding not implemented)`. Runner cases 2–6 replaced by the fallback case (head present vs `--no-mtp` file: ids, PPL dumps, serve frames and streaming identical; the Q4_K head is named with 4.50 bpw): **`RESULT: ok (0 failures)`**. Open: whether llama.cpp's converter stores `enorm`/`hnorm` as `1 + w` on bartowski's file (our table says yes, consistent with its own fixture; verify on the real file before the head is ever used). Draft/verify (DeltaNet state rollback on rejection) is the work item if the owner says yes. |