# Integration test — several conversations at once on a GGUF source (`KV_SLOTS`, upstream v2.0.0) and the other post-sync contracts

Written 2026-10-14, **before** phase 7 (process gate in `../../04_Tasks/tasks.md`).
Specification: NFR-5 (container behaviour unchanged), NFR-6 (upstream sync stays routine),
FR-27/FR-30 (streaming accounting), spec §3 ("parity with Ollama's server features is out of
scope; the model function is compared"); architecture v2 §12 row "7 — consolidation";
`../SystemTest/upstream_sync.md` (the sync tool, dry merge: 9 conflicted files).

**Module under test:** the `06_Code` subtree **after** `tools/upstream_sync.py --apply` brings
colibrì **v2.0.0** (932 commits, `bf244291`) in, and the GGUF arm of `qwen36.c` on top of what
that release changed in the engine: `KV_SLOTS=n` (#1955, every text engine serves up to 16
conversations at once, one batched decode step per token across the active requests), the
Vulkan routed-expert tier (`COLI_VK_TIER*`, `vk_tier.c`) and the dense chain (`vk_chain.c`).
A GGUF source must serve several conversations exactly as a container does, and the features
the GGUF path does not implement must say so by name instead of misbehaving.
**Neighbours:** upstream's own check `tests/serve_mux_check.py` (arrives with the sync: frames
of every request against the same request served alone, byte for byte), the serve protocol
(`SUBMIT <id> <slot> <bytes> <max> …`), `expert_streaming.md` (the GGUF slot cache, `GGUF
reads:`), `cuda_tier_kquant.md` (the CUDA tier is unaffected by the slots), the equivalence
harness (unchanged).
**Inputs:** the tiny Qwen3.6 snapshot (`make_tiny_qwen36_hf.py`), its F32 GGUF, the 320-token
byte-level tokenizer the streaming runner writes, and — where torch exists (CI) — the int8
container of the same snapshot for the "container unchanged" half.
**Outputs:** identical frames multiplexed vs alone, the accounting line, refusals by name.

## Why this is the first test of phase 7

The sync is the one change that touches every file sylph also touched (`qwen36.c`,
`qwen36_tier.c`, `coli`, `doctor.py`, `family_registry.py`, `Makefile`, `ENVIRONMENT.md`,
`docs.json`, `CHANGELOG.md`). Its correctness has two halves: the **old** contracts (every
phase runner, `make check`, the CI oracles — they exist and stay the gates) and the **new**
behaviour a GGUF source inherits from v2.0.0. This test pins the second half, so the merge
has a target beyond "it compiles".

## Contracts fixed by this test

### `KV_SLOTS` on a GGUF source (`qwen36.c`)

- `SNAP=<file.gguf> SERVE=1 KV_SLOTS=n ./qwen36 <cap> 8` serves `n` conversations; every
  request's `ACCEPT`/`DATA`/`DONE` frames (ids, text, logprob tails) equal those of the same
  request served alone (`KV_SLOTS=1`) on the same file — `tests/serve_mux_check.py` passes
  with the GGUF as its snapshot, both waves (extend / go back inside / start over, one
  `STOP`, one `CANCEL`).
- The GGUF slot cache is shared across the conversations as the container's expert cache is
  (identity `(layer, eid)`); a step that routes the rows of `k` conversations reads an
  expert **once** — `GGUF reads:` after the multiplexed session counts fewer slices than the
  `n` single sessions together (`slices_mux < Σ slices_alone`), and `slices = 3 × misses`
  still holds.
- `KV_SLOTS` outside `[1, 16]` is refused with upstream's message; `KV_SLOTS=n` prints
  `[qwen36] serving n conversations at once (KV_SLOTS)` on a GGUF exactly as on a container;
  nothing drafts (`DRAFT` ignored) as upstream states.
- `PPL=1` and the reference mode ignore `KV_SLOTS` (one conversation).

### Features the GGUF path does not implement (phase 7 scope: say so)

| Knob (v2.0.0) | On a GGUF source | Message |
|---|---|---|
| `COLI_VK_TIER=1` (Vulkan expert tier) | ignored, CPU experts, ids unchanged | `[GGUF] COLI_VK_TIER ignored: the Vulkan tier uploads container formats; GGUF blocks stay on the CPU (CUDA tier: COLI_CUDA=1)` — or, on a build without Vulkan, upstream's own "built without Vulkan" note |
| `COLI_VK_CHAIN*` (dense chain) | ignored, ids unchanged | `[GGUF] COLI_VK_CHAIN ignored: …` |
| `COLI_CUDA=1` | the phase-5 tier (unchanged) | as `cuda_tier_kquant.md` |

### Container unchanged (NFR-5)

`tests/serve_mux_check.py ./qwen36 <int8 container> 3` passes as it does upstream; the
container's frames before and after the sync are byte-identical for the same request
(`expert_streaming.md` case 12's baseline rule: the CI job's container run is the record).

## Cases

Runner: `python3 07_Tests/IntegrationTest/run_multi_slot_gguf.py [--keep] [--no-build]`.

| # | Case | Expected |
|---|---|---|
| 0 | fixtures | tiny snapshot, tokenizer, F32 GGUF (`st2gguf`), `tests/serve_mux_check.py` present in the subtree (v2.0.0) |
| 1 | frames multiplexed == alone | `serve_mux_check.py ./qwen36 tiny_f32.gguf 3 TOK=<json> COLI_DENSE_I8=0` exits 0 (its own byte-for-byte comparison), its report shows decode steps of more than one row |
| 2 | accounting | `GGUF reads:` of the multiplexed session (3 slots, the first wave) shows `slices = 3 × misses` and fewer slices than the three single sessions together |
| 3 | refusals | `KV_SLOTS=17` → exit 2 with `KV_SLOTS must be between 1 and 16`; `KV_SLOTS=0` likewise |
| 4 | reference mode ignores slots | `KV_SLOTS=3` in reference mode: ids identical to the plain run |
| 5 | Vulkan knobs on a GGUF | `COLI_VK_TIER=1` / `COLI_VK_CHAIN=1`: a one-line message naming the knob, ids identical to the plain run |
| 6 | container unchanged | where the int8 container exists: `serve_mux_check.py ./qwen36 tiny_c 3` exits 0 |

**Pass criterion:** `RESULT: ok (0 failures)` locally and in the `gguf-oracle` job.

## State

| Date | Result |
|---|---|
| 2026-10-14 | document and runner written. Pre-implementation run: see the row below. |
| 2026-10-14 | pre-implementation run (`run_multi_slot_gguf.py --no-build`): `RESULT: FAIL (5 failures)`, all post-sync contracts — `tests/serve_mux_check.py` absent (v2.0.0 not merged yet, so case 1/2 skipped), `KV_SLOTS=17/0` not refused (the knob does not exist in the subtree's `qwen36.c`), the Vulkan knobs are silently ignored (no message naming them). Already passing and therefore the regression net: the fixtures, reference mode with `KV_SLOTS=3` (ids identical), ids unchanged under `COLI_VK_TIER=1` / `COLI_VK_CHAIN=1`. |
