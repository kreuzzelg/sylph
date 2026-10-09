# Integration test — expert streaming from a GGUF (`Slot.kq` ⟷ LRU / pin / pilot ⟷ `src.h` slices ⟷ split sets ⟷ sidecars ⟷ startup line)

Written 2026-10-08, **before** phase 4 (process gate in `../../04_Tasks/tasks.md`).
Specification: FR-27, FR-28, FR-29, FR-30, FR-36, NFR-3, NFR-5, NFR-7 (reported
numbers); architecture §8 and the 2026-10-06 amendment ("phase 3 as implemented").

**Modules under test:** the GGUF slot flavour in `qwen36.c` (`Slot{kq, ktype[3],
kbytes[3]}`, `load_expert_gguf`, `moe_gq_run`), the expert cache around it (`expert_get`,
LRU eviction, `HOT` pinning, `PILOT`/`PILOT_REAL` prefetch), the slice reads of `src.h`
(`ts_expert`, `ts_read_expert_slice`), split sets through `gguf.h`, the `[GGUF]` startup
line and the end-of-run statistics, the on-demand embedding path (`gq_embed_row`), the
sidecar rule for GGUF models (engine and `coli`), the CUDA-tier refusal note.
**Neighbours exercised:** the container arm of the same cache (NFR-5), `tools/st2gguf.py`
(split writer), `tools/convert_qwen36.py` (the container twin of the fixture; needs torch),
the serve protocol (`SERVE=1`) with the tokenizer from the metadata, `family_registry`
and `coli info`.
**Inputs:** the torch-free tiny Qwen3.6 snapshot of `make_tiny_qwen36_hf.py` (same
fixture as `src_facade.md`), a **320-token byte-level tokenizer** written by the runner
so that every id of the tiny vocabulary decodes and any ASCII prompt encodes, GGUFs
derived with `tools/st2gguf.py` (F32, all-Q8_0, experts-only Q8_0, a 3-part split of the
F32 file), the int8 container `convert_qwen36.py --ebits 8` writes from the same snapshot (the converter imports torch; where torch is absent the runner skips the container cases and the `gguf-oracle` CI job, which has torch, carries them).
**Outputs:** identical token ids and identical log-prob dumps across cache capacities,
thread counts, prefetch modes and file layouts; statistics that account for every byte
read; nothing written next to the model.

## What phase 3 already pinned, and what this test adds

Phase 3 made a GGUF run at all (`lossless_oracle.md`: token-exact against transformers,
cache hit rates equal to the container's at cap 1/2/8). This test pins the **streaming
behaviour** the owner will live with on a 22 GB file: that the LRU, the pins and the
pilot see a GGUF expert exactly as they see a container expert (identity `(layer, eid)`,
FR-27); that the bytes read are the slices and nothing more; that split files work;
that the model directory stays clean (FR-29); that the startup line says what runs
where (FR-30); and that the two knobs phase 4 adds (`COLI_GGUF_EMBED`, `--split`) do
not change a single number.

### FR-28, verified against the engine (2026-10-08)

FR-28 asks for `DIRECT`, mirror, split dirs and `URING` "where upstream's `qwen36`
engine has them". It has **none of them**: `grep -n 'O_DIRECT\|COLI_MODEL_MIRROR\|io_uring\|COLI_MODEL_DIRS'
06_Code/c/qwen36.c` is empty; those live in `colibri.c` (OLMoE), `glm53.c`, `kimi_k3.c`
and `qwen38.c`. `qwen36.c` reads experts with `pread` and drops the pages with
`posix_fadvise(DONTNEED)`, and the GGUF arm inherits exactly that. The requirement is
therefore satisfied vacuously for v1; this test checks only that the knobs are **ignored
identically** on both sources (case 9), and the fact is recorded for the architecture
(the mirror/split machinery would be a `qwen36` feature first, a GGUF feature second).
The same grep shows `qwen36.c` writes **no sidecar** of its own (`.coli_usage` lives in
`route_trace.h`, which this engine does not include; `kv_prefix.h` is in-memory); the
files `coli` writes beside a model (`.coli_kv`, `.coli_ssd`) belong to the GLM engine.
FR-29 is therefore a rule for the **paths**, fixed below, not a migration.

## Contracts fixed by this test

### `tools/st2gguf.py --split N` (phase-4 deliverable, stdlib)

```
python3 tools/st2gguf.py <hf_dir> --out <stem>.gguf --split N [the phase-3 options]
```

writes `<stem>-00001-of-0000N.gguf` … `<stem>-0000N-of-0000N.gguf` the way llama.cpp's
`gguf-split` does: every part carries `split.no` (0-based), `split.count`,
`split.tensors.count` and `general.alignment`; part 1 carries the full KV set
(architecture, tokenizer), parts > 1 carry only those four keys; tensors are assigned in
file order so that the parts' payload sizes are as equal as possible, every part holds at
least one tensor, and a tensor never straddles parts. `tests/test_gguf <any part>` (phase
1) indexes the set; `ggufinfo.summarize` reports `N parts`. Without `--split` nothing
changes.

### Engine (`qwen36.c`, `src.h`)

- **Startup line (FR-30)**, one line on stderr before the first token, machine-readable
  by label:

  ```
  [GGUF] <arch> · <B> blocks (<A> attention) · <P> part(s) · experts <tg>/<tu>/<td> <x.xx> MB each × <E> × <B> = <y.yy> GB · dense <type list> <z.zz> GB · token_embd <type> <on demand|f32 at load> · output <type> <kernel> · experts on CPU (gq_moe_run) · sidecars <dir>/.coli-<stem>/
  ```

  `<kernel>` is `gq_matmul` for K-quants, `matmul_q_gs` for the Q8_0 split, `matmul_d`
  for f32/int8-at-load; the dense type list is the per-type tensor count as today
  (`F32×73 Q8_0×76`). The phase-3 `[meta] from GGUF:` line stays.
- **Statistics (FR-30, NFR-7 reporting)**: the end-of-run block (reference mode and the
  `DONE … STAT` path's stderr summary) gains one line

  ```
  GGUF reads: <S> slices · <M> MB · <m> MB/token · <P> part(s) touched
  ```

  with `S = 3 × misses` (one `pread` per slice, FR-27) and `M` = Σ bytes of the slices
  read, so `M / misses` equals the expert size the startup line names (± rounding).
- **Expert cache unchanged (FR-27)**: `expert_get`, `pin_hot_experts`, `apply_resident`,
  the pilot worker and `slot_ensure_allocated` take no GGUF branch beyond the slot
  flavour already in place; hence the hit/miss counts of a GGUF run equal those of the
  container run on the same ids at the same cap (case 1), and `PILOT`, `PILOT_REAL`,
  `HOT` produce the same ids as the plain run (case 3).
- **Embedding on demand**: `token_embd.weight` is no longer dequantized to f32 at load;
  `step()` fetches each prompt token's row with `gq_embed_row` (F32/F16/BF16/Q8_0/K-quants).
  `COLI_GGUF_EMBED=0` restores the phase-3 f32 table (A/B knob, documented in
  `docs/ENVIRONMENT.md`). Both produce bit-identical log-prob dumps (case 8); the startup
  line names the mode. RSS saving on the real file: 2048 × 248320 × 4 B = 2.03 GB, reported
  in `08_Documents/`.
- **Sidecar rule (FR-29)**: for a GGUF source the sidecar directory is `<dir>/.coli-<stem>/`,
  where `<dir>` is the directory holding the file (or the parts) and `<stem>` is the file
  name without `.gguf` and without the `-0000k-of-0000N` suffix. `src.h` exposes
  `ts_sidecar_dir(const TensorSource*, char *out, size_t cap)`; `family_registry.py`
  exposes `sidecar_dir(model_path)` with the same rule (and the model directory itself
  for a container). Nobody creates the directory until something is written into it; the
  engine and `coli` never write beside a `.gguf`. `coli info --model <gguf>` prints the
  path with `(none yet)` while it does not exist.
- **Split sets**: `SNAP=<dir>` holding the parts, `SNAP=<part 1>` or `SNAP=<any part>`
  all load the same model; a missing part is refused by name (phase 1's message); the
  startup line counts the parts and the statistics count the parts touched.
- **CUDA tier note (FR-36)**: with `COLI_CUDA=1` and a GGUF source the engine prints
  `[qwen36] COLI_CUDA=1 ignored: the VRAM expert tier does not take GGUF K-quant experts yet (sylph phase 5); CPU path`
  once and continues with identical output (already in phase 3; pinned here).
- **Serve path**: `SERVE=1` with a GGUF works without `TOK=` (tokenizer from the metadata)
  and produces the same `DATA` frames and `logprobs=k` tails as the same GGUF with `TOK=<the json the GGUF was written from>`.

### Runner-side fixture: the 320-token byte-level tokenizer

`tiny_tok.json`: ids 0…255 are the 256 GPT-2 byte symbols in byte order (the mapping
`build_byte_sym` implements: printable ASCII and Latin-1 ranges map to themselves, the
rest to U+0100…), ids 256…318 are 63 merges of frequent letter pairs and `Ġ`-prefixed
words (valid BPE: each merge's parts exist before it), id 319 is the special
`<|im_end|>` (so `eos_token_id` defaults to the vocabulary's last id, which
`st2gguf.py` writes as `tokenizer.ggml.eos_token_id`). It is **not** Qwen's tokenizer;
it makes every id of the tiny model's vocabulary decodable, so container (`TOK=`) and
GGUF (metadata) runs can be compared byte for byte.

## Cases

| # | Case | Expected |
|---|---|---|
| 0 | Fixtures | `tiny_hf/` (317 F32 tensors), `tiny_tok.json`, container `tiny_c/` (`convert_qwen36.py --ebits 8`; skipped without torch), `tiny_f32.gguf`, `tiny_q8_0.gguf` (dense+experts), `tiny_xq8_0.gguf` (experts only), `tiny_split-0000{1,2,3}-of-00003.gguf` (`--split 3`), all with `--tokenizer tiny_tok.json`; `ggufinfo.summarize` on the split set says 3 parts and the same tensor count as the single file. |
| 1 | **Capacity parity (FR-27)** | reference-id runs (`./qwen36 <cap> 8 ref.json`, 5-id prompt, 16 generated) at cap 1, 2, 8: the GGUF F32 `C engine :` ids are identical across caps; the container's are identical across caps; per cap, GGUF and container print the same `Expert cache hit rate … (hit=… miss=…)` whenever their ids coincide (reported otherwise: int8 vs f32 experts may route differently on random weights). |
| 2 | **Thread determinism** | `PPL=1 PPL_DUMP` with `OMP_NUM_THREADS=1` and `=4` on `tiny_f32.gguf` and `tiny_q8_0.gguf`: dumps byte-identical (rank-ordered expert sum, deterministic kernels). |
| 3 | **Pilot and pins** | `PILOT=1`, `PILOT=1 PILOT_REAL=1`, `HOT=2` on the GGUF F32: ids identical to case 1, exit 0, `[HOT] Pinned …` line present for `HOT=2`; same for the container (NFR-5). |
| 4 | **Split set** | `SNAP=<dir of parts>`, `SNAP=…-00001-of-00003.gguf`, `SNAP=…-00002-of-00003.gguf`: ids and PPL dump identical to the single file; startup line `3 parts`; stats `parts touched` ≥ 2 at cap 1; renaming part 3 away → exit ≠ 0, message names `00003`. |
| 5 | **Startup line (FR-30)** | every label of the format above present on F32, Q8_0 and split runs; `token_embd Q8_0 on demand` on the Q8_0 file; `output F32 matmul_d` on F32; `sidecars` path equals `family_registry.sidecar_dir()`. |
| 6 | **Reads accounting** | cap 1 and cap 8 on the GGUF F32: `GGUF reads:` slices `= 3 × miss`, MB `= miss × expert bytes` within 1 %, `MB/token` = MB / 16. |
| 7 | **Sidecar hygiene and API (FR-29)** | after cases 1–6 and 8–11 the fixture directory holds exactly the files case 0 created (no `.coli_*`, no dumps, no `qwen36_logits.f32`); `sidecar_dir()` for the file, the parts directory and a part → `<dir>/.coli-tiny_f32/` resp. `<dir>/.coli-tiny_split/`; for `tiny_c/` → `tiny_c/`; `coli info --model tiny_f32.gguf` prints `sidecars: … (none yet)`. |
| 8 | **Embedding on demand** | `tiny_q8_0.gguf` PPL dump with default and with `COLI_GGUF_EMBED=0`: byte-identical; the startup line says `on demand` resp. `f32 at load`. |
| 9 | **FR-28 knobs ignored identically** | `DIRECT=1 URING=1 PIPE=1 COLI_MODEL_MIRROR=<empty dir> COLI_MODEL_DIRS=<empty dir>` on GGUF F32 and on the container: exit 0, ids identical to case 1, no `[MIRROR]`/`URING` lines (the engine has no such paths). |
| 10 | **CUDA note (FR-36)** | `COLI_CUDA=1` on the GGUF F32: the note line exactly once, ids identical to case 1, exit 0. |
| 11 | **Serve smoke** | `SERVE=1 SNAP=tiny_f32.gguf` (no `TOK`) vs `SERVE=1 SNAP=tiny_f32.gguf TOK=tiny_tok.json`: `READY`, `STAT`, then `SUBMIT 1 0 <plen> 24 0 1 logprobs=5` with an ASCII prompt and once with a UTF-8 prompt (`ä€🐦`): `ACCEPT … <np>` equal, every `DATA` frame (bytes and log-prob tail) identical, `DONE … STAT` present; the GGUF without `TOK` never prints `tokenizer.json required`. |
| 12 | **Container untouched (NFR-5)** | the container's ids, hit rates and `make check` are unchanged by phase 4: cases 1, 3, 9 on `tiny_c/` pass with the phase-3 binary and with the phase-4 binary (the runner records both when `--baseline <qwen36 binary>` is given). |

**Pass criterion:** cases 0–12 pass; the reported comparison in case 1 (GGUF vs container
hit rates when ids differ) and the RSS figures are recorded, not judged. Bit-identity in
cases 2, 4, 8 is the point: a tolerance would hide a wrong slice offset or a wrong
accumulation order.

## Runner

```sh
python3 07_Tests/IntegrationTest/run_expert_streaming.py             # all cases; builds qwen36 if missing
python3 07_Tests/IntegrationTest/run_expert_streaming.py --keep      # leave the fixtures in the scratch dir
python3 07_Tests/IntegrationTest/run_expert_streaming.py --baseline ./qwen36.phase3   # case 12 with a second binary
```

Standard library only; fixtures are regenerated into a temporary directory on every
run (seconds; the tiny model decodes at thousands of tokens per second). Until phase 4
lands, the cases that need the new contracts fail with `RESULT: FAIL` and name them
(`--split`, the startup-line labels, `GGUF reads:`, `sidecar_dir`, `COLI_GGUF_EMBED`);
cases 1–3, 9–11 should already pass on the phase-3 binary and are the regression net
while phase 4 is implemented.

## State

| Date | Result |
|---|---|
| 2026-10-08 | document and runner written; `--split`, the startup-line format, `GGUF reads:`, `sidecar_dir`/`ts_sidecar_dir`, `COLI_GGUF_EMBED` and the on-demand embedding are phase-4 deliverables. Pre-implementation run: see the row below. |
| 2026-10-08 | pre-implementation run on the phase-3 binary (container skipped, no torch here): `RESULT: FAIL (13 failures)`, all of them the phase-4 contracts (`--split` → cases 0/4, startup-line labels and sidecar path → 5, `GGUF reads:` → 6, embedding mode in the line → 8, `sidecar_dir` + `coli info` → 7). Already passing and therefore the regression net: capacity parity across caps (1), thread determinism of the dumps for F32 and Q8_0 (2), `PILOT`/`PILOT_REAL`/`HOT` ids (3), FR-28 knobs ignored (9), CUDA note (10), serve smoke with the metadata tokenizer == `TOK=json` on 24 frames incl. a UTF-8 prompt (11), nothing written beside the model (7, hygiene half). Runner fix during writing: the reference mode exits 1 when the ids differ from `ref.json`'s arbitrary ids, so a run is judged by its printed summary, not its exit code. |
| 2026-10-09 | **pass** — `RESULT: ok (0 failures)` on the phase-4 engine, all 13 cases (container skipped here; it runs in the `gguf-oracle` job). Runner fixes during implementation: the dump header names the `SNAP` path, so dumps are compared body to body (split set vs single file); `ggufinfo.summarize` keys are `tensors` and `typical_expert_bytes`. Implementation notes: the startup line is composed in `ts_describe` from the index plus the engine's two decisions (`embd_mode`, `out_kernel`); the read counters live in `TensorSource` and are incremented in `ts_read_expert_slice`; `GGUF reads:` is printed in reference, PPL and (per turn) serve mode. |
| 2026-10-09 | **pass in CI with the container** (`gguf-oracle` run 21): `RESULT: ok (0 failures)`, all 13 cases incl. the container arms of cases 1, 3 and 9. Case 1 hit/miss parity: at cap 8 the ids coincide and GGUF and container print identical counts (hit 256 / miss 64); at cap 1 and 2 the int8 container and the f32 GGUF route one or two experts differently on this random-weight fixture (GGUF 20/300 and 58/262 vs container 18/302 and 54/266), reported as the test foresees. Split set: 3 parts touched at cap 1; `GGUF reads` 900 slices = 3 × 300 misses, 7.37 MB. |
