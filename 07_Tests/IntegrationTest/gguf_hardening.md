# Integration test — hardening of the GGUF path: mutation fuzz, truncated payloads, split sets, one reader, the GLM tokenizer (phase 7)

Written 2026-10-14, **before** phase 7 (process gate in `../../04_Tasks/tasks.md`).
Specification: NFR-4 (untrusted input: bounded, overflow-checked, refusals name the rule),
FR-20 (tokenizer from the metadata, both engines), FR-26, housekeeping "consolidate the two
stdlib GGUF readers"; architecture v2 §12 row "7 — consolidation".

**Modules under test:** `c/gguf.h` (the reader every engine opens the file with), `c/ggufinfo.py`
and upstream's `c/tools/gguf_reader.py` (the two stdlib readers), the load paths of both engines
(`qwen36.c`: `ts_*` through `src.h`; `colibri.c`: the GLM arm), the GLM tokenizer constructor
`glm_tok_load` (phase 6, smoke-tested only), `coli gguf inspect` / `coli doctor` on damaged files.
**Neighbours:** `tests/test_gguf.c` (phase 1: a bit-flip sweep over the metadata region of one
synthetic file, mirror, split-set rules), `tests/test_ggufinfo.py`, `tests/test_gguf_reader.py`
(upstream), `glm_assembly.md` case 9 (the GLM tokenizer corpus test that had no runner).
**Inputs:** the tiny Qwen3.6 F32 GGUF (single and `--split 2`), the tiny GLM F32 GGUF
(`--vocab 320`, with the tokenizer of `make_tiny_glm_tok.py`), `fixtures/tok_corpus.txt`.
**Outputs:** every damaged input ends in a named refusal or a clean success — never a signal,
a hang or a foreign exception; two readers that agree, then one reader; the GLM tokenizer
equal to its JSON twin on the corpus.

## Contracts fixed by this test

### Mutations (`gguf.h`, `ggufinfo.py`, both engines)

A **seeded mutation sweep** (`--seed`, default 20261014; `N` mutations, default 160 per file)
over the metadata region of a valid file: single bit flips, byte fills (`0x00`, `0xFF`),
truncations at every structural boundary (inside the header, inside a KV, inside a tensor
info, at `data_off`, inside the payload), and **structural edits** written on purpose:
`tensor_count` / `kv_count` inflated past the file, a string length of 2^40, an array of
2^31 elements, nested arrays, `general.alignment` 0 / 3 / 2^31, a tensor `offset` beyond
the data region, two tensors with the same name, a tensor whose `ne` product overflows
int64, `n_dims` 5, an unknown ggml type id, GGUF version 2 and 4, a payload shorter than
the last tensor declares.

For every mutated file:

| Reader / engine | Must | Must not |
|---|---|---|
| `tests/test_gguf <file>` (C reader CLI) | exit 0 with a dump, or exit 1 with `gguf: <rule>` | crash (signal), hang (> 20 s), exit without a message |
| `ggufinfo.open_set` (in-process) | return, or raise `GgufError` naming the key/tensor/rule | raise any other exception (`struct.error`, `MemoryError`, `IndexError`, `UnicodeDecodeError`…) |
| `coli gguf inspect <file>` | exit 0, or exit non-zero printing `gguf:`/`refus` | a traceback |
| `./qwen36 8 8 ref.json` / `./colibri 8 16 16` on the mutated file | exit 0 (valid after the edit) or exit 1/2 with `[GGUF]`/`gguf:`/`config:` in stderr | signal, hang (> 60 s), silent exit |

The sweep prints `accepted / refused / <reader>` counts; **at least one** mutation is accepted
by each reader (the sweep is not trivially "everything fails") and the refusal rate of the C
reader equals `ggufinfo`'s on structural edits (they implement the same rules).

### Truncated payload (both engines)

A valid header whose payload is cut inside an expert tensor loads its dense set, then refuses
at the first slice read that runs past EOF with `[GGUF] short read of <name> …` — exit 1, no
partial output, no crash. A payload cut inside a dense tensor refuses at load.

### Split sets (`gguf.h`, `ggufinfo.py`, `coli doctor`)

| Case | Expected |
|---|---|
| a part is missing (`-00002-of-00003` deleted) | `[GGUF] … split.count 3 but 2 parts found` (or upstream's wording naming the missing part); `coli doctor` `model.gguf.splits` **fail** |
| a part is duplicated under a second name | refused naming the duplicate `split.no` |
| `split.count` of one part disagrees | refused naming the part |
| a directory with two different split sets | refused naming both stems (never silently picks one) |

### One reader (housekeeping)

`ggufinfo.py` and `tools/gguf_reader.py` parse the same bytes: on every fixture (single, split,
synthetic `tiny_glm_dsa`, the mutated files both accept) the KV dictionaries, tensor names,
types, dims and offsets are **equal** (differential test, must pass before the consolidation).
After the consolidation one of them is the parser and the other a thin view: `ggufinfo`
exposes `GGUF_READER_IMPL` naming the module that parses (`tools/gguf_reader.py`), and
`tests/test_ggufinfo.py` + upstream's `tests/test_gguf_reader.py` both stay green. `coli gguf
inspect`'s output is byte-identical before and after on the fixtures.

### The GLM tokenizer from the GGUF (`tests/test_tok_gguf_glm`, FR-20)

`glm_tok_load` (from `tokenizer.ggml.*`, `pre = glm4`) and `tok_load` (from `tokenizer.json`
with the cl100k `Split` + `ByteLevel` pre-tokenizer) encode every line of `fixtures/tok_corpus.txt`
to **identical ids** and decode every id to the same piece; the control tokens
(`<|endoftext|>`, `<|user|>`, `<|assistant|>`, `[gMASK]`, `<sop>`) resolve to the same ids by
name; a GGUF whose `tokenizer.ggml.pre` is not GLM's (`qwen35`) is **refused**, not guessed
(`--expect-refusal`). `--encode <text>` prints the ids (the FR-32 gate for the GLM engine).

## Cases

Runner: `python3 07_Tests/IntegrationTest/run_gguf_hardening.py [--keep] [--no-build] [--seed N] [--mutations N]`.

| # | Case | Expected |
|---|---|---|
| 0 | fixtures | tiny Qwen F32 GGUF (single + `--split 2`), tiny GLM F32 GGUF (`--vocab 320`, tokenizer from `make_tiny_glm_tok.py`), `ref.json`s, `tests/test_gguf`, `tests/test_tok_gguf_glm` built |
| 1 | mutation sweep, readers | `N` mutations of the Qwen file: C reader and `ggufinfo` never crash/hang/foreign-raise; both accept ≥ 1; structural edits: same accept/refuse verdict in both |
| 2 | mutation sweep, engines | a subset (every 8th mutation + every structural edit): `./qwen36` exits 0 or refuses with a message; the GLM file likewise through `./colibri` |
| 3 | truncated payload | cut inside an expert tensor: `qwen36` refuses with `short read`, exit 1, no `Matching tokens` line; cut inside a dense tensor: refused at load; same for `colibri` |
| 4 | split sets | missing part, duplicated part, disagreeing `split.count`, two stems in one directory — refused by name in `gguf.h`, `ggufinfo` and `coli doctor` (`model.gguf.splits fail`) |
| 5 | one reader | differential equality on every fixture; `ggufinfo.GGUF_READER_IMPL` names `gguf_reader.py`; `coli gguf inspect` byte-identical to the pre-consolidation record (first run writes it to the tmp dir only; the contract is the differential) |
| 6 | GLM tokenizer | `tests/test_tok_gguf_glm tiny_glm.gguf tiny_glm_tok.json fixtures/tok_corpus.txt` → `all passed`; `--encode` prints one line per corpus line; the Qwen F32 GGUF (`pre = qwen35`) → `--expect-refusal` exits 0 |
| 7 | doctor on damage | `coli doctor --model <truncated>` reports `model.gguf.payload fail` with `--deep` and no traceback |

**Pass criterion:** `RESULT: ok (0 failures)` locally and in CI (`make check`'s Linux job runs it;
no torch needed).

## State

| Date | Result |
|---|---|
| 2026-10-14 | document, runner, `make_tiny_glm_tok.py`, `make_tiny_glm_hf.py --vocab`, `tests/test_tok_gguf_glm.c` written. Pre-implementation run: see the row below. |
| 2026-10-14 | pre-implementation run (`run_gguf_hardening.py --no-build`, seed 20261014, 160 mutations + 18 structural edits = 178 inputs): `RESULT: FAIL (4 failures)`, all phase-7 contracts — `coli doctor` reports every damaged file (truncated payload, missing split part) under `config.arguments` instead of a `model.gguf.*` check (cases 3, 4, 7), and `ggufinfo.GGUF_READER_IMPL` does not exist (case 5, the consolidation). **Already passing** (the hardening net the phases 1–6 left behind): the C reader and `ggufinfo` never crash, hang, refuse silently or raise a foreign exception on any of the 178 inputs, accept the same 53 and refuse the same 125, and agree on every structural edit; every structural defect but the unknown type id is refused by the readers (the reader reports unknown types by design, both engines refuse them by name); `qwen36` runs the pristine file and gives a named refusal on every mutated pick and every broken edit (38), `colibri` on 24 GLM mutations; a payload cut inside an expert or dense tensor is refused **at open** naming the tensor ("runs past the end of the file") in both engines — stronger than the "first read" the contract allowed, recorded as the behaviour; the four split-set rules are refused by name in both readers; the differential test finds `ggufinfo` and `tools/gguf_reader.py` equal on 15 files (ggufinfo pads `ne` to 4 dims — trimmed for the comparison; `gguf_reader` additionally refuses a non-UTF-8 tensor name a mutation produced, allowed on mutated files); the GLM tokenizer from the GGUF equals its JSON twin on the whole corpus (`tests/test_tok_gguf_glm`: `all passed`, `--encode`, a `qwen35` pre-tokenizer refused). Runner fixes during the run: the "offset beyond data" fixture is patched into the header bytes (the writer would otherwise pad a terabyte); an engine run is graded by its oracle line, not its exit code (the exit code grades the random reference's match). |
