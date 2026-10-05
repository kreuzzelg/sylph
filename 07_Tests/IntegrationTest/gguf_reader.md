# Integration test — GGUF reader (`gguf.h` ⟷ `ggufinfo.py` ⟷ real files)

**Module under test:** `06_Code/c/gguf.h` and `06_Code/c/ggufinfo.py`.
**Inputs:** a GGUF file, a split set, or a directory; optional `COLI_MODEL_DIRS`.
**Outputs:** identical file/key/tensor tables from both readers; a passing
`coli doctor` report; refusals with the rule named.

## Cases

| # | Case | Expected |
|---|---|---|
| 1 | Synthetic demo file (alignment 32 and 64) and 3-part split set written by `tools/make_gguf_fixture.py` | C dump (`tests/test_gguf <path>`) equals the Python table: same parts (size, data offset, alignment), same keys (type, element type, length), same tensors (type, ne, file, absolute offset, byte size). |
| 2 | Tiny synthetic `glm-dsa` layout | `summarize()`: engine `glm`, 2 trunk + 1 nextn, expert bytes = gate+up+down slice sizes, MTP `Q8_0`, indexer layers, tokenizer facts. `coli doctor` all GGUF checks `pass`. |
| 3 | One malformed file per validation rule | both readers refuse with a message containing the rule's key word (`not a GGUF`, `version`, `limit`, `aligned`, `runs past`, `duplicate`, `power of two`, `nests deeper`, `not a multiple`). |
| 4 | Real model: `unsloth/GLM-5.2-GGUF` `UD-Q4_K_XL` metadata (part 1 complete, parts 2–11 header-only, sparse-extended) | 11 parts, 1809 tensors, arch `glm-dsa`, type mix inside the v1 set, MTP `Q8_0` at `blk.78`, 79 indexer layers, `coli doctor --deep` passes every `model.gguf.*` check. |

## Runner

```sh
python3 07_Tests/IntegrationTest/run_gguf_reader.py                 # cases 1–3 (self-contained)
python3 07_Tests/IntegrationTest/run_gguf_reader.py /path/to/gguf   # + case 4 on a real file/dir
```

Requires `make -C 06_Code/c tests/test_gguf` once (the runner builds it if missing).
