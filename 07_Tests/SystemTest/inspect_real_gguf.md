# System test — inspect a real GGUF set (phase 1)

**Input:** `unsloth/GLM-5.2-GGUF`, folder `UD-Q4_K_XL` (11 parts). Only metadata is
needed: part 1 in full (9.4 MB), the first 2 MB of parts 2–11, then each part
extended sparsely to the size the Hugging Face tree API reports.

**Commands**

```sh
06_Code/c/coli gguf inspect <dir>
06_Code/c/coli doctor --model <dir> --gpu none --deep
06_Code/c/tests/test_gguf <dir> | wc -l
```

**Pass criteria**

- `inspect`: `architecture glm-dsa (engine: glm)`, 11 parts, 1809 tensors, type mix
  limited to `Q4_K Q5_K Q6_K Q8_0 F32`, `78 trunk + 1 nextn`, `mtp blk.78 … Q8_0`,
  `indexer 79 layers`, tokenizer `gpt2 · pre glm4 · 154880 tokens`.
- `doctor --deep`: every `model.gguf.*` check and `model.tokenizer` `pass`;
  `memory.ram` reflects the host (fails on hosts with < 21 GB free, by design).
- C dump: 1809 `T` lines, 11 `F` lines, under one second.

**Result 2026-10-05:** pass — see `../../08_Documents/inspection-glm52-ud-q4_k_xl-2026-10-05.md`.
