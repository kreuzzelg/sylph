# SystemTest

End-to-end tests, written after the architecture review and before the phase
that satisfies them. Each test names its inputs, the command, and the exact
pass criterion.

| Test | Phase | Criterion | State |
|---|---|---|---|
| [`inspect_real_gguf.md`](inspect_real_gguf.md) | 1 | `coli gguf inspect` + `coli doctor --deep` pass on public GLM-5.2 and Qwen3.6 GGUFs | passes (2026-10-05) |
| `lossless_oracle.md` | 3 | tiny Qwen3.6 oracle (`tools/make_qwen36_oracle.py`) from an F32 `qwen35moe` GGUF reproduces `ref_qwen36.json`; container oracle unchanged | to write before phase 3 |
| `equivalence.md` | 4 | E1–E3 vs llama.cpp / Ollama on the same real GGUF (spec §4) | to write before phase 4 |
| `real_model.md` | 4 | `coli chat/serve/web` from a Qwen3.6-35B-A3B `Q4_K_M` GGUF; A/B vs Ollama and vs the gs64 container on the owner's hardware | to write before phase 4 |
