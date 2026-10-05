# SystemTest

End-to-end tests, written after the architecture review and before the phase
that satisfies them. Each test names its inputs, the command, and the exact
pass criterion.

| Test | Phase | Criterion | State |
|---|---|---|---|
| [`inspect_real_gguf.md`](inspect_real_gguf.md) | 1 | `coli gguf inspect` + `coli doctor --deep` pass on a public GLM-5.2 GGUF set | passes (2026-10-05) |
| `lossless_oracle.md` | 3 | `SNAP=glm_tiny.gguf TF=1 ./colibri 64 16 16` → 32/32 teacher-forced, 20/20 greedy; safetensors oracle unchanged | to write before phase 3 |
| `real_model.md` | 4 | `coli chat/serve/web` from `UD-Q4_K_XL` on Linux + macOS; A/B vs gs64 container published | to write before phase 4 |
