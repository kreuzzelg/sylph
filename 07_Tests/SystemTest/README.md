# SystemTest

End-to-end tests, written after the architecture review and before the phase
that satisfies them. Each test names its inputs, the command, and the exact
pass criterion.

| Test | Phase | Criterion | State |
|---|---|---|---|
| [`inspect_real_gguf.md`](inspect_real_gguf.md) | 1 | `coli gguf inspect` + `coli doctor --deep` pass on public GLM-5.2 and Qwen3.6 GGUFs | passes (2026-10-05) |
| [`kernel_throughput.md`](kernel_throughput.md) | 2 | NFR-7: `Q4_K` GEMV and `gq_moe_run` within 1.5× of upstream's planar int4 per byte; `Q6_K`/`Q8_0` lm_head reported; two hosts, report in `08_Documents/kernels/` | dev container 2026-10-06: target met (`Q4_K` 1.27–1.30×, layer 1.17–1.45×); owner's host pending |
| [`lossless_oracle.md`](lossless_oracle.md) | 3 | upstream's torch-built tiny Qwen3.6 (`make_qwen36_tiny.py --ref-mode full`) from an F32 `qwen35moe` GGUF is token-exact (16/16 at cap 1/2/8), E1 subset vs torch log-probs via `PPL_DUMP`, container run unchanged; automated as the `gguf-oracle` CI job | **passes** (CI run 13, 2026-10-06): 16/16 at cap 1/2/8 for container and F32 GGUF, façade cross-check bit-exact, E1 mean \|ΔNLL\| 6.25e-8 / max 1.0e-6 (one print ulp), thresholds calibrated to 1e-6 / 1e-5; F16 and Q8_0 GGUF 16/16 reported; summary in `08_Documents/equivalence/` |
| [`equivalence.md`](equivalence.md) | 4 | E1–E3 vs llama.cpp / Ollama on the same real GGUF (spec §4), calibrated thresholds, one `make equivalence` entry point, CI subset on the tiny oracle (FR-34); prompts `fixtures/e3_prompts.txt` | harness implemented 2026-10-09 (`equivalence/`); **CI subset green** (run 21: E1 mean 1.8e-7, exact KL 6.2e-8, E2 equal, E3 16/16); owner run pending (needs llama.cpp, Ollama, the real file) |
| [`real_model.md`](real_model.md) | 4 | `coli chat/serve/web` from a Qwen3.6-35B-A3B `Q4_K_M` GGUF (functional, FR-29 hygiene, split set, tokenizer vs `llama-tokenize`); A/B vs Ollama and vs the gs64 container per `docs/benchmarking.md`, report in `08_Documents/benchmarks/` | written 2026-10-08 before implementation; owner's machine |
