# IntegrationTest

One test per architecture module, written **after** the architecture review and
**before** the module is implemented. Unit tests live next to the code in
`06_Code/c/tests/`; these tests exercise a module against its neighbours and
against real-world input.

| Module | Test | State |
|---|---|---|
| `gguf.h` + `ggufinfo.py` (phase 1) | [`gguf_reader.md`](gguf_reader.md) · runner `run_gguf_reader.py` | written after the fact (phase 1 pre-dates the process); passes |
| `gq.h` (phase 2) | [`gq_kernels.md`](gq_kernels.md) · runner `run_gq_kernels.py` · golden `fixtures/e0/` (gguf-py, generator `make_e0_golden.py`) | written 2026-10-05 before implementation; **passes 2026-10-06** (0 failures, all 6 cases) |
| `src.h` / `qwen35_names.h` / `gguf_xform.h` / `ts_cfg` / tokenizer ctor / `st2gguf.py` / registry arms (phase 3) | [`src_facade.md`](src_facade.md) · runner `run_src_facade.py` · torch-free fixture `make_tiny_qwen36_hf.py` · `fixtures/tok_corpus.txt` | written 2026-10-06 before implementation; **passes 2026-10-06** (0 failures, all 8 cases) |
| expert streaming: `Slot.kq` cache behaviour, slices, split sets, sidecar rule, startup line, on-demand embedding, serve path (phase 4) | [`expert_streaming.md`](expert_streaming.md) · runner `run_expert_streaming.py` (320-token byte-level tokenizer built by the runner) | written 2026-10-08 before implementation; **passes 2026-10-09** (0 failures, all 13 cases; container cases in the CI job) |
| CUDA expert tier on raw ggml blocks: block formats, `qt_init_gguf`/`qt_note_kq`, GGUF dense placement, planner VRAM arithmetic, engine with the fake backend (phase 5) | [`cuda_tier_kquant.md`](cuda_tier_kquant.md) · runner `run_cuda_tier_kquant.py` · C tests `tests/test_cuda_block_fmt_guard.c`, `tests/test_qwen36_tier_kq.c`, `tests/test_qwen36_tier_kq_engine.c` · device oracle `tests/test_gq_cuda.cu` + `tests/gq_ref.c` (`make cuda-test-gq`, owner) | written 2026-10-09 before implementation; pre-implementation run recorded in its state table |
