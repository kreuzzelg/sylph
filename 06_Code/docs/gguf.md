# GGUF models (sylph)

sylph is a fork of colibrì that reads **llama.cpp GGUF files** of the `qwen35moe`
architecture (Qwen3.5/3.6 MoE, e.g. `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`) in the `qwen36`
engine, with the weight bytes exactly as the quantizer wrote them. No conversion, no
`config.json`, no `tokenizer.json`: everything comes from the file's metadata.

## Run

```sh
coli gguf inspect ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf     # metadata only, instant
coli doctor --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --deep
coli plan   --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf    # RAM arithmetic, slots per layer
coli chat   --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf    # also serve / web / run
```

`--model` may be the file, the first part of a split set, or a directory holding the
parts. The engine prints one line that says what runs where:

```
[GGUF] qwen35moe · 40 blocks (10 attention) · 1 part · experts Q4_K/Q4_K/Q5_K 1.90 MB each × 256 × 40 = 19.6 GB
       · dense Q8_0×… F32×… 2.56 GB · token_embd Q8_0 on demand · output Q6_K gq_matmul · experts on CPU (gq_moe_run)
       · sidecars ~/models/.coli-Qwen3.6-35B-A3B-UD-Q4_K_M/
```

and, after a run, `GGUF reads: <slices> slices · <MB> MB · <MB/token> MB/token · <parts> touched`
next to the expert-cache hit rate.

## What is supported (v1)

| | |
|---|---|
| architecture | `qwen35moe`; NextN/MTP blocks are detected and skipped |
| tensor types | `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` (everything in the public `Q4_K_M` files); anything else is refused by name |
| routed experts | streamed from the file as raw ggml blocks into the LRU expert cache (`--cap` slots per layer), computed on the CPU by `gq_moe_run`; `PILOT`, `HOT`/`PIN`, pins work as for containers |
| dense matrices | `Q8_0` → lossless int8 plane + per-32 scales on upstream's kernels; K-quants → `gq_matmul`; f32/f16/bf16 → the container's int8-at-load path (`COLI_DENSE_I8=0` for exact f32) |
| tokenizer | from `tokenizer.ggml.*` (`pre = qwen35`); identical ids to Qwen's `tokenizer.json` on the test corpus |
| sidecars | `<dir>/.coli-<stem>/` — nothing is ever written beside the `.gguf` |
| GPU | `COLI_CUDA=1` on a CUDA build (`make qwen36 CUDA=1`): the hot experts go to VRAM as the raw ggml blocks they are (`Q8_0`/`Q4_K`/`Q5_K`/`Q6_K`, backend formats 16 + type), computed by kernels that reproduce the CPU reference bit for bit; the dense trunk is placed by the same auto-placer as for containers (`output`, DeltaNet projections, attention, shared expert, as stored; a `Q8_0` split is re-joined). `coli plan --gpu 0` prices the trunk and the expert budget (`VRAM … trunk + … hot tier · ~N experts`). A CPU-only build says `COLI_CUDA=1 ignored: built without CUDA`; a type without a kernel is refused by name. Measured on the RTX 3070: `07_Tests/SystemTest/gpu_rtx3070.md` |

Knobs: `COLI_GGUF_EMBED`, `PPL_DUMP`, `PPL_DUMP_FULL`, `QWEN_EXPERT_ACT=i8` (the opt-in int8-activation twin on the GGUF path; f32 by default) in [ENVIRONMENT.md](ENVIRONMENT.md); the tier's knobs (`COLI_CUDA`, `COLI_GPUS`, `CUDA_EXPERT_GB`, `COLI_PLACE`, `HEAT_FILE`) in [qwen36-cuda-tier.md](qwen36-cuda-tier.md).

## Equivalence

The numbers are checked, not assumed. The lossless gate runs in CI (an F32 GGUF of a tiny
Qwen3.6 reproduces the transformers forward pass to the last printed digit); the real-model
harness compares sylph with llama.cpp and Ollama on the **same file**:

```sh
make -C c equivalence MODEL=<gguf> LLAMA=<llama.cpp bin dir> OLLAMA=http://127.0.0.1:11434 TAG=qwen3.6:35b TEXT=wiki.test.raw
```

Levels: E1 teacher-forced log-probs (ΔNLL, KL, top-1), E2 perplexity (llama.cpp's 16 × 512
protocol), E3 greedy generation to the first near-tie; thresholds are 3 × llama.cpp's own
noise floor (thread counts, CPU vs CUDA). Protocol and results:
`07_Tests/SystemTest/equivalence.md`, `08_Documents/equivalence/`.

## Making test files

`python3 c/tools/st2gguf.py <hf snapshot> --out model.gguf [--type f32|f16|bf16|q8_0]
[--expert-type …] [--tokenizer tokenizer.json] [--split N]` writes a `qwen35moe` GGUF from a
Hugging Face snapshot with llama.cpp's converter transforms (standard library only); it
exists for fixtures and oracles, not as a quantizer.
