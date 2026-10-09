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
| architecture | `qwen35moe` (engine `qwen36`) and, since phase 6, `glm-dsa` (GLM-5 / 5.2, engine `colibri`, see below); anything else is refused by name. A Qwen NextN/MTP block is detected and reported (`nextn blk.L eh_proj <type> (<bpw> bpw) present, not used`), never read |
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
exists for fixtures and oracles, not as a quantizer. A snapshot with `mtp_num_hidden_layers = 1`
gets its NextN block (`blk.<L>.nextn.*` + `blk.<L>.*`, `--mtp-type` for `eh_proj`, `--no-mtp`
to leave it out); `--arch glm-dsa` is the GLM arm (above).

## GLM-5.2 (`glm-dsa`) in the `colibri` engine (phase 6)

`SNAP=<file.gguf> ./colibri …` or `coli chat/serve/run --model <file.gguf>` (the family is
resolved from `general.architecture`) runs a llama.cpp `glm-dsa` GGUF on the GLM engine, the
same way the Qwen engine runs `qwen35moe` files. Contract and cases:
`07_Tests/IntegrationTest/glm_assembly.md`; the torch oracle: `07_Tests/SystemTest/glm_lossless_oracle.md`.

```
[GGUF] glm-dsa · 78 blocks (3 dense, 75 MoE) · 11 parts · experts Q4_K/Q4_K/Q6_K 22.81 MB each × 256 × 75 = 420.1 GB
       · dense Q8_0×… F32×… 19.6 GB · kv_b from attn_k_b/attn_v_b (k widened from F16, 3.1 GB) · indexer 78/78 blocks
       · nextn blk.78 eh_proj Q8_0 (8.50 bpw) loaded · experts on CPU (gq) · sidecars ~/models/.coli-GLM-5.2-UD-Q4_K_XL/
```

| | |
|---|---|
| config | from the `glm-dsa.*` keys (architecture v1 §7.1); `expert_gating_func ≠ 2` and `expert_group_count ≠ 1` are refused by key, like the container path's `n_group` rule |
| names | `c/glm_names.h` ⇄ `tools/glm_tensor_kinds.py`; no value transforms (GLM stores plain norms) |
| MLA (FR-18) | llama.cpp writes `kv_b_proj` as the absorbed split `attn_k_b {qk_nope, kv_lora, H}` (per head transposed) + `attn_v_b {kv_lora, v_head, H}`; the engine rebuilds its `kv_b` from them in f32 (`glm_kv_b_from_split`, lossless; the line reports the widening cost). A file with `attn_kv_b` uses it directly; a block with neither is refused by name |
| dense matrices | f32/f16/bf16 decoded exactly; `Q8_0` and the K-quants stay the author's raw blocks (`QT.fmt = 16 + type`) and run on `gq.h` (`matmul_qt_ex`) — nothing is re-quantized, `dbits` is ignored |
| routed experts | three raw slices per expert into the LRU slab (`cap` slots per layer), `gq_matmul` per expert; `GGUF reads:` accounts every slice; `PIN`/`AUTOPIN`/`HOT` work (history in the sidecar directory) |
| DSA indexer | active when the file carries `indexer.*` tensors (every block of the public files; an explicit `attention.indexer.types` array wins, `DSA=0` disables) |
| NextN / MTP | the block inside `block_count` is loaded when its `eh_proj` has ≥ 8 bits per weight (`Q8_0` in the public files); below that it is skipped unless `MTP=1`; `MTP=0` always skips. The draft is verified by the trunk (upstream's `mtp_draft`), so greedy ids are identical with and without it; `[MTP] proposed N accepted M` at the end of a run |
| tokenizer | from `tokenizer.ggml.*` with `pre = glm4` (cl100k family); another `pre` is refused, not guessed; stop ids from `eos/eot/eom_token_id` and the control tokens by name |
| not available on a GGUF | `COLI_MMAP`, `URING`, `DIRECT`, the dual-SSD mirror and `TRUNK_RESIDENT_LAYERS` (safetensors features; ignored or refused with a message); the CUDA/Metal/Vulkan dense paths (the GGUF matrices run on the CPU kernels) |

Test files: `python3 tools/st2gguf.py <hf dir> --arch glm-dsa --out model.gguf [--type …] [--expert-type …]
[--kv-b split|fused|none] [--mtp-type …] [--no-mtp] [--gating-func N] [--expert-groups N]`
(`--arch` defaults to the snapshot's `model_type`). `07_Tests/IntegrationTest/make_tiny_glm_hf.py`
writes a torch-free tiny snapshot in the geometry of upstream's `tools/make_glm_oracle.py`.

## Upstream sync (colibrì main → the `06_Code` subtree)

`06_Code/` is a `git subtree` of upstream colibrì. Every sylph change lives beside upstream
code in the same tree, so a sync is a real merge with conflicts to resolve and re-verify.
The procedure is a script that says what will happen before anything is written
(`07_Tests/SystemTest/upstream_sync.md`):

```sh
python3 06_Code/c/tools/upstream_sync.py --check          # 1. read-only: base, upstream head, commits behind,
                                                          #    tags since, the conflict forecast (paths both sides changed)
python3 06_Code/c/tools/upstream_sync.py --trial          # 2. dry merge in a temporary clone: clean / conflicted files;
                                                          #    exit 0 / 3; the working tree is untouched
python3 06_Code/c/tools/upstream_sync.py --apply          # 3. the real `git subtree pull --prefix=06_Code upstream main`
                                                          #    (+ --squash on request); writes 06_Code/.upstream
```

4. **Resolve** conflicts by policy: in sylph-owned code (`gq.h`, `gq_i8.h`, `src.h`, `gguf.h`,
   `qwen35_names.h`, `glm_names.h`, the GGUF arms of `qwen36.c` / `colibri.c` / `qwen36_tier.c`
   / `backend_cuda.cu`, `st2gguf.py`, `ggufinfo.py`, the GGUF branches of `coli` / `doctor.py` /
   `family_registry.py` / `resource_plan.py`) keep sylph's arm and re-apply upstream's change
   around it; everywhere else take upstream. Never re-quantize or rename a GGUF contract to
   make a merge easier — the phase runners are the contract.
5. **Gates** before pushing: `make -C 06_Code/c check`, the `gguf-oracle` CI recipe, and the
   phase runners (`run_gguf_reader.py`, `run_gq_kernels.py`, `run_expert_streaming.py`,
   `run_cuda_tier_kquant.py`, `run_glm_assembly.py`).
6. **Commit message**: `06_Code: merge upstream colibrì main (<tag>, <sha>) into the subtree`
   (the tool proposes it; `--check` reads the base back from the newest such message or from
   `06_Code/.upstream`). Record the merged tag in the state table of `upstream_sync.md`.
