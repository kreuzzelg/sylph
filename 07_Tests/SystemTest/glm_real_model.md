# System test — GLM-5.2 from a public GGUF on the owner's machine (phase 6; feasibility first)

Written 2026-10-09, before phase 6. Specification §3 ("GLM-5.2 follows once the Qwen3.6
path is measured"), §8 items 4–6 applied to `glm-dsa`, §9 open point (d); inspection
`08_Documents/inspection-glm52-ud-q4_k_xl-2026-10-05.md`; `../IntegrationTest/glm_assembly.md`.

## Feasibility (read before planning a run)

| Fact (inspection, 2026-10-05) | Consequence |
|---|---|
| `UD-Q4_K_XL`: 11 parts, **467 GB**; experts 446 GB (22.81 MB each, 76 MoE blocks × 256) | the file must sit on a local disk with ≈ 470 GB free; experts stream from it (`cap` slots per layer) |
| resident dense set **21.0 GB** at `Q8_0` (every attention, shared-expert, indexer, embedding matrix) | a 32 GB host has ≈ 8 GB for expert cache + KV; a 16 GB host cannot run this quant (NFR-3 forbids re-quantizing at load) |
| `UD-Q4_K_M` / `UD-Q4_K_S` / `UD-Q3_K_XL` (unsloth) keep attention at 4–5 bit | the survey (`ud_survey.md`) measures their dense set and type mix; the owner picks the file the host can hold |
| the owner's reference for GLM is unknown (spec §9 d) | Ollama's GLM-5.2 tag, if any, decides the A/B; otherwise llama.cpp only |

**Decision needed from the owner** before this test is scheduled: whether GLM-5.2 stays on
the roadmap and which file/host it runs on. Everything else in this document is ready to
run once that is settled.

## Cases (same shape as `real_model.md` and `equivalence.md`, engine `colibri`)

| # | Case | Expected |
|---|---|---|
| R0 | `coli gguf inspect MODEL`, `coli doctor --model MODEL --deep`, `coli plan`, `coli info` | as phase 1 on the real file; `plan` prices 22.81 MB per expert and the dense set as stored; `info` names `colibri (glm)` and the sidecar dir |
| R1 | `coli chat --model MODEL` | `[GGUF] glm-dsa · 78 blocks (3 dense, 75 MoE) · 11 parts …`, `kv_b from attn_k_b/attn_v_b (k widened f16, ~1.0 GB)`, `indexer 78/78`, `nextn blk.78 eh_proj Q8_0 (8.50 bpw) loaded`; three coherent turns; nothing written beside the parts |
| R2 | tokenizer | `tests/test_tok_gguf MODEL <GLM tokenizer.json> fixtures/tok_corpus.txt --llama-tokenize …`: `all passed` (`pre = glm4`) |
| R3 | E1–E3 vs llama.cpp (and Ollama if a tag exists) | `run_equivalence.py --engine colibri --model MODEL --llama …` (harness gains `--engine`, the GLM serve/PPL contracts are the same wire format): thresholds calibrated per §4.3; report in `08_Documents/equivalence/` |
| R4 | MTP | `MTP=0` vs default: identical greedy ids (lossless), `[MTP] proposed/accepted` printed, tok/s both ways |
| R5 | A/B | `docs/benchmarking.md` protocol vs llama.cpp / Ollama on the same file; CPU, and the tier on the RTX 3070 if the dense set leaves room (it does not on 8 GB with a `Q8_0` dense set: say so) |
| R6 | container unchanged | the owner's GLM container (`convert_fp8_to_int4.py`) runs as before, same tok/s within noise |

## State

| Date | Result |
|---|---|
| 2026-10-09 | written; blocked on the owner's decision (spec §9 d) and on a host with the disk/RAM the chosen quant needs; the survey (`ud_survey.md`) supplies the per-quant dense-set sizes. |
