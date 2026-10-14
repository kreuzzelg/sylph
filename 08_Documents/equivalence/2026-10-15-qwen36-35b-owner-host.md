# Equivalence — Qwen3.6-35B-A3B on the owner's host, 2026-10-15

Protocol: `../../07_Tests/SystemTest/equivalence.md` (spec §4, FR-31…FR-35), first execution on the
owner's machine (the repository was built in a cloud container without GPU or model files).
Host: AMD Ryzen Threadripper PRO 3945WX 12-Cores, 94 GB RAM, RTX 3070 8 GB + Quadro RTX 4000 8 GB, NVMe.
References: llama.cpp `version: 0.6.0-dev (build 1, commit a0e6dc7)` (CUDA build, archs 75;86, built 2026-10-14) and
Ollama 0.40.3. sylph commit: the working tree of this session (see `04_Tasks/tasks.md`, phase 7b).

## Which file

The owner's Ollama tag `qwen3.6:35b` runs **Ollama's own conversion** (`blobs/sha256-f5ee307a…`, 23.9 GB,
1194 tensors with the vision tower; spec §9 a). Upstream llama.cpp refuses that file
(`qwen35moe.rope.dimension_sections has wrong array length; expected 4, got 3`), so llama.cpp can be the
reference only on a llama.cpp-converted file. Two runs therefore:

1. **unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`** (sha256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`): sylph vs llama.cpp
   (E1, E2, E3) and vs Ollama (E3, the same file registered as tag `qwen36-unsloth-ud-q4km`). 16 × 512 wikitext-2
   windows, 32 prompts × 128 tokens, `--cap 128`, 12 threads.
2. **Ollama's blob**: sylph vs Ollama, E3 only (8 prompts; no id-level gate without llama-tokenize).

## Results in one table (unsloth file)

| level | measured | ggml's own floor (llama.cpp CPU vs CUDA, same file) | gate (3 × floor, spec §4.3) | verdict |
|---|---|---|---|---|
| tokenizer gate | 297 193 ids identical to `llama-tokenize` | — | identical | **pass** (after the `unicode_cat.h` fix; 5 pieces differed before) |
| E1 mean \|ΔNLL\| | 0.0589 nat over 4080 positions | 0.0671 | ≤ 0.2013 | pass |
| E1 max \|ΔNLL\| | 1.958 | 2.460 | ≤ 7.379 | pass |
| E1 top-1 agreement | 0.9703 | 0.9667 | ≥ 0.9000 | pass |
| E1 exact KL (chunk 0, 255 positions, llama's 16-bit tail lumped) | 0.0820 mean / 0.259 max | 0.0035 | ≤ 0.0104 | **FAIL** |
| E1 coarsened KL (top-10 ∪ rest) | 0.00355 | — | — | reported |
| E2 PPL | llama.cpp 7.1113 · sylph 7.0946 (sylph -0.24 %) | 1.83e-04 rel | ≤ 5.48e-04 | **FAIL** on the relative gate; sign test p = 0.45 (no systematic direction, 6 of 16 chunks sylph worse) |
| E3 vs llama-server | 21 of 32 prompts identical to 128 tokens or to a near-tie (< 0.05 nat) | — | ≥ 31 of 32 | **FAIL** |
| E3 vs Ollama (same file) | 24 of 32 | — | ≥ 31 of 32 | **FAIL** |
| E3 sylph-i8 vs sylph-f32 (the engine against its own int8-activation twin) | 22 of 32 | — | — | calibration: the same spread as against ggml |

Deliberate differences (FR-35): `QWEN_EXPERT_ACT=i8` E1 mean 0.0632 / max 1.969, PPL 7.1169
(+0.08 % vs llama.cpp). On the 2-window smoke run (same file, same reference) the other
knobs measured: `COLI_GGUF_RAW=0` (exact f32 decode of every quantized dense matrix) E1 mean 0.04046 vs default
0.04045 — **the raw-block dense kernels are not the source of any difference**; `COLI_GGUF_EMBED=0` identical;
`COLI_DENSE_I8=1` (upstream's int8-at-load of the float dense tensors) E1 mean 0.0518, PPL 5.7775 — worse, hence
off by default on a GGUF source since this session.

For R1/R2: the owner's #1370 numbers on this protocol were Ollama Q4_K_M **7.147** and colibrì gs64 int4 **7.325**;
here llama.cpp on the unsloth file prints **7.1113** and sylph on the same bytes **7.0946**.

## Reading

- **At the target-token level sylph is inside ggml's own spread.** Mean and max |ΔNLL| and top-1 agreement against
  llama.cpp-CPU are all better than llama.cpp-CPU vs llama.cpp-CUDA on the same file; the PPL difference (−0.24 %)
  has no systematic sign across chunks.
- **At the distribution level sylph differs systematically from ggml**: exact KL 0.082 nat against ggml's internal
  0.0035. The dense kernels, the embedding path and the Q8_0 split were excluded as causes (f32 A/B identical); the
  phase-3 torch oracle says the engine reproduces HF transformers to 1e-7 on a tiny model. The remaining structural
  difference is that **ggml quantizes the activations of every quantized matmul to 8 bits (Q8_K / q8_1)** while sylph
  computes them in f32; both ggml backends share that rounding, so the CPU-vs-CUDA floor does not measure it — which
  is why the KL and E2 gates calibrated on that floor fail while every target-level number passes. sylph's own int8
  twin (per-32 scales, independent rounding) lands at the same distance from ggml (0.063 vs 0.059) and from
  sylph-f32 (E3 22/32), as independent 8-bit noise would.
- **E3 at 128 greedy tokens is dominated by this noise**: 21/32 and 24/32 prompts stay identical or diverge at a
  near-tie; the engine against its own int8 twin gives 22/32. The ≥ 31/32 criterion of the spec assumes an agreement
  ggml does not have with itself across backends.
- **Decision for the owner (spec §4.3, R4)**: either the floor gains a sample that varies ggml's activation precision
  (none exists in llama.cpp) and the KL/E2 gates are set from sylph-i8 vs sylph-f32 (0.06 nat mean, KL to be measured),
  or sylph's exact activations are accepted as the one *deliberate difference* in its favour, with the numbers above
  as its record. Nothing found in this session points at an error in sylph's GGUF path; the two real bugs it did find
  (tokenizer classes, CUDA group path) are fixed and pinned.

## Ollama's own blob (E3, 8 prompts, Ollama greedy without its default penalties)

5 of 8 prompts identical to the end or near-tie; the three divergences happen at Ollama top-2 margins of
0.055, 0.28 and 0.39 nat with the two candidates swapped by ≈ 0.1–0.3 nat between the engines (e.g. ` Wasser` −0.547 vs
` du` −0.933 in Ollama, ` du` −0.628 vs ` Wasser` −0.818 in sylph). With Ollama's defaults (`repeat_penalty 1.1`,
`top_k 40`, `top_p 0.9`) every prompt diverged within 20–250 characters — the harness now switches them off.

## Raw reports

The harness's own reports follow verbatim (`run_equivalence.py`; raw dumps under the run directories on the host).

---

### unsloth file, 16 windows, 32 prompts

# Equivalence report — Qwen3.6-35B-A3B-UD-Q4_K_M.gguf — sylph vs llama.cpp vs Ollama

## Setup

- model: `/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` sha256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`
- qwen35moe · 40 blocks · 1 part(s) · 733 tensors · 22.13 GB · {'Q4_K': {'bytes': 12079595520, 'tensors': 80}, 'Q5_K': {'bytes': 6828326912, 'tensors': 37}, 'Q8_0': {'bytes': 2033254400, 'tensors': 251}, 'Q6_K': {'bytes': 1077780480, 'tensors': 4}, 'F32': {'bytes': 104581632, 'tensors': 361}}
- llama.cpp: version: 0.6.0-dev (build 1, commit a0e6dc7)
- ollama: 0.40.3
- ollama_show: {'from': '/usr/share/ollama/.ollama/models/blobs/sha256-ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61', 'details': {'parent_model': 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', 'format': 'gguf', 'family': 'qwen35moe', 'families': ['qwen35moe'], 'parameter_size': '34.7B', 'quantization_level': 'Q4_K_M', 'runner': 'llamacpp'}, 'parameters': None}
- platform: Linux-6.8.0-142-generic-x86_64-with-glibc2.39
- machine: x86_64
- cpus: 24
- python: 3.12.9
- cpu: AMD Ryzen Threadripper PRO 3945WX 12-Cores
- sylph cap: 128 · threads: 12 · mode: real

## Tokenizer gate

- pass: 297193 ids compared

## Noise floor and thresholds

- floor sample -t 1 vs -t 12: mean |ΔNLL| 0, max 0, top-1 disagreement 0, KL 0; E2 |ΔPPL|/PPL 0
- floor sample CPU vs CUDA: mean |ΔNLL| 0.0671, max 2.46, top-1 disagreement 0.03333, KL 0.003473; E2 |ΔPPL|/PPL 1.827e-04
- floor E1 (max over samples): mean |ΔNLL| 0.0671, max 2.46, top-1 disagreement 0.03333, KL 0.003473; E2 |ΔPPL|/PPL 1.827e-04 (-t 1 vs -t 12, CPU vs CUDA)
- thresholds (calibrated): E1 mean ≤ 0.2013, max ≤ 7.379, top-1 ≥ 0.9, KL ≤ 0.01042; E2 |ΔPPL|/PPL ≤ 5.480e-04

## E1 — teacher-forced log-probs

| positions | mean \|ΔNLL\| | max \|ΔNLL\| | top-1 | top-5 | KL exact (mean / max, n) | KL coarsened mean | verdict |
|---|---|---|---|---|---|---|---|
| 4080 | 0.05889 | 1.958 | 0.9703 | 0.7966 | 0.08197 / 0.2591 (255) | 0.003555 | **FAIL** |

Worst positions: pos (6, 474) Δ 1.958 (ref -16.1576, cand -18.1155), pos (2, 398) Δ 1.861 (ref -5.4915, cand -3.6306), pos (10, 291) Δ 1.477 (ref -7.8052, cand -6.3285), pos (2, 338) Δ 1.38 (ref -16.1633, cand -17.5434), pos (12, 283) Δ 1.336 (ref -11.0459, cand -9.7102)

## E2 — perplexity

| windows | PPL reference | PPL sylph | \|ΔPPL\|/PPL | chunks sylph worse | sign test p | verdict |
|---|---|---|---|---|---|---|
| 16 | 7.1113 | 7.0946 | 0.00235 | 6 of 16 | 0.454 | **FAIL** |

## E3 — greedy generation

### vs llama-server: 21 of 32 prompts identical or near-tie — **FAIL**

| prompt | compared by | prefix | ref len | sylph len | margin at divergence | verdict |
|---|---|---|---|---|---|---|
| 0 | ids | 13 | 128 | 128 | 0.006588 | near-tie |
| 1 | ids | 0 | 128 | 128 | 0.308 | diverged |
| 2 | ids | 104 | 128 | 128 | 0.1719 | diverged |
| 3 | ids | 128 | 128 | 128 | — | identical |
| 4 | ids | 26 | 128 | 128 | 0.009635 | near-tie |
| 5 | ids | 34 | 128 | 128 | 0.09246 | diverged |
| 6 | ids | 72 | 128 | 128 | 0.09013 | diverged |
| 7 | ids | 128 | 128 | 128 | — | identical |
| 8 | ids | 128 | 128 | 128 | — | identical |
| 9 | ids | 128 | 128 | 128 | — | identical |
| 10 | ids | 120 | 128 | 128 | 0.05484 | diverged |
| 11 | ids | 128 | 128 | 128 | — | identical |
| 12 | ids | 62 | 128 | 128 | 0.02678 | near-tie |
| 13 | ids | 128 | 128 | 128 | — | identical |
| 14 | ids | 14 | 15 | 14 | — | identical |
| 15 | ids | 12 | 128 | 128 | 0.02689 | near-tie |
| 16 | ids | 29 | 102 | 104 | 0.274 | diverged |
| 17 | ids | 128 | 128 | 128 | — | identical |
| 18 | ids | 128 | 128 | 128 | — | identical |
| 19 | ids | 128 | 128 | 128 | — | identical |
| 20 | ids | 11 | 128 | 128 | 0.1141 | diverged |
| 21 | ids | 79 | 128 | 128 | 0.163 | diverged |
| 22 | ids | 13 | 128 | 128 | 0.00923 | near-tie |
| 23 | ids | 1 | 128 | 128 | 0.06845 | diverged |
| 24 | ids | 57 | 58 | 57 | — | identical |
| 25 | ids | 5 | 8 | 56 | 0.008638 | near-tie |
| 26 | ids | 32 | 128 | 128 | 0.001152 | near-tie |
| 27 | ids | 128 | 128 | 128 | — | identical |
| 28 | ids | 118 | 128 | 128 | 0.08256 | diverged |
| 29 | ids | 8 | 128 | 128 | 0.08922 | diverged |
| 30 | ids | 49 | 128 | 128 | 0.03003 | near-tie |
| 31 | ids | 107 | 108 | 107 | — | identical |

### vs ollama: 24 of 32 prompts identical or near-tie — **FAIL**

| prompt | compared by | prefix | ref len | sylph len | margin at divergence | verdict |
|---|---|---|---|---|---|---|
| 0 | text | 80 | 599 | 599 | 0.04538 | near-tie |
| 1 | text | 0 | 486 | 442 | 0.02502 | near-tie |
| 2 | text | 220 | 473 | 473 | 2.366e-04 | near-tie |
| 3 | text | 531 | 531 | 531 | — | identical |
| 4 | text | 177 | 538 | 537 | 0.1131 | diverged |
| 5 | text | 152 | 554 | 556 | 1.498 | diverged |
| 6 | text | 113 | 387 | 398 | 0.01616 | near-tie |
| 7 | text | 376 | 376 | 376 | — | identical |
| 8 | text | 226 | 226 | 226 | — | identical |
| 9 | text | 622 | 622 | 622 | — | identical |
| 10 | text | 146 | 433 | 379 | 0.002525 | near-tie |
| 11 | text | 343 | 343 | 343 | — | identical |
| 12 | text | 267 | 498 | 503 | 0.03127 | near-tie |
| 13 | text | 575 | 575 | 575 | — | identical |
| 14 | text | 65 | 65 | 65 | — | identical |
| 15 | text | 352 | 574 | 537 | 0.004158 | near-tie |
| 16 | text | 140 | 582 | 542 | 0.2026 | diverged |
| 17 | text | 742 | 742 | 742 | — | identical |
| 18 | text | 440 | 440 | 440 | — | identical |
| 19 | text | 460 | 460 | 460 | — | identical |
| 20 | text | 14 | 209 | 208 | 0.0874 | diverged |
| 21 | text | 195 | 475 | 446 | 0.07013 | diverged |
| 22 | text | 32 | 571 | 528 | 0.05256 | diverged |
| 23 | text | 212 | 572 | 571 | 0.01493 | near-tie |
| 24 | text | 5 | 675 | 261 | 0.04082 | near-tie |
| 25 | text | 21 | 29 | 271 | 0.1593 | diverged |
| 26 | text | 153 | 566 | 630 | 0.1673 | diverged |
| 27 | text | 591 | 591 | 591 | — | identical |
| 28 | text | 492 | 523 | 521 | 0.04457 | near-tie |
| 29 | text | 0 | 0 | 541 | — | identical |
| 30 | text | 0 | 0 | 510 | — | identical |
| 31 | text | 0 | 0 | 507 | — | identical |

## Deliberate differences (FR-35)

- `QWEN_EXPERT_ACT=i8`: E1 mean 0.06317 / max 1.969, E2 PPL 7.117 (7.869e-04 rel), E3 vs default sylph 22/32 prompts to the first near-tie

## Verdict

- E1: **FAIL**
- E2: **FAIL**
- E3: **FAIL**

## Files

- `e3_llama.jsonl` 1191174 bytes
- `e3_ollama.jsonl` 1364689 bytes
- `e3_sylph.jsonl` 829253 bytes
- `e3_sylph_QWEN_EXPERT_ACT_i8.jsonl` 820822 bytes
- `llama/cuda.chunk0.full` 253286451 bytes
- `llama/cuda.chunk0.tsv` 46305 bytes
- `llama/cuda.chunk1.tsv` 46348 bytes
- `llama/cuda.chunk10.tsv` 46517 bytes
- `llama/cuda.chunk11.tsv` 46403 bytes
- `llama/cuda.chunk12.tsv` 46393 bytes
- `llama/cuda.chunk13.tsv` 46648 bytes
- `llama/cuda.chunk14.tsv` 46119 bytes
- `llama/cuda.chunk15.tsv` 46453 bytes
- `llama/cuda.chunk2.tsv` 46430 bytes
- `llama/cuda.chunk3.tsv` 46546 bytes
- `llama/cuda.chunk4.tsv` 45928 bytes
- `llama/cuda.chunk5.tsv` 46458 bytes
- `llama/cuda.chunk6.tsv` 46377 bytes
- `llama/cuda.chunk7.tsv` 46435 bytes
- `llama/cuda.chunk8.tsv` 46463 bytes
- `llama/cuda.chunk9.tsv` 46585 bytes
- `llama/cuda.kld` 2026356628 bytes
- `llama/cuda.log` 1135 bytes
- `llama/t1.chunk0.full` 253286451 bytes
- `llama/t1.chunk0.tsv` 46343 bytes
- `llama/t1.chunk1.tsv` 46345 bytes
- `llama/t1.chunk10.tsv` 46515 bytes
- `llama/t1.chunk11.tsv` 46415 bytes
- `llama/t1.chunk12.tsv` 46432 bytes
- `llama/t1.chunk13.tsv` 46684 bytes
- `llama/t1.chunk14.tsv` 46112 bytes
- `llama/t1.chunk15.tsv` 46436 bytes
- `llama/t1.chunk2.tsv` 46465 bytes
- `llama/t1.chunk3.tsv` 46546 bytes
- `llama/t1.chunk4.tsv` 45909 bytes
- `llama/t1.chunk5.tsv` 46457 bytes
- `llama/t1.chunk6.tsv` 46363 bytes
- `llama/t1.chunk7.tsv` 46432 bytes
- `llama/t1.chunk8.tsv` 46491 bytes
- `llama/t1.chunk9.tsv` 46545 bytes
- `llama/t1.kld` 2026356628 bytes
- `llama/t1.log` 990 bytes
- `llama/tN.chunk0.full` 253286451 bytes
- `llama/tN.chunk0.tsv` 46343 bytes
- `llama/tN.chunk1.tsv` 46345 bytes
- `llama/tN.chunk10.tsv` 46515 bytes
- `llama/tN.chunk11.tsv` 46415 bytes
- `llama/tN.chunk12.tsv` 46432 bytes
- `llama/tN.chunk13.tsv` 46684 bytes
- `llama/tN.chunk14.tsv` 46112 bytes
- `llama/tN.chunk15.tsv` 46436 bytes
- `llama/tN.chunk2.tsv` 46465 bytes
- `llama/tN.chunk3.tsv` 46546 bytes
- `llama/tN.chunk4.tsv` 45909 bytes
- `llama/tN.chunk5.tsv` 46457 bytes
- `llama/tN.chunk6.tsv` 46363 bytes
- `llama/tN.chunk7.tsv` 46432 bytes
- `llama/tN.chunk8.tsv` 46491 bytes
- `llama/tN.chunk9.tsv` 46545 bytes
- `llama/tN.kld` 2026356628 bytes
- `llama/tN.log` 990 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_0.tsv` 27163 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_1.tsv` 27134 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_10.tsv` 27208 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_11.tsv` 27149 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_12.tsv` 27201 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_13.tsv` 27276 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_14.tsv` 26995 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_15.tsv` 27138 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_2.tsv` 27259 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_3.tsv` 27246 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_4.tsv` 26924 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_5.tsv` 27184 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_6.tsv` 27107 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_7.tsv` 27187 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_8.tsv` 27193 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_9.tsv` 27227 bytes
- `sylph/default/win_0.full` 253286451 bytes
- `sylph/default/win_0.tsv` 27165 bytes
- `sylph/default/win_1.tsv` 27140 bytes
- `sylph/default/win_10.tsv` 27227 bytes
- `sylph/default/win_11.tsv` 27154 bytes
- `sylph/default/win_12.tsv` 27207 bytes
- `sylph/default/win_13.tsv` 27284 bytes
- `sylph/default/win_14.tsv` 26988 bytes
- `sylph/default/win_15.tsv` 27144 bytes
- `sylph/default/win_2.tsv` 27265 bytes
- `sylph/default/win_3.tsv` 27240 bytes
- `sylph/default/win_4.tsv` 26934 bytes
- `sylph/default/win_5.tsv` 27175 bytes
- `sylph/default/win_6.tsv` 27112 bytes
- `sylph/default/win_7.tsv` 27190 bytes
- `sylph/default/win_8.tsv` 27191 bytes
- `sylph/default/win_9.tsv` 27235 bytes


---

### Ollama's blob, E3 only, 8 prompts

# Equivalence report — sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d — sylph vs llama.cpp vs Ollama

## Setup

- model: `/usr/share/ollama/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` sha256 `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`
- qwen35moe · 40 blocks · 1 part(s) · 1194 tensors · 23.94 GB · {'Q4_K': {'bytes': 13451378688, 'tensors': 349}, 'Q6_K': {'bytes': 9490145280, 'tensors': 94}, 'F32': {'bytes': 90665472, 'tensors': 269}, 'F16': {'bytes': 167936, 'tensors': 41}}
- llama.cpp: not given
- ollama: 0.40.3
- ollama_show: {'from': '/usr/share/ollama/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen35moe', 'families': ['qwen35moe'], 'parameter_size': '36.0B', 'quantization_level': 'Q4_K_M'}, 'parameters': 'top_k                          20\ntop_p                          0.95\nmin_p                          0\npresence_penalty               1.5\nrepeat_penalty                 1\ntemperature                    1'}
- platform: Linux-6.8.0-142-generic-x86_64-with-glibc2.39
- machine: x86_64
- cpus: 24
- python: 3.12.9
- cpu: AMD Ryzen Threadripper PRO 3945WX 12-Cores
- sylph cap: 128 · threads: 12 · mode: real

## Tokenizer gate

- FAIL: None ids compared

## Noise floor and thresholds

- no floor measured (proposal values of spec §4.1 used)
- thresholds (proposal): E1 mean ≤ 1.000e-04, max ≤ 0.001, top-1 ≥ 0.995, KL ≤ 0.001; E2 |ΔPPL|/PPL ≤ 0.003

## E3 — greedy generation

### vs ollama: 5 of 8 prompts identical or near-tie — **FAIL**

| prompt | compared by | prefix | ref len | sylph len | margin at divergence | verdict |
|---|---|---|---|---|---|---|
| 0 | text | 525 | 525 | 525 | — | identical |
| 1 | text | 239 | 395 | 368 | 0.05505 | diverged |
| 2 | text | 361 | 471 | 474 | 0.2792 | diverged |
| 3 | text | 522 | 522 | 522 | — | identical |
| 4 | text | 541 | 541 | 541 | — | identical |
| 5 | text | 36 | 568 | 464 | 0.3859 | diverged |
| 6 | text | 113 | 377 | 277 | 0.01753 | near-tie |
| 7 | text | 337 | 337 | 337 | — | identical |

## Verdict

- E3: **FAIL**

## Files

- `e3_ollama.jsonl` 340178 bytes
- `e3_sylph.jsonl` 207724 bytes


---

### unsloth file, 2-window smoke run with four deviations

# Equivalence report — Qwen3.6-35B-A3B-UD-Q4_K_M.gguf — sylph vs llama.cpp

## Setup

- model: `/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` sha256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`
- qwen35moe · 40 blocks · 1 part(s) · 733 tensors · 22.13 GB · {'Q4_K': {'bytes': 12079595520, 'tensors': 80}, 'Q5_K': {'bytes': 6828326912, 'tensors': 37}, 'Q8_0': {'bytes': 2033254400, 'tensors': 251}, 'Q6_K': {'bytes': 1077780480, 'tensors': 4}, 'F32': {'bytes': 104581632, 'tensors': 361}}
- llama.cpp: version: 0.6.0-dev (build 1, commit a0e6dc7)
- platform: Linux-6.8.0-142-generic-x86_64-with-glibc2.39
- machine: x86_64
- cpus: 24
- python: 3.12.9
- cpu: AMD Ryzen Threadripper PRO 3945WX 12-Cores
- sylph cap: 64 · threads: 12 · mode: real

## Tokenizer gate

- pass: 297193 ids compared

## Noise floor and thresholds

- floor sample -t 1 vs -t 12: mean |ΔNLL| 0, max 0, top-1 disagreement 0, KL 0; E2 |ΔPPL|/PPL 0
- floor sample CPU vs CUDA: mean |ΔNLL| 0.05004, max 0.7546, top-1 disagreement 0.01765, KL 0.003473; E2 |ΔPPL|/PPL 0.004468
- floor E1 (max over samples): mean |ΔNLL| 0.05004, max 0.7546, top-1 disagreement 0.01765, KL 0.003473; E2 |ΔPPL|/PPL 0.004468 (-t 1 vs -t 12, CPU vs CUDA)
- thresholds (calibrated): E1 mean ≤ 0.1501, max ≤ 2.264, top-1 ≥ 0.9471, KL ≤ 0.01042; E2 |ΔPPL|/PPL ≤ 0.0134

## E1 — teacher-forced log-probs

| positions | mean \|ΔNLL\| | max \|ΔNLL\| | top-1 | top-5 | KL exact (mean / max, n) | KL coarsened mean | verdict |
|---|---|---|---|---|---|---|---|
| 510 | 0.04045 | 0.5734 | 0.9725 | 0.8196 | 0.08197 / 0.2591 (255) | 0.001956 | **FAIL** |

Worst positions: pos (0, 322) Δ 0.5734 (ref -2.8279, cand -3.4013), pos (0, 306) Δ 0.4933 (ref -1.9372, cand -2.4304), pos (0, 321) Δ 0.4187 (ref -0.9495, cand -1.3682), pos (0, 323) Δ 0.4129 (ref -2.8188, cand -2.4058), pos (1, 439) Δ 0.3624 (ref -8.4475, cand -8.8100)

## E2 — perplexity

| windows | PPL reference | PPL sylph | \|ΔPPL\|/PPL | chunks sylph worse | sign test p | verdict |
|---|---|---|---|---|---|---|
| 2 | 5.7301 | 5.7341 | 6.924e-04 | 1 of 2 | 1.000 | **pass** |

## Deliberate differences (FR-35)

- `QWEN_EXPERT_ACT=i8`: E1 mean 0.04692 / max 1.242, E2 PPL 5.747 (0.003017 rel)
- `COLI_GGUF_RAW=0`: E1 mean 0.04046 / max 0.5734, E2 PPL 5.734 (7.482e-04 rel)
- `COLI_GGUF_EMBED=0`: E1 mean 0.04045 / max 0.5734, E2 PPL 5.734 (6.924e-04 rel)
- `COLI_DENSE_I8=1`: E1 mean 0.05181 / max 1.053, E2 PPL 5.778 (0.008265 rel)

## Verdict

- E1: **FAIL**
- E2: **pass**

## Files

- `llama/cuda.chunk0.full` 253286451 bytes
- `llama/cuda.chunk0.tsv` 46306 bytes
- `llama/cuda.chunk1.tsv` 46349 bytes
- `llama/cuda.kld` 253294596 bytes
- `llama/cuda.log` 988 bytes
- `llama/kvf32.chunk0.full` 253286451 bytes
- `llama/kvf32.chunk0.tsv` 46318 bytes
- `llama/kvf32.chunk1.tsv` 46320 bytes
- `llama/kvf32.kld` 253294596 bytes
- `llama/kvf32.log` 818 bytes
- `llama/t1.chunk0.full` 253286451 bytes
- `llama/t1.chunk0.tsv` 46344 bytes
- `llama/t1.chunk1.tsv` 46346 bytes
- `llama/t1.kld` 253294596 bytes
- `llama/t1.log` 841 bytes
- `llama/tN.chunk0.full` 253286451 bytes
- `llama/tN.chunk0.tsv` 46344 bytes
- `llama/tN.chunk1.tsv` 46346 bytes
- `llama/tN.kld` 253294596 bytes
- `llama/tN.log` 843 bytes
- `sylph/COLI_DENSE_I8_1/win_0.tsv` 27152 bytes
- `sylph/COLI_DENSE_I8_1/win_1.tsv` 27122 bytes
- `sylph/COLI_GGUF_EMBED_0/win_0.tsv` 27165 bytes
- `sylph/COLI_GGUF_EMBED_0/win_1.tsv` 27140 bytes
- `sylph/COLI_GGUF_RAW_0/win_0.tsv` 27149 bytes
- `sylph/COLI_GGUF_RAW_0/win_1.tsv` 27124 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_0.tsv` 27163 bytes
- `sylph/QWEN_EXPERT_ACT_i8/win_1.tsv` 27134 bytes
- `sylph/default/win_0.full` 253286451 bytes
- `sylph/default/win_0.tsv` 27165 bytes
- `sylph/default/win_1.tsv` 27140 bytes
