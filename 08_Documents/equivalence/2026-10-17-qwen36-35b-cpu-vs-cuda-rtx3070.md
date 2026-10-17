# sylph-CPU vs sylph-CUDA on the RTX 3070 — G3 of `gpu_rtx3070.md`, 2026-10-17

Owner's host (Threadripper PRO 3945WX, RTX 3070 = CUDA device 0), unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
(sha256 `ac0e2c11…`), `qwen36` built with `CUDA=1` at commit 13efd861 (phase 8 kernels, the chain init race fixed),
12 threads, cap 256. The harness (`run_equivalence.py`, 16 wikitext-2 windows of 512 tokens, 255 scored each,
32 E3 prompts at 128 greedy tokens) ran sylph's default CPU path and two deviation arms of the same binary:

- **chain**: `COLI_CUDA=1,COLI_GPUS=0,COLI_PLACE=auto,COLI_TRUNK_PROBE=0,CUDA_EXPERT_GB=auto,COLI_CUDA_CHAIN=1` — the
  owner's fast configuration (53 tok/s): every layer on the device, the resident experts through the raw-block
  kernels, DeltaNet and attention through the chain's own kernels;
- **tier-only**: the same without `COLI_CUDA_CHAIN` — the resident experts and the dense trunk on the device,
  DeltaNet and attention on the CPU.

The harness gained two things for this run: a deviation arm may set several variables (`--deviation K=V,K2=V2`),
and every arm is compared **against sylph's own default run** (`vs_sylph` in `report.json`, the "against default
sylph" clause in `report.md`) and not only against the llama.cpp reference. Command:

```sh
python3 07_Tests/SystemTest/equivalence/run_equivalence.py --model <unsloth file> --llama llama.cpp/build/bin \
  --text wiki.test.raw --chunks 16 --cap 256 --threads 12 --levels E1,E2,E3 --out equiv_unsloth_full \
  --deviation "COLI_CUDA=1,COLI_GPUS=0,COLI_PLACE=auto,COLI_TRUNK_PROBE=0,CUDA_EXPERT_GB=auto,COLI_CUDA_CHAIN=1" \
  --deviation "COLI_CUDA=1,COLI_GPUS=0,COLI_PLACE=auto,COLI_TRUNK_PROBE=0,CUDA_EXPERT_GB=auto"
```

## Results

| arm, against sylph-CPU (same binary) | E1 mean \|ΔNLL\| | E1 max | top-1 agreement | \|ΔPPL\|/PPL | E3 identical to 128 tokens |
|---|---|---|---|---|---|
| G3 bound as written (2026-10-09, for the tier) | ≤ 1e-5 | ≤ 1e-4 | 100 % | ≤ 1e-5 | ≥ 31/32 |
| **chain** | 2.15e-4 | 0.081 | 99.975 % (1 of 4080) | 1.2e-5 | **32/32** |
| **tier-only** | 0.000137 | 0.0813 | 99.975 % (1 of 4080) | 1.54e-05 | **32/32** |

For context, the same run's sylph-CPU against llama.cpp-CPU (G3 a, "unchanged by this phase"): E1 mean 0.0590 /
max 1.954 / top-1 97.0 %, exact KL 0.082; E2 PPL 7.0958 vs 7.1113 (0.22 %, sign test p 0.45); E3 21/32 against
llama-server — the 2026-10-15 picture (0.0589 / 1.958 / 97.0 %; 7.0946; 21/32) within a hair: the CPU baseline moved
by 1.7e-4 in PPL between the two dates (cap 128 → 256 and the colibrì `dev` merge lie in between; not pursued).

**Where the chain's differences are.** 13 of the 16 windows agree with the CPU to the dump's precision
(mean |ΔNLL| 1e-6, max 0.0 at six decimals). Three windows (4, 7, 14) carry sporadic positions with larger
deltas (window maxima 0.0085, 0.081, 0.081; window means 1.7e-4, 1.2e-3, 2.0e-3) from their first scored position
on, not a drift that grows from zero — the signature of the router choosing a different expert at a near-tie once
the device's DeltaNet/attention arithmetic (the chain's own kernels, a different summation order than the CPU's)
moves a logit by 1e-6: a discrete event that colours the positions after it through the recurrent state. The
raw-block expert kernels themselves are bit-exact (`make cuda-test-gq`; `QT_VERIFY=1` found 1 370 of 1 370 device
expert results equal to `gq_dot_row_ref` on the real file). The greedy text does not see it: 32 of 32 prompts
identical to 128 tokens, and the 128/256-token decode of the tuning report byte-identical to the CPU's.

**Verdict against G3 as written.** The bounds (E1 mean ≤ 1e-5, max ≤ 1e-4, top-1 100 %) were set on 2026-10-09 for
the *tier* — "the device `expf` and the `Q8_0` dense split are the only sources" — before the chain existed on a
GGUF. The chain arm does not meet them (2e-4 / 0.08 / 99.975 %) and is still two orders of magnitude inside llama.cpp's
own CPU-vs-CUDA floor measured on this host on 2026-10-15 (0.067 / 2.46 / 96.7 %). The tier-only arm does not meet them either (mean 0.000137 / max 0.0813 / top-1 99.98 %): its windows 4 and 14 carry the *same* deltas as the chain's (0.0085 and 0.0813 at the same positions), so those two flips come from the device expert/dense path both arms share — the device `expf` and the rejoined `Q8_0` split move a router logit by ~1e-6 at a near-tie — while the chain adds window 7 from its own DeltaNet/attention arithmetic; the other 14 (tier) / 13 (chain) windows agree to ~1e-6. A mean over 4 080 positions is not robust to a single near-tie flip, which is what the 2026-10-09 bound did not foresee. The owner decides
whether G3's bound stays the tier's (then the chain gets its own row: "sylph-CPU vs chain within llama's CPU-vs-CUDA
floor, E3 ≥ 31/32") or is loosened; the report recommends the former.

**Harness note.** The "noise floor" row of this run's `report.md` reads 0.0 for llama.cpp CPU vs CUDA: the cached
floor directory of 2026-10-15 was reused and the comparison inside it degenerated (the CUDA floor run had fallen
back to the CPU); the floor quoted above is the 2026-10-15 measurement. The harness's own `report.md` of this run is
appended below.

## Harness report

## Equivalence report — Qwen3.6-35B-A3B-UD-Q4_K_M.gguf — sylph vs llama.cpp

### Setup

- model: `/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` sha256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`
- qwen35moe · 40 blocks · 1 part(s) · 733 tensors · 22.13 GB · {'Q4_K': {'bytes': 12079595520, 'tensors': 80}, 'Q5_K': {'bytes': 6828326912, 'tensors': 37}, 'Q8_0': {'bytes': 2033254400, 'tensors': 251}, 'Q6_K': {'bytes': 1077780480, 'tensors': 4}, 'F32': {'bytes': 104581632, 'tensors': 361}}
- llama.cpp: version: 0.6.0-dev (build 1, commit a0e6dc7)
- platform: Linux-6.8.0-142-generic-x86_64-with-glibc2.39
- machine: x86_64
- cpus: 24
- python: 3.12.9
- cpu: AMD Ryzen Threadripper PRO 3945WX 12-Cores
- sylph cap: 256 · threads: 12 · mode: real

### Tokenizer gate

- pass: 297193 ids compared

### Noise floor and thresholds

- floor sample -t 1 vs -t 12: mean |ΔNLL| 0, max 0, top-1 disagreement 0, KL 0; E2 |ΔPPL|/PPL 0
- floor E1 (max over samples): mean |ΔNLL| 0, max 0, top-1 disagreement 0, KL 0; E2 |ΔPPL|/PPL 0 (-t 1 vs -t 12)
- thresholds (calibrated): E1 mean ≤ 1.000e-06, max ≤ 1.000e-05, top-1 ≥ 0.995, KL ≤ 0.001; E2 |ΔPPL|/PPL ≤ 1.000e-06

### E1 — teacher-forced log-probs

| positions | mean \|ΔNLL\| | max \|ΔNLL\| | top-1 | top-5 | KL exact (mean / max, n) | KL coarsened mean | verdict |
|---|---|---|---|---|---|---|---|
| 4080 | 0.05904 | 1.954 | 0.9699 | 0.7951 | 0.08197 / 0.2591 (255) | 0.003567 | **FAIL** |

Worst positions: pos (6, 474) Δ 1.954 (ref -16.1576, cand -18.1120), pos (2, 398) Δ 1.861 (ref -5.4915, cand -3.6306), pos (10, 291) Δ 1.487 (ref -7.8052, cand -6.3185), pos (2, 338) Δ 1.38 (ref -16.1633, cand -17.5434), pos (12, 283) Δ 1.336 (ref -11.0459, cand -9.7103)

### E2 — perplexity

| windows | PPL reference | PPL sylph | \|ΔPPL\|/PPL | chunks sylph worse | sign test p | verdict |
|---|---|---|---|---|---|---|
| 16 | 7.1113 | 7.0958 | 0.002183 | 6 of 16 | 0.454 | **FAIL** |

### E3 — greedy generation

#### vs llama-server: 21 of 32 prompts identical or near-tie — **FAIL**

| prompt | compared by | prefix | ref len | sylph len | margin at divergence | verdict |
|---|---|---|---|---|---|---|
| 0 | ids | 20 | 128 | 128 | 0.1966 | diverged |
| 1 | ids | 0 | 128 | 128 | 0.181 | diverged |
| 2 | ids | 63 | 128 | 128 | 0.01013 | near-tie |
| 3 | ids | 2 | 58 | 128 | 0.04844 | near-tie |
| 4 | ids | 10 | 128 | 128 | 0.001756 | near-tie |
| 5 | ids | 34 | 128 | 128 | 0.007034 | near-tie |
| 6 | ids | 44 | 128 | 128 | 0.03349 | near-tie |
| 7 | ids | 128 | 128 | 128 | — | identical |
| 8 | ids | 128 | 128 | 128 | — | identical |
| 9 | ids | 128 | 128 | 128 | — | identical |
| 10 | ids | 17 | 128 | 128 | 0.06688 | diverged |
| 11 | ids | 128 | 128 | 128 | — | identical |
| 12 | ids | 62 | 128 | 128 | 0.2005 | diverged |
| 13 | ids | 128 | 128 | 128 | — | identical |
| 14 | ids | 14 | 15 | 14 | — | identical |
| 15 | ids | 30 | 128 | 128 | 0.03308 | near-tie |
| 16 | ids | 33 | 118 | 104 | 0.1757 | diverged |
| 17 | ids | 58 | 128 | 128 | 0.05927 | diverged |
| 18 | ids | 49 | 128 | 128 | 0.01034 | near-tie |
| 19 | ids | 63 | 128 | 128 | 0.07176 | diverged |
| 20 | ids | 11 | 128 | 128 | 0.006765 | near-tie |
| 21 | ids | 89 | 128 | 128 | 0.01076 | near-tie |
| 22 | ids | 14 | 128 | 128 | 0.05015 | diverged |
| 23 | ids | 1 | 128 | 128 | 0.1058 | diverged |
| 24 | ids | 37 | 55 | 57 | 0.001707 | near-tie |
| 25 | ids | 5 | 8 | 56 | 0.07593 | diverged |
| 26 | ids | 15 | 128 | 128 | 0.0206 | near-tie |
| 27 | ids | 128 | 128 | 128 | — | identical |
| 28 | ids | 60 | 128 | 128 | 0.1215 | diverged |
| 29 | ids | 7 | 128 | 128 | 0.02991 | near-tie |
| 30 | ids | 128 | 128 | 128 | — | identical |
| 31 | ids | 107 | 108 | 107 | — | identical |

### Deliberate differences (FR-35)

- `COLI_CUDA=1,COLI_GPUS=0,COLI_PLACE=auto,COLI_TRUNK_PROBE=0,CUDA_EXPERT_GB=auto,COLI_CUDA_CHAIN=1`: E1 mean 0.0591 / max 1.954, E2 PPL 7.096 (0.00217 rel); **against default sylph**: mean |ΔNLL| 2.152e-04 / max 0.08131, top-1 agreement 0.9998, |ΔPPL|/PPL 1.220e-05 over 4080 positions, E3 vs default sylph 32/32 prompts to the first near-tie
- `COLI_CUDA=1,COLI_GPUS=0,COLI_PLACE=auto,COLI_TRUNK_PROBE=0,CUDA_EXPERT_GB=auto`: E1 mean 0.05906 / max 1.954, E2 PPL 7.096 (0.002167 rel); **against default sylph**: mean |ΔNLL| 1.370e-04 / max 0.08132, top-1 agreement 0.9998, |ΔPPL|/PPL 1.538e-05 over 4080 positions, E3 vs default sylph 32/32 prompts to the first near-tie

### Verdict

- E1: **FAIL**
- E2: **FAIL**
- E3: **FAIL**

### Files

- `e3_llama.jsonl` 1172478 bytes
- `e3_ollama.jsonl` 1364689 bytes
- `e3_sylph.jsonl` 829372 bytes
- `e3_sylph_COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1.jsonl` 832207 bytes
- `e3_sylph_COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto.jsonl` 831292 bytes
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
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_0.tsv` 27156 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_1.tsv` 27119 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_10.tsv` 27218 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_11.tsv` 27145 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_12.tsv` 27200 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_13.tsv` 27269 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_14.tsv` 26989 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_15.tsv` 27139 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_2.tsv` 27256 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_3.tsv` 27234 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_4.tsv` 26940 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_5.tsv` 27174 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_6.tsv` 27104 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_7.tsv` 27188 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_8.tsv` 27199 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto/win_9.tsv` 27230 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_0.tsv` 27154 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_1.tsv` 27136 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_10.tsv` 27231 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_11.tsv` 27144 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_12.tsv` 27201 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_13.tsv` 27268 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_14.tsv` 26982 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_15.tsv` 27142 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_2.tsv` 27262 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_3.tsv` 27231 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_4.tsv` 26936 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_5.tsv` 27179 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_6.tsv` 27107 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_7.tsv` 27192 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_8.tsv` 27202 bytes
- `sylph/COLI_CUDA_1+COLI_GPUS_0+COLI_PLACE_auto+COLI_TRUNK_PROBE_0+CUDA_EXPERT_GB_auto+COLI_CUDA_CHAIN_1/win_9.tsv` 27232 bytes
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
- `sylph/default/win_0.tsv` 27161 bytes
- `sylph/default/win_1.tsv` 27129 bytes
- `sylph/default/win_10.tsv` 27230 bytes
- `sylph/default/win_11.tsv` 27149 bytes
- `sylph/default/win_12.tsv` 27193 bytes
- `sylph/default/win_13.tsv` 27276 bytes
- `sylph/default/win_14.tsv` 26999 bytes
- `sylph/default/win_15.tsv` 27145 bytes
- `sylph/default/win_2.tsv` 27263 bytes
- `sylph/default/win_3.tsv` 27234 bytes
- `sylph/default/win_4.tsv` 26946 bytes
- `sylph/default/win_5.tsv` 27176 bytes
- `sylph/default/win_6.tsv` 27103 bytes
- `sylph/default/win_7.tsv` 27196 bytes
- `sylph/default/win_8.tsv` 27198 bytes
- `sylph/default/win_9.tsv` 27218 bytes
- `sylph/default_2026-10-15/win_0.full` 253286451 bytes
- `sylph/default_2026-10-15/win_0.tsv` 27165 bytes
- `sylph/default_2026-10-15/win_1.tsv` 27140 bytes
- `sylph/default_2026-10-15/win_10.tsv` 27227 bytes
- `sylph/default_2026-10-15/win_11.tsv` 27154 bytes
- `sylph/default_2026-10-15/win_12.tsv` 27207 bytes
- `sylph/default_2026-10-15/win_13.tsv` 27284 bytes
- `sylph/default_2026-10-15/win_14.tsv` 26988 bytes
- `sylph/default_2026-10-15/win_15.tsv` 27144 bytes
- `sylph/default_2026-10-15/win_2.tsv` 27265 bytes
- `sylph/default_2026-10-15/win_3.tsv` 27240 bytes
- `sylph/default_2026-10-15/win_4.tsv` 26934 bytes
- `sylph/default_2026-10-15/win_5.tsv` 27175 bytes
- `sylph/default_2026-10-15/win_6.tsv` 27112 bytes
- `sylph/default_2026-10-15/win_7.tsv` 27190 bytes
- `sylph/default_2026-10-15/win_8.tsv` 27191 bytes
- `sylph/default_2026-10-15/win_9.tsv` 27235 bytes

