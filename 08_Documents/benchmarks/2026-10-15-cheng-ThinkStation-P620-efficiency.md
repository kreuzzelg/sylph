# Efficiency report — Qwen3.6-35B-A3B-UD-Q4_K_M.gguf on cheng-ThinkStation-P620 (2026-10-15)

## Host

- commit `7ed1e69-dirty` (sylph); engine `qwen36` built from this tree
- CPU AMD Ryzen Threadripper PRO 3945WX 12-Cores · 24 logical CPUs · threads used: 12
- RAM 94.1 GB total · 65.6 GB available at start
- GPU Quadro RTX 4000, 8192 MiB, NVIDIA GeForce RTX 3070, 8192 MiB · sylph/container GPU use: cuda
- model volume /dev/nvme0n1p1 mounted at /media/cheng/nvme1 · buffered I/O (pread, no O_DIRECT in this engine)
- OS Linux 6.8.0-142-generic · Python 3.12.9

## Systems

| system | source | sha256 / id | command / request |
|---|---|---|---|
| S sylph-GGUF | `/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` | `/media/cheng/nvme1/SylphTest/sylph/06_Code/c/qwen36 256 8` with {'SNAP': '/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', 'SERVE': '1', 'COLI_TEMP': '0', 'OMP_NUM_THREADS': '12'} |
| C colibrì gs64 container | `/media/cheng/nvme1/ColibriTest/downloads/qwen36_i4_gs64` | `None` (index) | `/media/cheng/nvme1/SylphTest/sylph/06_Code/c/qwen36 256 8` with {'SNAP': '/media/cheng/nvme1/ColibriTest/downloads/qwen36_i4_gs64', 'SERVE': '1', 'COLI_TEMP': '0', 'OMP_NUM_THREADS': '12'} |
| O Ollama 0.40.3 | tag `qwen36-unsloth-ud-q4km` | blob `/usr/share/ollama/.ollama/models/blobs/sha256-ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` | `/api/generate` raw, options {'temperature': 0, 'seed': 1, 'num_predict': 64, 'num_ctx': 4096, 'repeat_penalty': 1.0, 'repeat_last_n': 0, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.0}, keep_alive 30m |
| sylph_cuda | `/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | — | `/media/cheng/nvme1/SylphTest/sylph/06_Code/c/qwen36 256 8` with {'SNAP': '/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', 'SERVE': '1', 'COLI_TEMP': '0', 'OMP_NUM_THREADS': '12', 'COLI_CUDA': '1'} |
| container_cuda | `/media/cheng/nvme1/ColibriTest/downloads/qwen36_i4_gs64` | — | `/media/cheng/nvme1/SylphTest/sylph/06_Code/c/qwen36 256 8` with {'SNAP': '/media/cheng/nvme1/ColibriTest/downloads/qwen36_i4_gs64', 'SERVE': '1', 'COLI_TEMP': '0', 'OMP_NUM_THREADS': '12', 'COLI_CUDA': '1'} |

Protocol: 16 prompts (`/media/cheng/nvme1/SylphTest/sylph/07_Tests/SystemTest/fixtures/e3_prompts.txt`), 64 new tokens each, greedy (temperature 0); 2 passes — pass 1 cold, passes 2… warm in the same process; prefill from one 2000-token prompt; expert cache `--cap 256` slots per layer (sylph and container); request sizes: prompts of 16–68 characters.
Cache state: page cache not dropped (--cold none): 'cold' means a fresh process with an empty expert cache

## Results

| system | state | decode tok/s (median, min–max) | TTFT (median) | prefill tok/s | bytes/token read | RSS | VRAM | expert hit rate | TF-NLL (PPL) first window | ids stable |
|---|---|---|---|---|---|---|---|---|---|---|
| sylph | cold | 10.55 (7.87–12.60) tok/s | 0.86 s | — | 19.5 MB | 20.27 GB | — | 97.3 % | 1.3993 (4.052) | True |
| sylph | warm | 12.31 (11.53–12.80) tok/s | 0.64 s | 17.2 tok/s | — | 20.28 GB | — | 100.0 % | 1.3993 (4.052) | True |
| container | cold | 12.13 (4.32–14.46) tok/s | 0.69 s | — | 18.1 MB | 20.26 GB | — | 97.3 % | 1.4112 (4.101) | True |
| container | warm | 14.82 (14.42–15.20) tok/s | 0.39 s | 25.1 tok/s | — | 20.26 GB | — | 100.0 % | 1.4112 (4.101) | True |
| sylph_cuda | cold | 18.57 (15.31–19.87) tok/s | 0.60 s | — | 20.1 MB | 21.16 GB | — | 98.5 % | 1.3993 (4.052) | True |
| sylph_cuda | warm | 19.89 (18.15–21.04) tok/s | 0.50 s | 22.3 tok/s | — | 21.17 GB | — | 100.0 % | 1.3993 (4.052) | True |
| container_cuda | cold | 11.89 (10.18–12.23) tok/s | 0.59 s | — | 19.0 MB | 41.96 GB | — | 98.3 % | 1.4065 (4.082) | False |
| container_cuda | warm | 12.01 (10.52–12.23) tok/s | 0.58 s | 22.1 tok/s | — | 42.07 GB | — | 100.0 % | 1.4065 (4.082) | False |
| ollama | cold | 43.99 (39.46–47.61) tok/s | 0.17 s | — | — | 9.62 GB | 13.17 GB | — | — | True |
| ollama | warm | 43.94 (40.83–46.54) tok/s | 0.15 s | 593.7 tok/s | — | 9.62 GB | 13.17 GB | — | — | True |

bytes/token: S from the engine's `GGUF reads:` lines; C = misses × bytes per expert as stored; O not observable. tok/s and RSS are the engines' own `DONE … STAT` fields; TTFT is the tool's clock from request to first token (Ollama: `prompt_eval_duration`, plus the model load on its first cold request).

## Ratios

| ratio (warm) | decode tok/s | TTFT | prefill tok/s | bytes/token | RSS |
|---|---|---|---|---|---|
| S/C | 0.83 | 1.64 | 0.69 | — | 1.00 |
| S/O | 0.28 | 4.16 | 0.03 | — | 2.11 |
| sylph_cuda/S | 1.61 | 0.78 | 1.30 | — | 1.04 |
| container_cuda/S | 0.98 | 0.89 | 1.28 | — | 2.07 |

Deliberate differences of S (spec §4.2): no re-quantization (the file's blocks as stored), f32 activations on the expert kernels (upstream's container runs int8), exact float dense tensors (`COLI_DENSE_I8` off on a GGUF), `token_embd` decoded per token.

## Verdict

It runs: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf served 16 prompts × 2 passes with 64 new tokens each, ids stable across passes = True, 12.31 tok/s warm decode, TTFT 0.64 s, RSS 20.28 GB. Equivalence: E1 FAIL, E2 FAIL, E3 FAIL (from report.json; the reading of those gates is in 08_Documents/equivalence/). Efficiency (sylph relative to the others, warm): vs gs64 container: decode 0.83×, TTFT 1.64×, prefill 0.69×, RSS 1.00×; vs Ollama: decode 0.28×, TTFT 4.16×, prefill 0.03×, RSS 2.11×. The limiting factor in sylph's own profile is attention / DeltaNet (disk (expert reads) 0.0 s, MoE compute 22.5 s, attention / DeltaNet 44.2 s, lm_head 0.0 s).

Equivalence source: `/media/cheng/nvme1/SylphTest/equiv_unsloth_full/report.json` — E1 FAIL, E2 FAIL, E3 FAIL

## Raw

```
{"sylph": {"tokens": 1948, "tok_s": 12.314, "tok_s_min": 11.533, "tok_s_max": 12.799, "ttft_s": 0.644987940788269, "rss_gb": 20.28, "bytes_per_token": 9733639.630390143, "bytes_source": "GGUF reads", "hit_rate": 0.9865045691906005, "hits": 725436, "misses": 9924, "expert_bytes": 2039808, "reads": {"bytes": 18961130000.0, "slices": 29772, "parts_touched": 1}, "prof": {"wall_s": 172.765238, "disk_s": 12.573236, "moe_s": 45.375563, "attn_s": 89.05777499999999, "head_s": 0.0}, "scope": {"accounting": "all passes", "speed": "warm passes"}, "label": "sylph", "cold": {"tokens": 974, "tok_s": 10.548, "tok_s_min": 7.869, "tok_s_max": 12.596, "ttft_s": 0.8559070825576782, "rss_gb": 20.27, "bytes_per_token": 19467279.260780286, "bytes_source": "GGUF reads", "hit_rate": 0.9730091383812011, "hits": 357756, "misses": 9924, "expert_bytes": 2039808, "reads": {"bytes": 18961130000.0, "slices": 29772, "parts_touched": 1}, "prof": {"wall_s": 93.28468, "disk_s": 12.573236, "moe_s": 22.83578, "attn_s": 44.865742, "head_s": 0.0}}, "warm": {"tokens": 974, "tok_s": 12.314, "tok_s_min": 11.533, "tok_s_max": 12.799, "ttft_s": 0.644987940788269, "rss_gb": 20.28, "bytes_per_token": 0.0, "bytes_source": "GGUF reads", "hit_rate": 1.0, "hits": 367680, "misses": 0, "expert_bytes": 2039808, "reads": {"bytes": 0.0, "slices": 0, "parts_touched": 1}, "prof": {"wall_s": 79.480558, "disk_s": 0.0, "moe_s": 22.539783, "attn_s": 44.192033, "head_s": 0.0}}, "ids_stable": true, "load_s": 2.447277784347534, "rss_after_load_gb": 2.52, "prefill_tok_s": 17.220275943346795, "prefill_tokens": 1862, "prefill_ttft_s": 108.12834858894348, "startup": ["[meta] from GGUF: q_heads=16 kv_heads=2 head_dim=256 q_head_dim=512 o_in=4096 rotary_dim=64 n_experts=256 topk=8 inter=512 shared_inter=512 attn_output_gate=1 n_active=10 | DeltaNet vheads=32 kheads=16 kdim=128 vdim=128 convk=4 conv_dim=8192", "resident weights loaded in 2.2s | RSS after load: 2.52 GB", "[qwen36] expert format on disk: GGUF ggml blocks as stored (gq_moe_run on the CPU; CUDA tier fmt 16+type)", "[GGUF] qwen35moe \u00b7 40 blocks (10 attention) \u00b7 1 part \u00b7 experts Q4_K/Q4_K/Q5_K 1.95 MB each \u00d7 256 \u00d7 40 = 18.22 GB \u00b7 dense F32\u00d7361 Q8_0\u00d7251 Q6_K\u00d71 2.38 GB \u00b7 token_embd Q8_0 on demand \u00b7 output Q6_K gq_matmul \u00b7 experts on CPU (gq_moe_run) \u00b7 sidecars /media/cheng/nvme1/SylphTest/models/.coli-Qwen3.6-35B-A3B-UD-Q4_K_M/"], "env": {"SNAP": "/media/cheng/nvme1/SylphTest/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "SERVE": "1", "COLI_TEMP": "0", "OMP_NUM_THREADS": "12"}, "cmd": ["/media/cheng/nvme1/SylphTest/sylph/06_Code/c/qwen36", "256", "8"], "cold_note": "page cache not dropped (--cold none): 'cold' means a fresh process with an empty expert cache", "nll": {"tf_nll": 1.3993, "n": 255, "ppl": 4.052362320160094, "dump": "../../08_Documents/benchmarks/2026-10-15-cheng-ThinkStation-P620-efficiency_work/sylph/win_0.tsv"}, "first_ids": [[279, 20616, 314, 279, 12386, 5260, 303, 220, 20, 15, 24, 17501, 11, 948, 279, 1483, 11148, 314, 21047, 11, 23211, 438, 205172, 7070, 10032, 11, 557, 888, 83457, 13, 1061, 12495], [25, 271, 248068, 271, 248069, 271, 760, 8684, 557, 2512, 2972, 92592, 220, 16, 16, 159034, 271, 332, 1536, 11956, 64700, 198, 9, 256, 2972, 31010, 2544, 64700, 5589, 220, 16, 21], [271, 248068, 198, 90700, 8340, 25, 271, 16, 13, 220, 2972, 1847, 7355, 279, 9640, 64700, 198, 262, 348, 256, 31819, 25, 19253, 5143, 21892, 13, 198, 262, 348, 256, 13855, 25], [271, 248068, 198, 8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983, 2570, 5396, 64700, 198, 256, 471, 2972, 14162, 64700, 79091, 303, 1330, 22157, 3069, 279, 12515], [712, 36181, 220, 18, 23, 307, 13, 50454, 13, 2659, 245097, 77298, 84354, 4449, 52851, 258, 704, 2280, 318, 3580, 5840, 8, 211815, 32630, 13, 8148, 52277, 25275, 13644, 2607, 427, 82628], [271, 248068, 271, 248069, 271, 608, 598, 5251, 13507, 11, 71286, 5810, 12981, 9528, 51799, 7029, 41371, 205385, 6993, 16927, 55746, 27795, 6846, 1
```
