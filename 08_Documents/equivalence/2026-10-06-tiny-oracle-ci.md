# Lossless oracle — tiny Qwen3.6, CI run 13, 2026-10-06

Protocol: `../../07_Tests/SystemTest/lossless_oracle.md` (FR-25, FR-31, FR-34, NFR-5).
Record: the `gguf-oracle` job of `.github/workflows/check.yml`, run 13 on `main`
(commit `2375ec1`, "06_Code: the GGUF source in the Qwen3.6 engine (phase 3)"),
https://github.com/kreuzzelg/sylph/actions/runs/37533479870. ubuntu-latest, 2 vCPUs
(the engine used 1 thread), gcc, Python 3.13.16, torch 2.14.1+cpu, transformers 5.19.0.
This file is the copy of the summary that criterion 6 asks for.

## Fixture

Upstream's `tools/make_qwen36_tiny.py --ref-mode full` (seeded): hidden 64, 8 blocks with
Gated Attention at blocks 3 and 7 and Gated DeltaNet elsewhere, 4 q / 2 kv heads of 16,
rope 4 dims, 8 experts top-2 of inter 32, vocab 320, DeltaNet 4 key / 8 value heads of 8
(so the value-head permutation is non-trivial: `vh/vk = 2`). ~0.6 M parameters. Reference
ids from the transformers forward pass: prompt `1 2 3 4 5`, 16 greedy continuations
`32 146 258 130 236 174 55 104 68 141 234 74 308 166 305 301`.

Sources under test, all written from the same safetensors snapshot:

| Source | Writer | Tensors | Size |
|---|---|---|---|
| `qwen36_tiny_c/` | `tools/convert_qwen36.py --ebits 8` (upstream's container; experts int8) | 8 layers + 3 globals | — |
| `qwen36_tiny_f32.gguf` | `tools/st2gguf.py --type f32` | 149 (F32) | 2.52 MB |
| `qwen36_tiny_f16.gguf` | `tools/st2gguf.py --type f16` | 76 F16 + 73 F32 | 1.29 MB |
| `qwen36_tiny_q8_0.gguf` | `tools/st2gguf.py --type q8_0 --expert-type q8_0` | 76 Q8_0 + 73 F32 | 0.71 MB |

The GGUFs carry the placeholder tokenizer (320 pieces, no merges); the reference-id mode
does not need a tokenizer.

## Results

**Façade cross-check** (`tests/test_gguf_load <gguf> qwen36_tiny`): `Cfg` from the GGUF
equals `config.json` in every field (11 checks); 317 tensors / expert slices, 629 520
values, **bit-exact** for the F32 GGUF and **within one f16 ulp** for the F16 GGUF, after
undoing the converter's transforms (norm `1+w`, `ssm_a = −exp(A_log)`, the value-head
permutation on rows, elements and columns).

**Token-exact** (`./qwen36 <cap> 8 qwen36_tiny/ref_full.json`, `COLI_DENSE_I8=0`):

| Source | cap 1 | cap 2 | cap 8 | Expert cache hit rate (cap 1 / 2 / 8) |
|---|---|---|---|---|
| container (unchanged, NFR-5) | 16/16 | 16/16 | 16/16 | 5.9 % / 21.9 % / 80.0 % |
| GGUF F32 | 16/16 | 16/16 | 16/16 | 5.9 % / 21.9 % / 80.0 % |
| GGUF F16 (reported) | — | 16/16 | — | 21.9 % |
| GGUF Q8_0, `COLI_DENSE_I8=0` (reported) | — | 16/16 | — | 21.6 % |
| GGUF Q8_0, `COLI_DENSE_I8=1` (reported) | — | 16/16 | — | 21.2 % |

The hit rates of the container and the F32 GGUF agree at every capacity, which they
must: routing is identical, so the LRU sees the same expert sequence. The Q8_0 rates
differ by one or two slots — the quantized router picks a different second expert at
one or two positions, without changing any emitted token on this fixture. Cap 1 evicts
on every routed expert, so the raw-slice loads and the `kq` slot bookkeeping ran 301
times per source.

**E1 subset** (`PPL=1 PPL_DUMP=dump_f32.tsv … ./qwen36 8 8 …`, then
`compare_logprobs.py qwen36_tiny/ref_logprobs.tsv dump_f32.tsv --tf-nll 5.3197`):

| Check | Value |
|---|---|
| scored positions | 16, same targets on both sides |
| top-1 | identical at every position |
| mean \|ΔNLL\| | 6.25e-8 nat |
| max \|ΔNLL\| | 1.0e-6 nat (one unit of the 7-significant-digit dump format) |
| TF-NLL torch / engine | 5.319704 / 5.319704 (ppl 204.3235 both) |
| dump reproduces the printed `TF-NLL: 5.3197` | yes (1e-6 relative) |

Both dumps print log-probs with `%.7g`; around −5 nat the last digit is 1e-6, so a
single position differing by one print ulp is the whole observed difference. The engine
in f32 with its own summation order and the transformers fp32 forward agree below the
resolution of the dump.

## Thresholds

The protocol proposed mean ≤ 1e-4 and max ≤ 1e-3 and said the first run calibrates them
at 3× the observed values. 3× the observed max is three print ulps; a gate that narrow
measures the formatter. The thresholds are set at **mean ≤ 1e-6, max ≤ 1e-5** (10× the
format unit). For scale: on the torch-free tiny fixture of `src_facade.md`
(`make_tiny_qwen36_hf.py`, same geometry, random bf16-representable weights), the
engine's F16 GGUF differs from its F32 GGUF by mean 4.4e-7 / max 1.0e-6 (f16 holds those
weights exactly, so again print ulps), while the Q8_0 GGUF differs by mean 4.9e-2 / max
2.9e-1 nat with the same top-1 everywhere — the smallest real precision loss in the v1
type set sits four orders of magnitude above the gate. The job passes
`--mean 1e-6 --max 1e-5` explicitly; `compare_logprobs.py`'s defaults stay at the
generic E1 values for other uses.

## What this does and does not show

- Shown: for an F32 GGUF of a `qwen35moe` model with both layer kinds, routed experts
  and a non-trivial DeltaNet head permutation, the GGUF source of the engine is
  numerically the transformers model (E1 below print resolution, E3 token-exact), and
  the container path is untouched by the phase-3 changes (NFR-5).
- Shown: the F16 and Q8_0 GGUFs and the int8 container reproduce the same 16 tokens;
  their ΔNLL was not measured in run 13 (tokens only). The job now dumps and compares
  them as reported values; run 14 onward records them.
- Not shown: anything about K-quants on a real model (the tiny fixture's experts are
  F32/F16/Q8_0), tokenizer equivalence (placeholder tokenizer here; that is
  `tests/test_tok_gguf` on the real metadata, 52/52 lines), or equivalence with
  llama.cpp's own numerics on the same GGUF — that is E1–E3 of phase 4 on the owner's
  machine (`07_Tests/SystemTest/equivalence.md`, to be written).

## Same run, other jobs

linux and macos (`make check`) green. windows red: `tests/test_gguf_load.c` declared a
local `int far`, and `far` is an empty macro in the Windows headers that `qwen36.c`
pulls in on MinGW. Renamed; reproduced and re-verified locally with `-Dfar= -Dnear=`
on both new tests. No behaviour change.
