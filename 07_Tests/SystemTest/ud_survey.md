# System test — survey of the public quantizations (unsloth `UD-*`, bartowski): which types a reference file would need (phase 6)

Written 2026-10-09, before phase 6. Specification §3 (types in scope: `F32 F16 BF16 Q4_0
Q8_0 Q4_K Q5_K Q6_K`; "refused by name" otherwise), FR-9, FR-26, §9 risk "more types if a
reference file needs them"; tasks.md phase 6 "`UD-*` dynamic quants survey".

The question this test answers is a fact, not a feature: **for each public GGUF of the two
reference models, which tensor types does it use, how large is its resident dense set, and
would sylph refuse it today?** The answer decides whether phase 6 adds types (I-quants,
`Q2_K`/`Q3_K`, `IQ4_XS`, `MXFP4`) or not. Headers only are read (metadata, never weights),
exactly as the phase-1 inspections did; the dev container reaches Hugging Face.

## Method (`run_ud_survey.py`, stdlib)

```
python3 07_Tests/SystemTest/run_ud_survey.py [--repos unsloth/Qwen3.6-35B-A3B-GGUF,…] [--only UD-Q4_K_M,…]
                                             [--out 08_Documents/survey/<date>-ud-quants.md] [--cache <dir>]
```

1. `GET https://huggingface.co/api/models/<repo>/tree/main` (and the `tree/main/<dir>` of
   each split-set directory) lists the `.gguf` files with their sizes.
2. For each file the first 16 MB are fetched with a `Range` request (the metadata of a
   250 k-token tokenizer is ≈ 10 MB; the loop doubles the window while `ggufinfo` reports
   the data offset beyond it, up to 64 MB), the local copy is extended **sparsely** to the
   reported size (`os.truncate`) so the reader's bounds checks see the true geometry.
3. `ggufinfo.open_set` + `summarize` → architecture, type mix (per tensor family: experts
   gate/up/down, attention, shared expert, embeddings, output), `unsupported_v1`, MTP,
   bits per weight, and the **resident dense set** = Σ bytes of every tensor that is not a
   routed expert (what `coli plan` calls dense).
4. One Markdown table per repo in `08_Documents/survey/<date>-ud-quants.md`:
   `file · GiB · dense GiB · expert MiB · types (experts / attention / shared / embd-output) · outside the v1 set · verdict`
   (binary units; the 2026-10-05 inspections quoted decimal GB: 467 GB = 435 GiB)
   with verdict `runs` (every type in the set), `refused: <types>` otherwise; plus a
   summary: which types would unlock which files, and the smallest file whose dense set
   fits 8 / 16 / 32 GB.

Default repos: `unsloth/Qwen3.6-35B-A3B-GGUF`, `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`,
`unsloth/GLM-5.2-GGUF`. Default selection: every `UD-*` directory and every single-file
quant except `BF16`/`F16`/`Q8_0` duplicates of the type mix already known (configurable).

## Cases

| # | Case | Expected |
|---|---|---|
| 0 | API reachable | the three trees list; a `--only` filter narrows; a missing repo is reported, not fatal |
| 1 | header fetch | every selected file's metadata parses with `ggufinfo` after at most 64 MB; split sets open as one set (`split.count` consistent) |
| 2 | the two inspected files reproduce | `UD-Q4_K_M` (Qwen) and `UD-Q4_K_XL` (GLM) report the type mixes of the 2026-10-05 inspections (Qwen: `Q4_K 55 % · Q5_K 31 % · Q8_0 9 % · Q6_K 5 %`; GLM dense set 21.0 GB) |
| 3 | verdicts | `runs` for every file whose types are in the v1 set; `refused: IQ4_XS` etc. for the I-quant files, naming every type outside the set (ggml ids 10, 11, 15–23, 29 and `MXFP4`) |
| 4 | report | `08_Documents/survey/<date>-ud-quants.md` written with the tables, the summary and the reproduction command; `--check <report>` re-parses it |
| 5 | **conclusion recorded** | the summary states, per reference model, whether any file the owner would plausibly use needs a type outside the set — the input to the phase-6 decision "more types" |

**Pass criterion:** cases 0–4 green when run here; case 5 is the deliverable (a fact in
`08_Documents/`). The survey is re-run when a new quantization appears.

## State

| Date | Result |
|---|---|
| 2026-10-09 | document and runner written; first run: see the row below. |
| 2026-10-09 | **first run: `RESULT: ok (0 failures)`**, report `08_Documents/survey/2026-10-09-ud-quants.md` (42 files/sets: 18 unsloth Qwen3.6, 11 bartowski, 13 unsloth GLM-5.2; headers only, ≈ 600 MB fetched). Findings: every K-quant file at `Q4_K_M` and above runs with the supported set (unsloth `UD-Q4_K_S/M/XL`, `UD-Q5_K_*`, `UD-Q6_K*`, `UD-Q8_K_XL`, `Q8_0`; bartowski `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`; GLM `UD-Q4_K_S/M/XL`, `UD-Q5_K_*`, `UD-Q6_K_XL`, `UD-Q8_K_XL`); everything below Q4 needs I-quants: `IQ4_XS` would unlock 13 files, `IQ3_XXS` 9, `Q3_K` 9, `IQ3_S` 8 (unsloth's `UD-Q3_K_*`/`UD-Q2_K_XL` mix `IQ3_XXS`/`IQ4_XS` into their K-quant names); `MXFP4` only the one `MXFP4_MOE` file. Dense sets: Qwen 1.4–2.5 GiB (bartowski `Q4_K_M` 1.8, unsloth `UD-Q4_K_M` 2.4), GLM 14.6–19.8 GiB (`UD-Q4_K_XL` 19.6 GiB = the inspection's 21.0 GB). Runner fixes during the run: the proxy cut three 64 MB range reads (`IncompleteRead`) — the prefix is used; units labelled GiB. |
| 2026-10-09 | **phase-6 decision ("more types")**: none added. Every K-quant file at `Q4_K_M` and above runs with the supported set; the files below Q4 need `IQ4_XS` + `IQ3_S`/`IQ3_XXS` + `Q3_K` first — left to the owner (deliberate differences only; `st2gguf` refuses `Q2_K`/`Q3_K`/the I-quants by name). |