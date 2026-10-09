# System test — equivalence E1–E3 against llama.cpp and Ollama on the same real GGUF (phase 4)

Written 2026-10-08, **before** phase 4 (process gate in `../../04_Tasks/tasks.md`).
Specification §4.1–§4.3 (levels, deliberate differences, calibrated tolerances),
FR-31 (done, phase 3), FR-32, FR-33, FR-34 (as amended 2026-10-06), FR-35, NFR-2,
acceptance §8.3–§8.4; architecture §10. Owner's requirement R4 ("automatische Tests,
um sicherzustellen, dass sie äquivalent sind; Differenzen müssen bewusste Verbesserungen
sein").

**Under test:** the GGUF path of the `qwen36` engine as a *function* — the same file,
the same token ids, compared with ggml's computation of that file as llama.cpp and
Ollama expose it. **Not under test:** Ollama's server features, templates, sampling
(spec §3).

This test has two halves. The **owner's machine** runs the real model against the real
references (one entry point, one report). **CI** runs the same harness code on the tiny
torch-built model with the transformers forward pass as the reference (FR-34), so the
runners, the comparer and the report never rot between owner runs.

## Inputs

| Input | Where from | Notes |
|---|---|---|
| `MODEL` | the GGUF Ollama actually runs (`ollama show <tag> --modelfile` → `FROM <blob>`; `~/.ollama/models/blobs/sha256-…`) **or** `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | spec §9 (a): sylph and every reference read **this one file**; the report records its sha256 and `coli gguf inspect` summary |
| `LLAMA` | a llama.cpp build directory with `llama-tokenize`, `llama-perplexity`, `llama-server` (preferred) or `llama-cli`; version pinned in the report (`llama-cli --version`) | CPU build for the noise floor; a CUDA build is a second floor sample if present |
| `OLLAMA` | `http://host:11434`, `TAG` = the model tag whose blob is `MODEL` | version from `/api/version`; `logprobs`/`top_logprobs` used where the server offers them |
| `TEXT` | `wiki.test.raw` of wikitext-2-raw-v1 (llama.cpp's `scripts/get-wikitext-2.sh`) | E1/E2 corpus (the owner's #1370 setting) |
| `fixtures/e3_prompts.txt` | this repository | 32 prompts, raw unless prefixed `chat:` (then rendered with the GGUF's chat template by **both** sides: sylph through `coli`, Ollama through `/api/chat`, llama.cpp through `--jinja`) |
| `CHUNKS` | default 16 | windows of `n_ctx = 512`, second half scored (E1/E2 protocol) |
| `CAP` | default from `resource_plan` | sylph expert-cache slots per layer; recorded (the result must not depend on it: case 7) |

## Harness contract (phase-4 deliverable, stdlib Python; lives in `07_Tests/SystemTest/equivalence/`)

```
make -C 06_Code/c equivalence MODEL=<gguf> LLAMA=<dir> [OLLAMA=http://host:11434 TAG=<tag>] \
     [TEXT=wiki.test.raw] [CHUNKS=16] [CAP=<n>] [LEVELS=E1,E2,E3] [OUT=<report dir>] [KL_CHUNKS=1]
# = python3 ../../07_Tests/SystemTest/equivalence/run_equivalence.py --model … --llama … …
```

Files:

| File | Role |
|---|---|
| `run_equivalence.py` | driver: tokenizer gate → noise floor → E1 → E2 → E3 → report; every step idempotent and cached in `OUT/` (re-runs skip finished steps; `--force` redoes them) |
| `ref_llama.py` | `tokenize` (`llama-tokenize --ids`), `perplexity` (`llama-perplexity -f TEXT -c 512 --chunks N -b 512 --kl-divergence-base OUT/llama.kld`), `generate` (`llama-server` `/completion` with `n_probs 10`, `temperature 0`, `seed 1`, `n_predict 128`; `llama-cli --temp 0 -n 128` as the text-only fallback) |
| `ref_ollama.py` | `/api/version`, `/api/show` (blob digest check), `/api/generate` (`raw: true`, `options: {temperature: 0, seed: 1, num_predict: 128, num_ctx: 4096}`, `logprobs: true, top_logprobs: 10` when accepted), `/api/chat` for the `chat:` prompts |
| `sylph_runner.py` | drives `qwen36`: `tests/test_tok_gguf <gguf> --encode <text>` for ids, `PPL=1 PPL_DUMP=` per window, `PPL_DUMP_FULL=` for the exact-KL windows, `SERVE=1` greedy generation with `logprobs=10` |
| `compare.py` | the comparer (FR-33): extends `compare_logprobs.py`; E1 ΔNLL/KL/top-1/top-5, E2 PPL pair and chunk-wise sign test, E3 prefix lengths; thresholds from the noise floor; writes `report.md` + `report.json` |
| `formats.md` | the interchange formats below |

### Interchange formats

- **`ppl-dump v1`** (FR-31, `lossless_oracle.md`): per scored position `pos \t target \t logprob \t tail`. Every engine's E1 output is converted to this format: sylph writes it; `ref_llama.py` converts the `--kl-divergence-base` file (its layout is read from the pinned llama.cpp's `tools/perplexity/perplexity.cpp`; the converter **self-checks** by recomputing the PPL from the parsed log-probs and requiring it to match the PPL `llama-perplexity` printed within 1e-3 — the format is not trusted, it is verified).
- **`full-logprob v1`**: for exact KL, per scored position the full log-softmax vector as little-endian f32 (`vocab × 4` bytes), header line `# sylph full-logprob v1 vocab=<V> positions=<N>`. Written by the engine on `PPL_DUMP_FULL=<file>` (phase-4 deliverable), by `ref_llama.py` from the kld file, by torch in CI. 254 MB per 256-position window on the real model, hence `KL_CHUNKS` (default 1, chunk 0 only); the truncated KL over the union of both sides' top-10 plus the lumped remainder is reported for every chunk.
- **`e2-chunks v1`**: TSV `chunk \t n_scored \t sum_nll \t ppl_running` per engine.
- **`e3-gen v1`**: JSON lines, one per prompt and engine: `{"prompt": i, "engine": "...", "ids": [...], "text": "...", "top": [[[id, lp], …k], …]}` (`top` empty where the reference gives text only).
- **Report** `OUT/report.md` (copied to `08_Documents/equivalence/<date>-<model stem>-<host>.md`): Setup (file sha256, inspect summary, versions, host, `CAP`, threads), Tokenizer gate, Noise floor, E1, E2, E3 tables, Verdict per level with the thresholds used, list of raw files. `report.json` carries the same numbers for machines.

### Protocol per level

0. **Tokenizer gate** (FR-32, FR-20). `TEXT` and the 32 prompts tokenized by sylph
   (`test_tok_gguf --encode`) and by `llama-tokenize --ids`: identical id sequences,
   else stop ("E1–E3 would compare different inputs") — Ollama uses the same ggml
   tokenizer as llama.cpp. The gate also fixes `add_bos` (false for Qwen3.6) for both.
1. **Noise floor** (§4.3). `llama-perplexity` on `TEXT` with `-t 1` and `-t <cores>`
   (and the CUDA build if `LLAMA_CUDA=<dir>` is given): `ppl-dump` both, floor_E1 =
   (mean |ΔNLL|, max |ΔNLL|, top-1 disagreement rate), floor_E2 = |ΔPPL|/PPL. Threshold =
   max(3 × floor, one unit of the 7-digit format). Recorded in the report; the proposal
   of §4.1 (mean ≤ 3×floor, top-1 ≥ 99.5 %, KL ≤ 1e-3; |ΔPPL| ≤ 0.3 %) stays the
   starting point and is replaced by the calibrated values.
2. **E1 (logits, teacher-forced)** on the reference's ids: sylph `PPL=1 PPL_DUMP` per
   window (`prompt_ids` = first **257** ids, `full_ids` = all 512; `tf_nll` scores targets
   257…511 — 255 positions, exactly the ones `llama-perplexity` scores: its logits at
   indices 256…510 predict tokens 257…511); llama.cpp from the kld file.
   Pass: mean |ΔNLL| and max |ΔNLL| within threshold, top-1 agreement ≥ 99.5 % (or
   ≥ 1 − 3 × floor's disagreement rate, whichever is looser), exact KL on the
   `KL_CHUNKS` windows ≤ 1e-3 nat, truncated KL reported for all.
3. **E2 (perplexity)**: PPL over the 16 windows from the same dumps (sylph) and from
   `llama-perplexity`'s own print (llama.cpp); pass: |ΔPPL|/PPL ≤ threshold and the
   chunk-wise sign of ΔNLL has no systematic direction (two-sided binomial test,
   p > 0.05).
4. **E3 (generation)**: greedy 128 tokens from each of the 32 prompts, three engines.
   Divergence point = first position where ids differ. Pass: identical prefix up to the
   first position where the **reference's** top-2 margin (llama-server `n_probs`,
   Ollama `top_logprobs` where offered, else llama-server's margin on the same prefix,
   flagged) is < 0.05 nat; ≥ 95 % of prompts (≥ 31 of 32) reach 128 tokens or a
   justified near-tie. The `chat:` prompts additionally check that sylph's rendered
   prompt (via `coli`'s template path) tokenizes to the same ids as the reference's.
5. **Deliberate differences (FR-35)**: every opt-in deviation (today: `COLI_DENSE_I8=1`,
   int8-at-load dense; later the int8-activation kernels) is run as a second sylph
   configuration through E1/E2 and its deltas land in the same report under its switch
   name. Default configuration = none of them.
6. **Cost control**: `--levels`, `--chunks 4` for a first pass, `--prompts 8`; the driver
   prints the estimated token count before starting; sylph windows are cached per
   `(sha256, cap, config)`.

## Cases

| # | Case | Expected |
|---|---|---|
| 0 | **CI subset (FR-34)** — `run_equivalence.py --ci --model qwen36_tiny_f32.gguf --torch-ref qwen36_tiny` in the `gguf-oracle` job | E1 on `ref_full.json` (16 positions) and E2 on four synthetic 512-id windows (seeded ids from the 320 vocabulary; torch writes `ppl-dump` and `full-logprob` for them); comparer runs the full E1/E2 path incl. exact KL; thresholds = the `lossless_oracle.md` gate (mean 1e-6 / max 1e-5; |ΔPPL|/PPL ≤ 1e-6; KL ≤ 1e-9); `report.md`/`report.json` produced and parsed back; E3 against the torch greedy ids of `ref_full.json` (16 tokens, margin from torch) = 16/16. Green on every push. |
| 1 | Tokenizer gate on the real file | `TEXT` (≈ 300 k ids) and the 32 prompts: identical ids sylph vs `llama-tokenize`; `add_bos false` both. |
| 2 | Noise floor | floor_E1 and floor_E2 measured and printed; thresholds derived; CUDA sample if available. |
| 3 | **E1** | per criteria above over 16 × 255 positions; exact KL on chunk 0. |
| 4 | **E2** | |ΔPPL| within threshold; sign test p > 0.05; both PPLs and the per-chunk table in the report. The owner's #1370 numbers (gs64 7.325, mixed 7.281, int8 7.153 on this protocol) are quoted beside them for R1/R2. |
| 5 | **E3** | ≥ 31/32 prompts token-identical to 128 tokens or to a justified near-tie against llama-server; Ollama text-identical to the same points (ids where `top_logprobs` is offered). |
| 6 | Deliberate differences | `COLI_DENSE_I8=1` run reported with its ΔNLL/ΔPPL (expected worse than default; must not change the default's verdict). |
| 7 | Cache independence | E1 on chunk 0 with `CAP` = 8, 64 and the plan default: dumps byte-identical (streaming never changes a number). |
| 8 | Report | `report.md` committed under `08_Documents/equivalence/`, `report.json` validates (`compare.py --check report.json`), raw dumps listed with sizes and sha256. |

**Pass criterion:** case 0 green in CI; on the owner's machine cases 1–5, 7, 8 pass
with the calibrated thresholds, case 6 reported. A failure in E1 with the gate of
`lossless_oracle.md` still green points at the K-quant kernels or a transform only the
real file exercises (e.g. `Q5_K`/`Q6_K` `ffn_down`, `Q6_K` `output`); the comparer
prints the worst positions and the tensors most likely involved (by layer kind) to
start from.

## Cost estimate (owner's machine, CPU)

E1/E2: 16 windows × 512 tokens = 4 080 sylph positions (257 prefill + 255 single-token
steps per window) plus one `llama-perplexity` pass per floor sample. At the decode
rates of #1370 this is minutes per engine, not hours; with `CAP` ≥ 64 the expert
cache hit rate on wikitext is high enough that disk is not the bottleneck. E3: 3 engines
× 32 prompts × 128 tokens ≈ 12 k tokens. Exact KL on chunk 0: 254 MB per engine.

## Runner (owner)

```sh
# once: references
ollama show qwen3.6:35b --modelfile | grep FROM           # -> MODEL
bash <llama.cpp>/scripts/get-wikitext-2.sh                 # -> wiki.test.raw
# the run (first pass small, then full)
make -C 06_Code/c equivalence MODEL=~/.ollama/models/blobs/sha256-… LLAMA=~/llama.cpp/build/bin \
     OLLAMA=http://127.0.0.1:11434 TAG=qwen3.6:35b TEXT=~/wiki.test.raw CHUNKS=4 LEVELS=E1,E2
make -C 06_Code/c equivalence MODEL=… LLAMA=… OLLAMA=… TAG=… TEXT=… CHUNKS=16
# then: copy OUT/report.md to 08_Documents/equivalence/<date>-<stem>-<host>.md and commit
```

## State

| Date | Result |
|---|---|
| 2026-10-08 | document and `fixtures/e3_prompts.txt` written; the harness (`equivalence/`), `PPL_DUMP_FULL`, `test_tok_gguf --encode`, the `make equivalence` target and the CI case are phase-4 deliverables. Open: spec §9 (a) — the owner names the Ollama tag/blob. |
| 2026-10-09 | harness implemented (`equivalence/run_equivalence.py`, `ref_llama.py`, `ref_ollama.py`, `sylph_runner.py`, `compare.py`, `formats.md`), `PPL_DUMP_FULL`, `--encode`, `make equivalence`, CI case wired into the `gguf-oracle` job (first run pending). Validated locally without references by a sylph-made pseudo reference (`--ci` mode): same file → E1/E2/E3 pass with mean/max 0 and KL 0; F16 candidate → E1 fails on exact KL (5e-9 > 1e-9) while ΔNLL stays under the gate; all-Q8_0 candidate → E1/E2 fail (mean 7.5e-2). Correction found while writing the comparer: the engine's log-prob tail is ` <lp> <k> <id> <lp> …`, not `<id>:<lp>`; see `lossless_oracle.md`. The kld-file parser of `ref_llama.py` is written from the layout of `tools/perplexity/perplexity.cpp` and **self-checks** against the printed PPL on the owner's machine; it has not run against a real file here (no llama.cpp binary in the container). |
