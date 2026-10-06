# System test — lossless oracle: the tiny Qwen3.6 from an F32 GGUF reproduces the transformers reference (phase 3)

Written 2026-10-06, before phase 3. Specification FR-25, FR-31, FR-34 (as amended
2026-10-06), NFR-5, acceptance §8.2; architecture §8, §10.

This is the gate that proves the **plumbing** end to end: config from metadata,
names, un-transforms, expert slices, tokenizer-less reference mode, the engine's math
untouched. It runs the same torch-built tiny model upstream's own CI job runs
(`.github/workflows/ci.yml`, "Tiny Qwen3.6-shaped fixture + full-hybrid oracle"),
through the container as before and through a GGUF written by `tools/st2gguf.py`.

## Inputs

| Input | How it is made |
|---|---|
| `qwen36_tiny/` + `qwen36_tiny/ref_full.json` | `python3 tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json` (torch + transformers from `tools/oracle-requirements.txt`; seeded, so the reference ids are reproducible: 5 prompt ids, 16 greedy continuations through **both** layer kinds) |
| `qwen36_tiny_c/` | `python3 tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8` (upstream's container; experts int8, dense f16 → the engine's f32 with `COLI_DENSE_I8=0`) |
| `qwen36_tiny_f32.gguf`, `_f16.gguf`, `_q8_0.gguf` | `python3 tools/st2gguf.py qwen36_tiny --out qwen36_tiny_f32.gguf --type f32` (and `--type f16`; `--type q8_0 --expert-type q8_0`) |
| `qwen36_tiny/ref_logprobs.tsv` | `python3 07_Tests/SystemTest/make_tiny_ref_logprobs.py qwen36_tiny qwen36_tiny/ref_full.json` (torch): teacher-forced f32 forward over `full_ids`, per scored position the target log-prob and the top-5, in the `PPL_DUMP` format below |

## Commands

```sh
cd 06_Code/c && make qwen36
for cap in 1 2 8; do
  COLI_DENSE_I8=0 SNAP=qwen36_tiny_c        ./qwen36 "$cap" 8 qwen36_tiny/ref_full.json   # container, unchanged
  COLI_DENSE_I8=0 SNAP=qwen36_tiny_f32.gguf ./qwen36 "$cap" 8 qwen36_tiny/ref_full.json   # GGUF F32
done
COLI_DENSE_I8=0 SNAP=qwen36_tiny_f16.gguf  ./qwen36 2 8 qwen36_tiny/ref_full.json        # reported
COLI_DENSE_I8=0 SNAP=qwen36_tiny_q8_0.gguf ./qwen36 2 8 qwen36_tiny/ref_full.json        # reported
PPL=1 PPL_DUMP=dump_f32.tsv COLI_DENSE_I8=0 SNAP=qwen36_tiny_f32.gguf ./qwen36 8 8 qwen36_tiny/ref_full.json
python3 07_Tests/SystemTest/compare_logprobs.py qwen36_tiny/ref_logprobs.tsv dump_f32.tsv  # E1 subset
```

`SNAP` pointing at a `.gguf` file (or a directory holding one GGUF / a split set)
selects the GGUF source; everything else about the invocation is upstream's.

### `PPL_DUMP` format (FR-31, decision A4)

With `PPL=1`, `PPL_DUMP=<file>` makes `tf_nll` write, after the usual `TF-NLL:` line:

```
# sylph ppl-dump v1 model=<SNAP> vocab=<V> scored=<N> topk=5
<pos>\t<target_id>\t<logprob_target>\t<tail>
```

one line per scored position `pos` (0-based index into `full_ids`, the positions
`tf_nll` scores), `logprob_target` in nats with 7 significant digits, and `tail` the
text `coli_logprob_tail` produces for the serve protocol's `logprobs=k` channel
(` <lp> <id>:<lp> <id>:<lp> …`, top-k ids and their log-probs) — so the dump and the
serve channel can never disagree. `−Σ logprob_target / N` equals the printed TF-NLL
within 1e-6. The reference script writes the same format from torch.

## Pass criteria

1. **Container unchanged (NFR-5).** The container run prints `Matching tokens: 16/16`
   and exits 0 at cap 1, 2 and 8, exactly as upstream's job does today.
2. **GGUF F32 token-exact (FR-25).** The F32 GGUF run prints `Matching tokens: 16/16`
   and exits 0 at cap 1, 2 and 8 (cap 1 evicts on every routed expert: the slice loads
   and the `kq` slot bookkeeping are exercised). Its startup line names the source and
   the layout: `[GGUF] qwen35moe · 8 blocks (2 attention) · experts F32 … per expert ·
   dense F32 …`.
3. **E1 subset (FR-34).** `compare_logprobs.py`: top-1 identical at every scored
   position; mean |ΔNLL| ≤ 1e-4 nat and max |ΔNLL| ≤ 1e-3 nat between the F32 GGUF
   dump and the torch reference (proposal: f32 engine vs f32 torch on a tiny model;
   the first run records the observed values and the thresholds are set at 3× them,
   spec §4.3). The sum of the dump's log-probs reproduces the printed TF-NLL.
4. **Reported, not required.** F16 and Q8_0 GGUF: matching tokens and ΔNLL vs the
   reference, next to the int8 container's (`qwen36_tiny_c`) — the Q8_0 GGUF is
   expected within 3× of the container's ΔNLL (both are 8-bit experts).
5. **Tokenizer-less reference mode.** The GGUF written without `--tokenizer` loads
   and runs the reference-id mode (no `tokenizer.json` on disk); text mode refuses
   with the usual `[enc] no tokenizer` message.
6. **Automated.** All of the above runs as the `gguf-oracle` job of
   `.github/workflows/check.yml` (ubuntu, `pip install -r c/tools/oracle-requirements.txt`,
   the recipe of upstream's job plus the GGUF steps); the job is green on `main`.
   Its log is the record; a copy of the summary goes to `08_Documents/equivalence/`.

## Why the reference is torch, not the container

Spec FR-34 (v2) said the CI subset's reference would be "the F32 container path".
There is no such thing: `convert_qwen36.py` quantizes experts to int8 at best, so the
container is never lossless. The lossless pin is the F32 GGUF against the
**transformers forward pass**, which upstream's CI already runs on this fixture. FR-34
and architecture §10 were amended on 2026-10-06 accordingly.

## State

| Date | Result |
|---|---|
| 2026-10-06 | written; `tools/st2gguf.py`, the GGUF source in `qwen36.c`, `PPL_DUMP`, `make_tiny_ref_logprobs.py`, `compare_logprobs.py` and the `gguf-oracle` CI job are phase-3 deliverables. Later the same day: all of them implemented; the job is in `check.yml`, its first run decides this row. torch is not installable in the development container (download.pytorch.org is egress-blocked; the PyPI wheel downloads but is CUDA-linked and untested), so the first execution is the CI job, then the owner's machine. |
