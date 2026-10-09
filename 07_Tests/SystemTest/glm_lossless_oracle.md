# System test — GLM-5.2 lossless oracle: upstream's tiny `glm_moe_dsa` from an F32 GGUF matches the container run and the torch reference (phase 6)

Written 2026-10-09, before phase 6. Specification FR-18, FR-25 (twin for `glm-dsa`), NFR-3,
NFR-5; architecture v1 §7, §11.2; `../IntegrationTest/glm_assembly.md` for the plumbing.

Upstream's own GLM oracle (`tools/make_glm_oracle.py` → `glm_tiny/` + `ref_glm.json`;
`.github/workflows/ci.yml` "GLM oracle": `SNAP=./glm_tiny TF=1 COLI_TEMP=0 ORACLE_STRICT=1
ORACLE_TF_MAX_MISMATCHES=2 ./colibri 64 16 16`) is the reference. Upstream tolerates **two**
teacher-forcing mismatches ("FP near-ties are toolchain-dependent"); this test keeps that
tolerance against torch and adds the stricter statement sylph can make: the **GGUF run and
the container run of the same snapshot are identical**, because they are the same engine on
the same numbers.

## Inputs

| Input | How it is made |
|---|---|
| `glm_tiny/` (+ `ref_glm.json` beside it, in `c/`) | `python3 tools/make_glm_oracle.py` (torch + transformers, `tools/requirements-glm53-tiny.txt`-class deps; hidden 128, moe 32, 5 blocks, 8 experts, MLA + DSA indexer; seeded) — f32 safetensors + `config.json` |
| `glm_tiny_f32.gguf` | `python3 tools/st2gguf.py glm_tiny --arch glm-dsa --out glm_tiny_f32.gguf --type f32` (absorbed MLA split written as llama.cpp does) |
| `glm_tiny_q8_0.gguf`, `glm_tiny_xq8_0.gguf` | `--type q8_0 --expert-type q8_0`; `--type f32 --expert-type q8_0` (reported, not required) |
| `glm_tiny_fmt4/` | `python3 tools/make_glm_oracle.py --fmt4` → the quantized container twin (hidden 256) with its own `ref_glm.json`; and `glm_tiny_fmt4_q4k.gguf` from the **dequantized** snapshot it writes (reported) |
| container run | `SNAP=./glm_tiny` as upstream runs it (f32 safetensors read directly: `./colibri 64 16 16` = cap 64, ebits 16, dbits 16) |

## Commands

```sh
cd 06_Code/c && make colibri
python3 tools/make_glm_oracle.py
python3 tools/st2gguf.py glm_tiny --arch glm-dsa --out glm_tiny_f32.gguf --type f32
# container (upstream's recipe) and GGUF, teacher forcing and free decode
for src in ./glm_tiny glm_tiny_f32.gguf; do
  SNAP=$src REF=ref_glm.json TF=1 COLI_TEMP=0 ORACLE_STRICT=1 ORACLE_TF_MAX_MISMATCHES=2 ./colibri 64 16 16
  SNAP=$src REF=ref_glm.json COLI_TEMP=0 ORACLE_STRICT=1 ./colibri 64 16 16
done
for cap in 1 2 8; do SNAP=glm_tiny_f32.gguf REF=ref_glm.json COLI_TEMP=0 ./colibri $cap 16 16; done
```

## Pass criteria

1. **Container unchanged** (NFR-5): upstream's two oracle commands on `./glm_tiny` pass as
   before this phase (`ORACLE_STRICT=1`, ≤ 2 TF mismatches).
2. **GGUF == container**: the F32 GGUF run prints the same generated ids and the same TF
   prediction vector as the container run (byte-identical `C engine :` lines and TF
   summary), at cap 64 and at cap 1, 2, 8 — streaming never changes a number.
3. **GGUF vs torch**: `ORACLE_STRICT=1` passes on the F32 GGUF with upstream's tolerance
   (≤ 2 TF mismatches; free decode token-exact to the first near-tie as upstream's gate
   defines it).
4. **MLA split exercised**: the `[GGUF] glm-dsa …` line says `kv_b from attn_k_b/attn_v_b`;
   a run from a `--kv-b fused` file gives identical ids (policy 1 vs policy 2).
5. **Indexer**: `[DSA] indexer active` on both sources; `DSA=0` gives identical ids on both
   (the tiny model's `index_topk` exceeds the sequence, so the indexer is a no-op by
   construction — the point is that the GGUF arm wires it the same way).
6. Reported, not required: `Q8_0`, experts-only `Q8_0` and the `--fmt4` twin (dequantized
   snapshot → `Q4_K` experts vs the fmt4 container), with match counts.
7. Automated as a step of the `gguf-oracle` CI job (torch present there): container run,
   GGUF run, equality of their outputs, then `run_glm_assembly.py`.

## State

| Date | Result |
|---|---|
| 2026-10-09 | written; needs phase 6 (`st2gguf --arch glm-dsa`, the GGUF arm of `colibri.c`) — then the CI step. |
| 2026-10-09 | phase 6 implemented; the CI step is in `.github/workflows/check.yml` (job `gguf-oracle`): `make_glm_oracle.py`, `st2gguf --arch glm-dsa` (F32, fused, Q8_0), `tests/test_gguf_load_glm` suite + cross-checks, upstream's two oracle commands on `./glm_tiny` and on the F32 GGUF, `diff` of the `GLM C engine` lines (container == GGUF, cap 1/2/8, fused file, `DSA=0`), the TF mismatch counts compared, Q8_0 reported. No torch in the dev container: the first result is the next CI run (recorded in the tasks' state table). |
