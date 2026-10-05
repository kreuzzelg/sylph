# Inspection report: Qwen3.6-35B-A3B GGUFs (`qwen35moe`) and the converter's value transforms

Date: 2026-10-05 · Reader: phase 1 (`c/gguf.h`, `c/ggufinfo.py`) · Files: `unsloth/Qwen3.6-35B-A3B-GGUF`
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.13 GB) and `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
`Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` (22.29 GB). Method: first 48 MB of each file (metadata
region ≈ 20 MB with the 248k-token vocabulary), sparse-extended to the true size; selected
small tensors range-fetched for the value audit (~60 MB in total).

## Headline

- Architecture **`qwen35moe`** (not `qwen3next`), pre-tokenizer **`qwen35`**, 248 320 tokens,
  `eos 248046`, `bos 248044`, `add_bos_token false`, chat template embedded.
- Both files use only `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F32` (+ two `BF16` routers in
  bartowski's MTP block): **everything in the v1 type set**, `coli doctor` passes all GGUF
  checks on both (`memory.ram` fails on this 15 GB container, correctly).
- 40 blocks; 10 attention (`i % 4 == 3`), 30 Gated DeltaNet; every block has a 256-expert
  MoE + shared expert. Experts **1.90 MB** (unsloth) / **2.04 MB** (bartowski) each,
  19.6 / 20.4 GB in total; dense set **2.56 / 1.91 GB**.
- bartowski's file carries an **MTP block** (`blk.40.nextn.*`, `eh_proj Q8_0`); unsloth's does not.

## Metadata (`qwen35moe.*`)

| key | value | | key | value |
|---|---|---|---|---|
| `block_count` | 40 | | `attention.head_count` / `head_count_kv` | 16 / 2 |
| `embedding_length` | 2048 | | `attention.key_length` / `value_length` | 256 / 256 |
| `context_length` | 262 144 | | `attention.layer_norm_rms_epsilon` | 1e-6 |
| `expert_count` / `expert_used_count` | 256 / 8 | | `full_attention_interval` | 4 |
| `expert_feed_forward_length` | 512 | | `rope.dimension_count` | 64 |
| `expert_shared_feed_forward_length` | 512 | | `rope.dimension_sections` | [11, 11, 10, 0] |
| `ssm.conv_kernel` | 4 | | `rope.freq_base` | 1e7 |
| `ssm.group_count` (k heads) | 16 | | `ssm.state_size` (head dim) | 128 |
| `ssm.time_step_rank` (v heads) | 32 | | `ssm.inner_size` (value dim) | 4096 |

Absent (container defaults apply): `expert_weights_norm`, `expert_gating_func`,
`expert_weights_scale`, router bias tensors.

## Tensor inventory (unsloth; bartowski differs only in types and the MTP block)

```
token_embd Q8_0 {2048, 248320}     output Q6_K (unsloth) / Q8_0 (bartowski)     output_norm F32
DeltaNet block (30×):  attn_norm F32 · post_attention_norm F32
   attn_qkv Q8_0 {2048→8192}  attn_gate Q8_0 {2048→4096}  ssm_alpha F32 {2048→32}  ssm_beta F32 {2048→32}
   ssm_conv1d F32 {4, 8192}  ssm_dt.bias F32 [32]  ssm_a F32 [32]  ssm_norm F32 [128]  ssm_out Q8_0 {4096→2048}
Attention block (10×): attn_norm · post_attention_norm · attn_q Q8_0 {2048→8192} · attn_k/v Q8_0 {2048→512}
   attn_q_norm / attn_k_norm F32 [256] · attn_output Q8_0 {4096→2048}
MoE (40×): ffn_gate_inp F32 {2048→256} · ffn_gate_inp_shexp F32 {2048→1}
   ffn_gate_exps / ffn_up_exps Q4_K {2048, 512, 256} · ffn_down_exps Q5_K (37) | Q6_K (3) {512, 2048, 256}
   ffn_gate_shexp / ffn_up_shexp Q8_0 {2048→512} · ffn_down_shexp Q8_0 {512→2048}
bartowski extra: blk.40.* (MTP) all Q8_0/F32/BF16 + nextn.{eh_proj Q8_0 {4096→2048}, enorm, hnorm, shared_head_norm}
```

Type mix: unsloth `Q4_K 55% · Q5_K 31% · Q8_0 9% · Q6_K 5%`; bartowski
`Q4_K 70% · Q6_K 24% · Q8_0 5%` (down-projections `Q6_K` on 20 of 41 blocks).

## Value-transform audit (GGUF vs the owner's HF-named container)

Compared layer 0 (DeltaNet) and layer 3 (attention) of unsloth's GGUF with
`Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64` (dense tensors stored f16 under HF names).
Scripts: `scripts/transform_check.py`, `scripts/perm_check.py`.

| tensor pair | finding | max error |
|---|---|---|
| `attn_norm` vs `input_layernorm` | GGUF = **1 + HF** | 0 |
| `post_attention_norm` vs `post_attention_layernorm` | **1 + HF** | 0 |
| `attn_q_norm` / `attn_k_norm` vs `q_norm` / `k_norm` | **1 + HF** | 0 |
| `ssm_norm` vs `linear_attn.norm.weight` | identical (no offset) | 0 |
| `ssm_a` vs `A_log` | GGUF = **−exp(A_log)**, heads permuted | 1.7e-6 |
| `ssm_dt.bias` vs `dt_bias` | permuted | 0 |
| `ssm_alpha` / `ssm_beta` vs `in_proj_a` / `in_proj_b` | rows permuted (α↔a, β↔b) | 3e-8 |
| `ssm_conv1d` `{4,8192}` vs `conv1d.weight` `[8192,1,4]` | same row-major layout; **v channels** (4096..8191) permuted, q/k channels identity | 0 |
| `attn_qkv` vs `in_proj_qkv` | q/k rows identity, **v rows permuted** | rel 0.005 (Q8_0 noise) |
| `attn_gate` vs `in_proj_z` | rows permuted | rel 0.005 |
| `ssm_out` vs `out_proj` | **input columns** permuted | rel 0.006 |
| `ffn_gate_inp_shexp` vs `shared_expert_gate` | identical | 3e-8 |

**The permutation**: GGUF value-head `i` = HF value-head `perm[i]`,
`perm = [0,2,4,…,30, 1,3,5,…,31]` (even heads first, then odd). In HF order v-head `h`
pairs with k-head `h / 2`; in GGUF order v-head `i` pairs with k-head `i mod 16` — the
grouped layout llama.cpp's recurrence uses. Query/key heads are not permuted.
Stored as `scripts/qwen36_deltanet_perm.json`.

## Consequences (folded into the specification and architecture)

1. The engine can stay on HF order: undo the permutation and the `1+w` / `−exp` transforms
   at load (all lossless), so `deltanet()`, `attention()`, `rmsnorm_row` are unchanged and the
   container oracle keeps proving them.
2. `Q8_0` is the only dense quant type; a lossless split into int8 + per-32 f32 scales feeds
   upstream's group-scaled int8 kernels. The one new dense kernel is `Q6_K` for unsloth's
   `output.weight`.
3. Experts need `Q4_K` + `Q5_K`/`Q6_K` kernels and a slot that holds raw blocks.
4. Reference choice matters: unsloth and bartowski differ in `ffn_down_exps` precision and
   `output.weight`; Ollama's library blob (not inspectable from here) may differ again. The
   equivalence harness must always run both engines on the **same file**.

## Reproduction

```sh
B=https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main
curl -L -r 0-50331647 -o Qwen3.6-35B-A3B-UD-Q4_K_M.gguf $B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
python3 -c "import os; os.truncate('Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', 22134528992)"
06_Code/c/coli gguf inspect Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --tensors
06_Code/c/coli doctor --model Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --deep
python3 08_Documents/scripts/perm_check.py      # needs network; caches range fetches under ./work/cache
```
