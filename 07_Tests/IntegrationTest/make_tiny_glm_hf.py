#!/usr/bin/env python3
"""Write a tiny GLM-5.2-shaped (glm_moe_dsa) HF snapshot without torch (glm_assembly.md).

Geometry of upstream's tools/make_glm_oracle.py (hidden 128 / moe 32 by default; --wide
gives 256 / 256 so K-quant expert blocks fit): 5 blocks (3 dense + 2 MoE), 4 heads, 8 routed
experts top-2 + 1 shared, MLA (q_lora 64, kv_lora 32, qk_nope 24, qk_rope 8, v_head 32),
DSA indexer (2 heads x 16, top_k 4096) on every block, vocab 256. Seeded random weights,
bf16-representable, written as one float32 safetensors + config.json. --mtp adds the NextN
block as model.layers.5.* (eh_proj [hidden, 2*hidden], enorm, hnorm, shared_head.norm + a
full MoE block), the layout tools/convert_fp8_to_int4.py --mtp and llama.cpp both read.

    python3 07_Tests/IntegrationTest/make_tiny_glm_hf.py <out_dir> [--seed N] [--wide] [--mtp]

Standard library only; reuses the writer of make_tiny_qwen36_hf.py.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tiny_qwen36_hf import write_safetensors, bf16_round  # noqa: E402

GEOMETRY = dict(hidden=128, n_layers=5, first_dense=3, dense_inter=64, moe_inter=32, n_experts=8, topk=2,
                n_shared=1, heads=4, q_lora=64, kv_lora=32, qk_nope=24, qk_rope=8, v_head=32,
                index_topk=4096, index_hd=16, index_nh=2, vocab=256, eps=1e-5, theta=10000.0, routed_scale=2.5)


def tensor_shapes(g, mtp):
    H, L = g["hidden"], g["n_layers"]
    qk_head = g["qk_nope"] + g["qk_rope"]
    shapes = {"model.embed_tokens.weight": [g["vocab"], H], "model.norm.weight": [H], "lm_head.weight": [g["vocab"], H]}
    n_blocks = L + (1 if mtp else 0)
    for i in range(n_blocks):
        p = f"model.layers.{i}."
        shapes[p + "input_layernorm.weight"] = [H]
        shapes[p + "post_attention_layernorm.weight"] = [H]
        shapes[p + "self_attn.q_a_proj.weight"] = [g["q_lora"], H]
        shapes[p + "self_attn.q_a_layernorm.weight"] = [g["q_lora"]]
        shapes[p + "self_attn.q_b_proj.weight"] = [g["heads"] * qk_head, g["q_lora"]]
        shapes[p + "self_attn.kv_a_proj_with_mqa.weight"] = [g["kv_lora"] + g["qk_rope"], H]
        shapes[p + "self_attn.kv_a_layernorm.weight"] = [g["kv_lora"]]
        shapes[p + "self_attn.kv_b_proj.weight"] = [g["heads"] * (g["qk_nope"] + g["v_head"]), g["kv_lora"]]
        shapes[p + "self_attn.o_proj.weight"] = [H, g["heads"] * g["v_head"]]
        if i < g["first_dense"] and i < L:
            shapes[p + "mlp.gate_proj.weight"] = [g["dense_inter"], H]
            shapes[p + "mlp.up_proj.weight"] = [g["dense_inter"], H]
            shapes[p + "mlp.down_proj.weight"] = [H, g["dense_inter"]]
        else:
            shapes[p + "mlp.gate.weight"] = [g["n_experts"], H]
            shapes[p + "mlp.gate.e_score_correction_bias"] = [g["n_experts"]]
            for e in range(g["n_experts"]):
                shapes[p + f"mlp.experts.{e}.gate_proj.weight"] = [g["moe_inter"], H]
                shapes[p + f"mlp.experts.{e}.up_proj.weight"] = [g["moe_inter"], H]
                shapes[p + f"mlp.experts.{e}.down_proj.weight"] = [H, g["moe_inter"]]
            sI = g["moe_inter"] * g["n_shared"]
            shapes[p + "mlp.shared_experts.gate_proj.weight"] = [sI, H]
            shapes[p + "mlp.shared_experts.up_proj.weight"] = [sI, H]
            shapes[p + "mlp.shared_experts.down_proj.weight"] = [H, sI]
        if i < L:   # the indexer lives on the trunk blocks only
            shapes[p + "self_attn.indexer.wq_b.weight"] = [g["index_nh"] * g["index_hd"], g["q_lora"]]
            shapes[p + "self_attn.indexer.wk.weight"] = [g["index_hd"], H]
            shapes[p + "self_attn.indexer.weights_proj.weight"] = [g["index_nh"], H]
            shapes[p + "self_attn.indexer.k_norm.weight"] = [g["index_hd"]]
            shapes[p + "self_attn.indexer.k_norm.bias"] = [g["index_hd"]]
        if mtp and i == L:
            shapes[p + "eh_proj.weight"] = [H, 2 * H]
            shapes[p + "enorm.weight"] = [H]
            shapes[p + "hnorm.weight"] = [H]
            shapes[p + "shared_head.norm.weight"] = [H]
    return shapes


def raw_values(name, n, rng):
    if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
        return [1.0 + rng.uniform(-0.05, 0.05) for _ in range(n)]
    if name.endswith("k_norm.bias"):
        return [rng.uniform(-0.02, 0.02) for _ in range(n)]
    if name.endswith("e_score_correction_bias"):
        return [-0.1 + 0.2 * k / max(1, n - 1) for k in range(n)]
    if "mlp.gate.weight" in name or "weights_proj" in name:
        return [rng.gauss(0, 0.2) for _ in range(n)]
    return [rng.gauss(0, 0.05) for _ in range(n)]


def config(g, mtp):
    c = {"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "torch_dtype": "float32",
         "vocab_size": g["vocab"], "hidden_size": g["hidden"], "intermediate_size": g["dense_inter"],
         "moe_intermediate_size": g["moe_inter"], "num_hidden_layers": g["n_layers"], "first_k_dense_replace": g["first_dense"],
         "num_attention_heads": g["heads"], "num_key_value_heads": g["heads"], "n_routed_experts": g["n_experts"],
         "num_experts_per_tok": g["topk"], "n_shared_experts": g["n_shared"], "q_lora_rank": g["q_lora"], "kv_lora_rank": g["kv_lora"],
         "qk_nope_head_dim": g["qk_nope"], "qk_rope_head_dim": g["qk_rope"], "v_head_dim": g["v_head"],
         "index_topk": g["index_topk"], "index_head_dim": g["index_hd"], "index_n_heads": g["index_nh"],
         "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": g["routed_scale"],
         "rope_parameters": {"rope_type": "default", "rope_theta": g["theta"]}, "tie_word_embeddings": False,
         "rms_norm_eps": g["eps"], "attention_bias": False, "max_position_embeddings": 4096,
         "eos_token_id": [g["vocab"] - 1, g["vocab"] - 2, g["vocab"] - 3], "pad_token_id": 0}
    if mtp:
        c["num_nextn_predict_layers"] = 1
    return c


def main(argv):
    if len(argv) < 1:
        sys.exit(__doc__)
    out = Path(argv[0]); seed = 20261009; mtp = "--mtp" in argv
    g = dict(GEOMETRY)
    if "--wide" in argv:
        g["hidden"] = 256; g["moe_inter"] = 256
    for i, a in enumerate(argv):
        if a == "--seed": seed = int(argv[i + 1])
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    tensors = []
    for name, shape in tensor_shapes(g, mtp).items():
        n = 1
        for s in shape: n *= s
        tensors.append((name, shape, [bf16_round(v) for v in raw_values(name, n, rng)]))
    nbytes, _ = write_safetensors(out / "model.safetensors", tensors, "f32")
    (out / "config.json").write_text(json.dumps(config(g, mtp), indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": g["vocab"] - 1, "pad_token_id": 0}) + "\n")
    print(f"{out}: {len(tensors)} tensors, {nbytes / 1e6:.2f} MB, seed {seed}, hidden {g['hidden']}, mtp {'yes' if mtp else 'no'}")


if __name__ == "__main__":
    main(sys.argv[1:])
