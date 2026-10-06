#!/usr/bin/env python3
"""Write a tiny Qwen3.6-shaped HF snapshot without torch (see src_facade.md).

Same geometry and tensor names as upstream's tools/make_qwen36_tiny.py default
preset (hidden 64, 8 blocks with attention at 3 and 7, 4 query / 2 kv heads of
16, rope over 8 dims, 8 experts top-2 of width 32, shared expert 32, vocab 320,
DeltaNet 4 key heads / 8 value heads of 8, conv kernel 4), but the weights are
seeded random numbers (bf16-representable, as a real checkpoint's) written straight
into one float32 safetensors file plus config.json. No transformers run, so no reference ids: this fixture pins the
PLUMBING (two sources, one engine, identical tensors), the torch-built fixture
in SystemTest/lossless_oracle.md pins the MATH.

    python3 07_Tests/IntegrationTest/make_tiny_qwen36_hf.py <out_dir> [--seed N] [--dtype f32|bf16]

Standard library only.
"""
import json
import math
import random
import struct
import sys
from pathlib import Path

GEOMETRY = dict(hidden=64, n_layers=8, q_heads=4, kv_heads=2, head_dim=16, rope_dim=8,
                n_experts=8, topk=2, inter=32, shared_inter=32, vocab=320,
                dn_key_heads=4, dn_value_heads=8, dn_kdim=8, dn_vdim=8, conv_k=4,
                eps=1e-6, theta=10000.0)


def layer_types(g):
    return ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(g["n_layers"])]


def tensor_shapes(g):
    """HF tensor name -> shape, in the order transformers would list them."""
    H, V = g["hidden"], g["vocab"]
    conv_dim = 2 * g["dn_key_heads"] * g["dn_kdim"] + g["dn_value_heads"] * g["dn_vdim"]
    vtot = g["dn_value_heads"] * g["dn_vdim"]
    shapes = {"model.embed_tokens.weight": [V, H]}
    for i, kind in enumerate(layer_types(g)):
        p = f"model.layers.{i}."
        shapes[p + "input_layernorm.weight"] = [H]
        shapes[p + "post_attention_layernorm.weight"] = [H]
        if kind == "full_attention":
            shapes[p + "self_attn.q_proj.weight"] = [g["q_heads"] * g["head_dim"] * 2, H]   # output gate doubles q
            shapes[p + "self_attn.k_proj.weight"] = [g["kv_heads"] * g["head_dim"], H]
            shapes[p + "self_attn.v_proj.weight"] = [g["kv_heads"] * g["head_dim"], H]
            shapes[p + "self_attn.o_proj.weight"] = [H, g["q_heads"] * g["head_dim"]]
            shapes[p + "self_attn.q_norm.weight"] = [g["head_dim"]]
            shapes[p + "self_attn.k_norm.weight"] = [g["head_dim"]]
        else:
            shapes[p + "linear_attn.in_proj_qkv.weight"] = [conv_dim, H]
            shapes[p + "linear_attn.in_proj_z.weight"] = [vtot, H]
            shapes[p + "linear_attn.in_proj_b.weight"] = [g["dn_value_heads"], H]
            shapes[p + "linear_attn.in_proj_a.weight"] = [g["dn_value_heads"], H]
            shapes[p + "linear_attn.conv1d.weight"] = [conv_dim, 1, g["conv_k"]]
            shapes[p + "linear_attn.dt_bias"] = [g["dn_value_heads"]]
            shapes[p + "linear_attn.A_log"] = [g["dn_value_heads"]]
            shapes[p + "linear_attn.norm.weight"] = [g["dn_vdim"]]
            shapes[p + "linear_attn.out_proj.weight"] = [H, vtot]
        shapes[p + "mlp.gate.weight"] = [g["n_experts"], H]
        for e in range(g["n_experts"]):
            shapes[p + f"mlp.experts.{e}.gate_proj.weight"] = [g["inter"], H]
            shapes[p + f"mlp.experts.{e}.up_proj.weight"] = [g["inter"], H]
            shapes[p + f"mlp.experts.{e}.down_proj.weight"] = [H, g["inter"]]
        shapes[p + "mlp.shared_expert.gate_proj.weight"] = [g["shared_inter"], H]
        shapes[p + "mlp.shared_expert.up_proj.weight"] = [g["shared_inter"], H]
        shapes[p + "mlp.shared_expert.down_proj.weight"] = [H, g["shared_inter"]]
        shapes[p + "mlp.shared_expert_gate.weight"] = [1, H]
    shapes["model.norm.weight"] = [H]
    shapes["lm_head.weight"] = [V, H]
    return shapes


def bf16_round(v):
    """Round to the nearest bf16-representable float: real checkpoints are bf16, and it
    keeps the converter's 1 + w exact in f32 (so the un-transform is bit-exact)."""
    b = struct.unpack("<I", struct.pack("<f", v))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


def values(name, n, rng):
    """Plausible magnitudes per tensor kind (what matters is that both sources see
    the same numbers; the kinds only keep the engine's math well-conditioned).
    Every value is bf16-representable, as in a real checkpoint."""
    return [bf16_round(v) for v in raw_values(name, n, rng)]


def raw_values(name, n, rng):
    if name.endswith("A_log"):
        return [math.log(rng.uniform(1.0, 16.0)) for _ in range(n)]
    if name.endswith("dt_bias"):
        return [rng.uniform(-4.0, -1.0) for _ in range(n)]
    if "norm" in name or "layernorm" in name:
        return [1.0 + rng.gauss(0.0, 0.05) for _ in range(n)]
    if name.endswith("conv1d.weight"):
        return [rng.gauss(0.0, 0.3) for _ in range(n)]
    return [rng.gauss(0.0, 0.02) for _ in range(n)]


def f32_to_bf16_bytes(vals):
    out = bytearray()
    for v in vals:
        b = struct.unpack("<I", struct.pack("<f", v))[0]
        b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16        # round to nearest even
        out += struct.pack("<H", b)
    return bytes(out)


def write_safetensors(path, tensors, dtype):
    """tensors: list of (name, shape, float list). One file, offsets contiguous."""
    header, blobs, off = {}, [], 0
    esize = 4 if dtype == "f32" else 2
    for name, shape, vals in tensors:
        blob = struct.pack(f"<{len(vals)}f", *vals) if dtype == "f32" else f32_to_bf16_bytes(vals)
        header[name] = {"dtype": "F32" if dtype == "f32" else "BF16", "shape": shape, "data_offsets": [off, off + len(blob)]}
        blobs.append(blob); off += len(blob)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for b in blobs: f.write(b)
    return off + 8 + len(hb), esize


def config(g, dtype):
    return {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "dtype": "float32" if dtype == "f32" else "bfloat16",
        "vocab_size": g["vocab"], "hidden_size": g["hidden"], "intermediate_size": g["hidden"] * 2,
        "num_hidden_layers": g["n_layers"], "num_attention_heads": g["q_heads"], "num_key_value_heads": g["kv_heads"],
        "head_dim": g["head_dim"], "attn_output_gate": True, "attention_bias": False, "attention_dropout": 0.0,
        "num_experts": g["n_experts"], "num_experts_per_tok": g["topk"], "moe_intermediate_size": g["inter"],
        "shared_expert_intermediate_size": g["shared_inter"], "norm_topk_prob": False,
        "layer_types": layer_types(g), "full_attention_interval": 4,
        "linear_conv_kernel_dim": g["conv_k"], "linear_key_head_dim": g["dn_kdim"], "linear_value_head_dim": g["dn_vdim"],
        "linear_num_key_heads": g["dn_key_heads"], "linear_num_value_heads": g["dn_value_heads"],
        "rms_norm_eps": g["eps"], "rope_theta": g["theta"], "partial_rotary_factor": g["rope_dim"] / g["head_dim"],
        "rope_parameters": {"rope_type": "default", "rope_theta": g["theta"], "partial_rotary_factor": g["rope_dim"] / g["head_dim"]},
        "max_position_embeddings": 512, "hidden_act": "silu", "tie_word_embeddings": False, "use_cache": True,
        "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": g["vocab"] - 1,
        "transformers_version": "synthetic (07_Tests/IntegrationTest/make_tiny_qwen36_hf.py)",
    }


def main(argv):
    if len(argv) < 1:
        sys.exit(__doc__)
    out = Path(argv[0]); seed = 20261006; dtype = "f32"
    for i, a in enumerate(argv):
        if a == "--seed": seed = int(argv[i + 1])
        if a == "--dtype": dtype = argv[i + 1]
    g = dict(GEOMETRY)
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    tensors = []
    for name, shape in tensor_shapes(g).items():
        n = 1
        for s in shape: n *= s
        tensors.append((name, shape, values(name, n, rng)))
    nbytes, _ = write_safetensors(out / "model.safetensors", tensors, dtype)
    (out / "config.json").write_text(json.dumps(config(g, dtype), indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps({"bos_token_id": 1, "eos_token_id": g["vocab"] - 1, "pad_token_id": 0}) + "\n")
    print(f"{out}: {len(tensors)} tensors, {nbytes / 1e6:.2f} MB, dtype {dtype}, seed {seed}")


if __name__ == "__main__":
    main(sys.argv[1:])
