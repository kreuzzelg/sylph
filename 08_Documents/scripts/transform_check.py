"""Compare layer-0 dense tensors between the owner's colibrì container (HF-named, f16 safetensors,
Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64) and the unsloth GGUF (qwen35moe) to learn which value
transforms llama.cpp's converter applied (norm +1? A_log -> -exp? conv layout? alpha/beta mapping?)."""
import json, struct, subprocess, sys, math
sys.path.insert(0, "../../06_Code/c")
import ggufinfo
R = "https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64/resolve/main/"
G = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

def fetch(url, start, end):
    out = subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-r", f"{start}-{end}", url], capture_output=True, check=True)
    return out.stdout

want = ["model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias",
        "model.layers.0.linear_attn.norm.weight", "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.linear_attn.in_proj_a.weight", "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.mlp.shared_expert_gate.weight", "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight"]
found = {}
for i in range(0, 41):
    url = f"{R}model-{i:05d}.safetensors"
    head = fetch(url, 0, 7)
    hlen = struct.unpack("<Q", head)[0]
    hdr = json.loads(fetch(url, 8, 8 + hlen - 1))
    hits = [w for w in want if w in hdr and w not in found]
    for w in hits:
        meta = hdr[w]; s, e = meta["data_offsets"]
        found[w] = (url, 8 + hlen + s, 8 + hlen + e - 1, meta["dtype"], meta["shape"])
    if len(found) == len(want): break
print("located", len(found), "of", len(want), "tensors in the colibrì container")

def f16s(b):
    import array
    a = array.array("H"); a.frombytes(b)
    out = []
    for h in a:
        s = -1.0 if h & 0x8000 else 1.0; e = (h >> 10) & 0x1F; m = h & 0x3FF
        if e == 0: out.append(s * (m / 1024.0) * 2.0 ** -14)
        elif e == 31: out.append(float("inf") * s if m == 0 else float("nan"))
        else: out.append(s * (1 + m / 1024.0) * 2.0 ** (e - 15))
    return out
def f32s(b):
    import array
    a = array.array("f"); a.frombytes(b); return list(a)
def bf16s(b):
    import array
    a = array.array("H"); a.frombytes(b)
    return [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in a]
def decode(dtype, b):
    return {"F16": f16s, "F32": f32s, "BF16": bf16s}[dtype](b)

hf = {}
for w, (url, s, e, dt, shape) in found.items():
    hf[w] = (decode(dt, fetch(url, s, e)), dt, shape)
    print(f"  HF {w:<52} {dt} {shape}")

parts = ggufinfo.open_set("./work/unsloth")
gg = {}
tnames = ["blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
          "blk.0.ssm_norm.weight", "blk.0.ssm_conv1d.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
          "blk.0.ffn_gate_inp_shexp.weight", "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight"]
tmap = {t.name: t for p in parts for t in p.tensors}
for n in tnames:
    t = tmap[n]
    assert t.type_name == "F32", (n, t.type_name)
    gg[n] = f32s(fetch(G, t.off, t.off + t.nbytes - 1))
    print(f"  GG {n:<40} F32 ne={t.ne[:2]}")

def stats(name, a, b, note=""):
    n = min(len(a), len(b))
    d = [abs(x - y) for x, y in zip(a[:n], b[:n])]
    print(f"{name:<58} n={n:<7} max|a-b|={max(d):.3e}  a[:3]={[round(x,5) for x in a[:3]]} b[:3]={[round(x,5) for x in b[:3]]} {note}")

print("\n=== value-transform audit (HF container vs GGUF) ===")
stats("attn_norm vs input_layernorm", gg["blk.0.attn_norm.weight"], hf["model.layers.0.input_layernorm.weight"][0])
stats("attn_norm vs 1+input_layernorm", gg["blk.0.attn_norm.weight"], [1 + x for x in hf["model.layers.0.input_layernorm.weight"][0]])
stats("post_attention_norm vs 1+post_ln", gg["blk.0.post_attention_norm.weight"], [1 + x for x in hf["model.layers.0.post_attention_layernorm.weight"][0]])
stats("ssm_norm vs linear_attn.norm.weight", gg["blk.0.ssm_norm.weight"], hf["model.layers.0.linear_attn.norm.weight"][0])
stats("ssm_norm vs 1+norm.weight", gg["blk.0.ssm_norm.weight"], [1 + x for x in hf["model.layers.0.linear_attn.norm.weight"][0]])
alog = hf["model.layers.0.linear_attn.A_log"][0]
stats("ssm_a vs A_log", gg["blk.0.ssm_a"], alog)
stats("ssm_a vs -exp(A_log)", gg["blk.0.ssm_a"], [-math.exp(x) for x in alog])
stats("ssm_dt.bias vs dt_bias", gg["blk.0.ssm_dt.bias"], hf["model.layers.0.linear_attn.dt_bias"][0])
conv_hf = hf["model.layers.0.linear_attn.conv1d.weight"]
print("  conv1d HF shape", conv_hf[2], "GGUF ne", tmap["blk.0.ssm_conv1d.weight"].ne[:2])
stats("ssm_conv1d vs conv1d (same order)", gg["blk.0.ssm_conv1d.weight"], conv_hf[0])
# HF [conv_dim, 1, k] flattened = [c][k]; GGUF ne0=4 => row-major [8192][4] identical if ne order is (k fastest)
stats("ssm_alpha vs in_proj_a", gg["blk.0.ssm_alpha.weight"], hf["model.layers.0.linear_attn.in_proj_a.weight"][0])
stats("ssm_alpha vs in_proj_b", gg["blk.0.ssm_alpha.weight"], hf["model.layers.0.linear_attn.in_proj_b.weight"][0])
stats("ssm_beta vs in_proj_b", gg["blk.0.ssm_beta.weight"], hf["model.layers.0.linear_attn.in_proj_b.weight"][0])
stats("ssm_beta vs in_proj_a", gg["blk.0.ssm_beta.weight"], hf["model.layers.0.linear_attn.in_proj_a.weight"][0])
stats("ffn_gate_inp_shexp vs shared_expert_gate", gg["blk.0.ffn_gate_inp_shexp.weight"], hf["model.layers.0.mlp.shared_expert_gate.weight"][0])
stats("attn_q_norm vs q_norm", gg["blk.3.attn_q_norm.weight"], hf["model.layers.3.self_attn.q_norm.weight"][0])
stats("attn_q_norm vs 1+q_norm", gg["blk.3.attn_q_norm.weight"], [1 + x for x in hf["model.layers.3.self_attn.q_norm.weight"][0]])
stats("attn_k_norm vs 1+k_norm", gg["blk.3.attn_k_norm.weight"], [1 + x for x in hf["model.layers.3.self_attn.k_norm.weight"][0]])
