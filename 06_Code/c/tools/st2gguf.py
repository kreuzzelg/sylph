#!/usr/bin/env python3
"""HF Qwen3.5/3.6 MoE snapshot -> `qwen35moe` GGUF, the way llama.cpp's converter writes it.

Standard library only. Written for the tiny oracle fixtures (07_Tests), where it
lets the engine be tested against an F32 GGUF without llama.cpp; it is not a
replacement for `convert_hf_to_gguf.py` on a 35B checkpoint (pure-Python
quantization, whole tensors in memory).

    python3 tools/st2gguf.py <hf_dir> --out <file.gguf> [--type f32|f16|bf16|q8_0]
                             [--expert-type f32|f16|bf16|q8_0] [--tokenizer tokenizer.json]
                             [--name NAME] [--dry-run]

What it reproduces (verified against unsloth's and bartowski's files, see
08_Documents/inspection-qwen36-gguf-2026-10-05.md and the architecture §7):
  * names per the HF <-> GGUF table (c/qwen35_names.h is the C copy), routed experts
    stacked into 3-D ffn_{gate,up,down}_exps, `output.weight` always written;
  * attn_norm / post_attention_norm / attn_q_norm / attn_k_norm stored as 1 + w;
  * ssm_a = -exp(A_log);
  * the DeltaNet value-head permutation h(j) = (j mod vk)*(vh/vk) + j div vk (GGUF head j
    holds HF head h(j)) on ssm_a, ssm_dt.bias, ssm_alpha/ssm_beta rows, the value third of
    attn_qkv rows and of ssm_conv1d channels, attn_gate rows and ssm_out input columns;
  * ssm_conv1d written {conv_k, conv_dim}; norms, routers, ssm vectors, conv1d and
    ffn_gate_inp_shexp always F32;
  * the KV set of the real files and, with --tokenizer, tokenizer.ggml.* from the JSON;
    without it, placeholder tokens (enough for reference-id mode).
Every tensor name is classified with tools/qwen36_tensor_kinds.py; an unknown name stops
the conversion, mtp.*/visual.* are skipped and counted.
"""
import argparse
import array
import json
import math
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gguf_fixture import GgufWriter, U32, I32, F32 as KV_F32, BOOL, STR, ARR, write_split_set  # noqa: E402
import qwen36_tensor_kinds as kinds  # noqa: E402

T_F32, T_F16, T_Q8_0, T_BF16 = 0, 1, 8, 30
TYPE_IDS = {"f32": T_F32, "f16": T_F16, "q8_0": T_Q8_0, "bf16": T_BF16}
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 7, "bf16": 32}        # general.file_type (llama_ftype)
TOK_NORMAL, TOK_CONTROL, TOK_USER_DEFINED, TOK_UNUSED = 1, 3, 4, 5


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---- safetensors -------------------------------------------------------------------------
class Snapshot:
    def __init__(self, hf_dir):
        self.dir = Path(hf_dir)
        self.tensors = {}       # name -> (path, dtype, shape, start, end)
        for shard in sorted(self.dir.glob("*.safetensors")):
            with open(shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                s, e = meta["data_offsets"]
                self.tensors[name] = (shard, meta["dtype"], meta["shape"], 8 + n + s, 8 + n + e)
        if not self.tensors:
            sys.exit(f"st2gguf: no safetensors shards in {hf_dir}")

    def read(self, name):
        """-> (shape, list of python floats, exact f32 values of the stored dtype)"""
        path, dtype, shape, s, e = self.tensors[name]
        with open(path, "rb") as f:
            f.seek(s); raw = f.read(e - s)
        n = (e - s) // {"F32": 4, "F16": 2, "BF16": 2}[dtype]
        if dtype == "F32":
            a = array.array("f"); a.frombytes(raw)
            if sys.byteorder != "little": a.byteswap()
            vals = a.tolist()
        elif dtype == "F16":
            vals = list(struct.unpack(f"<{n}e", raw))
        elif dtype == "BF16":
            u = array.array("H"); u.frombytes(raw)
            if sys.byteorder != "little": u.byteswap()
            vals = list(struct.unpack(f"<{n}f", struct.pack(f"<{n}I", *(v << 16 for v in u))))
        else:
            sys.exit(f"st2gguf: {name}: unsupported dtype {dtype}")
        return shape, vals


# ---- config -------------------------------------------------------------------------------
def load_config(hf_dir):
    cfg = json.loads((Path(hf_dir) / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    gen = {}
    gp = Path(hf_dir) / "generation_config.json"
    if gp.exists():
        try: gen = json.loads(gp.read_text(encoding="utf-8"))
        except ValueError: gen = {}
    g = {}
    g["hidden"] = int(text["hidden_size"]); g["n_layers"] = int(text["num_hidden_layers"]); g["vocab"] = int(text["vocab_size"])
    g["q_heads"] = int(text["num_attention_heads"]); g["kv_heads"] = int(text["num_key_value_heads"])
    g["head_dim"] = int(text.get("head_dim") or g["hidden"] // g["q_heads"])
    g["n_experts"] = int(text["num_experts"]); g["topk"] = int(text["num_experts_per_tok"])
    g["inter"] = int(text["moe_intermediate_size"]); g["shared_inter"] = int(text.get("shared_expert_intermediate_size") or g["inter"])
    g["ffn"] = int(text.get("intermediate_size") or g["inter"])
    g["vk"] = int(text["linear_num_key_heads"]); g["vh"] = int(text["linear_num_value_heads"])
    g["kdim"] = int(text["linear_key_head_dim"]); g["vdim"] = int(text["linear_value_head_dim"]); g["conv_k"] = int(text["linear_conv_kernel_dim"])
    g["eps"] = float(text.get("rms_norm_eps", 1e-6))
    rp = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    g["theta"] = float(rp.get("rope_theta", text.get("rope_theta", 10000.0)))
    prf = rp.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
    g["rope_dim"] = int(round(g["head_dim"] * float(prf)))
    g["mrope"] = rp.get("mrope_section", text.get("mrope_section"))
    g["ctx"] = int(text.get("max_position_embeddings", 32768))
    lt = text.get("layer_types")
    if isinstance(lt, list) and len(lt) == g["n_layers"]:
        g["layer_types"] = lt
    else:
        iv = int(text.get("full_attention_interval", 4))
        g["layer_types"] = ["full_attention" if (i + 1) % iv == 0 else "linear_attention" for i in range(g["n_layers"])]
    g["interval"] = int(text.get("full_attention_interval", 4))
    attn = [i for i, k in enumerate(g["layer_types"]) if k == "full_attention"]
    if attn and any((i + 1) % g["interval"] != 0 for i in attn) or len(attn) != g["n_layers"] // g["interval"]:
        print(f"st2gguf: note: layer_types do not follow full_attention_interval {g['interval']}; GGUF carries the interval only", file=sys.stderr)
    g["mtp"] = int(text.get("mtp_num_hidden_layers", 0) or 0)
    def tok_id(key, dflt):
        v = gen.get(key, text.get(key, cfg.get(key, dflt)))
        if isinstance(v, list): v = v[0] if v else dflt
        return int(v) if v is not None else dflt
    g["bos"] = tok_id("bos_token_id", 1); g["eos"] = tok_id("eos_token_id", g["vocab"] - 1); g["pad"] = tok_id("pad_token_id", 0)
    g["name"] = cfg.get("_name_or_path") or Path(hf_dir).name
    return g


# ---- transforms -----------------------------------------------------------------------------
def hf_head(j, vh, vk):
    """HF value head stored at GGUF head j."""
    r = vh // vk
    return (j % vk) * r + j // vk


def permute_head_rows(vals, vh, vk, per, cols, off=0):
    """rows [off, off + vh*per) are vh blocks of `per` rows; GGUF block j <- HF block hf_head(j)."""
    out = list(vals)
    for j in range(vh):
        h = hf_head(j, vh, vk)
        for r in range(per):
            d = (off + j * per + r) * cols; s = (off + h * per + r) * cols
            out[d:d + cols] = vals[s:s + cols]
    return out


def permute_head_cols(vals, rows, vh, vk, per):
    """columns are vh blocks of `per`; GGUF block j <- HF block hf_head(j)."""
    width = vh * per; out = list(vals)
    for row in range(rows):
        base = row * width
        for j in range(vh):
            h = hf_head(j, vh, vk)
            out[base + j * per: base + (j + 1) * per] = vals[base + h * per: base + (h + 1) * per]
    return out


# ---- output types -------------------------------------------------------------------------------
def pack_f32(vals):
    return struct.pack(f"<{len(vals)}f", *vals)


def pack_f16(vals):
    try:
        return struct.pack(f"<{len(vals)}e", *vals)
    except (OverflowError, struct.error):
        return b"".join(struct.pack("<e", max(-65504.0, min(65504.0, v))) for v in vals)


def pack_bf16(vals):
    bits = struct.unpack(f"<{len(vals)}I", pack_f32(vals))
    out = array.array("H", [((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF for b in bits])
    if sys.byteorder != "little": out.byteswap()
    return out.tobytes()


def pack_q8_0(vals):
    """ggml quantize_row_q8_0_ref in float32 arithmetic: d = amax/127, q = roundf(x/d)."""
    n = len(vals)
    if n % 32:
        raise ValueError(f"Q8_0 needs a multiple of 32 elements per row, got {n}")
    out = bytearray()
    for b in range(0, n, 32):
        blk = vals[b:b + 32]
        amax = 0.0
        for v in blk:
            av = abs(v)
            if av > amax: amax = av
        d = f32(amax / 127.0); idv = f32(1.0 / d) if d else 0.0
        out += struct.pack("<e", d)
        for v in blk:
            x = f32(f32(v) * idv)
            q = int(math.floor(abs(x) + 0.5)); q = -q if x < 0 else q
            out += struct.pack("<b", max(-128, min(127, q)))
    return bytes(out)


PACK = {"f32": pack_f32, "f16": pack_f16, "bf16": pack_bf16, "q8_0": pack_q8_0}


# ---- tokenizer -----------------------------------------------------------------------------------
def tokenizer_arrays(path, g):
    tj = json.loads(Path(path).read_text(encoding="utf-8"))
    model = tj.get("model", tj)
    vocab = model.get("vocab") or {}
    adds = tj.get("added_tokens") or []
    mx = max([v for v in vocab.values()] + [t["id"] for t in adds] + [g["vocab"] - 1])
    tokens = [None] * (mx + 1); types = [TOK_NORMAL] * (mx + 1)
    for piece, i in vocab.items():
        tokens[i] = piece
    for t in adds:
        tokens[t["id"]] = t["content"]; types[t["id"]] = TOK_CONTROL if t.get("special") else TOK_USER_DEFINED
    for i, p in enumerate(tokens):
        if p is None:
            tokens[i] = f"[PAD{i}]"; types[i] = TOK_UNUSED
    merges = []
    for m in model.get("merges") or []:
        merges.append(m if isinstance(m, str) else f"{m[0]} {m[1]}")
    return tokens, types, merges


def placeholder_tokens(g):
    return [f"t{i}" for i in range(g["vocab"])], [TOK_NORMAL] * g["vocab"], []


# ---- main --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hf_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="f32", choices=sorted(TYPE_IDS)); ap.add_argument("--expert-type", default=None, choices=sorted(TYPE_IDS))
    ap.add_argument("--tokenizer", default=None); ap.add_argument("--name", default=None); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--split", type=int, default=0, help="write N parts <stem>-0000k-of-0000N.gguf as llama.cpp's gguf-split does (expert_streaming.md)")
    a = ap.parse_args(argv)
    g = load_config(a.hf_dir)
    snap = Snapshot(a.hf_dir)
    dense_t, exp_t = a.type, a.expert_type or a.type
    prefix = kinds.resolve_prefix(snap.tensors.keys())

    # classify every name first (the converter's contract): layer -> {kind: name}
    layers = {}; globals_ = {}; skipped = {}
    for name in snap.tensors:
        what = kinds.classify(name, prefix)         # raises UnknownTensor
        if what[0] == "skip": skipped[what[1]] = skipped.get(what[1], 0) + 1
        elif what[0] == "global": globals_[what[1]] = name
        else: layers.setdefault(what[1], {})[what[2]] = name
    for need in ("embed_tokens.weight", "norm.weight", "lm_head.weight"):
        if need not in globals_: sys.exit(f"st2gguf: missing global tensor {need}")
    if sorted(layers) != list(range(g["n_layers"])):
        sys.exit(f"st2gguf: layers found {sorted(layers)[:5]}… do not match num_hidden_layers {g['n_layers']}")

    w = GgufWriter(alignment=32)
    w.add_str("general.architecture", "qwen35moe"); w.add_str("general.type", "model")
    w.add_str("general.name", a.name or g["name"]); w.add("general.file_type", U32, FILE_TYPE[dense_t])
    w.add("general.quantization_version", U32, 2)
    P = "qwen35moe."
    for key, val in (("block_count", g["n_layers"] + g["mtp"]), ("context_length", g["ctx"]), ("embedding_length", g["hidden"]), ("feed_forward_length", g["ffn"]),
                     ("attention.head_count", g["q_heads"]), ("attention.head_count_kv", g["kv_heads"]), ("attention.key_length", g["head_dim"]),
                     ("attention.value_length", g["head_dim"]), ("rope.dimension_count", g["rope_dim"]), ("expert_count", g["n_experts"]),
                     ("expert_used_count", g["topk"]), ("expert_feed_forward_length", g["inter"]), ("expert_shared_feed_forward_length", g["shared_inter"]),
                     ("full_attention_interval", g["interval"]), ("ssm.conv_kernel", g["conv_k"]), ("ssm.state_size", g["kdim"]),
                     ("ssm.group_count", g["vk"]), ("ssm.time_step_rank", g["vh"]), ("ssm.inner_size", g["vh"] * g["vdim"])):
        w.add(P + key, U32, int(val))
    if g["mtp"]: w.add(P + "nextn_predict_layers", U32, g["mtp"])
    w.add(P + "attention.layer_norm_rms_epsilon", KV_F32, g["eps"]); w.add(P + "rope.freq_base", KV_F32, g["theta"])
    if isinstance(g["mrope"], list): w.add_arr(P + "rope.dimension_sections", I32, list(g["mrope"]) + [0] * (4 - len(g["mrope"])))
    tokens, types, merges = tokenizer_arrays(a.tokenizer, g) if a.tokenizer else placeholder_tokens(g)
    w.add_str("tokenizer.ggml.model", "gpt2"); w.add_str("tokenizer.ggml.pre", "qwen35")
    w.add_arr("tokenizer.ggml.tokens", STR, tokens); w.add_arr("tokenizer.ggml.token_type", I32, types); w.add_arr("tokenizer.ggml.merges", STR, merges)
    w.add("tokenizer.ggml.bos_token_id", U32, g["bos"]); w.add("tokenizer.ggml.eos_token_id", U32, g["eos"]); w.add("tokenizer.ggml.padding_token_id", U32, g["pad"])
    w.add("tokenizer.ggml.add_bos_token", BOOL, False)

    stats = {}
    def emit(gname, vals, shape_hf, type_name, ne=None):
        """shape_hf [O, I] (or [n]) -> ne [I, O]; payload in type_name"""
        if ne is None: ne = list(reversed(shape_hf))
        payload = PACK[type_name](vals)
        w.add_tensor(gname, TYPE_IDS[type_name], ne, payload=payload)
        stats[type_name] = (stats.get(type_name, (0, 0))[0] + 1, stats.get(type_name, (0, 0))[1] + len(payload))

    vh, vk, vdim, kdim = g["vh"], g["vk"], g["vdim"], g["kdim"]
    H = g["hidden"]
    shape, vals = snap.read(globals_["embed_tokens.weight"]); emit("token_embd.weight", vals, shape, dense_t)
    shape, vals = snap.read(globals_["norm.weight"]); emit("output_norm.weight", [f32(1.0 + v) for v in vals], shape, "f32")
    shape, vals = snap.read(globals_["lm_head.weight"]); emit("output.weight", vals, shape, dense_t)
    for i in range(g["n_layers"]):
        L = layers[i]; b = f"blk.{i}."
        def rd(kind):
            if kind not in L: sys.exit(f"st2gguf: layer {i}: missing {kind}")
            return snap.read(L[kind])
        shape, vals = rd("input_layernorm.weight"); emit(b + "attn_norm.weight", [f32(1.0 + v) for v in vals], shape, "f32")
        shape, vals = rd("post_attention_layernorm.weight"); emit(b + "post_attention_norm.weight", [f32(1.0 + v) for v in vals], shape, "f32")
        if g["layer_types"][i] == "full_attention":
            for kind, gn in (("self_attn.q_proj.weight", "attn_q"), ("self_attn.k_proj.weight", "attn_k"), ("self_attn.v_proj.weight", "attn_v"), ("self_attn.o_proj.weight", "attn_output")):
                shape, vals = rd(kind); emit(f"{b}{gn}.weight", vals, shape, dense_t)
            for kind, gn in (("self_attn.q_norm.weight", "attn_q_norm"), ("self_attn.k_norm.weight", "attn_k_norm")):
                if kind in L:
                    shape, vals = rd(kind); emit(f"{b}{gn}.weight", [f32(1.0 + v) for v in vals], shape, "f32")
        else:
            shape, vals = rd("linear_attn.in_proj_qkv.weight")
            emit(b + "attn_qkv.weight", permute_head_rows(vals, vh, vk, vdim, H, off=2 * vk * kdim), shape, dense_t)
            shape, vals = rd("linear_attn.in_proj_z.weight"); emit(b + "attn_gate.weight", permute_head_rows(vals, vh, vk, vdim, H), shape, dense_t)
            shape, vals = rd("linear_attn.in_proj_a.weight"); emit(b + "ssm_alpha.weight", permute_head_rows(vals, vh, vk, 1, H), shape, "f32")
            shape, vals = rd("linear_attn.in_proj_b.weight"); emit(b + "ssm_beta.weight", permute_head_rows(vals, vh, vk, 1, H), shape, "f32")
            shape, vals = rd("linear_attn.conv1d.weight")          # [conv_dim, 1, k] -> {k, conv_dim}
            conv_dim, k = shape[0], shape[-1]
            emit(b + "ssm_conv1d.weight", permute_head_rows(vals, vh, vk, vdim, k, off=2 * vk * kdim), shape, "f32", ne=[k, conv_dim])
            shape, vals = rd("linear_attn.dt_bias"); emit(b + "ssm_dt.bias", permute_head_rows(vals, vh, vk, 1, 1), shape, "f32")
            shape, vals = rd("linear_attn.A_log"); emit(b + "ssm_a", permute_head_rows([f32(-math.exp(v)) for v in vals], vh, vk, 1, 1), shape, "f32")
            shape, vals = rd("linear_attn.norm.weight"); emit(b + "ssm_norm.weight", vals, shape, "f32")
            shape, vals = rd("linear_attn.out_proj.weight"); emit(b + "ssm_out.weight", permute_head_cols(vals, shape[0], vh, vk, vdim), shape, dense_t)
        shape, vals = rd("mlp.gate.weight"); emit(b + "ffn_gate_inp.weight", vals, shape, "f32")
        if "mlp.shared_expert_gate.weight" in L:
            shape, vals = rd("mlp.shared_expert_gate.weight"); emit(b + "ffn_gate_inp_shexp.weight", vals, shape, "f32")
        for kind, gn in (("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp"), ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp"), ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp")):
            shape, vals = rd(kind); emit(f"{b}{gn}.weight", vals, shape, dense_t)
        E = g["n_experts"]
        if "mlp.experts.gate_up_proj" in L:          # fused layout [E, 2*inter, H] / [E, H, inter]
            shape, gu = rd("mlp.experts.gate_up_proj"); _, dn = rd("mlp.experts.down_proj")
            inter = shape[1] // 2
            gate = []; up = []
            for e in range(E):
                base = e * 2 * inter * H
                gate += gu[base: base + inter * H]; up += gu[base + inter * H: base + 2 * inter * H]
            emit(b + "ffn_gate_exps.weight", gate, None, exp_t, ne=[H, inter, E]); emit(b + "ffn_up_exps.weight", up, None, exp_t, ne=[H, inter, E])
            emit(b + "ffn_down_exps.weight", dn, None, exp_t, ne=[inter, H, E])
        else:
            gate = []; up = []; down = []; inter = None
            for e in range(E):
                shape, v = rd(f"mlp.experts.{e}.gate_proj.weight"); inter = shape[0]; gate += v
                _, v = rd(f"mlp.experts.{e}.up_proj.weight"); up += v
                _, v = rd(f"mlp.experts.{e}.down_proj.weight"); down += v
            emit(b + "ffn_gate_exps.weight", gate, None, exp_t, ne=[H, inter, E]); emit(b + "ffn_up_exps.weight", up, None, exp_t, ne=[H, inter, E])
            emit(b + "ffn_down_exps.weight", down, None, exp_t, ne=[inter, H, E])

    total = sum(n for _, n in stats.values())
    for t, (cnt, nbytes) in sorted(stats.items()):
        print(f"  {t:5s} {cnt:5d} tensors {nbytes / 1e6:10.2f} MB")
    if skipped: print("  skipped: " + ", ".join(f"{k} ×{v}" for k, v in skipped.items()))
    print(f"{a.out}: qwen35moe, {g['n_layers']} blocks ({g['layer_types'].count('full_attention')} attention), {g['n_experts']} experts, "
          f"{sum(c for c, _ in stats.values())} tensors, {total / 1e6:.2f} MB, dense {dense_t}, experts {exp_t}, tokenizer {'from ' + a.tokenizer if a.tokenizer else 'placeholder'}")
    if a.dry_run: return 0
    if a.split and a.split > 1:
        # gguf-split layout: part 1 carries the model metadata, parts > 1 only the split.*
        # keys (write_split_set adds them); tensors in file order, payload bytes balanced,
        # every part holds at least one tensor and no tensor straddles parts.
        n = min(a.split, len(w.tensors))
        total = sum(len(t[3]) for t in w.tensors)
        groups, cur, acc = [], [], 0
        for i, t in enumerate(w.tensors):
            remaining_parts = n - len(groups)
            cur.append(t); acc += len(t[3])
            left = len(w.tensors) - i - 1
            if remaining_parts > 1 and (acc >= total / n or left == remaining_parts - 1):
                groups.append(cur); cur, acc = [], 0
        if cur: groups.append(cur)
        while len(groups) < n:                      # pathological: more parts than bytes allow
            big = max(range(len(groups)), key=lambda k: len(groups[k]))
            groups.append([groups[big].pop()])
        out = Path(a.out); stem = out.name[:-5] if out.name.lower().endswith(".gguf") else out.name
        parts = [(w.kv if k == 0 else [], [tuple(t[:4]) for t in grp]) for k, grp in enumerate(groups)]
        paths = write_split_set(str(out.parent), stem, parts, alignment=w.alignment)
        print(f"  split: {len(paths)} parts, {[Path(q).name for q in paths]}")
        return 0
    w.write(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
