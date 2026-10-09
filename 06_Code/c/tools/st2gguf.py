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
the conversion, visual.* is skipped and counted. The NextN (MTP) head (mtp.*) is written as
blk.<n_layers>.nextn.* + blk.<n_layers>.* unless --no-mtp (phase 6, qwen36_mtp.md).

--arch glm-dsa (default when config.json says model_type glm_moe_dsa): a GLM-5.2 snapshot
-> `glm-dsa` GGUF (07_Tests/IntegrationTest/glm_assembly.md): names per glm_tensor_kinds.py /
c/glm_names.h, the MLA split llama.cpp stores (attn_k_b {qk_nope, kv_lora, H} transposed per
head, attn_v_b {kv_lora, v_head, H}; --kv-b fused writes attn_kv_b instead, none writes
neither -- a refusal fixture), exp_probs_b.bias, the expert_* / leading_dense_block_count /
indexer keys, tokenizer.ggml.pre = glm4, the NextN block inside block_count.
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
import glm_tensor_kinds as gkinds  # noqa: E402

T_F32, T_F16, T_Q8_0, T_BF16 = 0, 1, 8, 30
T_Q4_K, T_Q5_K, T_Q6_K = 12, 13, 14
TYPE_IDS = {"f32": T_F32, "f16": T_F16, "q8_0": T_Q8_0, "bf16": T_BF16, "q4_k": T_Q4_K, "q5_k": T_Q5_K, "q6_k": T_Q6_K}
BLOCK = {"f32": 1, "f16": 1, "bf16": 1, "q8_0": 32, "q4_k": 256, "q5_k": 256, "q6_k": 256}
# accepted on the command line only to be refused by name (FR-9: outside the supported set)
UNSUPPORTED_TYPES = {"q2_k": "Q2_K", "q3_k": "Q3_K", "iq4_xs": "IQ4_XS", "iq3_xxs": "IQ3_XXS", "mxfp4": "MXFP4"}
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 7, "bf16": 32, "q4_k": 15, "q5_k": 17, "q6_k": 18}   # general.file_type (llama_ftype: *_M)
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


# ---- K-quants (fixtures only; cuda_tier_kquant.md) ---------------------------------------------
# Valid blocks in ggml's exact layouts, encoded with a direct per-sub-block min/max
# rule rather than ggml's iterative search (make_qkx2_quants / make_qx_quants): the
# tests that use these files compare two decoders of the SAME bytes, so the encoder
# only has to be valid and reasonable, not ggml-equal. E0 pins the decoder.
def _f16(v):
    return struct.pack("<e", max(-65504.0, min(65504.0, v)))


def _rnd(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _pack_q45_k(vals, five):
    n = len(vals)
    if n % 256:
        raise ValueError(f"{'Q5_K' if five else 'Q4_K'} needs a multiple of 256 elements per row, got {n}")
    qmax = 31 if five else 15
    out = bytearray()
    for b0 in range(0, n, 256):
        blk = vals[b0:b0 + 256]
        scales, mins = [], []
        for j in range(8):
            sb = blk[32 * j:32 * j + 32]
            lo, hi = min(sb), max(sb)
            if lo > 0: lo = 0.0                      # the min term only subtracts (value = d*sc*q - dmin*m)
            scales.append((hi - lo) / qmax if hi > lo else 0.0); mins.append(-lo)
        d = max(scales) / 63.0; dmin = max(mins) / 63.0
        d16 = struct.unpack("<e", _f16(d))[0]; dmin16 = struct.unpack("<e", _f16(dmin))[0]
        sc = [min(63, _rnd(s / d16)) if d16 else 0 for s in scales]
        mn = [min(63, _rnd(m / dmin16)) if dmin16 else 0 for m in mins]
        # 12 scale bytes, the layout get_scale_min_k4 reads
        q12 = bytearray(12)
        for j in range(4):
            q12[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6)
            q12[j + 4] = (mn[j] & 63) | ((mn[j + 4] >> 4) << 6)
            q12[j + 8] = (sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4)
        qs = bytearray(128); qh = bytearray(32)
        for j in range(4):
            for half in range(2):
                k = 2 * j + half
                eff_d = d16 * sc[k]; eff_m = dmin16 * mn[k]
                for l in range(32):
                    x = blk[32 * k + l]
                    q = _rnd((x + eff_m) / eff_d) if eff_d else 0
                    q = max(0, min(qmax, q))
                    if half: qs[32 * j + l] |= (q & 0xF) << 4
                    else: qs[32 * j + l] |= q & 0xF
                    if five and (q & 16): qh[l] |= 1 << k
        out += _f16(d16) + _f16(dmin16) + q12 + (qh if five else b"") + qs
    return bytes(out)


def pack_q4_k(vals): return _pack_q45_k(vals, False)
def pack_q5_k(vals): return _pack_q45_k(vals, True)


def pack_q6_k(vals):
    n = len(vals)
    if n % 256:
        raise ValueError(f"Q6_K needs a multiple of 256 elements per row, got {n}")
    out = bytearray()
    for b0 in range(0, n, 256):
        blk = vals[b0:b0 + 256]
        amax = [max(abs(v) for v in blk[16 * k:16 * k + 16]) for k in range(16)]
        scales = [a / 31.0 for a in amax]
        d = max(scales) / 127.0
        d16 = struct.unpack("<e", _f16(d))[0]
        sc = [max(-128, min(127, _rnd(s / d16))) if d16 else 0 for s in scales]
        ql = bytearray(128); qh = bytearray(64)
        for h in range(2):
            for r in range(128):
                k = 8 * h + (r >> 4)
                eff = d16 * sc[k]
                x = blk[128 * h + r]
                q = max(-32, min(31, _rnd(x / eff))) if eff else 0
                v = q + 32
                part, l = r >> 5, r & 31
                lo, hi = v & 0xF, v >> 4
                if part == 0: ql[64 * h + l] |= lo; qh[32 * h + l] |= hi << 0
                elif part == 1: ql[64 * h + l + 32] |= lo; qh[32 * h + l] |= hi << 2
                elif part == 2: ql[64 * h + l] |= lo << 4; qh[32 * h + l] |= hi << 4
                else: ql[64 * h + l + 32] |= lo << 4; qh[32 * h + l] |= hi << 6
        out += ql + qh + bytes((s & 0xFF) for s in sc) + _f16(d16)
    return bytes(out)


PACK = {"f32": pack_f32, "f16": pack_f16, "bf16": pack_bf16, "q8_0": pack_q8_0, "q4_k": pack_q4_k, "q5_k": pack_q5_k, "q6_k": pack_q6_k}


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
def tokens_for(a, g, pre):
    tokens, types, merges = tokenizer_arrays(a.tokenizer, g) if a.tokenizer else placeholder_tokens(g)
    return tokens, types, merges, pre


def add_tokenizer(w, tokens, types, merges, pre, g, extra_ids=()):
    w.add_str("tokenizer.ggml.model", "gpt2"); w.add_str("tokenizer.ggml.pre", pre)
    w.add_arr("tokenizer.ggml.tokens", STR, tokens); w.add_arr("tokenizer.ggml.token_type", I32, types); w.add_arr("tokenizer.ggml.merges", STR, merges)
    w.add("tokenizer.ggml.bos_token_id", U32, g["bos"]); w.add("tokenizer.ggml.eos_token_id", U32, g["eos"]); w.add("tokenizer.ggml.padding_token_id", U32, g["pad"])
    for key, val in extra_ids:
        w.add(key, U32, val)
    w.add("tokenizer.ggml.add_bos_token", BOOL, False)


class Emitter:
    """Adds tensors to the writer: shape_hf [O, I] (or [n]) -> ne [I, O]; payload in type_name.
    A quantized type whose block does not divide the row falls back to f16 (what llama.cpp's
    quantizer does for such rows), noted once per tensor name."""
    def __init__(self, w):
        self.w = w; self.stats = {}; self.fallbacks = []

    def __call__(self, gname, vals, shape_hf, type_name, ne=None):
        if ne is None: ne = list(reversed(shape_hf))
        if type_name in UNSUPPORTED_TYPES:
            sys.exit(f"st2gguf: {gname}: {UNSUPPORTED_TYPES[type_name]} is outside the supported type set "
                     f"(F32 F16 BF16 Q8_0 Q4_K Q5_K Q6_K) -- refusing (FR-9)")
        if ne[0] % BLOCK[type_name]:
            self.fallbacks.append(f"{gname}: ne0 {ne[0]} is not a multiple of the {type_name} block, written f16")
            type_name = "f16"
        payload = PACK[type_name](vals)
        self.w.add_tensor(gname, TYPE_IDS[type_name], ne, payload=payload)
        self.stats[type_name] = (self.stats.get(type_name, (0, 0))[0] + 1, self.stats.get(type_name, (0, 0))[1] + len(payload))

    def report(self, out, line, skipped):
        total = sum(n for _, n in self.stats.values())
        for t, (cnt, nbytes) in sorted(self.stats.items()):
            print(f"  {t:5s} {cnt:5d} tensors {nbytes / 1e6:10.2f} MB")
        for f in self.fallbacks: print("  note: " + f)
        if skipped: print("  skipped: " + ", ".join(f"{k} ×{v}" for k, v in skipped.items()))
        print(f"{out}: {line}, {sum(c for c, _ in self.stats.values())} tensors, {total / 1e6:.2f} MB")


def write_out(w, a):
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


# ---- qwen35moe ---------------------------------------------------------------------------------------
def convert_qwen(a, snap):
    g = load_config(a.hf_dir)
    dense_t, exp_t = a.type, a.expert_type or a.type
    down_t = a.down_type or exp_t
    mtp_t = a.mtp_type or dense_t
    prefix = kinds.resolve_prefix(snap.tensors.keys())

    # classify every name first (the converter's contract): layer -> {kind: name}
    layers = {}; globals_ = {}; skipped = {}; mtp = {}
    for name in snap.tensors:
        what = kinds.classify(name, prefix)         # raises UnknownTensor
        if what[0] == "skip": skipped[what[1]] = skipped.get(what[1], 0) + 1
        elif what[0] == "global": globals_[what[1]] = name
        elif what[0] == "mtp": mtp[what[1]] = name
        else: layers.setdefault(what[1], {})[what[2]] = name
    for need in ("embed_tokens.weight", "norm.weight", "lm_head.weight"):
        if need not in globals_: sys.exit(f"st2gguf: missing global tensor {need}")
    if sorted(layers) != list(range(g["n_layers"])):
        sys.exit(f"st2gguf: layers found {sorted(layers)[:5]}… do not match num_hidden_layers {g['n_layers']}")
    write_mtp = bool(g["mtp"]) and bool(mtp) and not a.no_mtp
    if mtp and not write_mtp:
        skipped["mtp"] = len(mtp)
    if g["mtp"] and not mtp and not a.no_mtp:
        print(f"st2gguf: note: config says mtp_num_hidden_layers {g['mtp']} but the snapshot has no mtp.* tensors; no NextN block written", file=sys.stderr)

    w = GgufWriter(alignment=32)
    w.add_str("general.architecture", "qwen35moe"); w.add_str("general.type", "model")
    w.add_str("general.name", a.name or g["name"]); w.add("general.file_type", U32, FILE_TYPE[dense_t])
    w.add("general.quantization_version", U32, 2)
    P = "qwen35moe."
    for key, val in (("block_count", g["n_layers"] + (1 if write_mtp else 0)), ("context_length", g["ctx"]), ("embedding_length", g["hidden"]), ("feed_forward_length", g["ffn"]),
                     ("attention.head_count", g["q_heads"]), ("attention.head_count_kv", g["kv_heads"]), ("attention.key_length", g["head_dim"]),
                     ("attention.value_length", g["head_dim"]), ("rope.dimension_count", g["rope_dim"]), ("expert_count", g["n_experts"]),
                     ("expert_used_count", g["topk"]), ("expert_feed_forward_length", g["inter"]), ("expert_shared_feed_forward_length", g["shared_inter"]),
                     ("full_attention_interval", g["interval"]), ("ssm.conv_kernel", g["conv_k"]), ("ssm.state_size", g["kdim"]),
                     ("ssm.group_count", g["vk"]), ("ssm.time_step_rank", g["vh"]), ("ssm.inner_size", g["vh"] * g["vdim"])):
        w.add(P + key, U32, int(val))
    if write_mtp: w.add(P + "nextn_predict_layers", U32, 1)
    w.add(P + "attention.layer_norm_rms_epsilon", KV_F32, g["eps"]); w.add(P + "rope.freq_base", KV_F32, g["theta"])
    if isinstance(g["mrope"], list): w.add_arr(P + "rope.dimension_sections", I32, list(g["mrope"]) + [0] * (4 - len(g["mrope"])))
    add_tokenizer(w, *tokens_for(a, g, "qwen35"), g)

    emit = Emitter(w)
    vh, vk, vdim, kdim = g["vh"], g["vk"], g["vdim"], g["kdim"]
    H = g["hidden"]
    shape, vals = snap.read(globals_["embed_tokens.weight"]); emit("token_embd.weight", vals, shape, dense_t)
    shape, vals = snap.read(globals_["norm.weight"]); emit("output_norm.weight", [f32(1.0 + v) for v in vals], shape, "f32")
    shape, vals = snap.read(globals_["lm_head.weight"]); emit("output.weight", vals, shape, dense_t)

    def emit_block(i, L, kind):
        """one transformer block (trunk layer i, or the NextN block at i = n_layers) under blk.i."""
        b = f"blk.{i}."
        def rd(k):
            if k not in L: sys.exit(f"st2gguf: block {i}: missing {k}")
            return snap.read(L[k])
        shape, vals = rd("input_layernorm.weight"); emit(b + "attn_norm.weight", [f32(1.0 + v) for v in vals], shape, "f32")
        shape, vals = rd("post_attention_layernorm.weight"); emit(b + "post_attention_norm.weight", [f32(1.0 + v) for v in vals], shape, "f32")
        if kind == "full_attention":
            for k, gn in (("self_attn.q_proj.weight", "attn_q"), ("self_attn.k_proj.weight", "attn_k"), ("self_attn.v_proj.weight", "attn_v"), ("self_attn.o_proj.weight", "attn_output")):
                shape, vals = rd(k); emit(f"{b}{gn}.weight", vals, shape, dense_t)
            for k, gn in (("self_attn.q_norm.weight", "attn_q_norm"), ("self_attn.k_norm.weight", "attn_k_norm")):
                if k in L:
                    shape, vals = rd(k); emit(f"{b}{gn}.weight", [f32(1.0 + v) for v in vals], shape, "f32")
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
        for k, gn in (("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp"), ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp"), ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp")):
            shape, vals = rd(k); emit(f"{b}{gn}.weight", vals, shape, dense_t)
        E = g["n_experts"]
        if "mlp.experts.gate_up_proj" in L:          # fused layout [E, 2*inter, H] / [E, H, inter]
            shape, gu = rd("mlp.experts.gate_up_proj"); _, dn = rd("mlp.experts.down_proj")
            inter = shape[1] // 2
            gate = []; up = []
            for e in range(E):
                base = e * 2 * inter * H
                gate += gu[base: base + inter * H]; up += gu[base + inter * H: base + 2 * inter * H]
            emit(b + "ffn_gate_exps.weight", gate, None, exp_t, ne=[H, inter, E]); emit(b + "ffn_up_exps.weight", up, None, exp_t, ne=[H, inter, E])
            emit(b + "ffn_down_exps.weight", dn, None, down_t, ne=[inter, H, E])
        else:
            gate = []; up = []; down = []; inter = None
            for e in range(E):
                shape, v = rd(f"mlp.experts.{e}.gate_proj.weight"); inter = shape[0]; gate += v
                _, v = rd(f"mlp.experts.{e}.up_proj.weight"); up += v
                _, v = rd(f"mlp.experts.{e}.down_proj.weight"); down += v
            emit(b + "ffn_gate_exps.weight", gate, None, exp_t, ne=[H, inter, E]); emit(b + "ffn_up_exps.weight", up, None, exp_t, ne=[H, inter, E])
            emit(b + "ffn_down_exps.weight", down, None, down_t, ne=[inter, H, E])

    for i in range(g["n_layers"]):
        emit_block(i, layers[i], g["layer_types"][i])
    if write_mtp:
        # the NextN head: llama.cpp's blk.<L>.nextn.{eh_proj, enorm, hnorm, shared_head_norm}
        # + the block itself under blk.<L>.* (qwen36_mtp.md); norms as 1 + w like every norm
        Lm = g["n_layers"]; b = f"blk.{Lm}.nextn."
        def rdm(k):
            if k not in mtp: sys.exit(f"st2gguf: NextN head: missing mtp.{k}")
            return snap.read(mtp[k])
        shape, vals = rdm("fc.weight"); emit(b + "eh_proj.weight", vals, shape, mtp_t)
        for k, gn in (("pre_fc_norm_embedding.weight", "enorm"), ("pre_fc_norm_hidden.weight", "hnorm"), ("norm.weight", "shared_head_norm")):
            shape, vals = rdm(k); emit(f"{b}{gn}.weight", [f32(1.0 + v) for v in vals], shape, "f32")
        block = {k[len("layers.0."):]: n for k, n in mtp.items() if k.startswith("layers.0.")}
        emit_block(Lm, block, "full_attention")

    emit.report(a.out, f"qwen35moe, {g['n_layers']} blocks ({g['layer_types'].count('full_attention')} attention)"
                       f"{' + 1 nextn (eh_proj ' + mtp_t + ')' if write_mtp else ''}, {g['n_experts']} experts, dense {dense_t}, experts {exp_t}"
                       f"{'' if down_t == exp_t else ' (down ' + down_t + ')'}, tokenizer {'from ' + a.tokenizer if a.tokenizer else 'placeholder'}", skipped)
    return write_out(w, a)


# ---- glm-dsa ------------------------------------------------------------------------------------------
def load_config_glm(hf_dir):
    cfg = json.loads((Path(hf_dir) / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    gen = {}
    gp = Path(hf_dir) / "generation_config.json"
    if gp.exists():
        try: gen = json.loads(gp.read_text(encoding="utf-8"))
        except ValueError: gen = {}
    g = {}
    for k, key in (("hidden", "hidden_size"), ("n_layers", "num_hidden_layers"), ("vocab", "vocab_size"), ("heads", "num_attention_heads"),
                   ("n_experts", "n_routed_experts"), ("topk", "num_experts_per_tok"), ("moe_inter", "moe_intermediate_size"),
                   ("dense_inter", "intermediate_size"), ("first_dense", "first_k_dense_replace"), ("q_lora", "q_lora_rank"),
                   ("kv_lora", "kv_lora_rank"), ("qk_nope", "qk_nope_head_dim"), ("qk_rope", "qk_rope_head_dim"), ("v_head", "v_head_dim"),
                   ("n_shared", "n_shared_experts")):
        g[k] = int(text[key])
    g["kv_heads"] = int(text.get("num_key_value_heads") or g["heads"])
    g["n_group"] = int(text.get("n_group", 1) or 1); g["topk_group"] = int(text.get("topk_group", 1) or 1)
    g["norm_topk"] = bool(text.get("norm_topk_prob", False)); g["routed_scale"] = float(text.get("routed_scaling_factor", 1.0))
    g["gating"] = 2 if str(text.get("scoring_func", "sigmoid")) == "sigmoid" else 1
    g["eps"] = float(text.get("rms_norm_eps", 1e-5))
    rp = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    g["theta"] = float(rp.get("rope_theta", text.get("rope_theta", 10000.0)))
    g["ctx"] = int(text.get("max_position_embeddings", 131072))
    g["index_topk"] = int(text.get("index_topk", 0) or 0); g["index_nh"] = int(text.get("index_n_heads", 0) or 0); g["index_hd"] = int(text.get("index_head_dim", 0) or 0)
    g["nextn"] = int(text.get("num_nextn_predict_layers", 0) or 0)
    stops = []
    for src in (text.get("eos_token_id"), gen.get("eos_token_id")):
        for v in (src if isinstance(src, list) else [src]):
            if isinstance(v, int) and v not in stops: stops.append(v)
    g["stops"] = stops or [g["vocab"] - 1]
    g["eos"] = g["stops"][0]
    v = gen.get("bos_token_id", text.get("bos_token_id")); g["bos"] = int(v) if isinstance(v, int) else 0
    v = gen.get("pad_token_id", text.get("pad_token_id")); g["pad"] = int(v) if isinstance(v, int) else 0
    g["name"] = cfg.get("_name_or_path") or Path(hf_dir).name
    return g


def convert_glm(a, snap):
    g = load_config_glm(a.hf_dir)
    dense_t, exp_t = a.type, a.expert_type or a.type
    down_t = a.down_type or exp_t
    mtp_t = a.mtp_type or dense_t
    layers = {}; globals_ = {}; skipped = {}
    for name in snap.tensors:
        what = gkinds.classify(name)                 # raises UnknownTensor
        if what[0] == "skip": skipped[what[1]] = skipped.get(what[1], 0) + 1
        elif what[0] == "global": globals_[what[1]] = name
        else: layers.setdefault(what[1], {})[what[2]] = name
    for need in ("embed_tokens.weight", "norm.weight", "lm_head.weight"):
        if need not in globals_: sys.exit(f"st2gguf: missing global tensor {need}")
    NL = g["n_layers"]
    has_mtp = NL in layers and "eh_proj.weight" in layers[NL]
    write_mtp = has_mtp and not a.no_mtp
    if has_mtp and not write_mtp:
        skipped["nextn block"] = len(layers.pop(NL))
    if sorted(layers) != list(range(NL + (1 if write_mtp else 0))):
        sys.exit(f"st2gguf: blocks found {sorted(layers)[:6]}… do not match num_hidden_layers {NL}{' + 1 nextn' if write_mtp else ''}")
    H, Hd, nope, rope, vh, kvl = g["hidden"], g["heads"], g["qk_nope"], g["qk_rope"], g["v_head"], g["kv_lora"]
    groups = a.expert_groups if a.expert_groups else g["n_group"]
    gating = a.gating_func if a.gating_func else g["gating"]

    w = GgufWriter(alignment=32)
    w.add_str("general.architecture", "glm-dsa"); w.add_str("general.type", "model")
    w.add_str("general.name", a.name or g["name"]); w.add("general.file_type", U32, FILE_TYPE[dense_t])
    w.add("general.quantization_version", U32, 2)
    P = "glm-dsa."
    for key, val in (("block_count", NL + (1 if write_mtp else 0)), ("context_length", g["ctx"]), ("embedding_length", H),
                     ("feed_forward_length", g["dense_inter"]), ("expert_feed_forward_length", g["moe_inter"]),
                     ("attention.head_count", Hd), ("attention.head_count_kv", g["kv_heads"]),
                     ("attention.q_lora_rank", g["q_lora"]), ("attention.kv_lora_rank", kvl),
                     ("attention.key_length", kvl + rope), ("attention.value_length", kvl),
                     ("attention.key_length_mla", nope + rope), ("attention.value_length_mla", vh),
                     ("rope.dimension_count", rope), ("expert_count", g["n_experts"]), ("expert_used_count", g["topk"]),
                     ("expert_shared_count", g["n_shared"]), ("expert_gating_func", gating),
                     ("expert_group_count", groups), ("expert_group_used_count", min(groups, g["topk_group"]) if groups > 1 else 1),
                     ("leading_dense_block_count", g["first_dense"]), ("vocab_size", g["vocab"])):
        w.add(P + key, U32, int(val))
    if write_mtp: w.add(P + "nextn_predict_layers", U32, 1)
    w.add(P + "expert_weights_scale", KV_F32, g["routed_scale"]); w.add(P + "expert_weights_norm", BOOL, g["norm_topk"])
    w.add(P + "attention.layer_norm_rms_epsilon", KV_F32, g["eps"]); w.add(P + "rope.freq_base", KV_F32, g["theta"])
    if g["index_topk"] and g["index_nh"] and g["index_hd"]:
        w.add(P + "attention.indexer.head_count", U32, g["index_nh"]); w.add(P + "attention.indexer.key_length", U32, g["index_hd"])
        w.add(P + "attention.indexer.top_k", U32, g["index_topk"])
    extra = []
    if len(g["stops"]) > 1: extra.append(("tokenizer.ggml.eot_token_id", g["stops"][1]))
    if len(g["stops"]) > 2: extra.append(("tokenizer.ggml.eom_token_id", g["stops"][2]))
    add_tokenizer(w, *tokens_for(a, g, "glm4"), g, extra)

    emit = Emitter(w)
    shape, vals = snap.read(globals_["embed_tokens.weight"]); emit("token_embd.weight", vals, shape, dense_t)
    shape, vals = snap.read(globals_["norm.weight"]); emit("output_norm.weight", vals, shape, "f32")
    shape, vals = snap.read(globals_["lm_head.weight"]); emit("output.weight", vals, shape, dense_t)
    for i in sorted(layers):
        L = layers[i]; b = f"blk.{i}."; is_mtp = i == NL
        def rd(k):
            if k not in L: sys.exit(f"st2gguf: block {i}: missing {k}")
            return snap.read(L[k])
        shape, vals = rd("input_layernorm.weight"); emit(b + "attn_norm.weight", vals, shape, "f32")
        shape, vals = rd("post_attention_layernorm.weight"); emit(b + "ffn_norm.weight", vals, shape, "f32")
        for k, gn, t in (("self_attn.q_a_proj.weight", "attn_q_a.weight", dense_t), ("self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight", "f32"),
                         ("self_attn.q_b_proj.weight", "attn_q_b.weight", dense_t), ("self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight", dense_t),
                         ("self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight", "f32"), ("self_attn.o_proj.weight", "attn_output.weight", dense_t)):
            shape, vals = rd(k); emit(b + gn, vals, shape, t)
        shape, kvb = rd("self_attn.kv_b_proj.weight")
        if shape != [Hd * (nope + vh), kvl]:
            sys.exit(f"st2gguf: block {i}: kv_b_proj shape {shape} != [{Hd}*({nope}+{vh}), {kvl}]")
        if a.kv_b == "fused":
            emit(b + "attn_kv_b.weight", kvb, shape, dense_t)
        elif a.kv_b == "split":
            # llama.cpp's absorbed split: k_b = kv_b.view(H, nope+v, kvl)[:, :nope].transpose(1, 2) -> {nope, kvl, H};
            # v_b = kv_b.view(H, nope+v, kvl)[:, nope:] -> {kvl, v, H}
            kb = []; vb = []
            for h in range(Hd):
                K = kvb[h * (nope + vh) * kvl: h * (nope + vh) * kvl + nope * kvl]
                for r in range(kvl):
                    kb += [K[j * kvl + r] for j in range(nope)]
                vb += kvb[(h * (nope + vh) + nope) * kvl: (h + 1) * (nope + vh) * kvl]
            emit(b + "attn_k_b.weight", kb, None, dense_t, ne=[nope, kvl, Hd]); emit(b + "attn_v_b.weight", vb, None, dense_t, ne=[kvl, vh, Hd])
        if not is_mtp and "self_attn.indexer.wq_b.weight" in L:
            for k, gn, t in (("self_attn.indexer.wq_b.weight", "indexer.attn_q_b.weight", dense_t), ("self_attn.indexer.wk.weight", "indexer.attn_k.weight", dense_t),
                             ("self_attn.indexer.weights_proj.weight", "indexer.proj.weight", "f32"), ("self_attn.indexer.k_norm.weight", "indexer.k_norm.weight", "f32"),
                             ("self_attn.indexer.k_norm.bias", "indexer.k_norm.bias", "f32")):
                shape, vals = rd(k); emit(b + gn, vals, shape, t)
        if "mlp.gate_proj.weight" in L:                # leading dense block
            for k, gn in (("mlp.gate_proj.weight", "ffn_gate.weight"), ("mlp.up_proj.weight", "ffn_up.weight"), ("mlp.down_proj.weight", "ffn_down.weight")):
                shape, vals = rd(k); emit(b + gn, vals, shape, dense_t)
        else:
            shape, vals = rd("mlp.gate.weight"); emit(b + "ffn_gate_inp.weight", vals, shape, "f32")
            shape, vals = rd("mlp.gate.e_score_correction_bias"); emit(b + "exp_probs_b.bias", vals, shape, "f32")
            for k, gn in (("mlp.shared_experts.gate_proj.weight", "ffn_gate_shexp.weight"), ("mlp.shared_experts.up_proj.weight", "ffn_up_shexp.weight"), ("mlp.shared_experts.down_proj.weight", "ffn_down_shexp.weight")):
                shape, vals = rd(k); emit(b + gn, vals, shape, dense_t)
            E = g["n_experts"]; gate = []; up = []; down = []; inter = None
            for e in range(E):
                shape, v = rd(f"mlp.experts.{e}.gate_proj.weight"); inter = shape[0]; gate += v
                _, v = rd(f"mlp.experts.{e}.up_proj.weight"); up += v
                _, v = rd(f"mlp.experts.{e}.down_proj.weight"); down += v
            emit(b + "ffn_gate_exps.weight", gate, None, exp_t, ne=[H, inter, E]); emit(b + "ffn_up_exps.weight", up, None, exp_t, ne=[H, inter, E])
            emit(b + "ffn_down_exps.weight", down, None, down_t, ne=[inter, H, E])
        if is_mtp:
            shape, vals = rd("eh_proj.weight"); emit(b + "nextn.eh_proj.weight", vals, shape, mtp_t)
            for k, gn in (("enorm.weight", "nextn.enorm.weight"), ("hnorm.weight", "nextn.hnorm.weight"), ("shared_head.norm.weight", "nextn.shared_head_norm.weight")):
                shape, vals = rd(k); emit(b + gn, vals, shape, "f32")

    emit.report(a.out, f"glm-dsa, {NL} blocks ({g['first_dense']} dense, {NL - g['first_dense']} MoE){' + 1 nextn (eh_proj ' + mtp_t + ')' if write_mtp else ''}, "
                       f"{g['n_experts']} experts, kv_b {a.kv_b}, dense {dense_t}, experts {exp_t}{'' if down_t == exp_t else ' (down ' + down_t + ')'}, "
                       f"tokenizer {'from ' + a.tokenizer if a.tokenizer else 'placeholder'}", skipped)
    return write_out(w, a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hf_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default=None, choices=["qwen35moe", "glm-dsa"], help="default: from config.json's model_type")
    ap.add_argument("--type", default="f32", choices=["bf16", "f16", "f32", "q8_0"]); ap.add_argument("--expert-type", default=None, choices=sorted(TYPE_IDS) + sorted(UNSUPPORTED_TYPES))
    ap.add_argument("--down-type", default=None, choices=sorted(TYPE_IDS), help="type of ffn_down_exps (default: --expert-type); the Q4_K_M files alternate Q5_K/Q6_K there")
    ap.add_argument("--mtp-type", default=None, choices=sorted(TYPE_IDS), help="type of the NextN head's eh_proj (default: --type)")
    ap.add_argument("--no-mtp", action="store_true", help="skip the NextN (MTP) block even when the snapshot carries one")
    ap.add_argument("--kv-b", default="split", choices=["split", "fused", "none"], help="glm-dsa: write attn_k_b/attn_v_b (llama.cpp), attn_kv_b, or neither (refusal fixture)")
    ap.add_argument("--gating-func", type=int, default=None, help="glm-dsa: expert_gating_func to write (default from the config: 2 = sigmoid)")
    ap.add_argument("--expert-groups", type=int, default=None, help="glm-dsa: expert_group_count to write (default from the config)")
    ap.add_argument("--tokenizer", default=None); ap.add_argument("--name", default=None); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--split", type=int, default=0, help="write N parts <stem>-0000k-of-0000N.gguf as llama.cpp's gguf-split does (expert_streaming.md)")
    a = ap.parse_args(argv)
    arch = a.arch
    if arch is None:
        cfg = json.loads((Path(a.hf_dir) / "config.json").read_text(encoding="utf-8"))
        mt = str(cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type") or "")
        arch = "glm-dsa" if mt.startswith("glm") else "qwen35moe"
    snap = Snapshot(a.hf_dir)
    return convert_glm(a, snap) if arch == "glm-dsa" else convert_qwen(a, snap)


if __name__ == "__main__":
    sys.exit(main())
