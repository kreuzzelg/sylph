#!/usr/bin/env python3
"""Generate the E0 golden fixtures for the gq.h kernels (see gq_kernels.md).

The oracle is llama.cpp's own reference, gguf-py (`gguf.quants.dequantize`),
which is what `llama-perplexity` and Ollama's ggml compute from. It runs ONCE,
here, on the maintainer's machine; the fixtures it writes are plain files the
stdlib-only runner (`run_gq_kernels.py`) and the C unit test read without numpy.

Two fixture families go to fixtures/e0/:

  real_*   rows cut out of the public Qwen3.6-35B-A3B GGUFs by HTTP range request
           (unsloth UD-Q4_K_M, bartowski Q4_K_M): every block type the engine will
           meet, as the authors quantized it.
  synth_*  deterministic random blocks for every type in the v1 set, with the f16
           scale fields forced finite and edge blocks appended (zero scale, negative
           zero, largest finite scale, all-ones quants, all-zero quants). Covers the
           types absent from the real files (F16, Q4_0, F32).

Each fixture is <name>.bin (raw ggml blocks) + <name>.f32 (little-endian float32,
what ggml's dequantize_row_<type> yields) and one line in manifest.json
(type, numel, provenance, sha256 of both files, generator versions).

    pip install "numpy>=1.26" "gguf>=0.19"      # not a project dependency
    python3 07_Tests/IntegrationTest/make_e0_golden.py [--offline]

--offline skips the range requests and only rewrites the synthetic fixtures.
"""
import hashlib
import json
import struct
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

import numpy as np
import gguf
from gguf import GGMLQuantizationType as T
from gguf import quants

HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures" / "e0"

UNSLOTH = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BARTOWSKI = "https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"

# (fixture name, url, tensor, ggml type, ne0 (row elements), absolute byte offset of
# the tensor in the file, first row, number of rows). Offsets are the ones
# `coli gguf inspect` / tests/test_gguf printed on 2026-10-05 (08_Documents/
# inspection-qwen36-gguf-2026-10-05.md); a 3-D expert tensor is addressed by the
# flat row index  expert * ne1 + row.
REAL = [
    ("real_unsloth_output_q6_k",        UNSLOTH,   "output.weight",             T.Q6_K, 2048, 10990048,   0,            4),
    ("real_unsloth_token_embd_q8_0",    UNSLOTH,   "token_embd.weight",         T.Q8_0, 2048, 428175840,  0,            4),
    ("real_unsloth_attn_qkv_q8_0",      UNSLOTH,   "blk.0.attn_qkv.weight",     T.Q8_0, 2048, 977441248,  4096,         4),
    ("real_unsloth_gate_exps_q4_k_e0",  UNSLOTH,   "blk.0.ffn_gate_exps.weight", T.Q4_K, 2048, 1180930528, 0,            4),
    ("real_unsloth_gate_exps_q4_k_e200", UNSLOTH,  "blk.0.ffn_gate_exps.weight", T.Q4_K, 2048, 1180930528, 200 * 512,    4),
    ("real_unsloth_down_exps_q5_k_e0",  UNSLOTH,   "blk.0.ffn_down_exps.weight", T.Q5_K, 512,  995267040,  0,            8),
    ("real_bartowski_down_exps_q6_k_e0", BARTOWSKI, "blk.0.ffn_down_exps.weight", T.Q6_K, 512,  732730496,  0,            8),
    ("real_bartowski_gate_inp_bf16",    BARTOWSKI, "blk.40.ffn_gate_inp.weight", T.BF16, 2048, 21987640960, 0,           2),
    ("real_unsloth_attn_norm_f32",      UNSLOTH,   "blk.0.attn_norm.weight",    T.F32,  2048, 977433056,  0,            1),
]

SYNTH_TYPES = [T.F32, T.F16, T.BF16, T.Q4_0, T.Q8_0, T.Q4_K, T.Q5_K, T.Q6_K]
SYNTH_NUMEL = 1024          # random part: 1024 elements per type
SEED = 20261005

# byte offsets of the f16 scale fields inside one block, per type
F16_FIELDS = {T.Q4_0: [0], T.Q8_0: [0], T.Q4_K: [0, 2], T.Q5_K: [0, 2], T.Q6_K: [208]}


def block_spec(t):
    bs, nbytes = gguf.GGML_QUANT_SIZES[t]
    return bs, nbytes


def finite_f16(rng, n):
    """Random finite f16 bit patterns over the whole magnitude range, both signs,
    subnormals included; never inf/NaN (exponent 31 excluded)."""
    sign = rng.integers(0, 2, n, dtype=np.uint16) << 15
    exp = rng.integers(0, 31, n, dtype=np.uint16) << 10
    mant = rng.integers(0, 1024, n, dtype=np.uint16)
    return (sign | exp | mant).astype("<u2")


def synth_blocks(t, rng):
    bs, nbytes = block_spec(t)
    if t == T.F32:
        vals = rng.standard_normal(SYNTH_NUMEL).astype("<f4") * 3
        edge = np.array([0.0, -0.0, 1.0, -1.0, 65504.0, 6.1e-5, 5.9e-8, 3.4e38], dtype="<f4")
        return np.concatenate([vals, edge]).tobytes()
    if t == T.F16:
        raw = finite_f16(rng, SYNTH_NUMEL)
        edge = np.array([0x0000, 0x8000, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x0001, 0x0400], dtype="<u2")
        return np.concatenate([raw, edge]).tobytes()
    if t == T.BF16:
        f = (rng.standard_normal(SYNTH_NUMEL).astype("<f4") * 3).view("<u4") >> 16
        edge = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x0001, 0x0080], dtype="<u4")
        return np.concatenate([f, edge]).astype("<u2").tobytes()
    nb = SYNTH_NUMEL // bs
    blocks = rng.integers(0, 256, (nb, nbytes), dtype=np.uint8)
    for off in F16_FIELDS[t]:
        blocks[:, off:off + 2] = finite_f16(rng, nb).view(np.uint8).reshape(nb, 2)
    # edge blocks: scale 0, -0, largest finite, smallest subnormal; quants all 0 / all 1s
    edges = []
    for d_bits, fill in ((0x0000, 0xFF), (0x8000, 0x00), (0x7BFF, 0xFF), (0x0001, 0xAA), (0x3C00, 0x00), (0xBC00, 0xFF)):
        b = np.full(nbytes, fill, dtype=np.uint8)
        for off in F16_FIELDS[t]:
            b[off:off + 2] = np.frombuffer(struct.pack("<H", d_bits), np.uint8)
        edges.append(b)
    return np.concatenate([blocks, np.stack(edges)]).tobytes()


def dequant(raw, t, numel):
    if t == T.F32:
        return np.frombuffer(raw, "<f4").copy()
    if t == T.F16:
        return np.frombuffer(raw, "<f2").astype("<f4")
    bs, nbytes = block_spec(t)
    arr = np.frombuffer(raw, np.uint8).reshape(-1, nbytes)
    out = quants.dequantize(arr, t).astype("<f4").reshape(-1)
    assert out.size == numel, (t, out.size, numel)
    return out


def fetch(url, start, length):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": "sylph-e0-golden"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != length:
        raise SystemExit(f"short read from {url}: {len(data)} of {length}")
    return data


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main(argv):
    offline = "--offline" in argv
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"fixtures": []}
    entries = {e["name"]: e for e in manifest["fixtures"]}
    gen = {"gguf": metadata.version("gguf"), "numpy": np.__version__, "python": sys.version.split()[0]}

    rng = np.random.default_rng(SEED)
    for t in SYNTH_TYPES:
        raw = synth_blocks(t, rng)
        bs, nbytes = block_spec(t)
        numel = len(raw) // nbytes * bs
        exp = dequant(raw, t, numel)
        name = f"synth_{t.name.lower()}"
        (OUT / f"{name}.bin").write_bytes(raw)
        (OUT / f"{name}.f32").write_bytes(exp.tobytes())
        entries[name] = {"name": name, "type": t.name, "type_id": int(t), "numel": int(numel), "raw_bytes": len(raw),
                         "source": {"kind": "synthetic", "seed": SEED, "note": "random blocks, finite f16 scales, 6 edge blocks appended"},
                         "sha256_bin": sha(raw), "sha256_f32": sha(exp.tobytes()), "generator": gen}
        print(f"{name:40s} {t.name:5s} {numel:6d} elems  {len(raw):6d} B")

    if not offline:
        for name, url, tensor, t, ne0, toff, row0, nrows in REAL:
            bs, nbytes = block_spec(t)
            assert ne0 % bs == 0
            row_bytes = ne0 // bs * nbytes
            start = toff + row0 * row_bytes
            raw = fetch(url, start, nrows * row_bytes)
            numel = nrows * ne0
            exp = dequant(raw, t, numel)
            (OUT / f"{name}.bin").write_bytes(raw)
            (OUT / f"{name}.f32").write_bytes(exp.tobytes())
            entries[name] = {"name": name, "type": t.name, "type_id": int(t), "numel": int(numel), "raw_bytes": len(raw),
                             "source": {"kind": "range", "url": url, "tensor": tensor, "ne0": ne0, "tensor_offset": toff,
                                        "first_row": row0, "rows": nrows, "byte_range": [start, start + len(raw) - 1]},
                             "sha256_bin": sha(raw), "sha256_f32": sha(exp.tobytes()), "generator": gen}
            finite = np.isfinite(exp).all()
            print(f"{name:40s} {t.name:5s} {numel:6d} elems  {len(raw):6d} B  |max| {np.abs(exp).max():.4g}  finite={finite}")

    manifest["fixtures"] = [entries[k] for k in sorted(entries)]
    manifest["oracle"] = "llama.cpp gguf-py gguf.quants.dequantize (reference for ggml dequantize_row_*)"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(e["raw_bytes"] + 4 * e["numel"] for e in entries.values())
    print(f"{len(entries)} fixtures, {total / 1024:.0f} KiB on disk -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
