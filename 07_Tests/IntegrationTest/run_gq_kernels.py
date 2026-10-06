#!/usr/bin/env python3
"""Integration test for the gq.h kernel module (phase 2). See gq_kernels.md.

Standard library only. Exercises `06_Code/c/tests/test_gq_kernels` (built here if
missing) against the committed E0 golden fixtures (fixtures/e0/, generated once by
llama.cpp's gguf-py via make_e0_golden.py), against upstream's numpy dequantizer
when numpy is installed, and against OpenMP (thread-count independence of the
layer runner). Exit 0 iff every case passes.

    python3 07_Tests/IntegrationTest/run_gq_kernels.py            # all cases
    python3 07_Tests/IntegrationTest/run_gq_kernels.py --no-build # use the existing binary
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "06_Code" / "c"
FIX = Path(__file__).resolve().parent / "fixtures" / "e0"
EXE = C / "tests" / ("test_gq_kernels.exe" if sys.platform == "win32" else "test_gq_kernels")
UNSUPPORTED = ["Q2_K", "Q3_K", "IQ4_NL", "MXFP4", "TQ1_0"]   # must be refused by name

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, text=False, check=False, **kw)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def case0_fixtures_intact(manifest):
    print("case 0: golden fixtures intact (sha256 per manifest)")
    for e in manifest["fixtures"]:
        raw = (FIX / f"{e['name']}.bin").read_bytes()
        exp = (FIX / f"{e['name']}.f32").read_bytes()
        ok = sha(raw) == e["sha256_bin"] and sha(exp) == e["sha256_f32"] and len(exp) == 4 * e["numel"] and len(raw) == e["raw_bytes"]
        check(ok, f"{e['name']} ({e['type']}, {e['numel']} elements)")


def case1_unit_suite():
    print("case 1: tests/test_gq_kernels self-contained suite (E0 synthetic, SIMD == scalar, split, runner == GEMVs)")
    r = run([EXE], cwd=C)
    tail = (r.stdout.decode(errors="replace").strip().splitlines() or [""])[-1]
    check(r.returncode == 0, f"exit 0, last line: {tail!r}")


def case2_golden(manifest):
    print("case 2: E0 bit-exact dequantization vs llama.cpp gguf-py golden (real Qwen3.6 rows + synthetic blocks)")
    for e in manifest["fixtures"]:
        r = run([EXE, "deq", e["type"], FIX / f"{e['name']}.bin", e["numel"]])
        exp = (FIX / f"{e['name']}.f32").read_bytes()
        if r.returncode != 0:
            check(False, f"{e['name']}: deq {e['type']} failed: {r.stderr.decode(errors='replace').strip()[:160]}")
            continue
        got = r.stdout
        nd = sum(1 for i in range(0, min(len(got), len(exp)), 4) if got[i:i + 4] != exp[i:i + 4]) + abs(len(got) - len(exp)) // 4
        check(got == exp, f"{e['name']}: {e['type']} {e['numel']} values bit-exact ({nd} differ)")


def case3_live_ggufpy(manifest):
    print("case 3: fresh random blocks vs gguf-py (live cross-check; needs numpy + gguf)")
    try:
        import numpy as np
        from gguf import GGMLQuantizationType as T, GGML_QUANT_SIZES, quants
    except ImportError as ex:
        print(f"  skip  {ex}")
        return
    import tempfile
    rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    f16_fields = {"Q4_0": [0], "Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}
    with tempfile.TemporaryDirectory() as tmp:
        for name, offs in f16_fields.items():
            t = T[name]
            bs, nbytes = GGML_QUANT_SIZES[t]
            nb = 16
            blocks = rng.integers(0, 256, (nb, nbytes), dtype=np.uint8)
            for off in offs:   # finite f16 scales (exponent < 31)
                bits = (rng.integers(0, 2, nb, dtype=np.uint16) << 15) | (rng.integers(0, 31, nb, dtype=np.uint16) << 10) | rng.integers(0, 1024, nb, dtype=np.uint16)
                blocks[:, off:off + 2] = bits.astype("<u2").view(np.uint8).reshape(nb, 2)
            p = Path(tmp) / f"{name}.bin"
            p.write_bytes(blocks.tobytes())
            exp = quants.dequantize(blocks, t).astype("<f4").reshape(-1).tobytes()
            r = run([EXE, "deq", name, p, nb * bs])
            check(r.returncode == 0 and r.stdout == exp, f"{name}: {nb} random blocks bit-exact vs gguf.quants.dequantize")


def case4_upstream_oracle(manifest):
    print("case 4: upstream tools/gguf_dequant.py agrees with the golden (ties the two oracles; needs numpy)")
    try:
        import numpy as np
    except ImportError as ex:
        print(f"  skip  {ex}")
        return
    sys.path.insert(0, str(C / "tools"))
    import gguf_reader, gguf_dequant  # noqa: E402
    for e in manifest["fixtures"]:
        tid = getattr(gguf_reader, f"GGML_TYPE_{e['type']}", None)
        if tid is None or e["type"] == "Q4_0":
            print(f"  skip  {e['name']}: upstream module has no {e['type']}")
            continue
        raw = (FIX / f"{e['name']}.bin").read_bytes()
        exp = (FIX / f"{e['name']}.f32").read_bytes()
        got = gguf_dequant.dequantize(raw, tid, e["numel"]).astype("<f4").reshape(-1).tobytes()
        check(got == exp, f"{e['name']}: upstream numpy dequant bit-exact")


def case5_refusals():
    print("case 5: unsupported types refused by name")
    some = FIX / "synth_q8_0.bin"
    for name in UNSUPPORTED:
        r = run([EXE, "deq", name, some, 256])
        err = r.stderr.decode(errors="replace")
        check(r.returncode != 0 and "unsupported" in err.lower() and name in err, f"deq {name}: non-zero exit, message names 'unsupported' and {name}")


def case6_threads():
    print("case 6: gq_moe_run digest identical under OMP_NUM_THREADS=1,2,4 (rank-ordered reduction)")
    digests = {}
    for n in ("1", "2", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=n)
        r = run([EXE, "moe-digest"], env=env)
        digests[n] = r.stdout.decode(errors="replace").strip() if r.returncode == 0 else f"exit {r.returncode}"
    check(len(set(digests.values())) == 1 and not digests["1"].startswith("exit"), f"digests {digests}")


def main(argv):
    manifest = json.loads((FIX / "manifest.json").read_text())
    case0_fixtures_intact(manifest)
    if "--no-build" not in argv and not EXE.exists():
        b = run(["make", "-C", C, f"tests/{EXE.name}"])
        if b.returncode != 0:
            msg = (b.stderr.decode(errors="replace").strip().splitlines() or ["?"])[-1]
            check(False, f"build tests/{EXE.name}: {msg}")
            print("RESULT: FAIL (module not built — expected until phase 2 lands gq.h and tests/test_gq_kernels.c)")
            return 1
    case1_unit_suite()
    case2_golden(manifest)
    case3_live_ggufpy(manifest)
    case4_upstream_oracle(manifest)
    case5_refusals()
    case6_threads()
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
