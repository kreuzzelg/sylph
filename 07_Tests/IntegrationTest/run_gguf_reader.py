#!/usr/bin/env python3
"""Integration test for the GGUF reader module: C reader vs Python reader vs real files.
See gguf_reader.md. Standard library only; builds 06_Code/c/tests/test_gguf if needed."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "06_Code" / "c"
sys.path.insert(0, str(C))
import ggufinfo  # noqa: E402
from tools.make_gguf_fixture import demo_writer, tiny_glm_dsa, write_split_set, GgufWriter, U32, ARR, U8  # noqa: E402

EXE = C / "tests" / ("test_gguf.exe" if sys.platform == "win32" else "test_gguf")
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def c_dump(path):
    out = subprocess.run([str(EXE), str(path)], capture_output=True, text=True, check=False)
    return out.returncode, out.stdout, out.stderr


def compare(path):
    parts = ggufinfo.open_set(path)
    rc, dump, err = c_dump(path)
    check(rc == 0, f"C reader accepts {Path(path).name}: {err.strip()}")
    if rc:
        return
    lines = dump.splitlines()
    files = [l.split() for l in lines if l.startswith("F ")]
    kv = {l.split()[1]: l.split()[2:5] for l in lines if l.startswith("K ")}
    tens = {l.split()[1]: l.split()[2:] for l in lines if l.startswith("T ")}
    check(len(files) == len(parts), "same number of parts")
    for rec, p in zip(files, parts):
        check((int(rec[3]), int(rec[4]), int(rec[5])) == (p.size, p.data_off, p.alignment), f"part {rec[1].split('/')[-1]}: size/data_off/align agree")
    kv_ok = all(k in kv and kv[k][0] == ggufinfo.VALUE_TYPES[t[0]][0] and kv[k][2] == str(t[2]) for k, t in parts[0].kv_types.items())
    check(kv_ok, f"{len(parts[0].kv_types)} metadata keys agree (type, element type, length)")
    py = {t.name: t for p in parts for t in p.tensors}
    check(set(py) == set(tens), f"{len(py)} tensor names agree")
    bad = [n for n, t in py.items() if n in tens and (int(tens[n][0]), [int(x) for x in tens[n][2:6]], int(tens[n][6]), int(tens[n][7]), int(tens[n][8])) != (t.type, t.ne, t.file, t.off, -1 if t.nbytes is None else t.nbytes)]
    check(not bad, f"tensor type/shape/file/offset/bytes agree ({len(bad)} mismatches)")


def main(argv):
    if not EXE.exists():
        subprocess.run(["make", "-C", str(C), f"tests/{EXE.name}"], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("case 1: synthetic demo files and split set")
        demo_writer().write(tmp / "demo32.gguf")
        demo_writer(alignment=64).write(tmp / "demo64.gguf")
        d = demo_writer()
        chunks = [[], [], []]
        for i, t in enumerate(d.tensors):
            chunks[i % 3].append(t[:4])
        write_split_set(tmp / "split", "demo", [(d.kv if i == 0 else [], c) for i, c in enumerate(chunks)])
        for p in (tmp / "demo32.gguf", tmp / "demo64.gguf", tmp / "split"):
            compare(p)
        print("case 2: tiny glm-dsa layout")
        tiny_glm_dsa().write(tmp / "glm.gguf")
        compare(tmp / "glm.gguf")
        s = ggufinfo.summarize(ggufinfo.open_set(tmp / "glm.gguf"))
        check(s["engine"] == "glm" and s["trunk_layers"] == 2 and s["mtp"]["eh_proj_type"] == "Q8_0", "summary: engine glm, 2 trunk layers, MTP Q8_0")
        print("case 3: refusals name the rule")
        bad = {
            "not a GGUF": GgufWriter(magic=b"GGUX").add_str("a", "b"),
            "version": GgufWriter(version=2).add_str("a", "b"),
            "power of two": GgufWriter().add("general.alignment", U32, 48),
            "nests deeper": GgufWriter().add_arr("x", ARR, [(ARR, [(U8, [1])])]),
            "not a multiple": GgufWriter().add_tensor("t", "Q4_K", [100, 2], payload=b"\0" * 64),
            "aligned": GgufWriter().add_tensor("t", "F32", [4, 4], payload=b"\0" * 64, offset=16),
            "runs past": GgufWriter().add_tensor("t", "F32", [4, 4], payload=b"", offset=1 << 20),
            "duplicate": GgufWriter().add_tensor("t", "F32", [4]).add_tensor("t", "F32", [4]),
        }
        for word, w in bad.items():
            p = tmp / "bad.gguf"
            w.write(p)
            try:
                ggufinfo.read_part(p)
                py_ok = False
            except ggufinfo.GgufError as e:
                py_ok = word in str(e)
            rc, _, err = c_dump(p)
            check(py_ok and rc != 0 and word in err, f"both readers refuse '{word}'")
    if len(argv) > 1:
        print(f"case 4: real model {argv[1]}")
        compare(argv[1])
        s = ggufinfo.summarize(ggufinfo.open_set(argv[1], os.environ.get("COLI_MODEL_DIRS", "")))
        check(not s["unknown_types"], "no unknown ggml types")
        print(f"       {s['architecture']} · {s['parts']} parts · {s['tensors']} tensors · "
              f"{s['trunk_layers']} trunk layers · unsupported in v1: {len(s['unsupported_v1'])}")
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
