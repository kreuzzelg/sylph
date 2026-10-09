#!/usr/bin/env python3
"""Test of the upstream sync tooling (upstream_sync.md). Read-only unless --trial (temp clone).

    python3 07_Tests/SystemTest/run_upstream_sync.py [--trial]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "06_Code" / "c" / "tools" / "upstream_sync.py"
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True).stdout.strip()


def main(argv):
    print("case 0: --check --json")
    if not check(TOOL.exists(), f"{TOOL.relative_to(ROOT)} exists"):
        print("RESULT: FAIL (1 failures)"); return 1
    r = subprocess.run([sys.executable, str(TOOL), "--check", "--json"], capture_output=True, text=True, cwd=ROOT)
    if not check(r.returncode == 0, f"--check exits 0 ({(r.stderr.strip().splitlines() or [''])[-1][:160]})"):
        print("RESULT: FAIL"); return 1
    try:
        j = json.loads(r.stdout)
    except ValueError:
        check(False, "--json output parses"); print("RESULT: FAIL"); return 1
    check(j.get("base", "").startswith("ce370e87") or Path(ROOT / "06_Code" / ".upstream").exists(), f"base {j.get('base', '')[:12]}")
    check(isinstance(j.get("behind"), int) and j["behind"] >= 0, f"behind {j.get('behind')}")
    check(isinstance(j.get("tags_since"), list), f"tags since: {j.get('tags_since', [])[:5]}")
    check(isinstance(j.get("overlap"), list), f"overlap: {len(j.get('overlap', []))} paths")
    print("case 1: the forecast is honest (both sides touched every overlap path)")
    base = j.get("base", "ce370e87"); head = j.get("upstream_head", "upstream/main")
    ours = set(git("diff", "--name-only", "9db7d4ff", "HEAD", "--", "06_Code").splitlines())
    ours = {p[len("06_Code/"):] for p in ours}
    theirs = set(git("diff", "--name-only", base, head).splitlines())
    bad = [p for p in j.get("overlap", []) if p not in ours or p not in theirs]
    check(not bad, f"every overlap path changed on both sides ({bad[:5] or 'yes'})")
    if "--trial" in argv:
        print("case 2: --trial in a temporary clone")
        before = git("status", "--porcelain")
        r = subprocess.run([sys.executable, str(TOOL), "--trial", "--json"], capture_output=True, text=True, cwd=ROOT)
        check(r.returncode in (0, 3), f"--trial exits 0 (clean) or 3 (conflicts): {r.returncode}")
        try:
            t = json.loads(r.stdout); conf = t.get("conflicts", [])
            check(all(p in j.get("overlap", []) for p in conf), f"conflicts ⊆ overlap ({len(conf)} files)")
        except ValueError:
            check(False, "--trial --json parses")
        check(git("status", "--porcelain") == before, "working tree unchanged")
    print("case 3: documentation")
    doc = (ROOT / "06_Code" / "docs" / "gguf.md").read_text() if (ROOT / "06_Code" / "docs" / "gguf.md").exists() else ""
    check("subtree pull" in doc and "upstream_sync" in doc, "docs/gguf.md documents check -> trial -> apply -> gates")
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
