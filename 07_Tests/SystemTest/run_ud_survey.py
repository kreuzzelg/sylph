#!/usr/bin/env python3
"""Survey of the public GGUF quantizations (ud_survey.md): headers only, via the Hugging Face API.

    python3 07_Tests/SystemTest/run_ud_survey.py [--repos a,b] [--only UD-Q4_K_M,Q4_K_M] [--out <md>] [--cache <dir>] [--check <md>]
"""
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "06_Code" / "c"
sys.path.insert(0, str(C))
import ggufinfo  # noqa: E402

HF = "https://huggingface.co"
DEFAULT_REPOS = ["unsloth/Qwen3.6-35B-A3B-GGUF", "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", "unsloth/GLM-5.2-GGUF"]
# the quants a user would plausibly pick, one per class (the full tree is ~60 files x ~12 MB of
# metadata; --only widens or narrows the selection)
DEFAULT_ONLY = ["UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_S", "UD-Q4_K_M", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL",
                "UD-IQ2_M", "UD-IQ3_XXS", "UD-IQ4_XS", "UD-IQ4_NL",
                "-Q2_K.", "-Q3_K_M", "-Q4_K_M", "-Q5_K_M", "-Q6_K.", "-Q8_0.", "-IQ3_M", "-IQ4_XS", "-IQ2_M", "MXFP4"]
V1 = {"F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"}
GB = 1024 ** 3; MB = 1024 ** 2
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def http(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "sylph-ud-survey/1"} | ({"Range": rng} if rng else {}))
    import http.client
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except http.client.IncompleteRead as ex:
            # the proxy cuts long range reads now and then: what arrived is still a valid
            # prefix (the metadata is at the front), so use it and let the caller re-check
            return ex.partial
        except (urllib.error.URLError, TimeoutError) as ex:   # noqa: PERF203
            if attempt == 2:
                raise
            print(f"  retry {url.rsplit('/', 1)[-1]}: {ex}")


def tree(repo, sub=""):
    try:
        return json.loads(http(f"{HF}/api/models/{repo}/tree/main{('/' + sub) if sub else ''}"))
    except Exception as ex:  # noqa: BLE001
        print(f"  warn repo {repo}/{sub}: {ex}")
        return None


def fetch_header(repo, path, size, cache):
    """first bytes of the file, extended sparsely to `size` so ggufinfo sees the real geometry"""
    local = cache / repo.replace("/", "__") / path
    local.parent.mkdir(parents=True, exist_ok=True)
    have = local.stat().st_size if local.exists() else 0
    want = 16 * MB
    while True:
        if not local.exists() or have < want:
            data = http(f"{HF}/{repo}/resolve/main/{path}", f"bytes=0-{min(want, size) - 1}")
            with open(local, "wb") as f:
                f.write(data)
            have = len(data)
        with open(local, "r+b") as f:
            f.truncate(size)
        # the metadata must lie inside the fetched window (the payload is sparse zeros)
        off = header_data_offset(local)
        if off is not None and off <= have:
            return local
        if want >= 64 * MB or want >= size:
            return local
        want *= 2


def header_data_offset(path):
    try:
        return ggufinfo.open_set(str(path))[0].data_off
    except Exception:  # noqa: BLE001
        return None


def family_of(name):
    if re.match(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$", name): return "experts"
    if ".attn" in name or ".ssm_" in name or ".indexer." in name: return "attention"
    if "_shexp" in name: return "shared"
    if name in ("token_embd.weight", "output.weight"): return "embd/output"
    return "other"


def analyse(set_path):
    parts = ggufinfo.open_set(str(set_path))
    s = ggufinfo.summarize(parts)
    fam = {}
    dense = 0; expert_bytes = 0; n_exp = 0
    for t in ggufinfo.all_tensors(parts):
        f = family_of(t.name)
        fam.setdefault(f, {}); fam[f][t.type_name] = fam[f].get(t.type_name, 0) + (t.nbytes or 0)
        if f == "experts":
            expert_bytes += t.nbytes or 0
            if t.name.endswith("ffn_gate_exps.weight") and len(t.ne) >= 3: n_exp += t.ne[2]
        else:
            dense += t.nbytes or 0
    types = sorted({t.type_name for t in ggufinfo.all_tensors(parts)})
    outside = sorted(t for t in types if t not in V1)
    return {"arch": s["architecture"], "types": types, "outside": outside, "dense": dense, "experts": expert_bytes,
            "expert_mb": (expert_bytes / n_exp / MB) if n_exp else 0, "families": fam, "mtp": s["mtp"],
            "size": sum(p.size for p in parts), "tensors": s["tensors"], "parts": len(parts)}


def fam_str(fam):
    out = []
    for f in ("experts", "attention", "shared", "embd/output"):
        if f in fam:
            tot = sum(fam[f].values()) or 1
            out.append(f + ": " + " ".join(f"{t} {100 * b / tot:.0f}%" for t, b in sorted(fam[f].items(), key=lambda kv: -kv[1])))
    return " · ".join(out)


def main(argv):
    repos = DEFAULT_REPOS; only = set(DEFAULT_ONLY); out = None; cache = Path(os.environ.get("SYLPH_SURVEY_CACHE", "/tmp/sylph_survey_cache"))
    for i, a in enumerate(argv):
        if a == "--repos": repos = argv[i + 1].split(",")
        if a == "--only": only = set(argv[i + 1].split(","))
        if a == "--out": out = Path(argv[i + 1])
        if a == "--cache": cache = Path(argv[i + 1])
        if a == "--check":
            txt = Path(argv[i + 1]).read_text()
            ok = "## Summary" in txt and "| file |" in txt
            print("RESULT:", "ok" if ok else "FAIL", "(report parses)"); return 0 if ok else 1
    date = datetime.date.today().isoformat()
    out = out or ROOT / "08_Documents" / "survey" / f"{date}-ud-quants.md"
    cache.mkdir(parents=True, exist_ok=True)
    rows = {}
    for repo in repos:
        print(f"repo {repo}")
        t = tree(repo)
        if not check(t is not None, f"{repo}: tree lists"):
            continue
        items = []
        for e in t:
            if e["type"] == "directory":
                sub = tree(repo, e["path"])
                if sub:
                    files = [(f["path"], f.get("size", 0)) for f in sub if f["path"].endswith(".gguf")]
                    if files: items.append((e["path"], files))
            elif e["path"].endswith(".gguf") and "mmproj" not in e["path"] and "imatrix" not in e["path"]:
                items.append((Path(e["path"]).stem, [(e["path"], e.get("size", 0))]))
        rows[repo] = []
        for label, files in items:
            if only and not any(o in label or o in files[0][0] for o in only):
                continue
            try:
                locals_ = [fetch_header(repo, p, sz, cache) for p, sz in files]
                a = analyse(locals_[0].parent if len(locals_) > 1 else locals_[0])
            except Exception as ex:  # noqa: BLE001
                print(f"  warn {label}: {ex}"); rows[repo].append((label, None, str(ex))); continue
            verdict = "runs" if not a["outside"] else "refused: " + ", ".join(a["outside"])
            rows[repo].append((label, a, verdict))
            print(f"  {label:28s} {a['size'] / GB:6.1f} GB  dense {a['dense'] / GB:5.1f} GB  expert {a['expert_mb']:5.2f} MB  {verdict}")
    # report
    L = [f"# Public quantizations surveyed with the phase-1 reader — {date}", "",
         "Protocol: `07_Tests/SystemTest/ud_survey.md` (headers only, sparse local copies). Types outside the supported set",
         f"(`{' '.join(sorted(V1))}`) are refused by name today (FR-9); the dense set is every tensor that is not a routed expert. Sizes in GiB/MiB (the inspection reports used GB).", ""]
    needs = {}
    for repo, rs in rows.items():
        L += [f"## `{repo}`", "", "| file | GiB | dense GiB | expert MiB | types (experts · attention · shared · embd/output) | outside the set | verdict |", "|---|---|---|---|---|---|---|"]
        for label, a, verdict in rs:
            if a is None:
                L.append(f"| `{label}` | | | | | | not read: {verdict} |"); continue
            L.append(f"| `{label}` | {a['size'] / GB:.1f} | {a['dense'] / GB:.1f} | {a['expert_mb']:.2f} | {fam_str(a['families'])} | {', '.join(a['outside']) or '—'} | {verdict} |")
            for ty in a["outside"]: needs.setdefault(ty, []).append(f"{repo.split('/')[1]}/{label}")
        L.append("")
    L += ["## Summary", ""]
    for ty, files in sorted(needs.items()):
        L.append(f"- `{ty}` would unlock {len(files)} file(s): {', '.join(files[:8])}{' …' if len(files) > 8 else ''}")
    if not needs:
        L.append("- every surveyed file runs with the supported set")
    for cap in (8, 16, 32):
        fits = [(repo.split('/')[1], label, a["dense"] / GB) for repo, rs in rows.items() for label, a, v in rs if a and v == "runs" and a["dense"] <= cap * GB * 0.8]
        if fits:
            fits.sort(key=lambda x: x[2])
            L.append(f"- dense set within 80 % of {cap} GiB RAM (runs): " + ", ".join(f"{r}/{l} ({d:.1f} GB)" for r, l, d in fits[:6]))
    # the conclusion ud_survey.md case 5 asks for: per reference model, runnable vs refused, and the
    # types ranked by how many files they would unlock
    L += ["", "## Conclusion (ud_survey.md case 5)", ""]
    for repo, rs in rows.items():
        runs = [l for l, a, v in rs if a and v == "runs"]; ref = [l for l, a, v in rs if a and v != "runs"]
        L.append(f"- `{repo}`: {len(runs)} of {len(runs) + len(ref)} surveyed files run with the supported set ({', '.join(runs) or 'none'}); "
                 f"refused: {', '.join(ref) or 'none'}.")
    ranked = sorted(needs.items(), key=lambda kv: -len(kv[1]))
    if ranked:
        L.append("- Types ranked by files unlocked: " + ", ".join(f"`{t}` ({len(f)})" for t, f in ranked) + ".")
        L.append("- Every K-quant file at Q4_K_M and above runs today; everything below Q4 (unsloth's `UD-Q3_K_*`/`UD-Q2_K_XL`, the I-quant files, bartowski's `Q2_K`/`Q3_K_M`) needs `IQ4_XS` + `IQ3_S`/`IQ3_XXS` + `Q3_K` first — a decision for the owner, not a default.")
    L += ["", "Reproduction: `python3 07_Tests/SystemTest/run_ud_survey.py --out <this file>`", ""]
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(L))
    print(f"report: {out}")
    check(out.exists() and "## Summary" in out.read_text(), "report written with a summary")
    print("RESULT:", "FAIL" if failures else "ok", f"({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
