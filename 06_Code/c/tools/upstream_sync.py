#!/usr/bin/env python3
"""Sync of upstream colibrì (JustVugg/colibri, branch main) into the 06_Code subtree.

    python3 06_Code/c/tools/upstream_sync.py --check [--json]      # read-only: fetch upstream, report
    python3 06_Code/c/tools/upstream_sync.py --trial [--json]      # dry merge in a temporary clone
    python3 06_Code/c/tools/upstream_sync.py --apply [--squash]    # the real `git subtree pull`, then the gates

Contract and cases: 07_Tests/SystemTest/upstream_sync.md. Standard library + git only.

06_Code/ is a `git subtree` of colibrì. The base (the upstream commit last merged) is read
from 06_Code/.upstream when --apply wrote one, else from the newest merge commit on
06_Code whose message follows the convention
    06_Code: merge upstream colibrì main (<tag>, <sha>) into the subtree
--check reports base, upstream head, commits behind, tags since, and the conflict forecast:
the paths upstream changed since the base that sylph also changed since the merge.
--trial clones the repository (`git clone --shared`) into a temporary directory and runs
the subtree pull there; the working tree is never touched. Exit 0 clean, 3 conflicts,
1 error.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "06_Code"
UPSTREAM_URL = "https://github.com/JustVugg/colibri"
FALLBACK_BASE = "ce370e87"
MERGE_RE = re.compile(r"^06_Code: merge upstream colibr[iì] main \(([^,]+), ([0-9a-f]{7,40})\) into the subtree", re.M)
GATES = [
    "make -C 06_Code/c check",
    "the gguf-oracle recipe (.github/workflows/check.yml, job gguf-oracle) or a push and the CI run",
    "python3 07_Tests/IntegrationTest/run_gguf_reader.py",
    "python3 07_Tests/IntegrationTest/run_gq_kernels.py",
    "python3 07_Tests/IntegrationTest/run_expert_streaming.py",
    "python3 07_Tests/IntegrationTest/run_cuda_tier_kquant.py",
    "python3 07_Tests/IntegrationTest/run_glm_assembly.py",
]


class SyncError(Exception):
    pass


def git(*args, cwd=None, check=True, quiet=False):
    r = subprocess.run(["git", "-C", str(cwd or ROOT)] + [str(a) for a in args], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        if check and not quiet:
            raise SyncError(f"git {' '.join(str(a) for a in args)} failed: {(r.stderr or r.stdout).strip()[-400:]}")
        return None                      # a failed command never reads as empty output
    return r.stdout.strip()


def upstream_remote(cwd=None):
    """The name of the upstream remote, added when missing (a fetch target, nothing else)."""
    remotes = (git("remote", cwd=cwd) or "").split()
    if "upstream" in remotes:
        return "upstream"
    for name in remotes:
        url = git("remote", "get-url", name, cwd=cwd, check=False, quiet=True) or ""
        if "JustVugg/colibri" in url:
            return name
    git("remote", "add", "upstream", UPSTREAM_URL, cwd=cwd)
    return "upstream"


def fetch_upstream(cwd=None):
    remote = upstream_remote(cwd)
    git("fetch", "-q", remote, "main", cwd=cwd)
    return remote, git("rev-parse", f"{remote}/main", cwd=cwd)


def read_base():
    """(upstream base sha, local merge commit sha, tag) -- .upstream first, else the merge convention."""
    f = ROOT / PREFIX / ".upstream"
    if f.exists():
        rec = {}
        for ln in f.read_text().splitlines():
            if "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1); rec[k.strip()] = v.strip()
        if rec.get("upstream"):
            return rec["upstream"], rec.get("merge") or None, rec.get("tag")
    log = git("log", "--format=%H%x00%B%x01", "--merges", "--", PREFIX)
    for entry in log.split("\x01"):
        if "\x00" not in entry:
            continue
        sha, body = entry.strip().split("\x00", 1)
        m = MERGE_RE.search(body)
        if m:
            return m.group(2), sha, m.group(1)
    return FALLBACK_BASE, None, None


def local_merge_commit(base_short):
    """The local merge commit whose message names the base (for the 'ours' range)."""
    log = git("log", "--format=%H%x00%s", "--merges", "--", PREFIX)
    for ln in log.splitlines():
        sha, _, subj = ln.partition("\x00")
        if base_short[:7] in subj:
            return sha
    return None


def tags_since(remote, base, head):
    """Upstream tags that are reachable from head but not from base (ls-remote: nothing is written)."""
    out = git("ls-remote", "--tags", remote, check=False, quiet=True) or ""
    peeled = {}
    for ln in out.splitlines():
        sha, _, ref = ln.partition("\t")
        if not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/"):]
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            peeled.setdefault(name, sha)
    tags = []
    for name, sha in peeled.items():
        if git("cat-file", "-e", f"{sha}^{{commit}}", check=False, quiet=True) is None:
            continue
        if git("merge-base", "--is-ancestor", sha, head, check=False, quiet=True) is None:
            continue
        if git("merge-base", "--is-ancestor", sha, base, check=False, quiet=True) is not None:
            continue
        tags.append(name)

    def vkey(t):
        return [int(x) if x.isdigit() else x for x in re.split(r"[.\-]", t.lstrip("v"))]
    try:
        tags.sort(key=vkey)
    except TypeError:
        tags.sort()
    return tags


def check(args):
    remote, head = fetch_upstream()
    base_ref, merge, tag = read_base()
    base = git("rev-parse", f"{base_ref}^{{commit}}", check=False, quiet=True)
    if base is None:
        raise SyncError(f"base {base_ref} is not a commit in this repository (fetch upstream first)")
    if merge is None:
        merge = local_merge_commit(base_ref)
    behind = int(git("rev-list", "--count", f"{base}..{head}") or 0)
    theirs = set((git("diff", "--name-only", base, head) or "").splitlines())
    ours = set()
    if merge:
        ours = {p[len(PREFIX) + 1:] for p in (git("diff", "--name-only", merge, "HEAD", "--", PREFIX) or "").splitlines() if p.startswith(PREFIX + "/")}
    overlap = sorted(ours & theirs)
    head_date = git("show", "-s", "--format=%cs", head)
    facts = {
        "base": base, "base_tag": tag, "merge_commit": merge, "upstream_head": head, "upstream_head_date": head_date,
        "behind": behind, "tags_since": tags_since(remote, base, head),
        "ours_changed": len(ours), "theirs_changed": len(theirs), "overlap": overlap,
        "command": f"git subtree pull --prefix={PREFIX} {remote} main" + (" --squash" if args.squash else ""),
    }
    return facts


def print_check(f):
    print(f"subtree  {PREFIX}/  base {f['base'][:12]}" + (f" ({f['base_tag']})" if f.get("base_tag") else "") + (f"  merged in {f['merge_commit'][:12]}" if f.get("merge_commit") else ""))
    print(f"upstream {f['upstream_head'][:12]} ({f['upstream_head_date']})  {f['behind']} commits ahead of the base")
    print(f"tags since the base: {', '.join(f['tags_since']) or '(none)'}")
    print(f"paths changed: upstream {f['theirs_changed']}, sylph {f['ours_changed']}, both (conflict forecast) {len(f['overlap'])}")
    for p in f["overlap"]:
        print(f"  {p}")
    print(f"next: {f['command']}  (--trial first)")


def subtree_pull(cwd, remote, squash, message):
    cmd = ["subtree", "pull", f"--prefix={PREFIX}", remote, "main", "-m", message]
    if squash:
        cmd.append("--squash")
    r = subprocess.run(["git", "-C", str(cwd)] + cmd, capture_output=True, text=True, errors="replace")
    conflicts = [p for p in (git("diff", "--name-only", "--diff-filter=U", cwd=cwd, check=False, quiet=True) or "").splitlines()]
    return r.returncode, conflicts, (r.stderr + r.stdout).strip()


def merge_message(facts):
    tag = facts["tags_since"][-1] if facts["tags_since"] else "untagged"
    return f"06_Code: merge upstream colibrì main ({tag}, {facts['upstream_head'][:7]}) into the subtree"


def trial(args):
    facts = check(args)
    tmp = Path(tempfile.mkdtemp(prefix="sylph_upstream_trial_"))
    try:
        subprocess.run(["git", "clone", "-q", "--shared", str(ROOT), str(tmp / "repo")], check=True, capture_output=True)
        repo = tmp / "repo"
        git("config", "user.name", "sylph trial", cwd=repo); git("config", "user.email", "trial@invalid", cwd=repo)
        remote = upstream_remote(repo)
        git("fetch", "-q", remote, "main", cwd=repo)
        rc, conflicts, log = subtree_pull(repo, remote, args.squash, merge_message(facts))
        conflicts = [p[len(PREFIX) + 1:] if p.startswith(PREFIX + "/") else p for p in conflicts]
        facts["trial"] = {"exit": rc, "conflicts": conflicts, "clean": rc == 0 and not conflicts}
        facts["conflicts"] = conflicts
        if rc != 0 and not conflicts:
            facts["trial"]["error"] = log[-600:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return facts


def apply(args):
    facts = check(args)
    if git("status", "--porcelain", check=False, quiet=True):
        raise SyncError("the working tree is not clean; commit or stash first")
    remote = upstream_remote()
    rc, conflicts, log = subtree_pull(ROOT, remote, args.squash, merge_message(facts))
    conflicts = [p[len(PREFIX) + 1:] if p.startswith(PREFIX + "/") else p for p in conflicts]
    rec = (ROOT / PREFIX / ".upstream")
    if rc == 0 and not conflicts:
        write_record(rec, facts)
        git("add", str(rec.relative_to(ROOT)))
        git("commit", "-q", "--amend", "--no-edit")
        print(f"merged: {merge_message(facts)}")
    else:
        print(f"the subtree pull stopped with {len(conflicts)} conflicted file(s):")
        for p in conflicts:
            print(f"  {PREFIX}/{p}")
        print("resolve them (policy: keep sylph's arm in sylph-owned files, take upstream elsewhere), `git add`, then")
        print(f"  git commit -m \"{merge_message(facts)}\"")
        print(f"and write {rec.relative_to(ROOT)}:")
        print(record_text(facts), end="")
        if not conflicts:
            print(log[-600:])
    print("gates before pushing:")
    for g in GATES:
        print(f"  {g}")
    return 0 if (rc == 0 and not conflicts) else 3


def record_text(facts):
    tag = facts["tags_since"][-1] if facts["tags_since"] else ""
    return (f"# written by tools/upstream_sync.py --apply; read by --check\n"
            f"upstream={facts['upstream_head']}\ntag={tag}\ndate={facts['upstream_head_date']}\n"
            f"merge=\napplied={datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n")


def write_record(path, facts):
    path.write_text(record_text(facts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true"); g.add_argument("--trial", action="store_true"); g.add_argument("--apply", action="store_true")
    ap.add_argument("--json", action="store_true"); ap.add_argument("--squash", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.check:
            f = check(a)
            if a.json:
                print(json.dumps(f, indent=1))
            else:
                print_check(f)
            return 0
        if a.trial:
            f = trial(a)
            if a.json:
                print(json.dumps(f, indent=1))
            else:
                print_check(f)
                t = f["trial"]
                print("trial: " + ("clean merge" if t["clean"] else f"{len(t['conflicts'])} conflicted file(s): " + ", ".join(t["conflicts"]) if t["conflicts"] else f"error: {t.get('error', '')}"))
            return 0 if f["trial"]["clean"] else (3 if f["trial"]["conflicts"] else 1)
        return apply(a)
    except SyncError as ex:
        print(f"upstream_sync: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
