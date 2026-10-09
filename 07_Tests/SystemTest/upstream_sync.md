# System test — upstream sync: colibrì's `main` merged into the `06_Code` subtree, documented and exercised (phase 6 / housekeeping)

Written 2026-10-09, before the tooling exists. Tasks: housekeeping "upstream sync procedure
documented and exercised once more (`git subtree pull`)"; architecture v2 §12 row "6 —
breadth: upstream sync tooling"; owner requirement "Ollama/llama.cpp as reference, colibrì
as the base".

`06_Code/` is a `git subtree` of `kreuzzelg/colibri` (fork of `JustVugg/colibri`), last
merged at upstream `ce370e8` (v1.12.1, commit `9db7d4ff`, 2026-10-05). On 2026-10-09
`upstream/main` is **932 commits ahead** (v2.0.0). Every sylph change lives beside
upstream code in the same tree (`gq.h`, `src.h`, `qwen36.c` arms, tier, tests), so a sync
is a real merge with conflicts to resolve and re-verify; the procedure must be a script
that says what will happen before anything is written.

## Contract (`06_Code/c/tools/upstream_sync.py`, stdlib + git)

```
python3 06_Code/c/tools/upstream_sync.py --check [--json]      # read-only: fetch upstream, report
python3 06_Code/c/tools/upstream_sync.py --trial [--json]      # dry merge in a temporary clone, report conflicts
python3 06_Code/c/tools/upstream_sync.py --apply [--squash]    # owner: the real `git subtree pull`, then the gates
```

- `--check` prints: the subtree's last merged upstream commit (read from the merge commit
  `9db7d4ff`'s message or a recorded `06_Code/.upstream` file written by `--apply`), the
  upstream head, the number of commits behind, the upstream tags since, and the list of
  upstream-touched paths that sylph also modified (`git diff --name-only <base> HEAD --
  06_Code` ∩ upstream's changed paths) — the conflict forecast.
- `--trial` clones the repository into a temporary directory (`git clone --shared`), adds
  the upstream remote, runs `git subtree pull --prefix=06_Code upstream main` (with
  `--squash` when asked) and reports: clean / N conflicted files (listed), without touching
  the working tree. Exit 0 when clean, 3 when conflicts, 1 on error.
- `--apply` performs the pull in the real tree, writes `06_Code/.upstream` (the upstream
  commit, date, tag), and prints the gate list to run before pushing: `make -C 06_Code/c
  check`, the `gguf-oracle` recipe, the five phase runners.
- `--json` emits the same facts as one JSON object (the runner parses it).
- Nothing in `--check`/`--trial` needs write access to the repository or the network
  beyond `git fetch`.

## Cases

Runner: `python3 07_Tests/SystemTest/run_upstream_sync.py [--trial]`.

| # | Case | Expected |
|---|---|---|
| 0 | tool present, `--check --json` | fields `base` (= `ce370e87…` until the first apply), `upstream_head`, `behind` (≥ 932 today), `tags_since` (contains `v2.0.0`), `overlap` (list of paths; contains `c/qwen36.c`, `c/Makefile`, `CHANGELOG.md` if upstream touched them) |
| 1 | the forecast is honest | every path in `overlap` was changed by both sides (checked by the runner with `git log` on both ranges) |
| 2 | `--trial` (with `--trial`) | runs in a temp clone; the repository's `git status` is unchanged afterwards; exit 0 or 3 with the conflict list; the list is a subset of `overlap` |
| 3 | documentation | `06_Code/docs/gguf.md` (or `08_Documents/`) carries the procedure: check → trial → apply → gates → commit message convention (`06_Code: merge upstream colibrì main (<tag>, <sha>) into the subtree`), and the policy for resolving conflicts in sylph-owned files (keep sylph's arm, re-run the phase runners) |
| 4 | exercised once | the owner (or the next session with network) runs `--apply`, resolves, runs the gates, and records the result in the state table below with the upstream tag merged |

**Pass criterion:** cases 0–3 green here; case 4 is the housekeeping item itself.

## State

| Date | Result |
|---|---|
| 2026-10-09 | written; `git fetch upstream` works from the dev container: `ce370e87..bf244291`, 932 commits behind, upstream at v2.0.0. The tool is a phase-6 deliverable. |
| 2026-10-09 | pre-implementation run (`run_upstream_sync.py`): `RESULT: FAIL (1 failures)` — `06_Code/c/tools/upstream_sync.py` does not exist yet, as expected. |
