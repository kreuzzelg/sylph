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
| 2026-10-09 | **implemented: `RESULT: ok (0 failures)`** (`run_upstream_sync.py --trial`). `--check`: base `ce370e87` (v1.12.1, merged in `9db7d4ff`), upstream `bf244291` (2026-10-06, v2.0.0), 932 commits behind, tags since: `v2.0.0`; 512 upstream-changed paths, 49 sylph-changed, 15 in both (`CHANGELOG.md`, `c/Makefile`, `c/backend_cuda.cu/.h`, `c/coli`, `c/doctor.py`, `c/family_registry.py`, `c/qwen36.c`, `c/qwen36_tier.c/.h`, `c/resource_plan.py`, `c/tests/qwen36_fake_cuda.h`, `docs/ENVIRONMENT.md`, `docs/docs.json`, `docs/qwen36-cuda-tier.md`). `--trial` (4 s, temporary `--shared` clone): exit 3, **9 conflicted files** ⊆ overlap (`CHANGELOG.md`, `c/Makefile`, `c/coli`, `c/doctor.py`, `c/family_registry.py`, `c/qwen36.c`, `c/qwen36_tier.c`, `docs/ENVIRONMENT.md`, `docs/docs.json`); working tree unchanged. Procedure in `06_Code/docs/gguf.md`. Case 4 (`--apply`) is the housekeeping item. Tool fix during the run: a failed git command must never read as empty output (`merge-base --is-ancestor`). |
| 2026-10-14 | phase 7 makes the `--apply` (v2.0.0) its first item; the post-sync contracts beyond these gates are `../IntegrationTest/multi_slot_gguf.md` (`KV_SLOTS` on a GGUF, Vulkan knobs by name, container unchanged). |
| 2026-10-15 | **`--apply` done on the owner's host**: `git subtree pull --prefix=06_Code upstream main` (v2.0.0, `bf24429`, 932 commits; the tool needed a configured git identity). 13 files conflicted (the trial forecast 13): `CHANGELOG.md`, `c/.gitignore`, `c/Makefile` (5 hunks), `c/coli`, `c/colibri.c` (4), `c/doctor.py` (3), `c/family_registry.py`, `c/qwen36.c` (**22 hunks**: upstream's QW/Slot structs, dense_resolve/dense_has/load_t_affine, load_norm_n with `zero_centered_norms`, the Vulkan dense and tier arms, `moe_ex` with `use_qq`/`use_vk`/`expert_hold`, Clef, the KV_SLOTS mux, `qwen36_resolved_cap`), `c/qwen36_tier.c` (2: in-place LFRU overwrite — the raw-block experts keep free-then-upload), `c/tests/test_qwen36_tensor_kinds.py`, `c/tools/qwen36_tensor_kinds.py` (upstream now converts the vision tower and skips mtp; sylph's `("mtp", kind)` stays for st2gguf and `convert_qwen36.py` counts it as skipped), `docs/ENVIRONMENT.md`, `docs/docs.json`. Policy applied as documented: sylph's arms kept, upstream everywhere else; `cfg_validate` (shared CKR block) gained upstream's new indexer check; `dense_has`/`dense_resolve` route a GGUF to the façade; `cfg_from_gguf` sets `zero_centered_norms` (the façade already hands back w); `uclass` keeps llama.cpp's classes for a GGUF tokenizer (`g_tok_llama`) and upstream's tables (incl. its new `g_tok_exact`) for containers; the `CACHE`/`GGUF reads:` lines also end a KV_SLOTS mux turn; sylph's Makefile rules dropped their hand-listed headers (upstream's `-MMD`, `test_makefile_deps`). Gates after the merge: `test_unicode_cat`, `test_gguf_load`, `test_gq_kernels`, `test_gguf`, `test_gguf_load_glm` pass; all seven phase runners `RESULT: ok`; `run_efficiency.py` ok; wikitext tokenizer gate 297 193 ids identical; the blob and the unsloth file load and answer as before the merge (teacher-forced window 0: TF-NLL 1.3993 unchanged, per-position log-probs within 1e-6 of the pre-merge dump — upstream's new batched shared-expert path reorders a few sums); CUDA tier on the RTX 3070 16.1 tok/s with the CPU's greedy text. `make check` green after the merge (every C suite, 2363 Python tests, 21 min; two merge slips it caught: an extra `memory_available()` call in `run_doctor` that broke upstream's cgroup admission tests, and `$(VK_OBJ)` missing from four sylph link rules). Not yet exercised: `multi_slot_gguf.md` (KV_SLOTS on a GGUF). |

