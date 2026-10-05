# sylph

**sylph** is an experimental fork of [colibrì](https://github.com/JustVugg/colibri)
(the pure-C, zero-dependency engine that runs GLM-5.2's 744B Mixture-of-Experts
model by streaming experts across VRAM, RAM and NVMe) that adds **direct GGUF
support**: the engine reads llama.cpp GGUF files (K-quants and simple block
types) and streams the routed experts straight from them, computing on the
blocks as stored — no conversion, no re-quantization.

Upstream colibrì deliberately does not want GGUF in its tree, so this work lives
here. Everything that is not GGUF-specific is upstream's and is kept in sync via
`git subtree` (see below). License: Apache-2.0, see `LICENSE` and `NOTICE`.

Named after the sylphs, the long-tailed hummingbirds of the Andes.

## Repository layout

The project follows a fixed folder structure; each folder has a single purpose.

| Folder | Content |
|---|---|
| `01_Requirements/` | Requirements, **written by the project owner only**. |
| `02_Specifications/` | What the GGUF support must do (functional and non-functional requirements, acceptance criteria). |
| `03_Architecture/` | How it is built: modules, interfaces, data flow (Markdown + PlantUML). |
| `04_Tasks/` | `tasks.md`, the single source of truth for implementation progress. |
| `05_Prompt/` | Prompts used to drive the work. |
| `06_Code/` | The engine: colibrì as a git subtree plus the GGUF modules. Unit tests live here (`06_Code/c/tests/`). |
| `07_Tests/` | `IntegrationTest/` (per module), `SystemTest/` (end to end), `UserTest/` (requested by the owner). |
| `08_Documents/` | Reports, inspection logs, release notes. |

## Building and running

The engine is unchanged in use; everything happens under `06_Code/`:

```sh
make -C 06_Code/c check                     # portable CPU build + all unit tests (no model needed)
make -C 06_Code/c colibri                   # the GLM-5.2 engine
06_Code/c/coli gguf inspect /path/to/model.gguf     # phase 1: metadata view of any GGUF
06_Code/c/coli doctor --model /path/to/model.gguf --deep
```

Running a model *from* a GGUF is phase 3/4 work; see `04_Tasks/tasks.md` for the
current state and `03_Architecture/` for the plan.

## Keeping up with upstream colibrì

`06_Code/` was added with `git subtree` and keeps the full upstream history, so
upstream changes can be merged in place:

```sh
git remote add upstream https://github.com/JustVugg/colibri
git fetch upstream
git subtree pull --prefix=06_Code upstream main --no-squash
```

GGUF-specific files (`06_Code/c/gguf.h`, `ggufinfo.py`, the GGUF tests, the
`coli gguf` subcommand, the GGUF branch of `doctor.py`) do not exist upstream and
never conflict; shared files (`colibri.c`, `coli`, `doctor.py`, `Makefile`) may.
