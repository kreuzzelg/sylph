# 2026-10-05 — separate repository, structured layout

**Owner:**

> Das Repo https://github.com/kreuzzelg/colibri ist ein Fork von
> https://github.com/JustVugg/colibri . Da JustVugg den GGUF ungerne unterstützen
> möchte, macht Sinn, dass wir ein separates Repo verwenden, um diese GGUF Variant
> von Colibri zu entwickeln. Es kommt dazu, dass ich die Projektstruktur des neuen
> Repo nach dem Skill (siehe unten) organisieren möchte.
>
> 1. Kannst du einen neuen Repo legen?
> 2. Das Projekt braucht ein Name. Hast du ein Vorschlag? Zum Beispiel gibt es eine
>    bekannte Art von Colibri als Name?
> 3. In dem Repo soll die Verzeichnis wie in Skill beschrieben organisiert werden.
>    Das Verzeichnis "06_Code" entspricht dann https://github.com/JustVugg/colibri

followed by the `structured-project` skill text (folder layout `01_Requirements` …
`08_Documents`, Markdown + PlantUML, folder rules), then:

> Sylph sieht gut aus

**Result:** this repository (`kreuzzelg/sylph`): skeleton, `06_Code/` as a git
subtree of the fork at `gguf/p1-reader`, documents moved into the numbered
folders, `04_Tasks/tasks.md`, phase-1 integration and system tests.

---

# 2026-10-05 — owner requirements, v2 documents

**Owner** (after editing `01_Requirements/README.md` on GitHub):

> Ich habe die Requirement editiert. Passen Sie bitte entsprechend Spezifikation und Architektur an.

**Result:** v2 of `02_Specifications/gguf-specification.md` and
`03_Architecture/gguf-architecture.md` (Qwen3.6-35B-A3B on the `qwen36` engine; Ollama/llama.cpp
as reference with equivalence levels E0–E3), upstream colibrì v1.12.1 merged into `06_Code/`,
`08_Documents/inspection-qwen36-gguf-2026-10-05.md` with the converter value-transform audit.
