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
