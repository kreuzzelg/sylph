# 2026-10-04 — kickoff

**Owner:**

> Ich möchte eine Version von Colibri erstellen, die auch GGUF Format unterstützt.
> Diese sollte in einem separaten Branch probiert werden. Erstell bitte eine
> Anforderungsdokument und eine Architektur für diese Zweck. Wenn Du Fragen hast,
> bitte frag jetzt, damit diese Arbeit übernacht durchgeführt werden kann.

**Clarifications answered by the owner (multiple choice):**

- Direction: load GGUF directly in the engine.
- Quant types for v1: Q4_0 / Q8_0 / F16 **and** the K-quants Q4_K, Q5_K, Q6_K.
- Dependencies: no — own reader and kernels, pure C.
- Deliverable: English documents only.

**Result:** `02_Specifications/gguf-specification.md`, `03_Architecture/gguf-architecture.md`
(originally `docs/gguf/REQUIREMENTS.md` and `ARCHITECTURE.md` on the fork branch
`claude/epic-edison-u3ncsq`).
