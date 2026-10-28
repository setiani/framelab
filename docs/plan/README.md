# FrameLab: Implementation Plan

> **Every phone is a physics lab.** FrameLab turns a phone video of a simple experiment
> (a falling ball, a pendulum) into measured, uncertainty-quantified, *verified* scientific
> data, and then designs the next experiment.
>
> Built for the **Meta Global AI Developer Hackathon 2026** (Indie track, problem statement
> *Push the Frontier*). Building period: 27 Oct – 4 Nov 2026. Submission deadline:
> **Wed 4 Nov 2026, 21:00 ET = Thu 5 Nov 2026, 09:00 WIB**.

This folder is the **single source of truth** for implementing FrameLab. It is written
for a coding agent (Muse Code, Claude Code, OpenCode, Cursor, …) and for the human who
steers it. If the code and this plan disagree, either fix the code or update the plan in
the same commit. Do not let them drift silently.

---

## Reading order

| # | File | What it defines | Read when |
|---|------|-----------------|-----------|
| — | [`AGENTS.md`](AGENTS.md) | Rules for the coding agent. **Copy to the repo root.** | Always, first |
| 00 | [`00-product-brief.md`](00-product-brief.md) | What we build, for whom, scope, non-goals, judging fit | Before any work |
| 01 | [`01-architecture.md`](01-architecture.md) | System diagram, components, run lifecycle, folder layout, repo tree, tech stack | Before any work |
| 02 | [`02-meta-api-reference.md`](02-meta-api-reference.md) | Exact Meta Model API usage (SAM 3.1, Muse Spark, Muse Wonder, Voice, Image, Baseten) with code | Before writing any client code |
| 03 | [`03-data-contracts.md`](03-data-contracts.md) | Pydantic models, run-folder files, JSONL formats, config | Before writing schemas |
| 04 | [`04-pipeline-and-measurement.md`](04-pipeline-and-measurement.md) | Ingest, transcription, segmentation, mask→measurement, calibration, overlay, report | M2–M3, M8 |
| 05 | [`05-physics-and-uncertainty.md`](05-physics-and-uncertainty.md) | Free fall, pendulum, series regression, bootstrap, error propagation, event detection | M3–M4 |
| 06 | [`06-agents-and-verification.md`](06-agents-and-verification.md) | Planner, Analyst, Auditor, Designer, tool schemas, full prompts, gates, Claim Ledger | M5–M7 |
| 07 | [`07-backend-api-and-frontend.md`](07-backend-api-and-frontend.md) | REST + SSE contract, UI layout and behaviour | M9–M10 |
| 08 | [`08-testing-and-evaluation.md`](08-testing-and-evaluation.md) | Unit, golden, live tests; benchmark protocol and ground truth | Every milestone |
| 09 | [`09-milestones-and-tasks.md`](09-milestones-and-tasks.md) | Ordered task list with acceptance criteria and ready-to-paste agent prompts | Daily |
| 10 | [`10-deployment-demo-submission.md`](10-deployment-demo-submission.md) | Docker, hosting, demo mode, demo video script, submission checklist | M11–M12 |
| 11 | [`11-risks-and-open-questions.md`](11-risks-and-open-questions.md) | Unknowns to verify in the first spike, risks, fallbacks | M1, and whenever something breaks |

---

## How to use this plan with a coding agent

1. Create the repository folder next to this plan, e.g. `meta-hackathon-2026/framelab/`.
2. Copy `plan/` into the repo as `docs/plan/` (so the agent can read it inside its
   workspace and so the plan is public with the code), and copy `plan/AGENTS.md` to the
   repo root as `AGENTS.md`.
3. Work **one milestone at a time** from `09-milestones-and-tasks.md`. Each milestone has a
   "Prompt for the coding agent" block. Paste it, let the agent work, then check the
   acceptance criteria yourself before moving on.
4. After each milestone: run `make test`, commit, and push (the repo must be public).
5. When the agent proposes something that contradicts the plan (different framework, LLM
   computing numbers, skipping gates…), point it back to the relevant section.

### Golden rules (repeated in AGENTS.md)

1. **The LLM decides, code computes.** No number shown to the user may be produced by a
   language model. Every number comes from deterministic Python and is recorded in the
   Claim Ledger.
2. **Every claim has evidence.** Report text references claims (`{{c03}}`). The renderer
   fills values from the ledger. Bare numbers in model-written text fail gate G5.
3. **Never tune data toward the textbook value.** Gate G4 (plausibility) only triggers a
   diagnosis. It never changes or hides data.
4. **Everything is logged** in `events.jsonl` and replayable offline.
5. **Ship the MVP first** (free fall end-to-end on the live site), then pendulum, then
   stretch goals.

---

## Ringkasan singkat (Bahasa Indonesia)

Folder ini berisi rencana implementasi FrameLab yang lengkap untuk coding agent. Dokumennya
ditulis dalam bahasa Inggris supaya coding agent mengikutinya dengan akurat, dan karena repo
serta juri hackathon berbahasa Inggris.

**Cara memakainya:**

1. Buat folder repo `framelab/` di dalam folder hackathon.
2. Salin folder `plan/` ke `framelab/docs/plan/`, lalu salin `AGENTS.md` ke root repo.
3. Kerjakan milestone satu per satu sesuai `09-milestones-and-tasks.md`. Setiap milestone
   punya blok "Prompt for the coding agent" yang bisa langsung ditempel ke Muse Code.
4. Setelah setiap milestone, cek *acceptance criteria*, jalankan tes, lalu commit dan push.
5. **Milestone M1 (spike verifikasi API) wajib dikerjakan pertama.** Beberapa perilaku API
   (indeks frame SAM, batas fps/durasi video, format structured output di Responses API)
   belum terdokumentasi dan harus diuji dulu. Daftarnya ada di
   `11-risks-and-open-questions.md`.

Aturan terpenting: model AI **tidak boleh** menghasilkan angka hasil pengukuran. Semua angka
dihitung kode Python dan dicatat di *Claim Ledger*.
