"""Lab report: Jinja2 HTML → PDF (Playwright Chromium), ID and EN.

Spec: docs/plan/04-pipeline-and-measurement.md §9–§10
Status: skeleton (M0). Implement in milestone M8.
"""

from __future__ import annotations

from pathlib import Path

from framelab.schemas import ReportDraft


def render_html(run_dir: Path, draft: ReportDraft) -> str:
    """Fill templates/report.html.j2; claim placeholders rendered via agent.narrative."""
    raise NotImplementedError("Milestone M8: see docs/plan/09-milestones-and-tasks.md")


def html_to_pdf(html: str, out_pdf: Path) -> Path:
    """Playwright: page.set_content(html); page.pdf(format='A4', print_background=True)."""
    raise NotImplementedError("Milestone M8: see docs/plan/09-milestones-and-tasks.md")


def export_csv(run_dir: Path) -> Path:
    """outputs/data.csv in SI units with # header comments (04 §10)."""
    raise NotImplementedError("Milestone M8: see docs/plan/09-milestones-and-tasks.md")
