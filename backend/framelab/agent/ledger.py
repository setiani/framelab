"""Claim Ledger: evidence (from tools) and claims (values copied by code).

Spec: docs/plan/06-agents-and-verification.md §1, §4; docs/plan/03-data-contracts.md §6
Status: skeleton (M0). Implement in milestone M3 (evidence) and M6 (claims).
"""

from __future__ import annotations

from pathlib import Path

from framelab.schemas import Claim, Evidence

_TODO = "Milestone M3/M6: see docs/plan/09-milestones-and-tasks.md"


class Ledger:
    """Append-only ``analysis/ledger.jsonl`` holding evidence and claims."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def next_id(self, prefix: str) -> str:
        """e.g. ``'fit_'`` → ``'fit_03'``; ``'c'`` → ``'c07'``."""
        raise NotImplementedError(_TODO)

    def add_evidence(self, evidence: Evidence) -> Evidence:
        """Append an evidence record (produced only by tools/pipeline code)."""
        raise NotImplementedError(_TODO)

    def add_claim(
        self,
        claim_type: str,
        evidence_id: str,
        statement_en: str,
        statement_id: str,
        quantity: str | None = None,
    ) -> Claim:
        """Create a claim whose value/ci/unit are COPIED from the evidence record."""
        raise NotImplementedError(_TODO)

    def evidence(self) -> list[Evidence]:
        raise NotImplementedError(_TODO)

    def claims(self) -> list[Claim]:
        raise NotImplementedError(_TODO)
