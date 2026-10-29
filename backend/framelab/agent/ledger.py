"""Claim Ledger: evidence (from tools/pipeline code) and claims (values copied by code).

Spec: docs/plan/06-agents-and-verification.md §1, §4; docs/plan/03-data-contracts.md §6.
Status: evidence implemented (M3); claims arrive with the Analyst tools (M6).

``analysis/ledger.jsonl`` is append-only. Evidence ids are ``<prefix>_<NN>`` (``fit_01``);
claim ids are ``cNN``. Numbers in evidence are produced ONLY by deterministic code.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from framelab.schemas import Claim, Evidence

EVIDENCE_PREFIXES = ("seg", "trk", "cal", "win", "fit", "ci", "der", "gate", "fig", "cmp", "spk")
_ID_RE = re.compile(r"^(?P<prefix>[a-z]+)_(?P<num>\d+)$|^c(?P<claim>\d+)$")


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and tuples so evidence data is plain JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Ledger:
    """Append-only ``analysis/ledger.jsonl`` holding evidence and claims."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    # --- reading ------------------------------------------------------------------------
    def _records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def evidence(self) -> list[Evidence]:
        return [Evidence.model_validate(r) for r in self._records() if r.get("kind") == "evidence"]

    def claims(self) -> list[Claim]:
        return [Claim.model_validate(r) for r in self._records() if r.get("kind") == "claim"]

    def get(self, evidence_id: str) -> Evidence:
        for ev in self.evidence():
            if ev.id == evidence_id:
                return ev
        raise KeyError(evidence_id)

    def ids(self) -> set[str]:
        return {r["id"] for r in self._records()}

    # --- writing ------------------------------------------------------------------------
    def next_id(self, prefix: str) -> str:
        """e.g. ``'fit'`` → ``'fit_03'``; ``'c'`` → ``'c07'``."""
        prefix = prefix.rstrip("_")
        highest = 0
        for rid in self.ids():
            m = _ID_RE.match(rid)
            if not m:
                continue
            if prefix == "c" and m.group("claim"):
                highest = max(highest, int(m.group("claim")))
            elif m.group("prefix") == prefix:
                highest = max(highest, int(m.group("num")))
        return f"c{highest + 1:02d}" if prefix == "c" else f"{prefix}_{highest + 1:02d}"

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def add_evidence(
        self,
        prefix: str,
        tool: str,
        data: dict[str, Any],
        summary: str,
        inputs: list[str] | None = None,
    ) -> Evidence:
        """Allocate an id and append an evidence record. Inputs must already exist."""
        if prefix not in EVIDENCE_PREFIXES:
            raise ValueError(f"unknown evidence prefix {prefix!r}")
        with self._lock:
            known = self.ids()
            missing = [i for i in inputs or [] if i not in known]
            if missing:
                raise KeyError(f"unknown input evidence {missing}")
            ev = Evidence(
                id=self.next_id(prefix),
                tool=tool,
                inputs=list(inputs or []),
                data=_jsonable(data),
                summary=summary,
            )
            self._append(json.loads(ev.model_dump_json()))
        return ev

    def add_claim(
        self,
        claim_type: str,
        evidence_id: str,
        statement_en: str,
        statement_id: str,
        quantity: str | None = None,
    ) -> Claim:
        """Create a claim whose value/ci/unit are COPIED from the evidence record (M6)."""
        raise NotImplementedError("Milestone M6: see docs/plan/09-milestones-and-tasks.md")
