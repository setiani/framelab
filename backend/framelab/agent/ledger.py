"""Claim Ledger: evidence (from tools/pipeline code) and claims (values copied by code).

Spec: docs/plan/06-agents-and-verification.md §1, §4; docs/plan/03-data-contracts.md §6.
Status: evidence (M3) and claims with copied values (M6).

``analysis/ledger.jsonl`` is append-only. Evidence ids are ``<prefix>_<NN>`` (``fit_01``);
claim ids are ``cNN``. Numbers in evidence are produced ONLY by deterministic code.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, get_args

import numpy as np

from framelab.agent.narrative import statement_errors
from framelab.schemas import Claim, ClaimType, Evidence

EVIDENCE_PREFIXES = (
    "seg",
    "trk",
    "cal",
    "win",
    "fit",
    "ci",
    "der",
    "gate",
    "fig",
    "cmp",
    "spk",
    "pred",
)
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

    def get_claim(self, claim_id: str) -> Claim:
        for claim in self.claims():
            if claim.id == claim_id:
                return claim
        raise KeyError(claim_id)

    def add_claim(
        self,
        claim_type: str,
        evidence_id: str,
        statement_en: str,
        statement_id: str,
        quantity: str | None = None,
    ) -> Claim:
        """Create a claim whose value, CI and unit are COPIED from the evidence (06 §4).

        The statements are templates: ``{value}``, ``{ci}``, ``{unit}``, ``{outcome}`` and
        ``{rel_diff}`` are filled at render time. Any other digit is rejected here, so a number can
        only enter a claim through the evidence.
        """
        if claim_type not in get_args(ClaimType):
            raise ValueError(f"unknown claim_type {claim_type!r}")
        for text in (statement_en, statement_id):
            errors = statement_errors(text)
            if errors:
                raise ValueError("; ".join(errors))
        with self._lock:
            try:
                source = self.get(evidence_id)
            except KeyError:
                raise KeyError(f"unknown evidence {evidence_id!r}") from None
            value, ci95, unit = copy_numbers(source)
            claim = Claim(
                id=self.next_id("c"),
                claim_type=claim_type,  # type: ignore[arg-type]
                quantity=quantity,
                value=value,
                ci95=ci95,
                unit=unit,
                evidence=[evidence_id],
                statement_en=statement_en,
                statement_id=statement_id,
            )
            self._append(json.loads(claim.model_dump_json()))
        return claim


def copy_numbers(ev: Evidence) -> tuple[float | None, tuple[float, float] | None, str | None]:
    """The value, 95 % interval and unit a claim takes from an evidence record."""
    data = ev.data
    raw_value = data.get(
        "value", data.get("m")
    )  # cmp_ hypothesis_test keeps the measured value as m
    value = float(raw_value) if isinstance(raw_value, int | float) else None
    ci: tuple[float, float] | None = None
    if data.get("ci95") is not None:
        ci = (float(data["ci95"][0]), float(data["ci95"][1]))
    elif data.get("lo") is not None and data.get("hi") is not None:
        ci = (float(data["lo"]), float(data["hi"]))
    unit = data.get("unit")
    return value, ci, str(unit) if unit is not None else None
