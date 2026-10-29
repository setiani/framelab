from __future__ import annotations

import numpy as np
import pytest

from framelab.agent.ledger import Ledger


def test_ids_and_evidence_roundtrip(tmp_path) -> None:
    ledger = Ledger(tmp_path / "analysis" / "ledger.jsonl")
    a = ledger.add_evidence(
        "fit", "fit_model", {"r2": np.float64(0.99), "p": np.array([1, 2])}, "fit"
    )
    b = ledger.add_evidence("fit", "fit_model", {}, "fit 2", inputs=[a.id])
    c = ledger.add_evidence("ci", "bootstrap_ci", {"lo": 1, "hi": 2}, "ci", inputs=[a.id, b.id])
    assert (a.id, b.id, c.id) == ("fit_01", "fit_02", "ci_01")
    assert ledger.next_id("c") == "c01"
    again = Ledger(ledger.path)
    assert [e.id for e in again.evidence()] == ["fit_01", "fit_02", "ci_01"]
    assert again.get("fit_01").data == {"r2": 0.99, "p": [1, 2]}


def test_rejects_unknown_inputs_and_prefixes(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(KeyError):
        ledger.add_evidence("der", "derive", {}, "x", inputs=["fit_99"])
    with pytest.raises(ValueError):
        ledger.add_evidence("bogus", "x", {}, "x")
