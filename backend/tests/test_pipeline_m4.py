"""M4 offline: synthetic pendulum video + synthetic SAM records → T, T0, g; series CLI."""

from __future__ import annotations

import json
import math
import shutil

import pytest
from typer.testing import CliRunner

from framelab.agent.ledger import Ledger
from framelab.cli import app as cli_app
from framelab.physics.pendulum import analyze_pendulum
from framelab.schemas import Stage
from framelab.store import RunStore, write_json_atomic
from tests.fakes import MEDIA, pendulum_fake
from tests.synth import make_pendulum

G_TRUE = 9.81  # pendulum_60fps_12s: L = 0.60 m, θ0 = 15°, γ = 0.02 s⁻¹ (exact nonlinear ODE)
L_TRUE = 0.60
T0_TRUE = 2 * math.pi * math.sqrt(L_TRUE / G_TRUE)
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _run(tmp_path, *extra: str):
    runs = tmp_path / "runs"
    result = CliRunner().invoke(
        cli_app,
        ["run", str(MEDIA / "pendulum_60fps_12s.mp4"), "--no-planner", "--experiment", "pendulum",
         "--concept", "red ball", "--runs-dir", str(runs), *extra],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    run_dir = next(runs.glob("r_*"))
    return result.output, run_dir, json.loads((run_dir / "state.json").read_text())


@needs_ffmpeg
def test_pendulum_with_spoken_length_ruler_and_prediction(tmp_path, monkeypatch) -> None:
    fake = pendulum_fake().install(monkeypatch)
    out, run_dir, state = _run(
        tmp_path, "--string-length-m", "0.60", "--ref-concept", "ruler", "--ref-length-m", "1.0",
        "--predict", "1.55 s",
    )  # fmt: skip
    assert ("image", "string") in fake.calls
    assert "g = 9.8" in out and "hypothesis: close" in out  # T = 1.559 s, within 5 %
    h = state["headline"]
    assert h["experiment"] == "pendulum" and h["quantity"] == "g"
    assert h["value"] == pytest.approx(G_TRUE, rel=0.005)
    assert h["ci95"][0] <= G_TRUE <= h["ci95"][1]
    assert h["period0"] == pytest.approx(T0_TRUE, rel=0.001)
    assert h["length_source"] == "spoken" and 10 < h["theta_mean_deg"] < 15
    assert h["n_cycles"] > 7 and h["scale_method"] == "ruler"
    assert h["scale_agreement_pct"] < 2.0  # ruler vs spoken length / pivot radius

    ledger = Ledger(run_dir / "analysis" / "ledger.jsonl")
    tools = {e.id: e.tool for e in ledger.evidence()}
    assert "calibrate_pivot" in tools.values() and "hypothesis_test" in tools.values()
    der = ledger.get(h["evidence"])
    assert der.data["quantity"] == "g" and der.data["value"] == h["value"]
    pend = json.loads((run_dir / "analysis" / "pendulum.json").read_text())
    assert pend["pivot"]["reliable"] and pend["pivot"]["r_px"] == pytest.approx(276, abs=3)
    assert "pivot_string_mismatch" not in pend["pivot"]["warnings"]


@needs_ffmpeg
def test_pendulum_period_only_and_measured_length(tmp_path, monkeypatch) -> None:
    pendulum_fake().install(monkeypatch)
    out, _run_dir, state = _run(tmp_path)
    h = state["headline"]
    assert h["quantity"] == "period" and h["unit"] == "s" and "T = 1.55" in out
    assert h["period0"] == pytest.approx(T0_TRUE, rel=0.001)
    assert any("String length unknown" in w for w in state["warnings"])

    out, _run_dir, state = _run(tmp_path / "b", "--ref-concept", "ruler", "--ref-length-m", "1.0")
    h = state["headline"]
    assert h["length_source"] == "measured"
    assert h["length_m"] == pytest.approx(L_TRUE, rel=0.01)
    assert h["value"] == pytest.approx(G_TRUE, rel=0.015)


# --- series ------------------------------------------------------------------------------------
def _fake_pendulum_run(store: RunStore, length: float | None, seed: int, experiment="pendulum"):
    state = store.create_run(f"pendulum_{length}.mp4")
    t, x, _y = make_pendulum(L=length or 0.5, theta0_deg=10, seed=seed)
    r = analyze_pendulum(
        t, x, length_m=length, sigma_length_m=0.003 if length else None,
        length_source="spoken" if length else None, radius_px=(length or 0.5) * 460,
        bootstrap_n=60, seed=seed,
    )  # fmt: skip
    write_json_atomic(store.run_dir(state.run_id) / "analysis" / "pendulum.json", r.to_dict())
    state.experiment = experiment  # type: ignore[assignment]
    state.stage = Stage.ANALYZING
    store.save_state(state)
    return state.run_id


def test_series_cli_end_to_end(tmp_path) -> None:
    runs = tmp_path / "runs"
    store = RunStore(runs)
    ids = [_fake_pendulum_run(store, length, i) for i, length in enumerate((0.3, 0.5, 0.8))]
    cli = CliRunner()
    args = ["--runs-dir", str(runs)]
    res = cli.invoke(cli_app, ["series", "create", *sum((["--run", r] for r in ids), []), *args])
    assert res.exit_code == 0, res.output
    sid = res.stdout.strip().splitlines()[0]
    assert store.load_state(ids[0]).series_id == sid

    res = cli.invoke(cli_app, ["series", "analyze", sid, *args])
    assert res.exit_code == 0, res.output
    assert "√L law supported" in res.output
    s = store.load_series(sid)
    assert s.status == "done"
    assert s.headline["value"] == pytest.approx(9.78, rel=0.005)
    assert s.headline["n"] == pytest.approx(0.5, abs=0.02)
    ledger = Ledger(store.series_dir(sid) / "analysis" / "ledger.jsonl")
    assert ledger.get(s.headline["evidence"]).data["value"] == s.headline["value"]

    # re-analysis is idempotent (ledger rebuilt, same ids)
    res = cli.invoke(cli_app, ["series", "analyze", sid, *args])
    assert res.exit_code == 0 and store.load_series(sid).headline["evidence"] == "der_01"


def test_series_rejects_unsuitable_runs(tmp_path) -> None:
    runs = tmp_path / "runs"
    store = RunStore(runs)
    good = [_fake_pendulum_run(store, length, i) for i, length in enumerate((0.3, 0.5))]
    no_len = _fake_pendulum_run(store, None, 5)
    ff = _fake_pendulum_run(store, 0.4, 6, experiment="free_fall")
    cli = CliRunner()
    for bad in (no_len, ff):
        res = cli.invoke(
            cli_app,
            ["series", "create", "--run", good[0], "--run", good[1], "--run", bad,
             "--runs-dir", str(runs)],
        )  # fmt: skip
        assert res.exit_code == 1
    assert (
        "length unknown"
        in cli.invoke(
            cli_app, ["series", "create", "--run", no_len, "--runs-dir", str(runs)]
        ).output
    )

    res = cli.invoke(cli_app, ["series", "create", "--run", good[0], "--run", good[1],
                               "--runs-dir", str(runs)])  # fmt: skip
    sid = res.stdout.strip().splitlines()[0]
    res = cli.invoke(cli_app, ["series", "analyze", sid, "--runs-dir", str(runs)])
    assert res.exit_code == 1 and "at least 3 runs" in res.output
    assert store.load_series(sid).status == "failed"
