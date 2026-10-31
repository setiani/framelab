"""Spike (S-3): Muse Spark Open on Baseten, live, before turning it on for the Designer.

Checks
  B1  plain chat with reasoning_effort="low": answer, reasoning field, tokens, latency
  B2  response_format json_schema: Baseten documents no structured outputs for Spark Open
  B3  the real Designer on Spark Open for free fall and for the pendulum (predict tool, then
      submit_proposal tool), with the same validation as a run; falls back to Muse Spark 1.3
      only if MODEL_API_KEY is set and Spark Open fails

Usage (needs BASETEN_API_KEY in .env; MODEL_API_KEY only for the fallback)
  uv run python scripts/spikes/spike_spark_open.py
Approx. cost: a few cents of Baseten credits. Writes scripts/spikes/out/spike_spark_open.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from framelab.agent import designer  # noqa: E402
from framelab.agent.ledger import Ledger  # noqa: E402
from framelab.clients import baseten  # noqa: E402
from framelab.config import get_settings  # noqa: E402
from framelab.schemas import ExperimentSpec  # noqa: E402

OUT = ROOT / "scripts" / "spikes" / "out"


async def b1() -> dict:
    t0 = time.perf_counter()
    resp = await baseten.chat(
        messages=[
            {
                "role": "user",
                "content": "In one sentence: why does a pendulum's period not depend on its mass?",
            }
        ],  # fmt: skip
        purpose="spike",
        reasoning_effort="low",
    )
    msg = resp.choices[0].message
    reasoning = baseten.reasoning_text(msg)
    return {
        "ok": True,
        "latency_s": round(time.perf_counter() - t0, 2),
        "answer": msg.content,
        "reasoning_field": bool(reasoning),
        "reasoning_chars": len(reasoning or ""),
        "usage": resp.usage.model_dump() if resp.usage else None,
    }


async def b2() -> dict:
    schema = {"type": "object", "properties": {"g": {"type": "number"}}, "required": ["g"],
              "additionalProperties": False}  # fmt: skip
    try:
        resp = await baseten.chat(
            messages=[{"role": "user", "content": "Give g on Earth as JSON."}],
            purpose="spike",
            reasoning_effort="low",
            response_format={"type": "json_schema",
                             "json_schema": {"name": "g", "schema": schema, "strict": True}},
        )  # fmt: skip
    except baseten.BasetenError as exc:
        return {"ok": False, "supported": False, "error": str(exc)[:300]}
    return {"ok": True, "supported": "unclear", "content": resp.choices[0].message.content}


async def b3(experiment: str) -> dict:
    events: list[tuple[str, dict]] = []
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        settings = SimpleNamespace(designer_backend="baseten",
                                   baseten_api_key=get_settings().baseten_api_key)  # fmt: skip
        ctx = SimpleNamespace(settings=settings, run_dir=run_dir,
                              emit=lambda t, **p: events.append((t, p)))  # fmt: skip
        spec = ExperimentSpec(experiment=experiment, target_concept="ball",
                              analysis_window_s=(0, 2), motion_axis="y", confidence=0.9)  # fmt: skip
        session = SimpleNamespace(ctx=ctx, spec=spec, ledger=Ledger(run_dir / "ledger.jsonl"))
        t0 = time.perf_counter()
        proposal = await designer.run(session, headline={"value": 9.79},
                                      gate_warnings=["G3 slight residual correlation"])  # fmt: skip
        latency = round(time.perf_counter() - t0, 2)
    calls = [p for t, p in events if t == "model_result"]
    warnings = [p.get("message") for t, p in events if t == "warning"]
    if proposal is None:
        return {"ok": False, "latency_s": latency, "warnings": warnings, "calls": calls}
    return {
        "ok": True,
        "latency_s": latency,
        "models": proposal.models,
        "title": proposal.title,
        "values_m": [q.value for q in proposal.new_values],
        "prediction_en": designer.render_text(proposal.prediction_en,
                                              proposal.predicted_values, "en"),
        "prediction_id": designer.render_text(proposal.prediction_id,
                                              proposal.predicted_values, "id"),
        "steps_en": proposal.protocol_steps_en,
        "safety_en": proposal.safety_notes_en,
        "warnings": warnings,
        "calls": [{k: c.get(k) for k in ("model", "latency_s", "usage", "cost_usd", "ok")}
                  for c in calls],
    }  # fmt: skip


async def main() -> int:
    if not get_settings().baseten_api_key:
        sys.exit("BASETEN_API_KEY is not set. Put it in .env (see .env.example).")
    results: dict[str, dict] = {}
    for name, job in (("B1", b1), ("B2", b2), ("B3_free_fall", lambda: b3("free_fall")),
                      ("B3_pendulum", lambda: b3("pendulum"))):  # fmt: skip
        print(f"--- {name}")
        try:
            results[name] = await job()
        except Exception as exc:  # noqa: BLE001 - a spike reports, it does not crash
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                             "trace": traceback.format_exc()[-1500:]}  # fmt: skip
        print(json.dumps(results[name], ensure_ascii=False, indent=1, default=str)[:3000])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "spike_spark_open.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), "utf-8")
    designed = [k for k in ("B3_free_fall", "B3_pendulum")
                if results.get(k, {}).get("models") == [designer.SPARK_OPEN.model]]  # fmt: skip
    print(f"\nSaved {path}")
    print(f"Spark Open designed both proposals on its own: {len(designed) == 2} ({designed})")
    return 0 if len(designed) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
