"""Designer: the next experiment, one variable changed; predictions computed by code.

Spec: docs/plan/06-agents-and-verification.md §6, §9.4; docs/plan/03-data-contracts.md §7.

The model picks the variable and the test values through one tool call, ``predict``. Code
computes every predicted value from the measured g and writes it to the ledger as ``pred_``
evidence. The proposal's prediction texts refer to those values as ``{{p1}}``, ``{{p2}}``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from framelab.agent import narrative
from framelab.agent.tools import ToolResult
from framelab.clients import meta as meta_client
from framelab.clients.meta import MetaClientError, strict_schema
from framelab.config import MODEL_SPARK
from framelab.physics.uncertainty import fmt
from framelab.schemas import DesignerProposal, Quantity
from framelab.store import write_json_atomic

if TYPE_CHECKING:
    from framelab.agent.session import AnalysisSession

PROMPT_PATH = Path(__file__).parent / "prompts" / "designer.md"
MAX_HEIGHT_M = 2.0  # 06 §6 safety: heights at most 2 m
MAX_VALUES = 5
MAX_ATTEMPTS = 2
VARIABLE = {"free_fall": "height", "pendulum": "string_length"}

PREDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "predict",
        "description": "Predicted values for the next experiment, computed by code from the "
        "measured g. Give 2–5 test values in metres (positive, at most 2 m).",
        "parameters": {
            "type": "object",
            "properties": {
                "changed_variable": {"type": "string", "enum": ["height", "string_length"]},
                "values": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["changed_variable", "values"],
        },
    },
}


class DesignerWire(BaseModel):
    """What the model writes (strict json_schema). Values and predictions come from code."""

    model_config = ConfigDict(extra="forbid")

    title: str
    changed_variable: str
    controlled_variables: list[str]
    prediction_en: str
    prediction_id: str
    protocol_steps_en: list[str]
    protocol_steps_id: list[str]
    safety_notes_en: list[str]
    safety_notes_id: list[str]


class DesignerUnavailable(RuntimeError):
    """No usable proposal after the retries. The run keeps its verdict without a proposal."""


def predict(experiment: str, changed_variable: str, values: list[float],
            g_measured: float) -> list[dict[str, Any]]:  # fmt: skip
    """pendulum: T = 2π·√(L/g); free fall: t_fall = √(2h/g). Small-angle, no drag (06 §6)."""
    if VARIABLE.get(experiment) != changed_variable:
        raise ValueError(
            f"{experiment} varies {VARIABLE.get(experiment)!r}, not {changed_variable!r}"
        )
    if not 1 <= len(values) <= MAX_VALUES:
        raise ValueError(f"give 1 to {MAX_VALUES} values")
    if g_measured <= 0 or not math.isfinite(g_measured):
        raise ValueError("the measured g must be positive")
    out = []
    for v in values:
        if not (0 < v <= MAX_HEIGHT_M):
            raise ValueError(f"value {v} m is outside (0, {MAX_HEIGHT_M}] m (safety limit)")
        if experiment == "pendulum":
            predicted = 2 * math.pi * math.sqrt(v / g_measured)
            formula = "T = 2π·√(L/g), small angle"
            unit_out = "s"
        else:
            predicted = math.sqrt(2 * v / g_measured)
            formula = "t = √(2h/g), no air drag"
            unit_out = "s"
        out.append({"input": float(v), "unit_in": "m", "predicted": predicted,
                    "unit_out": unit_out, "formula": formula})  # fmt: skip
    return out


def _assistant_call(call: Any) -> dict[str, Any]:
    """The assistant turn that made the ``predict`` call, replayed before its result."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call.id, "type": "function",
                        "function": {"name": "predict", "arguments": call.function.arguments}}],
    }  # fmt: skip


def load_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)


def render_text(text: str, predictions: list[dict[str, Any]], lang: str) -> str:
    """Replace ``{{pN}}`` with the formatted predicted value (N counts from 1)."""
    claims = {
        f"p{i}": fmt(p["predicted"], None, p["unit_out"], lang)
        for i, p in enumerate(predictions, start=1)
    }
    return narrative.render(text, claims)


async def _predict_step(
    session: AnalysisSession, experiment: str, g: float, gate_warnings: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ask the model for one ``predict`` call, run it, write the ``pred_`` evidence.

    Returns the predictions and the conversation so far (the next step continues it).
    """
    context = {
        "experiment": experiment,
        "measured_g_ms2": g,
        "allowed_variable": VARIABLE[experiment],
        "gate_warnings": gate_warnings,
        "safety": "household objects only, heights at most 2 m, no fire, no mains electricity, "
                  "no glass, adult supervision for children",
    }  # fmt: skip
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": load_prompt()},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    for _ in range(MAX_ATTEMPTS):
        resp = await meta_client.chat(model=MODEL_SPARK, messages=messages, tools=[PREDICT_TOOL],
                                      reasoning_effort="low", purpose="designer",
                                      emit=session.ctx.emit)  # fmt: skip
        msg = resp.choices[0].message
        calls = [c for c in (msg.tool_calls or []) if c.function.name == "predict"]
        if not calls:
            messages.append({"role": "user", "content": "Call the predict tool first."})
            continue
        call = calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
            predictions = predict(experiment, args["changed_variable"],
                                  [float(v) for v in args["values"]], g)  # fmt: skip
        except (ValueError, KeyError, TypeError) as exc:
            result = ToolResult(False, f"predict rejected: {exc}").to_json()
            messages += [
                _assistant_call(call),
                {"role": "tool", "tool_call_id": call.id, "content": result},
            ]
            continue
        for p in predictions:
            summary = f"{experiment}: {p['input']:g} m → {p['predicted']:.3f} s"
            ev = session.ledger.add_evidence("pred", "predict", p, summary)
            p["evidence_id"] = ev.id
        messages += [
            _assistant_call(call),
            {"role": "tool", "tool_call_id": call.id,
             "content": json.dumps({"ok": True, "predictions": [
                 {"id": f"p{i}", "input_m": p["input"], "predicted_s": round(p["predicted"], 4)}
                 for i, p in enumerate(predictions, start=1)]})},  # fmt: skip
        ]  # fmt: skip
        return predictions, messages
    raise DesignerUnavailable("the model did not call predict with valid values")


async def _write_step(
    session: AnalysisSession, messages: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> DesignerWire:
    """The proposal text, validated: prediction texts only use {{pN}} and no bare numbers."""
    messages = [*messages, {"role": "user", "content": "Write the proposal now as the JSON "
                "object. Prediction texts use the placeholders {{p1}}, {{p2}}… only."}]  # fmt: skip
    text_format = {"type": "json_schema", "json_schema": {
        "name": "designer_proposal", "schema": strict_schema(DesignerWire), "strict": True,
    }}  # fmt: skip
    for _ in range(MAX_ATTEMPTS):
        resp = await meta_client.chat(
            model=MODEL_SPARK,
            messages=messages,
            response_format=text_format,
            reasoning_effort="low",
            purpose="designer",
            emit=session.ctx.emit,
        )
        content = resp.choices[0].message.content or ""
        try:
            wire = DesignerWire.model_validate_json(content)
        except ValidationError as exc:
            problem = str(exc)[:300]
        else:
            problem = _prediction_problem(wire, len(predictions))
            if not problem:
                return wire
        messages += [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": f"Invalid proposal: {problem}. Fix it and return the JSON.",
            },
        ]
    raise DesignerUnavailable("the proposal was invalid after the retry")


def _prediction_problem(wire: DesignerWire, n: int) -> str:
    texts = [wire.prediction_en, wire.prediction_id]
    for text in texts:
        if narrative.bare_numbers(text):
            return f"prediction text has bare numbers {narrative.bare_numbers(text)}"
        for ref in re.findall(r"\{\{p(\d+)\}\}", text):
            if not 1 <= int(ref) <= n:
                return f"placeholder {{{{p{ref}}}}} does not exist"
    return ""


async def run(
    session: AnalysisSession, *, headline: dict[str, Any] | None, gate_warnings: list[str]
) -> DesignerProposal | None:
    """Propose the next experiment. ``None`` if the Designer cannot give a valid proposal."""
    if headline is None:
        return None
    experiment = session.spec.experiment
    g = float(headline["value"])
    try:
        predictions, messages = await _predict_step(session, experiment, g, gate_warnings)
        wire = await _write_step(session, messages, predictions)
    except (DesignerUnavailable, MetaClientError) as exc:
        session.ctx.emit("warning", message=f"designer unavailable: {exc}")
        return None
    values = [Quantity(value=p["input"], unit="m") for p in predictions]
    proposal = DesignerProposal(
        title=wire.title,
        changed_variable=VARIABLE[experiment],  # the code's choice, the model cannot change it
        controlled_variables=wire.controlled_variables,
        new_values=values,
        prediction_en=wire.prediction_en,
        prediction_id=wire.prediction_id,
        protocol_steps_en=wire.protocol_steps_en,
        protocol_steps_id=wire.protocol_steps_id,
        safety_notes_en=wire.safety_notes_en,
        safety_notes_id=wire.safety_notes_id,
        predicted_values=[{"id": f"p{i}", **p} for i, p in enumerate(predictions, start=1)],
    )  # fmt: skip
    write_json_atomic(session.ctx.run_dir / "analysis" / "designer.json", proposal)
    return proposal
