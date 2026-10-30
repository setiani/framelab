"""Analyst tools: JSON schemas, dispatch, compact results, evidence records.

Spec: docs/plan/06-agents-and-verification.md §4
Every tool computes its numbers with code (``AnalysisSession``), writes them to the ledger, and
returns a compact result. Tool results sent to the model are at most 2 KB (06 §3.1).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from framelab.agent import gates as gates_mod
from framelab.agent.ledger import copy_numbers
from framelab.agent.narrative import render_statement
from framelab.agent.session import AnalysisSession, ToolError
from framelab.physics.experiments import REGISTRY
from framelab.schemas import Claim, ReportDraft
from framelab.store import write_json_atomic

MAX_RESULT_BYTES = 2048
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}
_EVENT_ARGS_CHARS = 300


@dataclass
class ToolResult:
    ok: bool
    summary: str
    evidence_ids: list[str] = field(default_factory=list)
    data_excerpt: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    terminal: bool = False
    payload: Any = None

    def to_json(self) -> str:
        """Compact JSON for the model: at most 2 KB. Large fields are cut, never the status."""
        body: dict[str, Any] = {
            "ok": self.ok,
            "summary": self.summary,
            "evidence_ids": self.evidence_ids,
            "data": self.data_excerpt,
            "warnings": self.warnings,
        }
        text = json.dumps(body, ensure_ascii=False, default=str)
        if len(text.encode("utf-8")) <= MAX_RESULT_BYTES:
            return text
        body["data"] = {"truncated": True}
        body["evidence_ids"] = self.evidence_ids[:20]
        body["warnings"] = self.warnings[:5]
        body["summary"] = self.summary[:400]
        text = json.dumps(body, ensure_ascii=False, default=str)
        if len(text.encode("utf-8")) <= MAX_RESULT_BYTES:
            return text
        return json.dumps({"ok": self.ok, "summary": self.summary[:200]}, ensure_ascii=False)


@dataclass(frozen=True)
class RerecordRequest:
    reasons: list[str]
    tips_en: list[str]
    tips_id: list[str]


Handler = Callable[[AnalysisSession, dict[str, Any]], Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


_WINDOW_ID = {"type": "string", "description": "id of a set_fit_window result, e.g. win_01"}
_FIT_ID = {"type": "string", "description": "id of a fit_model result, e.g. fit_02"}
_DER_ID = {"type": "string", "description": "id of a derive result, e.g. der_03"}
_SCALE_ID = {"type": "string", "description": "id of a scale (cal_), e.g. cal_01"}
_REASON = {"type": "string", "description": "why, in a few words"}


# --------------------------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------------------------


async def _get_run_summary(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    evidence = [f"{e.id}: {e.summary[:90]}" for e in s.ledger.evidence()][-12:]
    claims = [f"{c.id}: {c.claim_type} {c.quantity or ''}" for c in s.ledger.claims()][-12:]
    return ToolResult(
        True,
        "run summary",
        data_excerpt={
            "experiment": s.spec.experiment,
            "round": s.round,
            "has_numeric_prediction": s.spec.hypothesis_prediction is not None,
            "track": s.track_summary(),
            "calibration": s.calibration_summary(),
            "windows": list(s.windows)[-4:],
            "fits": list(s.fits)[-4:],
            "evidence_latest": evidence,
            "claims": claims,
        },
    )


async def _choose_track(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    track = s.choose_track(a["track_id"])
    return ToolResult(
        True,
        f"track {track.track_id} is active",
        evidence_ids=[s.track_evidence_id],
        data_excerpt={"n_valid": sum(p.valid for p in track.points), "coverage": track.coverage},
    )


async def _resegment(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    concept = a.get("concept") or s.spec.target_concept
    out = await s.resegment(concept, a.get("reason", ""))
    return ToolResult(
        True,
        f"SAM re-segmented '{concept}'",
        evidence_ids=[out["seg_id"], s.track_evidence_id],
        data_excerpt={"track_id": out["track_id"]},
        warnings=["window and fits were cleared: set_fit_window again"],
    )


async def _calibrate_scale(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    data = s.use_scale(a["method"])
    return ToolResult(True, f"scale by {a['method']}", evidence_ids=[data["scale_id"]],
                      data_excerpt={k: v for k, v in data.items() if k != "scale_id"})  # fmt: skip


async def _set_fit_window(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    win_id, warnings = s.set_window(a["mode"], a.get("t_start"), a.get("t_end"))
    start, end = s.windows[win_id]
    return ToolResult(
        True,
        f"window {win_id} set ({end - start + 1} points)",
        evidence_ids=[win_id],
        data_excerpt={"n_points": end - start + 1},
        warnings=warnings,
    )


async def _fit_model(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    fit_id, result = s.fit(a["window_id"], a["model"])
    return ToolResult(
        True,
        f"{a['model']} fit {fit_id}",
        evidence_ids=[fit_id],
        data_excerpt={
            "r2": round(result.r2, 6),
            "rms_px": round(result.rms_px, 3),
            "durbin_watson": round(result.durbin_watson, 3),
            "n": result.n,
        },  # fmt: skip
    )


async def _bootstrap_ci(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    default = "period" if s.pendulum else "g"
    out = s.bootstrap(a["fit_id"], a.get("quantity", default))
    excerpt = {"value": out["value"], "ci95": out["ci95"]}
    if "n_cycles" in out:
        excerpt["n_cycles"] = out["n_cycles"]
    return ToolResult(True, "bootstrap CI computed", evidence_ids=[out["ci_id"]],
                      data_excerpt=excerpt, warnings=out.get("warnings", []))  # fmt: skip


async def _derive(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    der_id = s.derive(a["quantity"], a["fit_id"], a.get("scale_id"), a.get("length_source"))
    return ToolResult(True, f"{a['quantity']} derived", evidence_ids=[der_id])


async def _compare_reference(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    cmp_id = s.compare_reference(a["derived_id"])
    rel = s.ledger.get(cmp_id).data["rel_diff_pct"]
    return ToolResult(True, "compared with reference", evidence_ids=[cmp_id],
                      data_excerpt={"rel_diff_pct": round(rel, 3)})  # fmt: skip


async def _compare_hypothesis(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    cmp_id = s.compare_hypothesis(a["derived_id"])
    outcome = s.ledger.get(cmp_id).data["outcome"]
    return ToolResult(True, f"hypothesis {outcome}", evidence_ids=[cmp_id],
                      data_excerpt={"outcome": outcome})  # fmt: skip


async def _run_gates(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    results = gates_mod.run_gates(s, REGISTRY[s.spec.experiment].gates)
    return ToolResult(
        True,
        " · ".join(f"{g.gate} {g.status}" for g in results),
        evidence_ids=[g.evidence_id for g in results if g.evidence_id],
        data_excerpt={
            g.gate: {
                "status": g.status,
                "hard": g.hard,
                "remedy": g.remedy,
                "diagnosis": g.diagnosis,
            }
            for g in results
        },  # fmt: skip
        warnings=[g.message_en for g in results if g.status in ("fail", "warn")],
    )


async def _make_chart(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    fig_id = s.chart(a["kind"], a["fit_id"])
    return ToolResult(True, f"{a['kind']} chart", evidence_ids=[fig_id])


async def _add_claim(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    """The claim's numbers are copied from the evidence; the statements may hold only fields."""
    try:
        source = s.ledger.get(a["evidence_id"])
    except KeyError:
        return ToolResult(False, f"unknown evidence_id {a['evidence_id']!r}")
    if a["claim_type"] == "hypothesis_test" and source.tool != "hypothesis_test":
        return ToolResult(False, "a hypothesis_test claim must cite compare_hypothesis evidence")
    value, ci95, unit = copy_numbers(source)
    probe = Claim(
        id="c00", claim_type=a["claim_type"], quantity=a.get("quantity"), value=value,
        ci95=ci95, unit=unit, evidence=[source.id], statement_en=a["statement_en"],
        statement_id=a["statement_id"],
    )  # fmt: skip
    try:
        preview = render_statement(probe, source, s.spec.language)
        claim = s.ledger.add_claim(
            a["claim_type"], a["evidence_id"], a["statement_en"], a["statement_id"],
            quantity=a.get("quantity"),
        )  # fmt: skip
    except (ValueError, KeyError) as exc:
        return ToolResult(False, f"add_claim rejected: {exc}")
    s.ctx.emit("claim_added", claim_id=claim.id, claim_type=claim.claim_type,
               evidence=claim.evidence)  # fmt: skip
    return ToolResult(True, f"{claim.id} added", evidence_ids=[claim.id],
                      data_excerpt={"preview": preview[:200]})  # fmt: skip


async def _submit_report(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    try:
        draft = ReportDraft.model_validate(a)
    except ValidationError as exc:
        problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()][:5]
        return ToolResult(False, "report is not valid: " + "; ".join(problems))
    gate = gates_mod.gate_g5(draft, s.ledger, hypothesis=gates_mod.hypothesis_required(s.spec))
    ev = s.ledger.add_evidence("gate", "submit_report", gate.model_dump(mode="json"),
                               f"G5 {gate.status}")  # fmt: skip
    s.ctx.emit("gate_result", gate="G5", status=gate.status, evidence_id=ev.id)
    if gate.status != "pass":
        return ToolResult(False, "report rejected by G5: " + gate.metrics.get("violations", ""),
                          evidence_ids=[ev.id], warnings=[gate.message_en])  # fmt: skip
    write_json_atomic(s.ctx.run_dir / "draft_report.json", draft)
    return ToolResult(True, "report accepted", evidence_ids=[ev.id], terminal=True, payload=draft)


async def _request_rerecord(s: AnalysisSession, a: dict[str, Any]) -> ToolResult:
    request = RerecordRequest(
        reasons=list(a.get("reasons", [])),
        tips_en=list(a.get("tips_en", [])),
        tips_id=list(a.get("tips_id", [])),
    )
    return ToolResult(True, "re-recording requested", terminal=True, payload=request)


# --------------------------------------------------------------------------------------------
# Schemas and registry (06 §4 table)
# --------------------------------------------------------------------------------------------

_REPORT_SECTION = _obj(
    {"heading": {"type": "string"}, "body": {"type": "string"}}, ["heading", "body"]
)
_REPORT_PROPERTIES = {
    "lang": {"type": "string", "enum": ["id", "en"]},
    "title": {"type": "string"},
    "aim": {"type": "string"},
    "setup": {"type": "string"},
    "method": {"type": "string"},
    "results": {"type": "array", "items": _REPORT_SECTION},
    "hypothesis_evaluation": {"type": "string", "nullable": True},
    "error_analysis": {"type": "string"},
    "conclusion": {"type": "string"},
    "claims_used": {"type": "array", "items": {"type": "string"}},
}

TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "get_run_summary",
            "Spec, track, calibration, latest evidence and claims.",
            _obj({}),
            _get_run_summary,
        ),  # fmt: skip
        ToolSpec(
            "choose_track",
            "Make another candidate object the measured track.",
            _obj({"track_id": {"type": "string"}, "reason": _REASON}, ["track_id", "reason"]),
            _choose_track,
        ),  # fmt: skip
        ToolSpec(
            "resegment",
            "New SAM request on the analysis clip (max 2 per run). Clears windows and fits.",
            _obj({"concept": {"type": "string"}, "reason": _REASON}, ["reason"]),
            _resegment,
        ),  # fmt: skip
        ToolSpec(
            "calibrate_scale",
            "Compute the scale by one method and use it.",
            _obj(
                {
                    "method": {
                        "type": "string",
                        "enum": ["ruler", "object_diameter", "spoken_length"],
                    }
                },
                ["method"],
            ),
            _calibrate_scale,
        ),  # fmt: skip
        ToolSpec(
            "set_fit_window",
            "auto: free fall = release-to-impact window by event detection; pendulum = the "
            "whole swing with hand contact trimmed. manual: t_start and t_end in seconds.",
            _obj(
                {
                    "mode": {"type": "string", "enum": ["auto", "manual"]},
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "reason": _REASON,
                },
                ["mode", "reason"],
            ),
            _set_fit_window,
        ),  # fmt: skip
        ToolSpec(
            "fit_model",
            "Least-squares fit in pixels inside a window.",
            _obj(
                {
                    "window_id": _WINDOW_ID,
                    "model": {"type": "string", "enum": ["quadratic", "cubic", "damped_sine"]},
                },
                ["window_id", "model"],
            ),
            _fit_model,
        ),  # fmt: skip
        ToolSpec(
            "bootstrap_ci",
            "Block bootstrap and error propagation: g (free fall, quadratic fit) or the "
            "period (pendulum, damped_sine fit). Needs a fit from this round.",
            _obj(
                {"fit_id": _FIT_ID, "quantity": {"type": "string", "enum": ["g", "period"]}},
                ["fit_id"],
            ),  # fmt: skip
            _bootstrap_ci,
        ),  # fmt: skip
        ToolSpec(
            "derive",
            "A physical quantity from the bootstrapped fit. Free fall: g and v0 need scale_id; "
            "fall_time (release to impact, no scale needed). Pendulum: period (measured T), "
            "period0 (amplitude-corrected), g (needs length_source: spoken or measured).",
            _obj(
                {
                    "quantity": {
                        "type": "string",
                        "enum": ["g", "v0", "fall_time", "period", "period0"],
                    },
                    "fit_id": _FIT_ID,
                    "scale_id": _SCALE_ID,
                    "length_source": {"type": "string", "enum": ["spoken", "measured"]},
                },
                ["quantity", "fit_id"],
            ),  # fmt: skip
            _derive,
        ),  # fmt: skip
        ToolSpec(
            "compare_reference",
            "Compare a derived g with the reference value.",
            _obj({"derived_id": _DER_ID}, ["derived_id"]),
            _compare_reference,
        ),  # fmt: skip
        ToolSpec(
            "compare_hypothesis",
            "Compare a derived value with the student's numeric prediction (same unit: a "
            "prediction in seconds needs fall_time or period, in m/s² needs g).",
            _obj({"derived_id": _DER_ID}, ["derived_id"]),
            _compare_hypothesis,
        ),  # fmt: skip
        ToolSpec("run_gates", "Run G1–G4 on the current state.", _obj({}), _run_gates),
        ToolSpec(
            "make_chart",
            "Chart data for the report (trajectory or residuals).",
            _obj(
                {
                    "kind": {"type": "string", "enum": ["trajectory", "residuals"]},
                    "fit_id": _FIT_ID,
                },
                ["kind", "fit_id"],
            ),
            _make_chart,
        ),  # fmt: skip
        ToolSpec(
            "add_claim",
            "Create a claim. Value, CI and unit are copied from the evidence. Fields: {value}, "
            "{ci}, {unit} come from derive, bootstrap_ci and compare evidence; {rel_diff} from "
            "compare_reference; {outcome} from compare_hypothesis. Evidence from fit_model, "
            "set_fit_window, run_gates or calibrate_scale has none: write a plain sentence. No "
            "digits in statements.",
            _obj(
                {
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "measurement",
                            "derived",
                            "comparison",
                            "hypothesis_test",
                            "quality",
                            "interpretation",
                        ],
                    },
                    "quantity": {"type": "string"},
                    "evidence_id": {"type": "string"},
                    "statement_en": {"type": "string"},
                    "statement_id": {"type": "string"},
                },
                ["claim_type", "evidence_id", "statement_en", "statement_id"],
            ),
            _add_claim,
        ),  # fmt: skip
        ToolSpec(
            "submit_report",
            "Submit the ReportDraft. Checked by G5; rejected drafts come back with violations.",
            _obj(
                _REPORT_PROPERTIES,
                [
                    "lang",
                    "title",
                    "aim",
                    "setup",
                    "method",
                    "results",
                    "error_analysis",
                    "conclusion",
                    "claims_used",
                ],
            ),
            _submit_report,
        ),  # fmt: skip
        ToolSpec(
            "request_rerecord",
            "Stop: the data cannot support the measurement. Give tips.",
            _obj(
                {
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "tips_en": {"type": "array", "items": {"type": "string"}},
                    "tips_id": {"type": "array", "items": {"type": "string"}},
                },
                ["reasons", "tips_en", "tips_id"],
            ),
            _request_rerecord,
        ),  # fmt: skip
    ]
}
TOOL_NAMES = list(TOOLS)


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function-calling schemas for all tools (02 §6.1)."""
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description,
                                          "parameters": t.parameters}}
        for t in TOOLS.values()
    ]  # fmt: skip


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Required keys, JSON types and enums of the top-level arguments. Unknown keys are ignored."""
    errors: list[str] = []
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"missing {key}")
    for key, value in args.items():
        prop = props.get(key)
        if prop is None:
            continue
        kind = prop.get("type")
        if value is None and prop.get("nullable"):
            continue
        if kind in _TYPES:
            wrong_bool = isinstance(value, bool) and kind in ("number", "integer")
            if wrong_bool or not isinstance(value, _TYPES[kind]):
                errors.append(f"{key} must be {kind}")
        if "enum" in prop and value not in prop["enum"]:
            errors.append(f"{key} must be one of {prop['enum']}")
    return errors


def _brief(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False, default=str)[:_EVENT_ARGS_CHARS]


async def dispatch(session: AnalysisSession, name: str, args: dict[str, Any]) -> ToolResult:
    """Validate, run, emit ``tool_call``/``tool_result``. Errors come back as ``ok: false``."""
    emit = session.ctx.emit
    emit("tool_call", tool=name, args=_brief(args))
    spec = TOOLS.get(name)
    if spec is None:
        result = ToolResult(False, f"unknown tool {name!r}")
    elif errors := validate_args(spec.parameters, args):
        result = ToolResult(False, "invalid arguments: " + "; ".join(errors))
    else:
        try:
            result = await spec.handler(session, args)
        except ToolError as exc:
            result = ToolResult(False, str(exc))
        except (KeyError, ValueError) as exc:
            result = ToolResult(False, f"failed: {exc}")
    emit("tool_result", tool=name, ok=result.ok, summary=result.summary[:200],
         evidence_ids=result.evidence_ids[:10])  # fmt: skip
    return result
