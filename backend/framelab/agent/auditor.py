"""Auditor: independent review of the draft (Muse Spark; Wonder behind FRAMELAB_USE_WONDER).

Spec: docs/plan/06-agents-and-verification.md §5 (pre-checks, model input, post-checks), §8 (final
verdict), §9.3 (prompt); docs/plan/02-meta-api-reference.md §7 (Wonder decision).

The Auditor sees the draft (claims shown inline), the ledger, the gate results and the spec. It
never sees the Analyst's conversation: its messages are built from scratch here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from framelab.agent import narrative
from framelab.agent.gates import gate_g5, hypothesis_required
from framelab.clients import meta as meta_client
from framelab.clients.meta import MetaClientError, strict_schema
from framelab.config import MODEL_SPARK, MODEL_WONDER
from framelab.schemas import ExperimentSpec, GateResult, ReportDraft, Verdict
from framelab.store import write_json_atomic

if TYPE_CHECKING:
    from framelab.agent.ledger import Ledger
    from framelab.agent.session import AnalysisSession
    from framelab.agent.tools import RerecordRequest

PROMPT_PATH = Path(__file__).parent / "prompts" / "auditor.md"
MAX_ATTEMPTS = 2  # the model answer is parsed once; one retry with the errors (02 §7)
PASSING = ("PASS", "PASS_WITH_WARNINGS")


class AuditorOutput(BaseModel):
    """What the model returns (strict json_schema). Verdict minus the fields code fills in."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["PASS", "PASS_WITH_WARNINGS", "REJECT", "NEEDS_RERECORD"]
    unsupported_claims: list[str]
    overreaching_statements: list[str]
    gate_failures: list[str]
    fix_instructions: list[str]
    rerecord_tips_en: list[str]
    rerecord_tips_id: list[str]
    notes: str = Field(default="")


class AuditorUnavailable(RuntimeError):
    """The model did not return a valid verdict after the retry and the fallback."""


@dataclass
class PreCheck:
    """Deterministic checks run before the model sees the draft (06 §5)."""

    g5: GateResult
    hard_failures: list[GateResult] = field(default_factory=list)
    soft_warnings: list[GateResult] = field(default_factory=list)


def pre_check(draft: ReportDraft, ledger: Ledger, gates: list[GateResult],
              spec: ExperimentSpec) -> PreCheck:  # fmt: skip
    g5 = gate_g5(draft, ledger, hypothesis=hypothesis_required(spec))
    hard = [g for g in gates if g.hard and g.status == "fail"]
    if g5.status != "pass":
        hard.append(g5)
    soft = [g for g in gates if g.status == "warn"]
    return PreCheck(g5=g5, hard_failures=hard, soft_warnings=soft)


# --------------------------------------------------------------------------------------------
# Model input (06 §5): one user message built from draft, ledger, gates and spec
# --------------------------------------------------------------------------------------------


def _inline_claim(claim_id: str, ledger: Ledger, lang: str) -> str:
    claim = ledger.get_claim(claim_id)
    text = narrative.render_statement(claim, ledger.get(claim.evidence[0]), lang)
    return f"[{claim_id}: {text}]"


def draft_for_auditor(draft: ReportDraft, ledger: Ledger) -> str:
    """The draft with every placeholder replaced by ``[cNN: formatted claim]``."""
    lines = [f"Title: {draft.title}", f"Aim: {draft.aim}", f"Setup: {draft.setup}",
             f"Method: {draft.method}"]  # fmt: skip
    for section in draft.results:
        lines.append(f"Result section '{section.heading}': {section.body}")
    if draft.hypothesis_evaluation:
        lines.append(f"Hypothesis evaluation: {draft.hypothesis_evaluation}")
    lines += [f"Error analysis: {draft.error_analysis}", f"Conclusion: {draft.conclusion}"]
    text = "\n".join(lines)
    known = {c.id for c in ledger.claims()}
    mapping = {cid: _inline_claim(cid, ledger, draft.lang) for cid in known}
    return re.sub(
        r"\{\{(c\d{2,})\}\}",
        lambda m: mapping.get(m.group(1), f"[unknown {m.group(1)}]"),
        text,
    )  # fmt: skip


def _claims_block(ledger: Ledger, lang: str) -> str:
    rows = []
    for claim in ledger.claims():
        rows.append(
            f"{claim.id} type={claim.claim_type} quantity={claim.quantity} value={claim.value} "
            f"ci95={claim.ci95} unit={claim.unit} evidence={claim.evidence} "
            f"text={narrative.render_statement(claim, ledger.get(claim.evidence[0]), lang)}"
        )  # fmt: skip
    return "\n".join(rows) or "(no claims)"


def _evidence_block(ledger: Ledger) -> str:
    rows = []
    for ev in ledger.evidence():
        if ev.id.startswith(("gate_", "fig_", "spk_")):
            continue  # gates are listed separately; figures and spoken values are not claims
        rows.append(f"{ev.id} ({ev.tool}): {ev.summary}")
    return "\n".join(rows[-40:]) or "(no evidence)"


def _gates_block(gates: list[GateResult], pre: PreCheck) -> str:
    rows = []
    for g in gates:
        hard = "hard" if g.hard else "soft"
        rows.append(f"{g.gate} {g.name}: {g.status} ({hard}, remedy {g.remedy}, "
                    f"diagnosis {g.diagnosis}) — {g.message_en}")  # fmt: skip
    rows.append(f"G5 {pre.g5.name}: {pre.g5.status} — {pre.g5.message_en}")
    return "\n".join(rows)


def build_message(draft: ReportDraft, session: AnalysisSession, gates: list[GateResult],
                  pre: PreCheck) -> str:  # fmt: skip
    spec = session.spec
    spec_view = {
        "experiment": spec.experiment,
        "hypothesis": spec.hypothesis,
        "hypothesis_prediction": spec.hypothesis_prediction.model_dump() if spec.hypothesis_prediction else None,  # noqa: E501
        "spoken_params": {k: v.model_dump() for k, v in spec.spoken_params.items()},
        "setup_description": spec.setup_description,
        "planner_warnings": spec.warnings,
    }  # fmt: skip
    hard = ", ".join(g.gate for g in pre.hard_failures) or "none"
    return "\n\n".join([
        "REPORT DRAFT (claims shown as [cNN: value]):\n" + draft_for_auditor(draft, session.ledger),
        "CLAIMS:\n" + _claims_block(session.ledger, draft.lang),
        "EVIDENCE SUMMARIES:\n" + _evidence_block(session.ledger),
        "GATES:\n" + _gates_block(gates, pre),
        f"PRE-CHECK: hard failures: {hard}. Soft warnings: {len(pre.soft_warnings)}.",
        "SPEC:\n" + json.dumps(spec_view, ensure_ascii=False, default=str),
    ])  # fmt: skip


def load_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)


# --------------------------------------------------------------------------------------------
# Model call with one retry and a Spark fallback when Wonder is enabled (02 §7)
# --------------------------------------------------------------------------------------------


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())[:500]


async def _ask(model: str, message: str, session: AnalysisSession) -> AuditorOutput:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": load_prompt()},
        {"role": "user", "content": message},
    ]
    text_format = {"type": "json_schema", "json_schema": {
        "name": "verdict", "schema": strict_schema(AuditorOutput), "strict": True}}  # fmt: skip
    for attempt in range(MAX_ATTEMPTS):
        resp = await meta_client.chat(
            model=model,
            messages=messages,
            response_format=text_format,
            reasoning_effort="high",
            purpose="auditor",
            emit=session.ctx.emit,
        )
        content = resp.choices[0].message.content or ""
        try:
            return AuditorOutput.model_validate_json(content)
        except ValidationError as exc:
            if attempt + 1 == MAX_ATTEMPTS:
                raise AuditorUnavailable(f"invalid verdict from {model}: {_errors(exc)}") from exc
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Your verdict is invalid: {_errors(exc)}. "
                 "Return the complete Verdict JSON again."},
            ]  # fmt: skip
    raise AssertionError("unreachable")


async def ask(message: str, session: AnalysisSession) -> tuple[AuditorOutput, str]:
    """The verdict from Wonder (if enabled, then Spark on failure) or from Spark."""
    models = [MODEL_WONDER, MODEL_SPARK] if session.ctx.settings.use_wonder else [MODEL_SPARK]
    errors: list[str] = []
    for model in models:
        try:
            return await _ask(model, message, session), model
        except (AuditorUnavailable, MetaClientError) as exc:
            errors.append(str(exc))
            session.ctx.emit("warning", message=f"auditor {model} failed: {exc}")
    raise AuditorUnavailable("; ".join(errors))


# --------------------------------------------------------------------------------------------
# Post-checks (06 §5): code overrides the model when a hard gate failed
# --------------------------------------------------------------------------------------------


def _gate_messages(gates: list[GateResult]) -> tuple[list[str], list[str]]:
    return [g.message_en for g in gates], [g.message_id for g in gates]


def post_check(
    out: AuditorOutput,
    pre: PreCheck,
    *,
    rounds_left: bool,
    model: str,
) -> Verdict:
    """Apply the 06 §5 rules to the model's verdict. Every override is written to ``overrides``."""
    status = out.status
    fixes = list(out.fix_instructions)
    tips_en, tips_id = list(out.rerecord_tips_en), list(out.rerecord_tips_id)
    overrides: list[str] = []
    hard_ids = [g.gate for g in pre.hard_failures]

    if pre.hard_failures and status in ("PASS", "PASS_WITH_WARNINGS", "REJECT"):
        rerecord = any(g.remedy == "rerecord" for g in pre.hard_failures)
        # Only the text failed (G5) and no round is left: stay REJECT, so the orchestrator
        # replaces the text with the deterministic report instead of asking for a new video.
        text_only = all(g.gate == "G5" for g in pre.hard_failures)
        target = "REJECT" if not rerecord and (rounds_left or text_only) else "NEEDS_RERECORD"
        if target != status:
            overrides.append(f"hard gates {hard_ids} failed: {status} → {target}")
        status = target
    elif status == "PASS" and pre.soft_warnings:
        overrides.append("soft warnings present: PASS → PASS_WITH_WARNINGS")
        status = "PASS_WITH_WARNINGS"

    if status == "REJECT" and not fixes:
        fixes = [f"Revise the report to fix {g.gate}: {g.message_en}" for g in pre.hard_failures]
        fixes = fixes or ["Revise the flagged statements so each one cites a claim."]
        overrides.append("REJECT without fix instructions: generic instruction added")
    if status == "NEEDS_RERECORD":
        # the gate messages are always there, so the diagnosis never depends on the model's wording
        msgs_en, msgs_id = _gate_messages(pre.hard_failures)
        tips_en = tips_en + [m for m in msgs_en if m not in tips_en]
        tips_id = tips_id + [m for m in msgs_id if m not in tips_id]
        tips_en = tips_en or ["Record the video again."]
        tips_id = tips_id or ["Rekam ulang video."]

    return Verdict(
        status=status,
        unsupported_claims=out.unsupported_claims,
        overreaching_statements=out.overreaching_statements,
        gate_failures=hard_ids or out.gate_failures,
        fix_instructions=fixes,
        rerecord_tips_en=tips_en,
        rerecord_tips_id=tips_id,
        auditor_model=model,
        notes=out.notes,
        overrides=overrides,
    )


def unavailable_output(pre: PreCheck) -> AuditorOutput:
    """No model verdict: deterministic checks only, flagged as a warning (never a silent PASS)."""
    return AuditorOutput(
        status="PASS_WITH_WARNINGS",
        unsupported_claims=[], overreaching_statements=[], gate_failures=[],
        fix_instructions=[], rerecord_tips_en=[], rerecord_tips_id=[],
        notes="auditor_unavailable: only the deterministic checks ran",
    )  # fmt: skip


async def run(session: AnalysisSession, draft: ReportDraft, gates: list[GateResult],
              *, round_: int) -> Verdict:  # fmt: skip
    """Pre-checks, the model verdict, post-checks. Written to ``analysis/verdict_rNN.json``."""
    pre = pre_check(draft, session.ledger, gates, session.spec)
    message = build_message(draft, session, gates, pre)
    try:
        out, model = await ask(message, session)
    except AuditorUnavailable as exc:
        session.ctx.emit("warning", message=f"auditor_unavailable: {exc}")
        out, model = unavailable_output(pre), "none"
    rounds_left = round_ < session.ctx.settings.max_revise_rounds
    verdict = post_check(out, pre, rounds_left=rounds_left, model=model)
    write_json_atomic(session.ctx.run_dir / "analysis" / f"verdict_r{round_:02d}.json", verdict)
    session.ctx.emit("verdict", round=round_, status=verdict.status, model=model,
                     overrides=verdict.overrides)  # fmt: skip
    return verdict


def rerecord_message(request: RerecordRequest, session: AnalysisSession,
                     gates: list[GateResult]) -> str:  # fmt: skip
    """What the Auditor sees when the Analyst stopped with request_rerecord (no draft)."""
    spec = session.spec
    gate_rows = [
        f"{g.gate} {g.name}: {g.status} ({'hard' if g.hard else 'soft'}, remedy "
        f"{g.remedy}, diagnosis {g.diagnosis}) — {g.message_en}"
        for g in gates
    ]
    return "\n\n".join([
        "THE ANALYST REQUESTED A RE-RECORDING instead of writing a report. Decide whether the data "
        "really cannot support a measurement. Answer NEEDS_RERECORD if you agree. Answer REJECT "
        "with fix_instructions if the gates and evidence show the data is usable and the Analyst "
        "should finish the report.",
        "REASONS: " + ", ".join(request.reasons),
        "TIPS (EN): " + " | ".join(request.tips_en),
        "GATES:\n" + "\n".join(gate_rows),
        "EVIDENCE SUMMARIES:\n" + _evidence_block(session.ledger),
        "SPEC:\n" + json.dumps({"experiment": spec.experiment, "warnings": spec.warnings,
                                "setup_description": spec.setup_description},
                               ensure_ascii=False),  # fmt: skip
    ])  # fmt: skip


async def review_rerecord(session: AnalysisSession, request: RerecordRequest,
                          gates: list[GateResult], *, round_: int) -> Verdict:  # fmt: skip
    """06 §4: request_rerecord is terminal for the Analyst, but the Auditor still reviews it.

    A failed hard gate with remedy ``rerecord`` settles it in code. Otherwise the Auditor decides;
    if it disagrees, the verdict is REJECT so the Analyst finishes the report in the next round.
    """
    hard = [g for g in gates if g.hard and g.status == "fail"]
    msgs_en, msgs_id = _gate_messages(hard)
    tips_en, tips_id = list(request.tips_en) + msgs_en, list(request.tips_id) + msgs_id
    base = {"gate_failures": [g.gate for g in hard], "rerecord_tips_en": tips_en,
            "rerecord_tips_id": tips_id}  # fmt: skip
    if any(g.remedy == "rerecord" for g in hard):
        verdict = Verdict(status="NEEDS_RERECORD", auditor_model="code",
                          notes="Analyst requested re-recording; a hard gate with remedy rerecord "
                          "failed, so code confirms it.", **base)  # fmt: skip
    else:
        try:
            out, model = await ask(rerecord_message(request, session, gates), session)
        except AuditorUnavailable as exc:
            session.ctx.emit("warning", message=f"auditor_unavailable: {exc}")
            out, model = None, "none"
        if out is None or out.status == "NEEDS_RERECORD":
            verdict = Verdict(status="NEEDS_RERECORD", auditor_model=model,
                              notes=(out.notes if out else "auditor unavailable: the Analyst's "
                                     "request stands"), **base)  # fmt: skip
        else:
            fixes = list(out.fix_instructions) or [
                "The gates do not show a recording problem. Complete the analysis, add the "
                "claims and submit the report."
            ]
            verdict = Verdict(
                status="REJECT",
                auditor_model=model,
                fix_instructions=fixes,
                gate_failures=[g.gate for g in hard],
                notes=out.notes,
                overrides=[f"re-recording request not confirmed ({out.status})"],
            )
    write_json_atomic(session.ctx.run_dir / "analysis" / f"verdict_r{round_:02d}.json", verdict)
    session.ctx.emit("verdict", round=round_, status=verdict.status, model=verdict.auditor_model,
                     overrides=verdict.overrides)  # fmt: skip
    return verdict


def finalize_verdict(
    auditor: Verdict, gates: list[GateResult], *, fallback: bool = False
) -> tuple[str, list[str]]:
    """Final status per 06 §8 (never REJECT at DONE) and extra warnings.

    A report from the deterministic fallback is at most PASS_WITH_WARNINGS (06 §3.4).
    """
    warnings: list[str] = []
    hard_failed = [g for g in gates if g.hard and g.status == "fail"]
    if hard_failed:
        return "NEEDS_RERECORD", warnings
    if auditor.status == "REJECT":
        warnings.append("auditor_unresolved")
        return "PASS_WITH_WARNINGS", warnings
    if fallback and auditor.status == "PASS":
        warnings.append("analyst_fallback: verdict capped at PASS_WITH_WARNINGS")
        return "PASS_WITH_WARNINGS", warnings
    return auditor.status, warnings
