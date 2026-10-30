"""Analyst agent: tool-calling loop on Chat Completions (Muse Spark 1.3).

Spec: docs/plan/06-agents-and-verification.md §3 (loop, context, fallback), §9.2 (prompt)

The model chooses the tools; code computes every number (tools.py, session.py) and checks the
report (G5). If the step budget runs out, or the model API fails twice, the deterministic analysis
and a template report are used instead (§3.4).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from framelab.agent.gates import gate_g5
from framelab.agent.session import AnalysisSession
from framelab.agent.tools import RerecordRequest, ToolResult, dispatch, tool_schemas
from framelab.clients import meta as meta_client
from framelab.clients.meta import MetaClientError
from framelab.config import MODEL_SPARK
from framelab.physics.experiments import REGISTRY
from framelab.pipeline import analysis_auto
from framelab.schemas import ReportDraft
from framelab.store import write_json_atomic

PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst.md"
MAX_API_FAILURES = 2  # 06 §3.4: the model API fails twice → fallback


@dataclass
class AnalystOutcome:
    kind: Literal["draft", "rerecord"]
    draft: ReportDraft | None = None
    rerecord: RerecordRequest | None = None
    fallback: bool = False
    steps: int = 0


def load_prompt() -> str:
    """The system prompt without its version header comment."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)


def context_message(s: AnalysisSession, fix_instructions: list[str] | None = None) -> str:
    """One user message: spec, experiment definition, track, scale, pre-computed evidence (§3.2)."""
    definition = REGISTRY[s.spec.experiment]
    evidence = [f"{e.id}: {e.summary}" for e in s.ledger.evidence()]
    context = {
        "lang": s.spec.language,
        "spec": s.spec.model_dump(mode="json", exclude={"target_hint_box"}),
        "experiment": {
            "name": definition.name,
            "primary_model": definition.primary_model,
            "headline_quantity": definition.headline_quantity,
            "plausible_range_ms2": list(definition.plausible_range),
            "min_valid_points": definition.min_valid_points,
        },
        "gate_thresholds": definition.gates.model_dump(),
        "track": s.track_summary(),
        "calibration": s.calibration_summary(),
        "evidence": evidence,
        "required_sections": ["aim", "setup", "method", "results", "error_analysis", "conclusion"]
        + (["hypothesis_evaluation"] if s.spec.hypothesis else []),
        "round": s.round,
        # round > 0: the Auditor's fixes, and the claims that already exist (06 §3.2)
        "fix_instructions": fix_instructions or [],
        "existing_claims": [f"{c.id}: {c.statement_en}" for c in s.ledger.claims()],
    }
    return json.dumps(context, ensure_ascii=False, default=str)


def _assistant_message(msg: Any) -> dict[str, Any]:
    """The assistant turn exactly as the API needs it replayed before its tool results (02 §6.1)."""
    calls = [
        {"id": tc.id, "type": "function",
         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in (msg.tool_calls or [])
    ]  # fmt: skip
    return {"role": "assistant", "content": msg.content or "", "tool_calls": calls}


async def run(s: AnalysisSession, *, fix_instructions: list[str] | None = None) -> AnalystOutcome:
    """The Analyst loop: at most ``analyst_max_steps`` model turns (06 §3.1)."""
    settings = s.ctx.settings
    emit = s.ctx.emit
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": load_prompt()},
        {"role": "user", "content": context_message(s, fix_instructions)},
    ]
    failures = 0
    step = 0
    while step < settings.analyst_max_steps:
        step += 1
        left = settings.analyst_max_steps - step + 1
        if left == 3:  # budget signal: the model must finish, not explore
            messages.append({"role": "user", "content":
                             "Three steps are left. Create the remaining claims in one turn with "
                             "parallel calls, then call submit_report."})  # fmt: skip
        try:
            resp = await meta_client.chat(
                model=MODEL_SPARK,
                messages=messages,
                tools=tool_schemas(),
                reasoning_effort="high",
                purpose="analyst",
                emit=emit,
            )
        except MetaClientError as exc:
            failures += 1
            emit(
                "warning",
                message=f"Analyst model call failed ({failures}/{MAX_API_FAILURES}): {exc}",
            )
            if failures >= MAX_API_FAILURES:
                return fallback(s, steps=step)
            step -= 1  # the same step is tried again
            continue
        msg = resp.choices[0].message
        if getattr(msg, "content", None) and msg.tool_calls:
            emit("progress", message=f"analyst: {msg.content[:200]}")
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            nudge = "You must act through tools. Call submit_report or request_rerecord to finish."
            messages.append({"role": "user", "content": nudge})
            continue
        messages.append(_assistant_message(msg))
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                result = ToolResult(False, "arguments are not valid JSON")
            else:
                result = await dispatch(s, call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.to_json()})
            if result.terminal and isinstance(result.payload, ReportDraft):
                return AnalystOutcome("draft", draft=result.payload, steps=step)
            if result.terminal and isinstance(result.payload, RerecordRequest):
                write_json_atomic(
                    s.ctx.run_dir / "analysis" / "rerecord.json", asdict(result.payload)
                )
                return AnalystOutcome("rerecord", rerecord=result.payload, steps=step)
    return fallback(s, steps=step)


# --------------------------------------------------------------------------------------------
# Deterministic fallback (06 §3.4)
# --------------------------------------------------------------------------------------------

_TEXT = {
    "free_fall": {
        "en": {
            "title": "Free-fall measurement",
            "aim": "Measure the acceleration of a falling object from its video.",
            "setup": "A falling object was filmed with a phone camera and tracked frame by frame.",
            "method": (
                "The fit window runs from release to the first impact. A quadratic fit of the "
                "vertical position gives the acceleration. The scale comes from a reference in "
                "the plane of motion, and the uncertainty comes from a bootstrap."
            ),
            "error": (
                "The uncertainty combines the statistical spread, the spread from the choice of "
                "window and the scale uncertainty. This report is a template: the agent could not "
                "finish the analysis, so only the deterministic numbers are shown."
            ),
            "conclusion": "The measured acceleration and its uncertainty are given above.",
        },
        "id": {
            "title": "Pengukuran jatuh bebas",
            "aim": "Mengukur percepatan benda jatuh dari videonya.",
            "setup": "Benda yang jatuh direkam dengan kamera ponsel dan dilacak frame demi frame.",
            "method": (
                "Jendela fit dimulai dari saat lepas hingga benturan pertama. Fit kuadratik pada "
                "posisi vertikal memberi percepatan. Skala berasal dari referensi pada bidang "
                "gerak, dan ketidakpastian dihitung dengan bootstrap."
            ),
            "error": (
                "Ketidakpastian menggabungkan sebaran statistik, sebaran akibat pilihan jendela, "
                "dan ketidakpastian skala. Laporan ini berbentuk templat: agen tidak menyelesaikan "
                "analisis, sehingga hanya angka hasil deterministik yang ditampilkan."
            ),
            "conclusion": "Percepatan terukur dan ketidakpastiannya ada pada hasil di atas.",
        },
    },
    "pendulum": {
        "en": {
            "title": "Pendulum measurement",
            "aim": "Measure the period of a pendulum, and g when the string length is known.",
            "setup": (
                "A swinging pendulum was filmed with a phone camera and tracked frame by frame."
            ),
            "method": (
                "A damped sine fit of the horizontal position gives the period. The period is "
                "corrected for the swing amplitude, and g follows from the string length. The "
                "uncertainty comes from a bootstrap and the length uncertainty."
            ),
            "error": (
                "The uncertainty combines the statistical spread of the period, the amplitude "
                "correction and the length uncertainty. This report is a template: the agent could "
                "not finish the analysis, so only the deterministic numbers are shown."
            ),
            "conclusion": "The measured result and its uncertainty are given above.",
        },
        "id": {
            "title": "Pengukuran bandul",
            "aim": "Mengukur periode bandul, dan g bila panjang tali diketahui.",
            "setup": (
                "Bandul yang berayun direkam dengan kamera ponsel dan dilacak frame demi frame."
            ),
            "method": (
                "Fit sinus teredam pada posisi horizontal memberi periode. Periode dikoreksi untuk "
                "amplitudo ayunan, dan g dihitung dari panjang tali. Ketidakpastian berasal dari "
                "bootstrap dan ketidakpastian panjang."
            ),
            "error": (
                "Ketidakpastian menggabungkan sebaran statistik periode, koreksi amplitudo, dan "
                "ketidakpastian panjang. Laporan ini berbentuk templat: agen tidak menyelesaikan "
                "analisis, sehingga hanya angka hasil deterministik yang ditampilkan."
            ),
            "conclusion": "Hasil terukur dan ketidakpastiannya ada pada hasil di atas.",
        },
    },
}
_RESULT = {"en": "Result", "id": "Hasil"}
_HYP_TEXT = {"en": "Hypothesis test: {{%s}}.", "id": "Uji hipotesis: {{%s}}."}
_SYMBOL = {"g": "g", "period": "T"}
# Claim statements: the only place where the text may use the template fields.
_COMPARISON = {
    "en": "Compared with the reference value, the difference is {rel_diff}.",
    "id": "Dibandingkan nilai referensi, selisihnya {rel_diff}.",
}
_HYPOTHESIS = {
    "en": "The student's prediction is {outcome} by the measurement.",
    "id": "Prediksi siswa {outcome} oleh pengukuran.",
}


def fallback(s: AnalysisSession, *, steps: int) -> AnalystOutcome:
    """The deterministic analysis and a template report. Only claim placeholders, so G5 passes."""
    s.ctx.emit("warning", message="analyst_fallback: deterministic analysis and template report")
    headline = analysis_auto.run(s.ctx, s.spec, s.ing, s.seg, s.observations, s.summary)
    if headline is None:
        raise analysis_auto.AnalysisError("the deterministic analysis produced no headline")
    lang = s.spec.language
    text = _TEXT[s.spec.experiment][lang]
    ledger = s.ledger
    quantity = headline["quantity"]
    statement = f"{_SYMBOL.get(quantity, quantity)} = {{value}}"
    claims = [ledger.add_claim("derived", headline["evidence"], statement, statement,
                               quantity=quantity)]  # fmt: skip
    if headline.get("comparison_evidence"):
        claims.append(
            ledger.add_claim(
                "comparison",
                headline["comparison_evidence"],
                _COMPARISON["en"],
                _COMPARISON["id"],
                quantity="g",
            )
        )
    hyp = None
    if headline.get("hypothesis_evidence"):
        hyp = ledger.add_claim("hypothesis_test", headline["hypothesis_evidence"],
                               _HYPOTHESIS["en"], _HYPOTHESIS["id"])  # fmt: skip
        claims.append(hyp)
    body_claims = [c for c in claims if c is not hyp]
    draft = ReportDraft(
        lang=lang,
        title=text["title"],
        aim=text["aim"],
        setup=text["setup"],
        method=text["method"],
        results=[
            {"heading": _RESULT[lang], "body": " ".join(f"{{{{{c.id}}}}}" for c in body_claims)}
        ],  # fmt: skip
        hypothesis_evaluation=_HYP_TEXT[lang] % hyp.id if hyp is not None else None,
        error_analysis=text["error"],
        conclusion=text["conclusion"],
        claims_used=[c.id for c in claims],
    )
    gate = gate_g5(draft, ledger, hypothesis=hyp is not None)
    if gate.status != "pass":  # the templates hold no numbers, so this would be a bug
        raise RuntimeError(f"fallback report failed G5: {gate.message_en}")
    write_json_atomic(s.ctx.run_dir / "draft_report.json", draft)
    s.ctx.state.warnings.append("analyst_fallback")
    return AnalystOutcome("draft", draft=draft, fallback=True, steps=steps)
