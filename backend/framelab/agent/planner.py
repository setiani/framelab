"""Planner: video + transcript → ExperimentSpec (Muse Spark 1.3, Responses API).

Spec: docs/plan/06-agents-and-verification.md §2 and §9.1, docs/plan/02-meta-api-reference.md §6.3.

Wire format: Spark returns ``PlannerOutput``, not ``ExperimentSpec``. ``strict_schema()`` closes
every object, so a ``dict`` field would allow no keys at all, and spoken params become a list
with a ``key`` instead (docs/plan/11 §3). ``to_spec()`` converts the answer before any stage
reads it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

from framelab.agent.ledger import Ledger
from framelab.clients import meta as meta_client
from framelab.clients.meta import Emit, strict_schema
from framelab.config import MODEL_SPARK
from framelab.schemas import (
    ExperimentSpec,
    ExperimentType,
    Lang,
    ProbeInfo,
    Quantity,
    ReferenceObject,
    SpokenParamKey,
    Transcript,
)

PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.md"
ALLOWED_KEYS: frozenset[str] = frozenset(get_args(SpokenParamKey))
MIN_CONFIDENCE = 0.4  # 06 §2: below this the run stops with NEEDS_RERECORD
MIN_WINDOW_S = 0.5  # 06 §2
MAX_CONCEPT_WORDS = 4  # 06 §2
MAX_ATTEMPTS = 2  # 06 §2: one retry with the validation errors appended
CONCEPT_WORD_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


class PlannerError(RuntimeError):
    """The Planner answer was still invalid after the retry and cannot be repaired by dropping."""


class SpokenParam(BaseModel):
    """One value read from the transcript. ``source_turn`` must name the turn that says it."""

    model_config = ConfigDict(extra="forbid")

    key: SpokenParamKey
    value: float
    unit: str
    source_turn: int


class PlannerOutput(BaseModel):
    """What Spark returns (strict json_schema). ExperimentSpec minus the pipeline-only fields."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentType
    target_concept: str
    target_alt_concepts: list[str]
    reference: ReferenceObject | None
    object_diameter_m: float | None
    spoken_params: list[SpokenParam]
    analysis_window_start_s: float
    analysis_window_end_s: float
    motion_axis: Literal["x", "y", "xy"]
    hypothesis: str | None
    hypothesis_prediction: Quantity | None
    setup_description: str
    confidence: float
    warnings: list[str]


# --------------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------------


def _norm_concept(text: str) -> str:
    return text.strip().lower()


def to_spec(out: PlannerOutput, lang: Lang) -> ExperimentSpec:
    """Build the ExperimentSpec. Concepts are lowercased. A repeated spoken key keeps its first
    value and adds a warning."""
    warnings = list(out.warnings)
    spoken: dict[str, Quantity] = {}
    for param in out.spoken_params:
        if param.key in spoken:
            warnings.append(f"Planner gave spoken_params.{param.key} twice; kept the first")
            continue
        spoken[param.key] = Quantity(
            value=param.value, unit=param.unit, source_turn=param.source_turn
        )
    reference = out.reference
    if reference is not None:
        reference = reference.model_copy(update={"concept": _norm_concept(reference.concept)})
    return ExperimentSpec(
        experiment=out.experiment,
        target_concept=_norm_concept(out.target_concept),
        target_alt_concepts=[_norm_concept(c) for c in out.target_alt_concepts],
        reference=reference,
        object_diameter_m=out.object_diameter_m,
        spoken_params=spoken,
        analysis_window_s=(out.analysis_window_start_s, out.analysis_window_end_s),
        motion_axis=out.motion_axis,
        hypothesis=out.hypothesis,
        hypothesis_prediction=out.hypothesis_prediction,
        language=lang,
        setup_description=out.setup_description,
        confidence=out.confidence,
        warnings=warnings,
    )


def apply_overrides(spec: ExperimentSpec, overrides: dict[str, Any]) -> ExperimentSpec:
    """Typed user flags win over the Planner. ``spoken_params`` merge key by key."""
    if not overrides:
        return spec
    update = dict(overrides)
    if "spoken_params" in update:
        update["spoken_params"] = {**spec.spoken_params, **update["spoken_params"]}
    return spec.model_copy(update=update)


def needs_rerecord(spec: ExperimentSpec) -> bool:
    """06 §2: an unknown experiment or low confidence stops the run before measurement."""
    return spec.experiment == "unknown" or spec.confidence < MIN_CONFIDENCE


# --------------------------------------------------------------------------------------------
# Validation (06 §2)
# --------------------------------------------------------------------------------------------


_DIGIT_WORDS = {
    "nol": 0, "zero": 0, "satu": 1, "one": 1, "dua": 2, "two": 2, "tiga": 3, "three": 3,
    "empat": 4, "four": 4, "lima": 5, "five": 5, "enam": 6, "six": 6, "tujuh": 7, "seven": 7,
    "delapan": 8, "eight": 8, "sembilan": 9, "nine": 9,
}  # fmt: skip
_TEEN_WORDS = {
    "sepuluh": 10, "ten": 10, "sebelas": 11, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
_TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
_DECIMAL_SEPARATORS = frozenset({"koma", "point"})
_HALF_WORDS = frozenset({"setengah", "half"})


def _read_int(tokens: list[str], i: int) -> tuple[int | None, int]:
    """One whole number starting at ``tokens[i]``: (value, index after it), or (None, i)."""
    word = tokens[i]
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    if word in _DIGIT_WORDS:
        digit = _DIGIT_WORDS[word]
        if nxt == "puluh":  # dua puluh = 20
            return digit * 10, i + 2
        if nxt == "ratus":  # dua ratus = 200
            return digit * 100, i + 2
        if nxt == "belas":  # dua belas = 12
            return 10 + digit, i + 2
        return digit, i + 1
    if word in _TEEN_WORDS:
        return _TEEN_WORDS[word], i + 1
    if word in _TENS_WORDS:  # twenty five = 25
        if nxt in _DIGIT_WORDS and _DIGIT_WORDS[nxt]:
            return _TENS_WORDS[word] + _DIGIT_WORDS[nxt], i + 2
        return _TENS_WORDS[word], i + 1
    if word in ("seratus", "hundred"):
        return 100, i + 1
    return None, i


def spoken_values(text: str) -> list[float]:
    """Numbers in a transcript, as digits ("0.45", "0,45") or as words.

    Words: "satu meter" → 1, "sepuluh sentimeter" → 10, "setengah meter" → 0.5, and decimals
    such as "nol koma empat lima" or "zero point four five" → 0.45. Other compound forms are not
    read, so a value written that way is treated as untraceable (and dropped).
    """
    values = [float(m.replace(",", ".")) for m in re.findall(r"\d+(?:[.,]\d+)?", text)]
    tokens = re.findall(r"[a-z]+", text.lower())
    i = 0
    while i < len(tokens):
        if tokens[i] in _HALF_WORDS:
            values.append(0.5)
            i += 1
            continue
        whole, after = _read_int(tokens, i)
        if whole is None:
            i += 1
            continue
        if after < len(tokens) and tokens[after] in _DECIMAL_SEPARATORS:
            digits = ""
            j = after + 1
            while j < len(tokens) and tokens[j] in _DIGIT_WORDS:
                digits += str(_DIGIT_WORDS[tokens[j]])
                j += 1
            if digits:
                values.append(float(f"{whole}.{digits}"))
                i = j
                continue
        values.append(float(whole))
        i = after
    return values


def value_in_text(value: float, text: str) -> bool:
    """True if the number appears in the text, as digits or words (see ``spoken_values``)."""
    return any(
        math.isclose(candidate, value, rel_tol=1e-6, abs_tol=1e-9)
        for candidate in spoken_values(text)
    )


def concept_errors(field: str, concept: str) -> list[str]:
    """A concept is 1–4 plain lowercase words, e.g. "orange ball" or "a4 paper"."""
    words = concept.split()
    if not 1 <= len(words) <= MAX_CONCEPT_WORDS:
        return [f"{field}: {concept!r} must be 1–{MAX_CONCEPT_WORDS} words"]
    if not all(CONCEPT_WORD_RE.fullmatch(word) for word in words):
        return [f"{field}: {concept!r} must be lowercase words without punctuation"]
    return []


def quantity_errors(field: str, quantity: Quantity, transcript: Transcript) -> str | None:
    """Why a spoken value is unusable, or None. Checks units, the source turn and the number."""
    try:
        quantity.to_si()
    except ValueError:
        return f"{field}: unknown unit {quantity.unit!r}"
    turns = {turn.index: turn.text for turn in transcript.turns}
    if quantity.source_turn is None or quantity.source_turn not in turns:
        return f"{field}: source_turn {quantity.source_turn} is not a transcript turn"
    text = turns[quantity.source_turn]
    if not value_in_text(quantity.value, text):
        return f"{field}: {quantity.value} is not in turn {quantity.source_turn} ({text!r})"
    return None


def spoken_errors(spec: ExperimentSpec, transcript: Transcript) -> dict[str, str]:
    """Untraceable values, keyed by spoken key (``hypothesis_prediction`` included).

    Used by ``validate_spec`` and by the drop step after the retry.
    """
    bad: dict[str, str] = {}
    for key, quantity in spec.spoken_params.items():
        if key not in ALLOWED_KEYS:
            bad[key] = f"spoken_params.{key}: unknown key"
            continue
        message = quantity_errors(f"spoken_params.{key}", quantity, transcript)
        if message:
            bad[key] = message
    prediction = spec.hypothesis_prediction
    if prediction is not None:
        message = quantity_errors("hypothesis_prediction", prediction, transcript)
        if message:
            bad["hypothesis_prediction"] = message
    return bad


def validate_spec(spec: ExperimentSpec, transcript: Transcript, duration_s: float) -> list[str]:
    """Return validation errors (06 §2). An empty list means the spec may be used.

    A window longer than ``max_analysis_s`` is not an error here: the run clips it and warns.
    """
    errors = concept_errors("target_concept", spec.target_concept)
    for i, concept in enumerate(spec.target_alt_concepts):
        errors += concept_errors(f"target_alt_concepts[{i}]", concept)
    if spec.reference is not None:
        errors += concept_errors("reference.concept", spec.reference.concept)

    start, end = spec.analysis_window_s
    if start < 0 or end > duration_s + 1e-6:
        errors.append(
            f"analysis_window_s {start:.2f}–{end:.2f} must lie inside [0, {duration_s:.2f}] s"
        )
    elif end - start < MIN_WINDOW_S:
        errors.append(f"analysis_window_s is {end - start:.2f} s long, minimum is {MIN_WINDOW_S} s")

    errors += spoken_errors(spec, transcript).values()
    return errors


def drop_unverified(spec: ExperimentSpec, bad: dict[str, str]) -> tuple[ExperimentSpec, list[str]]:
    """Remove the values that cannot be traced to the transcript; return the spec and warnings."""
    spoken = {k: v for k, v in spec.spoken_params.items() if k not in bad}
    update: dict[str, Any] = {"spoken_params": spoken}
    if "hypothesis_prediction" in bad:
        update["hypothesis_prediction"] = None
    warnings = [f"Dropped a value the transcript does not support: {msg}" for msg in bad.values()]
    update["warnings"] = [*spec.warnings, *warnings]
    return spec.model_copy(update=update), warnings


# --------------------------------------------------------------------------------------------
# Call
# --------------------------------------------------------------------------------------------


def load_prompt() -> str:
    """The system prompt without its version header comment."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)


def _messages(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()]


def check_output(
    text: str, lang: Lang, transcript: Transcript, duration_s: float
) -> tuple[ExperimentSpec | None, list[str]]:
    """Parse Spark's JSON and validate it. Returns (spec or None, errors)."""
    try:
        out = PlannerOutput.model_validate_json(text)
    except ValidationError as exc:
        return None, _messages(exc)
    try:
        spec = to_spec(out, lang)
    except ValidationError as exc:
        return None, _messages(exc)
    return spec, validate_spec(spec, transcript, duration_s)


def retry_message(errors: list[str]) -> str:
    lines = "\n".join(f"- {error}" for error in errors)
    return (
        "Your previous JSON failed validation:\n"
        f"{lines}\n"
        "Return the complete corrected JSON object only. Keep every valid value."
    )


def record_spoken_evidence(spec: ExperimentSpec, transcript: Transcript, ledger: Ledger) -> None:
    """One ``spk_`` evidence record per spoken value, with the turn text it came from."""
    turns = {turn.index: turn.text for turn in transcript.turns}
    items: list[tuple[str, Quantity]] = list(spec.spoken_params.items())
    if spec.hypothesis_prediction is not None:
        items.append(("hypothesis_prediction", spec.hypothesis_prediction))
    for key, quantity in items:
        turn_text = turns.get(quantity.source_turn, "") if quantity.source_turn is not None else ""
        ledger.add_evidence(
            "spk",
            tool="planner",
            data={
                "key": key,
                "value": quantity.value,
                "unit": quantity.unit,
                "source_turn": quantity.source_turn,
                "turn_text": turn_text,
            },
            summary=(
                f"{key} = {quantity.value:g} {quantity.unit} (spoken, turn {quantity.source_turn})"
            ),
        )


async def run(
    file_id: str,
    transcript: Transcript,
    probe: ProbeInfo,
    lang: Lang,
    note: str | None,
    *,
    emit: Emit | None = None,
    ledger: Ledger | None = None,
) -> ExperimentSpec:
    """Call Spark with the video and the transcript, validate, retry once with the errors.

    If the retry is still invalid only because of spoken values, those values are dropped with a
    warning. Any other remaining problem raises ``PlannerError``.
    """
    user = {
        "requested_language": lang,
        "probe": {
            "duration_s": probe.duration_s,
            "fps": probe.avg_fps,
            "width": probe.width,
            "height": probe.height,
        },
        "transcript": transcript.model_dump(mode="json"),
        "user_note": note,
    }
    first_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": json.dumps(user, ensure_ascii=False)},
                {"type": "input_file", "file_id": file_id},
            ],
        }
    ]
    text_format = {
        "type": "json_schema",
        "name": "experiment_spec",
        "schema": strict_schema(PlannerOutput),
        "strict": True,
    }
    system = load_prompt()
    duration = probe.duration_s

    spec: ExperimentSpec | None = None
    errors: list[str] = []
    text = ""
    for attempt in range(MAX_ATTEMPTS):
        turn_input = first_input
        if attempt:
            if emit:
                emit("progress", message="Planner retry: " + "; ".join(errors)[:300])
            turn_input = [
                *first_input,
                {"role": "assistant", "content": text},
                {"role": "user", "content": retry_message(errors)},
            ]
        resp = await meta_client.responses(
            model=MODEL_SPARK,
            purpose="planner",
            emit=emit,
            reasoning={"effort": "medium"},
            instructions=system,
            input=turn_input,
            text={"format": text_format},
        )
        text = resp.output_text or ""
        spec, errors = check_output(text, lang, transcript, duration)
        if not errors:
            break

    if errors:
        if spec is None:
            raise PlannerError("Planner output invalid: " + "; ".join(errors))
        bad = spoken_errors(spec, transcript)
        spec, dropped = drop_unverified(spec, bad)
        if emit:
            for warning in dropped:
                emit("warning", message=warning)
        remaining = validate_spec(spec, transcript, duration)
        if remaining:
            raise PlannerError("Planner output invalid: " + "; ".join(remaining))

    if ledger is not None:
        record_spoken_evidence(spec, transcript, ledger)
    return spec
