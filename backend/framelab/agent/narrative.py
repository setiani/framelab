"""Placeholder rendering ({{cNN}}), claim statement templates, and the G5 bare-number check.

Spec: docs/plan/06-agents-and-verification.md §1 (rule 3), §4 (add_claim rules), §6 (rendering),
§7 (G5 bare-number rule). The model writes text with placeholders only; every digit is produced
by code from the ledger.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from framelab.physics.uncertainty import fmt, fmt_interval, fmt_pct, sigma_from_ci, unit_display

if TYPE_CHECKING:
    from framelab.schemas import Claim, Evidence

PLACEHOLDER_RE = re.compile(r"\{\{(c\d{2,}|p\d+)\}\}")
TEMPLATE_RE = re.compile(r"\{([a-z_]+)\}")
TEMPLATE_FIELDS = frozenset({"value", "ci", "unit", "outcome", "rel_diff"})

# Digits that are allowed in narrative text (06 §7). Matched and removed before the digit search.
_ALLOWED = (
    re.compile(r"\bG[1-5]\b"),
    re.compile(r"SAM 3\.1"),
    re.compile(r"Muse Spark 1\.3"),
    re.compile(r"muse-spark-1\.3"),
    re.compile(r"Muse Wonder 1\.0"),
    re.compile(r"95 ?%(?= (?:confidence|selang kepercayaan))"),
    re.compile(r"^\s*\d+\.(?=\s)", re.MULTILINE),  # ordinal heading at the start of a line
)
OUTCOME_LABELS = {
    "en": {"supported": "supported", "close": "close", "not_supported": "not supported"},
    "id": {"supported": "didukung", "close": "cukup dekat", "not_supported": "tidak didukung"},
}


def find_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def bare_numbers(text: str) -> list[str]:
    """Digits outside placeholders and the allow-list (06 §7 G5). Empty list = OK."""
    body = PLACEHOLDER_RE.sub(" ", text)
    for pattern in _ALLOWED:
        body = pattern.sub(" ", body)
    return re.findall(r"\d+", body)


def statement_errors(text: str) -> list[str]:
    """Why a claim statement template is not allowed (06 §4). Empty list = OK."""
    errors: list[str] = []
    if PLACEHOLDER_RE.search(text):
        errors.append("statements cannot contain {{cNN}} placeholders")
    unknown = sorted(set(TEMPLATE_RE.findall(text)) - TEMPLATE_FIELDS)
    if unknown:
        errors.append(f"unknown template field(s) {unknown}; use {sorted(TEMPLATE_FIELDS)}")
    numbers = bare_numbers(TEMPLATE_RE.sub(" ", text))
    if numbers:
        errors.append(f"bare number(s) {numbers} in statement: use {{value}} or {{ci}} instead")
    return errors


def render(text: str, claims: dict[str, str]) -> str:
    """Replace ``{{cNN}}`` placeholders with rendered claim text; an unknown id raises KeyError."""

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in claims:
            raise KeyError(f"unknown placeholder {{{{{key}}}}}")
        return claims[key]

    return PLACEHOLDER_RE.sub(substitute, text)


def _fields(claim: Claim, evidence: Evidence, lang: str, bare: bool) -> dict[str, str]:
    """Template fields for one claim. ``bare`` leaves the unit out of value and ci (the template
    places {unit} itself, so the unit would otherwise appear twice)."""
    unit = claim.unit or ""
    sigma = sigma_from_ci(*claim.ci95) if claim.ci95 else None
    value_unit = "" if bare else unit
    fields: dict[str, str] = {"unit": unit_display(unit)}
    if claim.value is not None:
        fields["value"] = fmt(claim.value, sigma, value_unit, lang)
    if claim.ci95 is not None and sigma:
        fields["ci"] = fmt_interval(claim.ci95[0], claim.ci95[1], value_unit, sigma, lang).strip()
    if "rel_diff_pct" in evidence.data:
        fields["rel_diff"] = fmt_pct(float(evidence.data["rel_diff_pct"]), lang, 2)
    if "outcome" in evidence.data:
        fields["outcome"] = OUTCOME_LABELS[lang].get(str(evidence.data["outcome"]), "")
    return fields


def _fill(template: str, fields: dict[str, str], claim: Claim, evidence: Evidence) -> str:
    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in fields:
            have = ", ".join(f"{{{name}}}" for name in sorted(fields)) or "no template fields"
            raise ValueError(
                f"statement needs {{{key}}}, but evidence {evidence.id} provides {have}; "
                "use only those fields, or plain text"
            )
        return fields[key]

    return TEMPLATE_RE.sub(substitute, template)


def render_statement(claim: Claim, evidence: Evidence, lang: str) -> str:
    """The claim sentence in ``lang`` (appendix A), with every template field filled."""
    template = claim.statement_id if lang == "id" else claim.statement_en
    fields = _fields(claim, evidence, lang, bare="{unit}" in template)
    return _fill(template, fields, claim, evidence)


def phrase(claim: Claim, evidence: Evidence, lang: str) -> str:
    """What ``{{cNN}}`` shows inside a sentence of the report (06 §6): the value with its unit
    ("g = 9.82 ± 0.03 m/s²"), a relative difference, or the outcome of a hypothesis test."""
    fields = _fields(claim, evidence, lang, bare=False)
    if claim.claim_type == "comparison" and "rel_diff" in fields:
        return (f"a difference of {fields['rel_diff']} from the reference value" if lang == "en"
                else f"selisih {fields['rel_diff']} dari nilai referensi")  # fmt: skip
    if claim.claim_type == "hypothesis_test" and "outcome" in fields:
        return fields["outcome"]
    if claim.value is not None:
        label = f"{claim.quantity} = " if claim.quantity else ""
        return label + fields["value"]
    template = claim.statement_id if lang == "id" else claim.statement_en
    return _fill(template, fields, claim, evidence)
