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


def render_statement(claim: Claim, evidence: Evidence, lang: str) -> str:
    """The claim sentence in ``lang``, with every template field filled from the ledger values."""
    template = claim.statement_id if lang == "id" else claim.statement_en
    unit = claim.unit or ""
    sigma = sigma_from_ci(*claim.ci95) if claim.ci95 else None
    fields: dict[str, str] = {"unit": unit_display(unit)}
    if claim.value is not None:
        fields["value"] = fmt(claim.value, sigma, unit, lang)
    if claim.ci95 is not None and sigma:
        fields["ci"] = fmt_interval(claim.ci95[0], claim.ci95[1], unit, sigma, lang)
    if "rel_diff_pct" in evidence.data:
        fields["rel_diff"] = fmt_pct(float(evidence.data["rel_diff_pct"]), lang, 2)
    if "outcome" in evidence.data:
        fields["outcome"] = OUTCOME_LABELS[lang].get(str(evidence.data["outcome"]), "")

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
