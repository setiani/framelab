"""Translator: the report draft in the second language (04 §9, 06 §9.5).

A second model call translates the draft JSON and keeps the placeholders. The result is used only
if the placeholder sets and the claim ids are identical, no new numbers appear, and G5 passes.
Otherwise the report stays in one language and the reason is recorded.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from framelab.agent.gates import draft_texts, gate_g5
from framelab.agent.ledger import Ledger
from framelab.agent.narrative import PLACEHOLDER_RE, bare_numbers
from framelab.clients import meta as meta_client
from framelab.clients.meta import Emit, MetaClientError, strict_schema
from framelab.config import MODEL_SPARK
from framelab.schemas import Lang, ReportDraft

PROMPT_PATH = Path(__file__).parent / "prompts" / "translator.md"


def load_prompt(src: Lang, dst: Lang) -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    text = re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.DOTALL)
    return text.replace("{src}", src).replace("{dst}", dst)  # str.replace: the text holds {{c03}}


def check_translation(src: ReportDraft, out: ReportDraft, ledger: Ledger) -> str:
    """Why the translation cannot be used, or '' if it can."""
    # counts, not sets: dropping one of two uses of {{c01}} is a change too
    src_ph = Counter(PLACEHOLDER_RE.findall("\n".join(draft_texts(src))))
    out_ph = Counter(PLACEHOLDER_RE.findall("\n".join(draft_texts(out))))
    if src_ph != out_ph:
        return f"placeholders differ: source {dict(src_ph)}, translation {dict(out_ph)}"
    if set(src.claims_used) != set(out.claims_used):
        return "claims_used differs from the source draft"
    numbers = [n for text in draft_texts(out) for n in bare_numbers(text)]
    if numbers:
        return f"translation adds numbers: {sorted(set(numbers))}"
    result = gate_g5(out, ledger, hypothesis=False)
    if result.status != "pass":
        return f"translation fails G5: {result.message_en}"
    return ""


async def translate(
    draft: ReportDraft, ledger: Ledger, *, dst: Lang, emit: Emit | None = None
) -> tuple[ReportDraft | None, str]:
    """(translated draft, '') on success; (None, reason) when the single-language report stays."""
    if dst == draft.lang:
        return None, "same language"
    messages = [
        {"role": "system", "content": load_prompt(draft.lang, dst)},
        {"role": "user", "content": json.dumps(draft.model_dump(mode="json"), ensure_ascii=False)},
    ]
    text_format = {"type": "json_schema", "json_schema": {
        "name": "report_draft", "schema": strict_schema(ReportDraft), "strict": True}}  # fmt: skip
    try:
        resp = await meta_client.chat(model=MODEL_SPARK, messages=messages,
                                      response_format=text_format, reasoning_effort="minimal",
                                      purpose="translator", emit=emit)  # fmt: skip
    except MetaClientError as exc:
        return None, f"translator call failed: {exc}"
    try:
        out = ReportDraft.model_validate_json(resp.choices[0].message.content or "")
    except ValueError as exc:
        return None, f"translation is not a valid draft: {str(exc)[:200]}"
    out = out.model_copy(update={"lang": dst})
    problem = check_translation(draft, out, ledger)
    return (None, problem) if problem else (out, "")
