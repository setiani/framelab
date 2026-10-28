"""Spike: structured output support (VERIFY V6, V7).

Variants
  S1  Spark,  Chat Completions, response_format json_schema          (baseline, cookbook-proven)
  S2  Spark,  Chat Completions, parse(response_format=PydanticModel)
  S3  Spark,  Responses API,    text.format json_schema strict=false   (V6)
  S4  Spark,  Responses API,    text.format json_schema strict=true    (V6)
  S5  Wonder, Chat Completions, response_format json_schema            (V7)
  S6  Wonder, Responses API,    text.format json_schema                (V7 fallback)

Usage
  uv run python scripts/spikes/spike_structured_output.py
Approx. cost: < $0.02
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from _common import (
    Report,
    async_client,
    describe_error,
    estimate_token_cost,
    fail_safe_main,
    sync_client,
    usage_dict,
)
from pydantic import BaseModel, Field, ValidationError

SPARK, WONDER = "muse-spark-1.3", "muse-wonder-1.0"


class MiniSpec(BaseModel):
    experiment: Literal["free_fall", "pendulum", "melting", "unknown"]
    target_concept: str = Field(description="1-4 word noun phrase for a segmentation model")
    spoken_length_cm: float | None
    confidence: float = Field(ge=0, le=1)


SCHEMA = MiniSpec.model_json_schema()
PROMPT = (
    "A student says: 'Panjang tali lima puluh senti, saya kira periodenya satu setengah detik.' "
    "The video shows a metal nut swinging on a string. Fill the schema."
)


def chat_json_schema(model: str):
    return sync_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "mini_spec", "schema": SCHEMA},
        },
    )


def chat_parse(model: str):
    client = sync_client()
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    return parse(
        model=model, messages=[{"role": "user", "content": PROMPT}], response_format=MiniSpec
    )


async def responses_json_schema(model: str, strict: bool):
    async with async_client() as api:
        return await api.responses.create(
            model=model,
            input=PROMPT,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "mini_spec",
                    "schema": SCHEMA,
                    "strict": strict,
                }
            },
        )


def validate(text: str) -> tuple[bool, str]:
    try:
        MiniSpec.model_validate_json(text)
        return True, ""
    except ValidationError as exc:
        return False, str(exc)[:300]
    except json.JSONDecodeError as exc:
        return False, f"not JSON: {exc}"


def main() -> None:
    report = Report("spike_structured_output")
    variants = {
        "S1_spark_chat_json_schema": (SPARK, lambda: chat_json_schema(SPARK), "chat"),
        "S2_spark_chat_parse": (SPARK, lambda: chat_parse(SPARK), "chat"),
        "S3_spark_responses_strict_false": (
            SPARK, lambda: asyncio.run(responses_json_schema(SPARK, False)), "responses"),
        "S4_spark_responses_strict_true": (
            SPARK, lambda: asyncio.run(responses_json_schema(SPARK, True)), "responses"),
        "S5_wonder_chat_json_schema": (WONDER, lambda: chat_json_schema(WONDER), "chat"),
        "S6_wonder_responses": (
            WONDER, lambda: asyncio.run(responses_json_schema(WONDER, False)), "responses"),
    }  # fmt: skip
    out = {}
    for name, (model, call, api) in variants.items():
        t0 = time.perf_counter()
        try:
            resp = call()
            text = resp.output_text if api == "responses" else resp.choices[0].message.content
            usage = usage_dict(resp.usage)
            report.cost(estimate_token_cost(model, usage))
            valid, err = validate(text or "")
            out[name] = {
                "ok": True,
                "valid_json_schema": valid,
                "validation_error": err,
                "latency_s": round(time.perf_counter() - t0, 2),
                "text": (text or "")[:600],
                "usage": usage,
            }
            report.note(f"{name}: ✓ valid={valid} {(text or '')[:160]!r}")
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": describe_error(exc)}
            report.note(f"{name}: ✗ {describe_error(exc)}")
    report.add("variants", out)
    report.save()


if __name__ == "__main__":
    fail_safe_main(main)
