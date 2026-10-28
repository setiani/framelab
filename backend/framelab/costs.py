"""Price table and per-call cost estimation (USD).

Spec: docs/plan/02-meta-api-reference.md §3 (prices as of 28 Oct 2026, Standard mode).
"""

from __future__ import annotations

from dataclasses import dataclass

from framelab.config import MODEL_IMAGE, MODEL_SAM, MODEL_SPARK, MODEL_VOICE, MODEL_WONDER


@dataclass(frozen=True)
class TokenPrice:
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


TOKEN_PRICES: dict[str, TokenPrice] = {
    MODEL_SPARK: TokenPrice(1.25, 0.15, 4.25),
    MODEL_WONDER: TokenPrice(4.00, 0.20, 20.00),
}

SAM_IMAGE_PER_1000 = 2.50
SAM_VIDEO_PER_1000_FRAMES = 0.20
SAM_VIDEO_MIN_FRAMES = 50
VOICE_PER_HOUR = 0.18
IMAGE_PER_IMAGE = 0.01


def token_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Cost of a text-model call. Unknown models (e.g. Baseten) return 0.0."""
    price = TOKEN_PRICES.get(model)
    if price is None:
        return 0.0
    uncached = max(input_tokens - cached_tokens, 0)
    return (
        uncached * price.input_per_m
        + cached_tokens * price.cached_input_per_m
        + output_tokens * price.output_per_m
    ) / 1_000_000


def sam_cost(*, frames: int = 0, images: int = 0) -> float:
    """SAM 3.1 cost. Video requests are billed for at least 50 frames."""
    video = 0.0
    if frames > 0:
        video = max(frames, SAM_VIDEO_MIN_FRAMES) * SAM_VIDEO_PER_1000_FRAMES / 1000
    return video + images * SAM_IMAGE_PER_1000 / 1000


def voice_cost(audio_seconds: float) -> float:
    """Muse Voice Transcribe, billed per whole second."""
    return int(round(audio_seconds)) * VOICE_PER_HOUR / 3600


def image_cost(n_images: int) -> float:
    return n_images * IMAGE_PER_IMAGE


def model_family(model: str) -> str:
    """Short label used in the UI agent trace."""
    return {
        MODEL_SPARK: "Muse Spark",
        MODEL_WONDER: "Muse Wonder",
        MODEL_SAM: "SAM 3.1",
        MODEL_VOICE: "Muse Voice",
        MODEL_IMAGE: "Muse Image",
    }.get(model, model)
