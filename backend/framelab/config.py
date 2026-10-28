"""Application settings and model identifiers.

Spec: docs/plan/01-architecture.md §8, docs/plan/03-data-contracts.md §10,
docs/plan/02-meta-api-reference.md §2.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Meta Model API -----------------------------------------------------------------------
META_BASE_URL = "https://api.meta.ai/v1"
META_TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"

MODEL_SPARK = "muse-spark-1.3"
MODEL_WONDER = "muse-wonder-1.0"
MODEL_SAM = "sam-3.1"
MODEL_VOICE = "muse-voice-transcribe-1.0"
MODEL_IMAGE = "muse-image-1.0"
# Hackathon-specific slug from the participant guide (NOT the docs' meta-models/muse-spark-open)
MODEL_SPARK_OPEN_BASETEN = "meta-models/muse-spark-open-1026"

# Reference gravity for comparison only (never used to alter data). Jakarta, see 05 §1.
G_REF_DEFAULT = 9.781


class Settings(BaseSettings):
    """Runtime settings. Read from environment variables prefixed ``FRAMELAB_`` and ``.env``."""

    model_config = SettingsConfigDict(env_prefix="FRAMELAB_", env_file=".env", extra="ignore")

    runs_dir: Path = Path("runs")
    demo: bool = False
    demo_allow_upload: bool = False
    max_upload_mb: int = 200
    max_analysis_s: float = 20.0
    max_long_edge_px: int = 960
    concurrency: int = 2
    use_wonder: bool = False  # M1: Wonder unreliable on structured tasks (docs/plan/02 §7)
    mask_dilation_px: float = 1.5  # M1: SAM masks ~1 px larger per side (docs/plan/04 §6.2)
    designer_backend: Literal["meta", "baseten"] = "meta"
    analyst_max_steps: int = 12
    max_revise_rounds: int = 2
    max_extra_sam_requests: int = 2
    bootstrap_n: int = 1000
    run_cost_guard_usd: float = 1.50
    daily_budget_usd: float = 10.0
    log_level: str = "INFO"
    live_tests: bool = False
    record_dir: Path | None = None

    @property
    def model_api_key(self) -> str | None:
        """Meta Model API key (env ``MODEL_API_KEY``, no prefix)."""
        return os.environ.get("MODEL_API_KEY") or _dotenv_value("MODEL_API_KEY")

    @property
    def baseten_api_key(self) -> str | None:
        """Baseten API key (env ``BASETEN_API_KEY``, no prefix). Stretch goal S-3 only."""
        return os.environ.get("BASETEN_API_KEY") or _dotenv_value("BASETEN_API_KEY")


def _dotenv_value(name: str, path: Path = Path(".env")) -> str | None:
    """Read an unprefixed variable from ``.env`` (pydantic-settings only reads prefixed ones)."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (tests construct ``Settings`` directly instead)."""
    return Settings()


def mask_secret(value: str | None) -> str:
    """Return a log-safe representation of a secret (never log full keys)."""
    if not value:
        return "<missing>"
    return f"{value[:6]}…{value[-4:]}" if len(value) > 12 else "<set>"
