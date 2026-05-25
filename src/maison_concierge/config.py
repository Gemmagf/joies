"""Centralised settings loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Locale = Literal["en", "fr"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-sonnet-4-6", alias="CLAUDE_MODEL")
    claude_max_tokens: int = Field(default=1024, alias="CLAUDE_MAX_TOKENS")

    chroma_dir: Path = Field(default=Path(".chroma"), alias="CHROMA_DIR")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    app_locale_default: Locale = Field(default="en", alias="APP_LOCALE_DEFAULT")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    app_escalation_threshold: float = Field(default=0.65, alias="APP_ESCALATION_THRESHOLD")

    clip_enabled: bool = Field(default=False, alias="CLIP_ENABLED")
    clip_model: str = Field(default="ViT-B-32", alias="CLIP_MODEL")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k", alias="CLIP_PRETRAINED")

    metrics_dir: Path = Field(default=Path("data/metrics"), alias="METRICS_DIR")

    # `auto` (default) → demo mode iff ANTHROPIC_API_KEY is missing or starts with "sk-ant-...".
    # `true` / `false` force the mode explicitly.
    demo_mode: str = Field(default="auto", alias="DEMO_MODE")

    @property
    def use_demo_mode(self) -> bool:
        if self.demo_mode.lower() == "true":
            return True
        if self.demo_mode.lower() == "false":
            return False
        key = (self.anthropic_api_key or "").strip()
        return not key or key in {"sk-ant-...", "missing-key", "test-key", "dummy-for-ci"}

    @property
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
