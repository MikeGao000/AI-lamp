"""Environment-backed application configuration with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(path: str | Path = ".env") -> None:
    """Load a minimal KEY=VALUE file without overriding explicit environment values."""
    file_path = Path(path)
    if not file_path.is_file():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip("'\""))


@dataclass(frozen=True)
class AppConfig:
    cloud_enabled: bool
    api_key: str | None
    api_base_url: str
    model: str
    still_seconds: float
    motion_threshold: float
    tts_voice: str
    tts_provider: str
    tts_model: str
    openai_tts_voice: str
    tts_instructions: str
    tts_timeout_s: float
    reading_language: str
    cloud_stream: bool
    cloud_reasoning_effort: str
    cloud_max_output_tokens: int

    @classmethod
    def from_environment(cls) -> "AppConfig":
        return cls(
            cloud_enabled=_bool(os.getenv("ENABLE_CLOUD_VISION", "false")),
            api_key=os.getenv("OPENAI_API_KEY") or None,
            api_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("VLM_MODEL", "gpt-5.6-luna"),
            still_seconds=float(os.getenv("CAMERA_STILL_SECONDS", "1.5")),
            motion_threshold=float(os.getenv("CAMERA_MOTION_THRESHOLD", "8.0")),
            tts_voice=os.getenv("TTS_VOICE", "da"),
            tts_provider=os.getenv("TTS_PROVIDER", "local").strip().lower(),
            tts_model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
            openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "marin"),
            tts_instructions=os.getenv(
                "TTS_INSTRUCTIONS",
                "Speak warmly and naturally to a young child. Match the language of the text. "
                "Use a gentle picture-book storytelling pace.",
            ),
            tts_timeout_s=float(os.getenv("TTS_TIMEOUT_SECONDS", "30")),
            reading_language=os.getenv("READING_LANGUAGE", "Danish"),
            cloud_stream=_bool(os.getenv("CLOUD_STREAM", "true")),
            cloud_reasoning_effort=os.getenv("CLOUD_REASONING_EFFORT", "none"),
            cloud_max_output_tokens=int(os.getenv("CLOUD_MAX_OUTPUT_TOKENS", "700")),
        )
