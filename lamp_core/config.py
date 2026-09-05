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
        )
