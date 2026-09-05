"""Offline speech-recognition adapter and safe command intent parsing.

Vosk is imported only when a real recognizer is constructed, so desktop
simulation and safety tests need no microphone, model, or external package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class VoiceIntent(str, Enum):
    NONE = "none"
    WAKE = "wake"
    STOP = "stop"
    READ = "read"
    TEACH_START = "teach_start"
    TEACH_STOP = "teach_stop"
    REPLAY = "replay"
    REST = "rest"


@dataclass(frozen=True)
class VoiceCommand:
    transcript: str
    intent: VoiceIntent


def parse_intent(transcript: str) -> VoiceCommand:
    """Map a recognized phrase to an allow-listed command; unknown speech does nothing."""
    clean = "".join(transcript.lower().strip().split())
    if not clean:
        return VoiceCommand("", VoiceIntent.NONE)
    mappings = (
        (VoiceIntent.STOP, ("急停", "停止", "停下", "别动", "stop")),
        (VoiceIntent.TEACH_START, ("开始示教", "开始录制", "录动作")),
        (VoiceIntent.TEACH_STOP, ("结束示教", "停止录制", "保存动作")),
        (VoiceIntent.REPLAY, ("回放动作", "播放动作", "重播")),
        (VoiceIntent.READ, ("开始读", "继续读", "讲故事", "读书")),
        (VoiceIntent.REST, ("休息", "回待机", "睡觉")),
        (VoiceIntent.WAKE, ("小灯", "台灯", "hello lamp")),
    )
    for intent, phrases in mappings:
        if any(phrase in clean for phrase in phrases):
            return VoiceCommand(transcript.strip(), intent)
    return VoiceCommand(transcript.strip(), VoiceIntent.NONE)


class VoskStreamingRecognizer:
    """Converts 16 kHz / 16-bit / mono PCM microphone chunks into commands."""

    def __init__(self, model_path: str | Path, sample_rate_hz: int = 16_000) -> None:
        try:
            from vosk import KaldiRecognizer, Model  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("install Vosk on the Raspberry Pi: python3 -m pip install vosk") from error
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Vosk model directory not found: {path}")
        self._recognizer = KaldiRecognizer(Model(str(path)), sample_rate_hz)

    def feed_pcm16_mono(self, pcm_data: bytes) -> VoiceCommand | None:
        """Return only finalized phrases; partial hypotheses never move the robot."""
        if not self._recognizer.AcceptWaveform(pcm_data):
            return None
        result = json.loads(self._recognizer.Result())
        return parse_intent(str(result.get("text", "")))
