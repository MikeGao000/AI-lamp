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
    ESTOP = "estop"
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


INTENT_PHRASES: tuple[tuple[VoiceIntent, tuple[str, ...]], ...] = (
    (VoiceIntent.ESTOP, ("急停", "emergency stop")),
    (VoiceIntent.STOP, ("停止", "停下", "别动", "stop")),
    (VoiceIntent.TEACH_START, ("开始示教", "开始录制", "录动作")),
    (VoiceIntent.TEACH_STOP, ("结束示教", "停止录制", "保存动作")),
    (VoiceIntent.REPLAY, ("回放动作", "播放动作", "重播")),
    (VoiceIntent.READ, ("开始读", "继续读", "讲故事", "读书")),
    (VoiceIntent.REST, ("休息", "回待机", "睡觉")),
    (VoiceIntent.WAKE, ("小灯", "台灯", "hello lamp")),
)


def _normalized(text: str) -> str:
    return "".join(text.lower().strip().split())


def _phrase_matches(clean: str) -> list[tuple[VoiceIntent, int, int, int]]:
    matches: list[tuple[VoiceIntent, int, int, int]] = []
    for priority, (intent, phrases) in enumerate(INTENT_PHRASES):
        for phrase in phrases:
            normalized_phrase = _normalized(phrase)
            start = 0
            while True:
                start = clean.find(normalized_phrase, start)
                if start < 0:
                    break
                end = start + len(normalized_phrase)
                matches.append((intent, start, end, priority))
                start += 1
    return matches


def parse_intent(transcript: str) -> VoiceCommand:
    """Map a recognized phrase to an allow-listed command; unknown speech does nothing."""
    clean = _normalized(transcript)
    if not clean:
        return VoiceCommand("", VoiceIntent.NONE)
    matches = _phrase_matches(clean)
    if not matches:
        return VoiceCommand(transcript.strip(), VoiceIntent.NONE)

    # An explicit emergency phrase always wins, regardless of other words.
    if any(intent is VoiceIntent.ESTOP for intent, _, _, _ in matches):
        return VoiceCommand(transcript.strip(), VoiceIntent.ESTOP)

    # A normal stop also wins unless it occurs only inside a more specific
    # command such as "停止录制". This preserves interruption without making
    # that compound teaching command unreachable.
    for intent, start, end, _ in matches:
        if intent is not VoiceIntent.STOP:
            continue
        contained = any(
            other_intent is not VoiceIntent.STOP
            and other_start <= start
            and end <= other_end
            and other_end - other_start > end - start
            for other_intent, other_start, other_end, _ in matches
        )
        if not contained:
            return VoiceCommand(transcript.strip(), VoiceIntent.STOP)

    best = min(matches, key=lambda item: (-(item[2] - item[1]), item[3]))
    return VoiceCommand(transcript.strip(), best[0])


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
