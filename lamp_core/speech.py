"""Speech adapters for local and OpenAI cloud voice output."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Protocol
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SpeechSink(Protocol):
    def speak(self, text: str) -> None: ...


class CloudSpeechError(RuntimeError):
    """A cloud TTS request failed without producing usable audio."""


@dataclass
class EspeakSpeech:
    voice: str = "da"

    def speak(self, text: str) -> None:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable is None:
            raise RuntimeError("install espeak-ng to enable local speech")
        subprocess.run([executable, "-v", self.voice, text], check=True)


@dataclass
class OpenAITtsSpeech:
    """OpenAI text-to-speech adapter that plays WAV audio on Raspberry Pi."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini-tts"
    voice: str = "marin"
    instructions: str | None = None
    timeout_s: float = 30.0

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise CloudSpeechError("cannot synthesize empty text")
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "wav",
            "stream_format": "audio",
        }
        if self.instructions:
            payload["instructions"] = self.instructions
        request = Request(
            self.base_url.rstrip("/") + "/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                audio = response.read()
        except HTTPError as error:
            raise CloudSpeechError(f"OpenAI TTS returned HTTP {error.code}") from error
        except TimeoutError as error:
            raise CloudSpeechError(f"OpenAI TTS timed out after {self.timeout_s:g}s") from error
        except URLError as error:
            raise CloudSpeechError("OpenAI TTS is unavailable") from error
        if not audio:
            raise CloudSpeechError("OpenAI TTS returned empty audio")
        return audio

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        player = shutil.which("aplay")
        if player is None:
            raise RuntimeError("install alsa-utils to play OpenAI TTS audio on Raspberry Pi")
        subprocess.run([player, "-q"], input=audio, check=True)


class QueuedSpeech:
    """Play speech in order on one worker, without blocking vision streaming."""

    _STOP = object()

    def __init__(self, sink: SpeechSink) -> None:
        self._sink = sink
        self._queue: Queue[str | object] = Queue()
        self._worker = threading.Thread(target=self._run, name="lamp-tts", daemon=True)
        self._worker.start()

    def speak(self, text: str) -> None:
        if text.strip():
            self._queue.put(text)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is self._STOP:
                    return
                self._sink.speak(str(text))
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._queue.put(self._STOP)
        self._queue.join()
        self._worker.join(timeout=1.0)


@dataclass
class WavFileSpeech:
    """The same local TTS engine as the Pi app, rendered to a file for testing."""

    output_path: Path
    voice: str = "da"
    _segments: list[str] | None = None

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        if self._segments is None:
            self._segments = []
        self._segments.append(text)

    def finalize(self) -> None:
        """Render queued segments as one WAV, preserving the real playback order."""

        if not self._segments:
            raise RuntimeError("no speech was queued for WAV output")
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable is None:
            raise RuntimeError("install espeak-ng to render a WAV file")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lamp-tts-", dir=self.output_path.parent) as directory:
            parts: list[Path] = []
            for index, text in enumerate(self._segments):
                part = Path(directory) / f"part-{index}.wav"
                subprocess.run([executable, "-v", self.voice, "-w", str(part), text], check=True)
                parts.append(part)
            with wave.open(str(parts[0]), "rb") as first:
                parameters = first.getparams()
            with wave.open(str(self.output_path), "wb") as combined:
                combined.setparams(parameters)
                for part in parts:
                    with wave.open(str(part), "rb") as source:
                        if source.getparams()[:4] != parameters[:4]:
                            raise RuntimeError("TTS segment format changed unexpectedly")
                        combined.writeframes(source.readframes(source.getnframes()))


@dataclass
class OpenAIWavFileSpeech:
    """OpenAI TTS rendered to one WAV file for the camera/speaker substitution test."""

    output_path: Path
    client: OpenAITtsSpeech
    _segments: list[str] | None = None

    def speak(self, text: str) -> None:
        if text.strip():
            if self._segments is None:
                self._segments = []
            self._segments.append(text)

    def finalize(self) -> None:
        if not self._segments:
            raise RuntimeError("no speech was queued for WAV output")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lamp-openai-tts-", dir=self.output_path.parent) as directory:
            parts: list[Path] = []
            for index, text in enumerate(self._segments):
                part = Path(directory) / f"part-{index}.wav"
                part.write_bytes(self.client.synthesize(text))
                parts.append(part)
            _combine_wav_files(parts, self.output_path)


def _combine_wav_files(parts: list[Path], output_path: Path) -> None:
    """Combine equal-format WAV chunks without changing playback order."""

    with wave.open(str(parts[0]), "rb") as first:
        parameters = first.getparams()
    with wave.open(str(output_path), "wb") as combined:
        combined.setparams(parameters)
        for part in parts:
            with wave.open(str(part), "rb") as source:
                if source.getparams()[:4] != parameters[:4]:
                    raise RuntimeError("TTS segment format changed unexpectedly")
                combined.writeframes(source.readframes(source.getnframes()))
