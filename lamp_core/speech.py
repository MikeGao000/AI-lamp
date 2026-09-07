"""Speech adapters for local and OpenAI cloud voice output."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import wave
import base64
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
    speed: float = 1.0
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
            "speed": self.speed,
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


@dataclass
class OpenAIRealtimeSpeech:
    """Speech adapter backed by an OpenAI Realtime WebSocket session.

    This is deliberately a separate adapter from ``OpenAITtsSpeech``: Realtime
    audio is returned as Base64 PCM chunks over a WebSocket, not by the
    request-based ``/audio/speech`` endpoint.
    """

    api_key: str
    model: str = "gpt-realtime-2.1-mini"
    voice: str = "marin"
    instructions: str | None = None
    timeout_s: float = 45.0
    safety_identifier: str = "smart-lamp-local"

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise CloudSpeechError("cannot synthesize empty text")
        websocket = self._connect()
        try:
            websocket.send(json.dumps(self._session_update()))
            websocket.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Read the following text aloud exactly as written.\n\n" + text.strip(),
                                }
                            ],
                        },
                    }
                )
            )
            websocket.send(json.dumps({"type": "response.create", "response": {"output_modalities": ["audio"]}}))
            pcm_parts: list[bytes] = []
            while True:
                event = json.loads(websocket.recv())
                event_type = event.get("type")
                if event_type == "response.output_audio.delta":
                    pcm_parts.append(base64.b64decode(event["delta"]))
                elif event_type == "error":
                    message = event.get("error", {}).get("message", "unknown Realtime API error")
                    raise CloudSpeechError(f"OpenAI Realtime returned an error: {message}")
                elif event_type == "response.done":
                    status = event.get("response", {}).get("status")
                    if status not in (None, "completed"):
                        raise CloudSpeechError(f"OpenAI Realtime response ended with status {status!r}")
                    break
        except CloudSpeechError:
            raise
        except Exception as error:
            raise CloudSpeechError("OpenAI Realtime TTS is unavailable or timed out") from error
        finally:
            websocket.close()
        if not pcm_parts:
            raise CloudSpeechError("OpenAI Realtime returned empty audio")
        return _pcm16_to_wav(b"".join(pcm_parts))

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        player = shutil.which("aplay")
        if player is None:
            raise RuntimeError("install alsa-utils to play OpenAI Realtime audio on Raspberry Pi")
        subprocess.run([player, "-q"], input=audio, check=True)

    def _session_update(self) -> dict:
        instructions = self.instructions or "Speak naturally and clearly."
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "voice": self.voice,
                    }
                },
                "instructions": (
                    instructions
                    + " Preserve the supplied reading text exactly: do not translate, add, omit, "
                    "summarize, or answer it. Express its mood with warm, natural spoken delivery."
                ),
            },
        }

    def _connect(self):
        try:
            import websocket
        except ImportError as error:
            raise RuntimeError("install websocket-client to enable OpenAI Realtime speech") from error
        return websocket.create_connection(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            header=[
                f"Authorization: Bearer {self.api_key}",
                f"OpenAI-Safety-Identifier: {self.safety_identifier}",
            ],
            timeout=self.timeout_s,
        )


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


@dataclass
class OpenAIRealtimeWavFileSpeech:
    """Realtime audio rendered to a WAV file for the hardware-substitution test."""

    output_path: Path
    client: OpenAIRealtimeSpeech
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
        with tempfile.TemporaryDirectory(prefix="lamp-openai-realtime-", dir=self.output_path.parent) as directory:
            parts: list[Path] = []
            for index, text in enumerate(self._segments):
                part = Path(directory) / f"part-{index}.wav"
                part.write_bytes(self.client.synthesize(text))
                parts.append(part)
            _combine_wav_files(parts, self.output_path)


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 24_000) -> bytes:
    """Wrap mono, little-endian PCM16 returned by Realtime in a WAV container."""

    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _combine_wav_files(parts: list[Path], output_path: Path) -> None:
    """Combine equal-format WAV chunks without changing playback order.

    Streaming TTS WAVs can legally mark their data length as ``0xffffffff``
    because the final size was unknown while the bytes were sent.  Read such
    files in bounded pieces until EOF; never trust that header's frame count.
    """

    with wave.open(str(parts[0]), "rb") as first:
        reference = (
            first.getnchannels(),
            first.getsampwidth(),
            first.getframerate(),
            first.getcomptype(),
            first.getcompname(),
        )
    with wave.open(str(output_path), "wb") as combined:
        combined.setnchannels(reference[0])
        combined.setsampwidth(reference[1])
        combined.setframerate(reference[2])
        combined.setcomptype(reference[3], reference[4])
        for part in parts:
            with wave.open(str(part), "rb") as source:
                actual = (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                    source.getcomptype(),
                    source.getcompname(),
                )
                if actual != reference:
                    raise RuntimeError("TTS segment format changed unexpectedly")
                while chunk := source.readframes(8_192):
                    combined.writeframesraw(chunk)
