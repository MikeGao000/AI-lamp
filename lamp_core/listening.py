"""Optional microphone VAD and multilingual cloud transcription for the Pi."""

from __future__ import annotations

import json
import math
import secrets
import threading
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TranscriptionError(RuntimeError):
    """Recorded speech could not be transcribed safely."""


@dataclass
class OpenAITranscriptionClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini-transcribe"
    timeout_s: float = 30.0

    def transcribe_wav(self, wav_audio: bytes) -> str:
        if not wav_audio:
            raise TranscriptionError("cannot transcribe empty audio")
        boundary = "----smart-lamp-" + secrets.token_hex(12)
        body = _multipart_audio_body(boundary, self.model, wav_audio)
        request = Request(
            self.base_url.rstrip("/") + "/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise TranscriptionError(f"transcription returned HTTP {error.code}") from error
        except (TimeoutError, URLError) as error:
            raise TranscriptionError("transcription service is unavailable or timed out") from error
        except json.JSONDecodeError as error:
            raise TranscriptionError("transcription response was not valid JSON") from error
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionError("transcription response lacked text")
        return text.strip()


class MicrophoneQuestionListener:
    """Detect speech locally, stop narration, then transcribe the completed turn.

    Amplitude VAD is deliberately local so interruption does not wait for the
    network.  The final captured WAV is sent only after the child becomes quiet.
    Acoustic echo cancellation should be enabled in the Pi audio stack; the
    higher barge-in threshold is only a conservative secondary guard.
    """

    def __init__(
        self,
        transcriber: OpenAITranscriptionClient,
        *,
        on_speech_started: Callable[[], bool],
        on_transcript: Callable[[str], None],
        should_listen: Callable[[], bool],
        system_is_speaking: Callable[[], bool],
        sample_rate_hz: int = 16_000,
        threshold_rms: int = 650,
        barge_in_multiplier: float = 1.8,
        silence_seconds: float = 0.65,
        max_question_seconds: float = 12.0,
        block_ms: int = 30,
        device: str | int | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.on_speech_started = on_speech_started
        self.on_transcript = on_transcript
        self.should_listen = should_listen
        self.system_is_speaking = system_is_speaking
        self.sample_rate_hz = sample_rate_hz
        self.threshold_rms = threshold_rms
        self.barge_in_multiplier = barge_in_multiplier
        self.silence_seconds = silence_seconds
        self.max_question_seconds = max_question_seconds
        self.block_ms = block_ms
        self.device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="lamp-microphone", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("install sounddevice to enable the Pi microphone") from error
        block_frames = max(1, self.sample_rate_hz * self.block_ms // 1000)
        start_needed = 2
        candidate: list[bytes] = []
        recording: list[bytes] = []
        loud_blocks = 0
        silent_blocks = 0
        with sd.RawInputStream(
            samplerate=self.sample_rate_hz,
            blocksize=block_frames,
            channels=1,
            dtype="int16",
            device=self.device,
        ) as stream:
            while not self._stop.is_set():
                data, _overflowed = stream.read(block_frames)
                pcm = bytes(data)
                # Once barge-in has been accepted the reading state becomes
                # LISTENING, so finish this already-started turn even though a
                # second question may not start until the answer is complete.
                if not recording and not self.should_listen():
                    candidate.clear()
                    recording.clear()
                    loud_blocks = 0
                    silent_blocks = 0
                    continue
                threshold = self.threshold_rms
                if self.system_is_speaking():
                    threshold = round(threshold * self.barge_in_multiplier)
                loud = pcm16_rms(pcm) >= threshold
                if not recording:
                    if loud:
                        candidate.append(pcm)
                        loud_blocks += 1
                    else:
                        candidate.clear()
                        loud_blocks = 0
                    if loud_blocks >= start_needed:
                        if self.on_speech_started():
                            recording.extend(candidate)
                        candidate.clear()
                        loud_blocks = 0
                    continue
                recording.append(pcm)
                silent_blocks = 0 if loud else silent_blocks + 1
                duration = len(recording) * self.block_ms / 1000
                quiet_long_enough = silent_blocks * self.block_ms / 1000 >= self.silence_seconds
                if not quiet_long_enough and duration < self.max_question_seconds:
                    continue
                audible_blocks = len(recording) - silent_blocks
                captured = recording
                recording = []
                silent_blocks = 0
                if audible_blocks * self.block_ms < 180:
                    continue
                try:
                    transcript = self.transcriber.transcribe_wav(
                        pcm16_mono_to_wav(b"".join(captured), self.sample_rate_hz)
                    )
                except TranscriptionError as error:
                    print(f"TRANSCRIPTION: {error}")
                else:
                    self.on_transcript(transcript)


def pcm16_rms(pcm: bytes) -> int:
    """Compute little-endian signed PCM16 RMS without NumPy."""

    if len(pcm) < 2:
        return 0
    even = len(pcm) - len(pcm) % 2
    samples = memoryview(pcm[:even]).cast("h")
    return round(math.sqrt(sum(int(sample) ** 2 for sample in samples) / len(samples)))


def pcm16_mono_to_wav(pcm: bytes, sample_rate_hz: int = 16_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm)
    return output.getvalue()


def _multipart_audio_body(boundary: str, model: str, wav_audio: bytes) -> bytes:
    marker = boundary.encode("ascii")
    return b"".join(
        (
            b"--" + marker + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\n',
            model.encode("utf-8"),
            b"\r\n--" + marker,
            b'\r\nContent-Disposition: form-data; name="file"; filename="question.wav"\r\n',
            b"Content-Type: audio/wav\r\n\r\n",
            wav_audio,
            b"\r\n--" + marker + b"--\r\n",
        )
    )
