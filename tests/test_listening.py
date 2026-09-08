import io
import json
import sys
import threading
import time
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from lamp_core.listening import (
    MicrophoneQuestionListener,
    OpenAITranscriptionClient,
    pcm16_mono_to_wav,
    pcm16_rms,
)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"text": "Hvad er det?"}).encode("utf-8")


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls = []

    def transcribe_wav(self, audio):
        self.calls.append(audio)
        return "这是什么？"


class FakeRawInputStream:
    def __init__(self, chunks, **kwargs):
        self.chunks = list(chunks)
        self.frames = kwargs["blocksize"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, frames):
        if self.chunks:
            return self.chunks.pop(0), False
        time.sleep(0.005)
        return b"\0\0" * frames, False


class ListeningTests(unittest.TestCase):
    def test_listener_finishes_turn_after_start_callback_changes_state(self):
        loud = (2000).to_bytes(2, "little", signed=True) * 480
        quiet = b"\0\0" * 480
        chunks = [loud] * 7 + [quiet] * 22
        transcriber = FakeTranscriber()
        received = []
        done = threading.Event()
        accepting_new_turn = [True]

        def started():
            accepting_new_turn[0] = False
            return True

        def transcript(text):
            received.append(text)
            done.set()

        listener = MicrophoneQuestionListener(
            transcriber,
            on_speech_started=started,
            on_transcript=transcript,
            should_listen=lambda: accepting_new_turn[0],
            system_is_speaking=lambda: False,
        )
        fake_module = SimpleNamespace(
            RawInputStream=lambda **kwargs: FakeRawInputStream(chunks, **kwargs)
        )
        with patch.dict(sys.modules, {"sounddevice": fake_module}):
            listener.start()
            self.assertTrue(done.wait(2.0))
            listener.close()
        self.assertEqual(["这是什么？"], received)
        self.assertEqual(1, len(transcriber.calls))

    def test_pcm_rms_detects_silence_and_voice_energy(self):
        self.assertEqual(0, pcm16_rms(b"\0\0" * 100))
        self.assertGreater(pcm16_rms((1000).to_bytes(2, "little", signed=True) * 100), 900)

    def test_wraps_microphone_pcm_as_16khz_mono_wav(self):
        audio = pcm16_mono_to_wav(b"\x01\x00\x02\x00")
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual(1, wav.getnchannels())
            self.assertEqual(16_000, wav.getframerate())
            self.assertEqual(b"\x01\x00\x02\x00", wav.readframes(2))

    def test_transcription_uses_audio_endpoint_and_multipart_wav(self):
        client = OpenAITranscriptionClient("secret", model="gpt-4o-mini-transcribe")
        with patch("lamp_core.listening.urlopen", return_value=FakeResponse()) as request:
            self.assertEqual("Hvad er det?", client.transcribe_wav(b"RIFF-test-wav"))
        sent = request.call_args.args[0]
        self.assertTrue(sent.full_url.endswith("/audio/transcriptions"))
        self.assertIn("multipart/form-data", sent.headers["Content-type"])
        self.assertIn(b'gpt-4o-mini-transcribe', sent.data)
        self.assertIn(b'filename="question.wav"', sent.data)
        self.assertIn(b"RIFF-test-wav", sent.data)


if __name__ == "__main__":
    unittest.main()
