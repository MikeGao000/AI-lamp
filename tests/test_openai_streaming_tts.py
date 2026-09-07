import io
import json
import unittest
import wave
from unittest.mock import patch

from lamp_core.cloud import OpenAIResponsesVisionClient
from lamp_core.speech import OpenAITtsSpeech


def wav_bytes() -> bytes:
    data = io.BytesIO()
    with wave.open(data, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\0\0\0\0")
    return data.getvalue()


class FakeResponse:
    def __init__(self, body: bytes | list[bytes]) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        assert isinstance(self.body, bytes)
        return self.body

    def __iter__(self):
        assert isinstance(self.body, list)
        return iter(self.body)


class OpenAIStreamingAndTtsTests(unittest.TestCase):
    def test_collects_text_deltas_and_calls_early_callback(self):
        output = []
        response = FakeResponse(
            [
                b'data: {"type":"response.output_text.delta","delta":"{\\"spoken_reading\\":\\"Hej\\"}"}\n',
                b'data: {"type":"response.completed"}\n',
            ]
        )
        self.assertEqual(
            '{"spoken_reading":"Hej"}',
            OpenAIResponsesVisionClient._read_stream(response, output.append),
        )
        self.assertEqual(['{"spoken_reading":"Hej"}'], output)

    def test_tts_posts_wav_request_with_selected_voice_and_instructions(self):
        client = OpenAITtsSpeech(
            "test-key", voice="marin", instructions="Warm, calm Danish reading."
        )
        with patch("lamp_core.speech.urlopen", return_value=FakeResponse(wav_bytes())) as request:
            audio = client.synthesize("Hej lille bjørn")
        self.assertEqual(wav_bytes(), audio)
        payload = json.loads(request.call_args.args[0].data.decode("utf-8"))
        self.assertEqual("gpt-4o-mini-tts", payload["model"])
        self.assertEqual("marin", payload["voice"])
        self.assertEqual("wav", payload["response_format"])
        self.assertEqual("Warm, calm Danish reading.", payload["instructions"])
