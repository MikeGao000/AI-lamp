import json
import tempfile
import unittest
from pathlib import Path

from app_main import accept_page_jpeg
from lamp_core.config import AppConfig
from lamp_core.coordinator import AppEvent, ReadingCompanionCoordinator
from lamp_core.page_memory import PageMemory
from lamp_core.virtual_hardware import SpeechStub, VirtualMotorBus
from simulate_system import LIMITS


def config() -> AppConfig:
    return AppConfig(
        True,
        "test-key",
        "https://example.invalid/v1",
        "vision-test",
        1.5,
        8.0,
        "da",
        "local",
        "tts-test",
        "marin",
        "warm",
        1.0,
        30.0,
        "Danish",
        "Danish",
        False,
        "none",
        700,
    )


class CountingStoryClient:
    def __init__(self) -> None:
        self.calls = 0

    def describe_page(self, jpeg: bytes, **kwargs) -> str:
        self.calls += 1
        return json.dumps(
            {
                "visible_text": "Hej måne",
                "spoken_reading": "Hej måne.",
                "teacher_story": "Se den runde måne. Den lyser blidt.",
                "image_description": "En måne over to bjørne.",
                "confidence": "high",
            }
        )


class PageMemoryTests(unittest.TestCase):
    def test_learns_only_close_visual_variants(self):
        hashes = {
            b"first": "0000000000000000",
            b"close": "0000000000000001",
            b"borderline": "000000000000007f",
            b"far": "ffffffffffffffff",
        }
        with tempfile.TemporaryDirectory() as directory:
            memory = PageMemory(directory, match_distance=7, fingerprint=hashes.__getitem__)
            stored = memory.store(
                b"first",
                recognition={"visible_text": "Hej"},
                speech_segments=("Hej.",),
                next_context="moon",
            )
            self.assertEqual(stored.page_id, memory.find(b"close").page_id)
            self.assertEqual(stored.page_id, memory.find(b"borderline").page_id)
            self.assertIsNone(memory.find(b"far"))
            record = json.loads(
                (Path(directory) / "pages" / stored.page_id / "page.json").read_text()
            )
            self.assertEqual(2, len(record["fingerprints"]))
            self.assertEqual(2, record["hit_count"])

    def test_real_acceptance_path_skips_cloud_on_repeat_page(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PageMemory(directory, fingerprint=lambda _: "0123456789abcdef")
            client = CountingStoryClient()
            first_speaker = SpeechStub()
            first = ReadingCompanionCoordinator(LIMITS, VirtualMotorBus(LIMITS), first_speaker)
            first.home()
            first.handle(AppEvent.BOOK_MOVED)
            accepted_first = accept_page_jpeg(
                first, client, config(), b"same-jpeg", page_memory=memory
            )

            second_speaker = SpeechStub()
            second = ReadingCompanionCoordinator(LIMITS, VirtualMotorBus(LIMITS), second_speaker)
            second.home()
            second.handle(AppEvent.BOOK_MOVED)
            accepted_second = accept_page_jpeg(
                second, client, config(), b"same-jpeg", page_memory=memory
            )

            self.assertEqual(1, client.calls)
            self.assertEqual(1.0, accepted_second.timing_s["cache_hit"])
            self.assertEqual(accepted_first.speech_segments, accepted_second.speech_segments)
            self.assertEqual(first_speaker.messages, second_speaker.messages)


if __name__ == "__main__":
    unittest.main()
