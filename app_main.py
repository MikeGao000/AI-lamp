"""Reading companion application entrypoint.

Run `python app_main.py --mode simulate` on a computer. Raspberry Pi camera
mode is opt-in and needs Picamera2, OpenCV, and a configured cloud key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

from lamp_core.cloud import CloudVisionError, OpenAIResponsesVisionClient, StaticStoryClient, StoryClient
from lamp_core.config import AppConfig, load_dotenv
from lamp_core.coordinator import AppEvent, ReadingCompanionCoordinator
from lamp_core.reading_prompt import PICTURE_BOOK_SYSTEM_INSTRUCTIONS, build_picture_book_prompt
from lamp_core.speech import EspeakSpeech
from lamp_core.vision import Picamera2FrameSource, StillnessGate
from simulate_system import LIMITS
from lamp_core.virtual_hardware import SpeechStub, VirtualMotorBus


@dataclass(frozen=True)
class AcceptedPage:
    """Production result of accepting one stable JPEG page."""

    next_context: str | None
    story_for_tts: str
    recognition: dict | None


def cloud_client(config: AppConfig) -> StoryClient:
    if not config.cloud_enabled:
        return StaticStoryClient()
    if not config.api_key:
        raise RuntimeError("ENABLE_CLOUD_VISION=true requires OPENAI_API_KEY")
    return OpenAIResponsesVisionClient(config.api_key, config.model, config.api_base_url)


def run_simulation(config: AppConfig) -> None:
    speaker = SpeechStub()
    controller = ReadingCompanionCoordinator(LIMITS, VirtualMotorBus(LIMITS), speaker)
    controller.home()
    controller.handle(AppEvent.BOOK_MOVED)
    # A simulated byte string is not a real image. Keep this path offline even
    # when cloud vision is configured for real camera or image tests.
    story = StaticStoryClient().describe_page(b"offline-simulated-jpeg")
    controller.set_pending_story(story)
    controller.handle(AppEvent.BOOK_STILL)
    print("\n".join(controller.log))
    for text in speaker.messages:
        print(f"SPEECH {text}")


def run_pi(config: AppConfig) -> None:
    """Validate camera/cloud/TTS flow on Pi while keeping motors simulated."""
    source = Picamera2FrameSource()
    gate = StillnessGate(config.still_seconds, config.motion_threshold)
    speaker = EspeakSpeech(config.tts_voice)
    controller = ReadingCompanionCoordinator(LIMITS, VirtualMotorBus(LIMITS), speaker)
    client = cloud_client(config)
    previous_page_context: str | None = None
    controller.home()
    print("Camera loop started. Motors remain virtual until a verified driver adapter is added.")
    try:
        while True:
            jpeg, motion = source.capture_jpeg_and_motion()
            if motion > config.motion_threshold:
                controller.handle(AppEvent.BOOK_MOVED)
            if gate.observe(motion):
                accepted_page = accept_page_jpeg(
                    controller, client, config, jpeg, previous_page_context
                )
                previous_page_context = accepted_page.next_context
                gate.reset()
            time.sleep(0.10)
    except KeyboardInterrupt:
        controller.handle(AppEvent.ESTOP)
    finally:
        source.close()


def _language_for_voice(voice: str) -> str:
    return {"da": "Danish", "zh": "Chinese", "cmn": "Chinese", "en": "English"}.get(
        voice.lower(), voice
    )


def accept_page_jpeg(
    controller: ReadingCompanionCoordinator,
    client: StoryClient,
    config: AppConfig,
    jpeg: bytes,
    previous_page_context: str | None = None,
) -> AcceptedPage:
    """Run the production page-acceptance path for a stable JPEG frame.

    The camera loop and the image-file hardware-substitution test both call this
    function, so cloud reading, prompt construction, safety coordination and TTS
    receive the same accepted page in either case.
    """

    if config.cloud_enabled:
        page = read_picture_book_page(
            client,
            jpeg,
            _language_for_voice(config.tts_voice),
            previous_page_context=previous_page_context,
        )
        story = page.get("teacher_story") or page.get("narration") or page["spoken_reading"]
        next_context = page_context_for_next_page(page)
        print("Picture-book page accepted:", page["confidence"])
        print("Recognized visible text:\n" + page.get("visible_text", "[no legible text returned]"))
    else:
        story = client.describe_page(jpeg)
        next_context = None
        page = None
    controller.set_pending_story(story)
    print("Narration to TTS:", story)
    controller.handle(AppEvent.BOOK_STILL)
    return AcceptedPage(next_context=next_context, story_for_tts=story, recognition=page)


def read_picture_book_page(
    client: StoryClient,
    jpeg: bytes,
    reply_language: str,
    age_range: str = "3-7",
    previous_page_context: str | None = None,
) -> dict:
    """Ask for one structured, safe page interpretation; never execute its behavior hint."""
    raw = client.describe_page(
        jpeg,
        prompt=build_picture_book_prompt(reply_language, age_range, previous_page_context),
        system_instructions=PICTURE_BOOK_SYSTEM_INSTRUCTIONS,
    )
    try:
        page = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CloudVisionError("picture-book model response was not valid JSON") from error
    if not isinstance(page, dict) or not isinstance(page.get("spoken_reading"), str):
        raise CloudVisionError("picture-book model response lacked spoken_reading")
    return page


def page_context_for_next_page(page: dict) -> str:
    """Keep only a small, already accepted context for one-page continuity."""

    visible_text = str(page.get("visible_text", "")).strip()
    description = str(page.get("image_description", "")).strip()
    context = f"Previous page visible text: {visible_text}\nPrevious page visible illustration: {description}"
    return context[:1200]


if __name__ == "__main__":
    # Windows PowerShell may use a legacy output encoding. Never let logging
    # prevent a safe simulation shutdown merely because a story contains Chinese.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("simulate", "pi"), default="simulate")
    args = parser.parse_args()
    load_dotenv()
    config = AppConfig.from_environment()
    if args.mode == "simulate":
        run_simulation(config)
    else:
        run_pi(config)
