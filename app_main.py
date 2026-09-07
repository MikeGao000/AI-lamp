"""Reading companion application entrypoint.

Run `python app_main.py --mode simulate` on a computer. Raspberry Pi camera
mode is opt-in and needs Picamera2, OpenCV, and a configured cloud key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from time import monotonic
from typing import Callable

from lamp_core.cloud import CloudVisionError, OpenAIResponsesVisionClient, StaticStoryClient, StoryClient
from lamp_core.config import AppConfig, load_dotenv
from lamp_core.coordinator import AppEvent, ReadingCompanionCoordinator
from lamp_core.question_prompt import (
    CHILD_QUESTION_SYSTEM_INSTRUCTIONS,
    build_child_question_prompt,
    detect_one_turn_reply_language,
)
from lamp_core.reading_prompt import PICTURE_BOOK_SYSTEM_INSTRUCTIONS, build_picture_book_prompt
from lamp_core.speech import EspeakSpeech, OpenAITtsSpeech, QueuedSpeech, SpeechSink
from lamp_core.vision import Picamera2FrameSource, StillnessGate
from simulate_system import LIMITS
from lamp_core.virtual_hardware import SpeechStub, VirtualMotorBus


@dataclass(frozen=True)
class AcceptedPage:
    """Production result of accepting one stable JPEG page."""

    next_context: str | None
    story_for_tts: str
    recognition: dict | None
    speech_segments: tuple[str, ...]
    timing_s: dict[str, float]


@dataclass(frozen=True)
class ChildQuestionAnswer:
    """A grounded answer that was queued on the same configured TTS voice."""

    answer: str
    recognition: dict
    next_turn: "ChildQuestionTurn"


@dataclass(frozen=True)
class ChildQuestionTurn:
    """Only the last grounded answer needed for a one-turn language rephrase."""

    last_answer: str


def cloud_client(config: AppConfig) -> StoryClient:
    if not config.cloud_enabled:
        return StaticStoryClient()
    if not config.api_key:
        raise RuntimeError("ENABLE_CLOUD_VISION=true requires OPENAI_API_KEY")
    return OpenAIResponsesVisionClient(
        config.api_key,
        config.model,
        config.api_base_url,
        stream=config.cloud_stream,
        reasoning_effort=config.cloud_reasoning_effort,
        max_output_tokens=config.cloud_max_output_tokens,
    )


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
    speaker = QueuedSpeech(create_production_speaker(config))
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
                try:
                    accepted_page = accept_page_jpeg(
                        controller, client, config, jpeg, previous_page_context
                    )
                except CloudVisionError as error:
                    # A cloud response must never terminate the camera loop or
                    # trigger motion.  Wait for the next stable frame instead.
                    print(f"CLOUD VISION: {error}")
                else:
                    previous_page_context = accepted_page.next_context
                finally:
                    gate.reset()
            time.sleep(0.10)
    except KeyboardInterrupt:
        controller.handle(AppEvent.ESTOP)
    finally:
        source.close()
        speaker.close()


def create_production_speaker(config: AppConfig) -> SpeechSink:
    """Choose the configured production playback adapter, never the test substitute."""

    if config.tts_provider == "openai":
        if not config.api_key:
            raise RuntimeError("TTS_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAITtsSpeech(
            api_key=config.api_key,
            base_url=config.api_base_url,
            model=config.tts_model,
            voice=config.openai_tts_voice,
            instructions=config.tts_instructions,
            speed=config.tts_speed,
            timeout_s=config.tts_timeout_s,
        )
    if config.tts_provider == "local":
        return EspeakSpeech(config.tts_voice)
    raise RuntimeError("TTS_PROVIDER must be 'local' or 'openai'")


def _language_for_voice(voice: str) -> str:
    """Legacy helper retained for existing prompt tests and local voice aliases."""

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

    started = monotonic()
    early_speech = StreamingReadingStarter(controller.speaker) if config.cloud_enabled and config.cloud_stream else None
    if config.cloud_enabled:
        page = read_picture_book_page(
            client,
            jpeg,
            config.reading_language,
            previous_page_context=previous_page_context,
            on_output_text_delta=early_speech.feed if early_speech is not None else None,
        )
        extension = page.get("teacher_story") or page.get("narration") or page["spoken_reading"]
        speech_segments = ((early_speech.spoken_reading,) if early_speech and early_speech.spoken_reading else ()) + (extension,)
        story = extension
        next_context = page_context_for_next_page(page)
        print("Picture-book page accepted:", page["confidence"])
        print("Recognized visible text:\n" + page.get("visible_text", "[no legible text returned]"))
    else:
        story = client.describe_page(jpeg)
        speech_segments = (story,)
        next_context = None
        page = None
    controller.set_pending_story(story)
    elapsed = monotonic() - started
    timing_s = {"page_complete": round(elapsed, 3)}
    if early_speech and early_speech.started_after_s is not None:
        timing_s["spoken_reading_queued"] = round(early_speech.started_after_s, 3)
    print("TTS segments:", len(speech_segments))
    print("Narration to TTS:", " ".join(speech_segments))
    print("Vision timing:", timing_s)
    controller.handle(AppEvent.BOOK_STILL)
    return AcceptedPage(
        next_context=next_context,
        story_for_tts=" ".join(speech_segments),
        recognition=page,
        speech_segments=speech_segments,
        timing_s=timing_s,
    )


def read_picture_book_page(
    client: StoryClient,
    jpeg: bytes,
    reply_language: str,
    age_range: str = "3-7",
    previous_page_context: str | None = None,
    on_output_text_delta: Callable[[str], None] | None = None,
) -> dict:
    """Ask for one structured, safe page interpretation; never execute its behavior hint."""
    raw = client.describe_page(
        jpeg,
        prompt=build_picture_book_prompt(reply_language, age_range, previous_page_context),
        system_instructions=PICTURE_BOOK_SYSTEM_INSTRUCTIONS,
        on_output_text_delta=on_output_text_delta,
    )
    try:
        page = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CloudVisionError("picture-book model response was not valid JSON") from error
    if not isinstance(page, dict) or not isinstance(page.get("spoken_reading"), str):
        raise CloudVisionError("picture-book model response lacked spoken_reading")
    return page


def answer_child_question(
    client: StoryClient,
    speaker: SpeechSink,
    config: AppConfig,
    jpeg: bytes,
    question: str,
    pointed_object: str | None = None,
    accepted_page_context: str | None = None,
    reply_language: str | None = None,
    previous_turn: ChildQuestionTurn | None = None,
) -> ChildQuestionAnswer:
    """Answer one child question without moving the lamp or advancing the story.

    Microphone transcription and pointing recognition are hardware adapters.  This
    production function receives their normalized text hints so the camera-free
    test harness and the future live hardware path share the exact same cloud,
    safety, and speech behavior.
    """

    if not question.strip():
        raise ValueError("child question must not be empty")
    requested_turn_language = detect_one_turn_reply_language(question)
    language = (reply_language or requested_turn_language or config.question_reply_language).strip()
    is_language_rephrase = requested_turn_language is not None and previous_turn is not None
    raw = client.describe_page(
        jpeg,
        prompt=build_child_question_prompt(
            question,
            language,
            pointed_object,
            accepted_page_context,
            previous_answer=previous_turn.last_answer if previous_turn else None,
            is_language_rephrase=is_language_rephrase,
        ),
        system_instructions=CHILD_QUESTION_SYSTEM_INSTRUCTIONS,
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CloudVisionError("child-question model response was not valid JSON") from error
    answer = result.get("answer") if isinstance(result, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise CloudVisionError("child-question model response lacked answer")
    spoken_answer = answer.strip()
    speaker.speak(spoken_answer)
    return ChildQuestionAnswer(spoken_answer, result, ChildQuestionTurn(spoken_answer))


@dataclass
class ChildQuestionSession:
    """Production session state for a page's spoken child questions."""

    client: StoryClient
    speaker: SpeechSink
    config: AppConfig
    previous_turn: ChildQuestionTurn | None = None

    def ask(
        self,
        jpeg: bytes,
        question: str,
        pointed_object: str | None = None,
        accepted_page_context: str | None = None,
    ) -> ChildQuestionAnswer:
        result = answer_child_question(
            self.client,
            self.speaker,
            self.config,
            jpeg,
            question,
            pointed_object,
            accepted_page_context,
            previous_turn=self.previous_turn,
        )
        self.previous_turn = result.next_turn
        return result

    def reset_for_new_page(self) -> None:
        self.previous_turn = None


class StreamingReadingStarter:
    """Queue the printed page text as soon as its JSON field completes in a stream."""

    _SPOKEN_READING = re.compile(r'"spoken_reading"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)

    def __init__(self, speaker: SpeechSink) -> None:
        self._speaker = speaker
        self._buffer = ""
        self.spoken_reading: str | None = None
        self._started = monotonic()
        self.started_after_s: float | None = None

    def feed(self, delta: str) -> None:
        if self.spoken_reading is not None:
            return
        self._buffer += delta
        match = self._SPOKEN_READING.search(self._buffer)
        if match is None:
            return
        try:
            spoken_reading = json.loads(f'"{match.group(1)}"').strip()
        except json.JSONDecodeError:
            return
        if spoken_reading:
            self.spoken_reading = spoken_reading
            self.started_after_s = monotonic() - self._started
            self._speaker.speak(spoken_reading)


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
