"""Hardware-substitution test for the real picture-book application path.

Run this on the Pi with a JPEG file. It uses the production JPEG acceptance
function from app_main, real configured cloud vision, the normal prompt and
coordinator, virtual motors, and writes local TTS output to a WAV file instead
of requiring a camera or speaker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app_main import accept_page_jpeg, cloud_client
from lamp_core.cloud import CloudVisionError
from lamp_core.config import AppConfig, load_dotenv
from lamp_core.coordinator import AppEvent, ReadingCompanionCoordinator
from lamp_core.page_memory import PageMemory
from lamp_core.speech import (
    CachedAudioSpeech,
    OpenAIRealtimeSpeech,
    OpenAIRealtimeWavFileSpeech,
    OpenAITtsSpeech,
    OpenAIWavFileSpeech,
    WavFileSpeech,
)
from lamp_core.virtual_hardware import VirtualMotorBus
from lamp_core.vision import Picamera2FrameSource
from simulate_system import LIMITS


def capture_production_camera_jpeg(output_path: Path, warmup_s: float = 2.0) -> None:
    """Capture one retained test page through the production Pi camera adapter."""

    source = Picamera2FrameSource()
    try:
        time.sleep(warmup_s)
        # Exercise the same high-resolution still path used by production after
        # its low-resolution tracking stream declares the page stable.
        source.capture_jpeg_and_motion()
        jpeg = source.capture_high_resolution_jpeg()
    finally:
        source.close()
    if not jpeg:
        raise RuntimeError("production camera returned an empty JPEG")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(jpeg)
    print(f"Production camera JPEG written: {output_path} ({len(jpeg)} bytes)")


def run_image_hardware_substitution_test(
    config: AppConfig,
    image_path: Path,
    wav_path: Path,
    result_json_path: Path,
    previous_page_context: str | None = None,
) -> None:
    """Run an image through the production app path with camera/speaker substituted."""

    if not config.cloud_enabled:
        raise RuntimeError("Set ENABLE_CLOUD_VISION=true: this test must use real cloud vision.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    jpeg = image_path.read_bytes()
    if not jpeg:
        raise ValueError("Image file is empty")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    speaker = create_test_speaker(config, wav_path)
    page_memory = (
        PageMemory(config.page_memory_dir, match_distance=config.page_match_distance)
        if config.page_memory_enabled
        else None
    )
    controller = ReadingCompanionCoordinator(LIMITS, VirtualMotorBus(LIMITS), speaker)
    controller.home()
    controller.handle(AppEvent.BOOK_MOVED)
    accepted_page = accept_page_jpeg(
        controller,
        cloud_client(config),
        config,
        jpeg,
        previous_page_context=previous_page_context,
        page_memory=page_memory,
    )
    speaker.finalize()
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    result_json_path.write_text(
        json.dumps(
            {
                "source_image": str(image_path),
                "recognized_page": accepted_page.recognition,
                "text_sent_to_tts": accepted_page.story_for_tts,
                "tts_segments": accepted_page.speech_segments,
                "timing_s": accepted_page.timing_s,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n".join(controller.log))
    print(f"WAV written: {wav_path}")
    print(f"Recognition result written: {result_json_path}")


def create_test_speaker(
    config: AppConfig, wav_path: Path
) -> WavFileSpeech | OpenAIWavFileSpeech | OpenAIRealtimeWavFileSpeech:
    """Use the same selected TTS provider, with WAV replacing physical playback."""

    if config.tts_provider == "openai":
        if not config.api_key:
            raise RuntimeError("TTS_PROVIDER=openai requires OPENAI_API_KEY")
        synthesizer = OpenAITtsSpeech(
            api_key=config.api_key,
            base_url=config.api_base_url,
            model=config.tts_model,
            voice=config.openai_tts_voice,
            instructions=config.tts_instructions,
            speed=config.tts_speed,
            timeout_s=config.tts_timeout_s,
        )
        if config.page_memory_enabled:
            synthesizer = CachedAudioSpeech(
                PageMemory(config.page_memory_dir).audio_dir, synthesizer
            )
        return OpenAIWavFileSpeech(wav_path, synthesizer)
    if config.tts_provider == "local":
        return WavFileSpeech(wav_path, config.tts_voice)
    if config.tts_provider == "openai-realtime":
        if not config.api_key:
            raise RuntimeError("TTS_PROVIDER=openai-realtime requires OPENAI_API_KEY")
        synthesizer = OpenAIRealtimeSpeech(
            api_key=config.api_key,
            model=config.tts_model,
            voice=config.openai_tts_voice,
            instructions=config.tts_instructions,
            timeout_s=config.tts_timeout_s,
        )
        if config.page_memory_enabled:
            synthesizer = CachedAudioSpeech(
                PageMemory(config.page_memory_dir).audio_dir, synthesizer
            )
        return OpenAIRealtimeWavFileSpeech(wav_path, synthesizer)
    raise RuntimeError("TTS_PROVIDER must be 'local', 'openai', or 'openai-realtime'")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, help="existing JPEG used in place of the Pi camera")
    parser.add_argument(
        "--camera-capture",
        type=Path,
        help="capture this JPEG with the production Picamera2 adapter, then run the real cloud/TTS path",
    )
    parser.add_argument("--wav", type=Path, default=Path("lamp-test.wav"), help="WAV output path")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("lamp-test-result.json"),
        help="JSON file containing recognized text and the exact text sent to TTS",
    )
    parser.add_argument(
        "--previous-page-context",
        help="Test-only accepted prior-page context; never substitutes for a live camera page.",
    )
    args = parser.parse_args()
    if (args.image is None) == (args.camera_capture is None):
        parser.error("provide exactly one existing image or --camera-capture OUTPUT.jpg")
    load_dotenv()
    try:
        image_path = args.image
        if args.camera_capture is not None:
            capture_production_camera_jpeg(args.camera_capture)
            image_path = args.camera_capture
        run_image_hardware_substitution_test(
            AppConfig.from_environment(),
            image_path,
            args.wav,
            args.result_json,
            args.previous_page_context,
        )
    except CloudVisionError as error:
        # The test uses the production cloud client but should report a concise,
        # actionable failure rather than a Python traceback.
        print(f"CLOUD VISION TEST FAILED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
