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
from pathlib import Path

from app_main import accept_page_jpeg, cloud_client
from lamp_core.cloud import CloudVisionError
from lamp_core.config import AppConfig, load_dotenv
from lamp_core.coordinator import AppEvent, ReadingCompanionCoordinator
from lamp_core.speech import OpenAITtsSpeech, OpenAIWavFileSpeech, WavFileSpeech
from lamp_core.virtual_hardware import VirtualMotorBus
from simulate_system import LIMITS


def run_image_hardware_substitution_test(
    config: AppConfig, image_path: Path, wav_path: Path, result_json_path: Path
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
    controller = ReadingCompanionCoordinator(LIMITS, VirtualMotorBus(LIMITS), speaker)
    controller.home()
    controller.handle(AppEvent.BOOK_MOVED)
    accepted_page = accept_page_jpeg(controller, cloud_client(config), config, jpeg)
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


def create_test_speaker(config: AppConfig, wav_path: Path) -> WavFileSpeech | OpenAIWavFileSpeech:
    """Use the same selected TTS provider, with WAV replacing physical playback."""

    if config.tts_provider == "openai":
        if not config.api_key:
            raise RuntimeError("TTS_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIWavFileSpeech(
            wav_path,
            OpenAITtsSpeech(
                config.api_key,
                config.api_base_url,
                config.tts_model,
                config.openai_tts_voice,
                config.tts_instructions,
                config.tts_timeout_s,
            ),
        )
    if config.tts_provider == "local":
        return WavFileSpeech(wav_path, config.tts_voice)
    raise RuntimeError("TTS_PROVIDER must be 'local' or 'openai'")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="JPEG image used in place of the Pi camera")
    parser.add_argument("--wav", type=Path, default=Path("lamp-test.wav"), help="WAV output path")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("lamp-test-result.json"),
        help="JSON file containing recognized text and the exact text sent to TTS",
    )
    args = parser.parse_args()
    load_dotenv()
    try:
        run_image_hardware_substitution_test(
            AppConfig.from_environment(), args.image, args.wav, args.result_json
        )
    except CloudVisionError as error:
        # The test uses the production cloud client but should report a concise,
        # actionable failure rather than a Python traceback.
        print(f"CLOUD VISION TEST FAILED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
