"""Camera/microphone/speaker substitution test for one child question.

The JPEG replaces a live camera frame, --question replaces speech-to-text, and
--pointed-object replaces pointing recognition.  The production question logic
and configured TTS provider are used unchanged; audio is written to a WAV file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app_main import answer_child_question, cloud_client
from app_main_test import create_test_speaker
from lamp_core.config import AppConfig, load_dotenv


def run_child_question_substitution_test(
    config: AppConfig,
    image_path: Path,
    question: str,
    pointed_object: str | None,
    reply_language: str | None,
    wav_path: Path,
    result_json_path: Path,
) -> None:
    if not config.cloud_enabled:
        raise RuntimeError("Set ENABLE_CLOUD_VISION=true: this test must use real cloud vision.")
    jpeg = image_path.read_bytes()
    if not jpeg:
        raise ValueError("Image file is empty")
    speaker = create_test_speaker(config, wav_path)
    result = answer_child_question(
        cloud_client(config),
        speaker,
        config,
        jpeg,
        question,
        pointed_object=pointed_object,
        reply_language=reply_language,
    )
    speaker.finalize()
    result_json_path.write_text(
        json.dumps(
            {
                "source_image": str(image_path),
                "question": question,
                "pointed_object": pointed_object,
                "requested_reply_language": reply_language or config.question_reply_language,
                "answer": result.recognition,
                "text_sent_to_tts": result.answer,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Question:", question)
    print("Answer to TTS:", result.answer)
    print(f"WAV written: {wav_path}")
    print(f"Result written: {result_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--question", default="这是什么？")
    parser.add_argument("--pointed-object", default="star")
    parser.add_argument("--reply-language", help="Override QUESTION_REPLY_LANGUAGE for this test")
    parser.add_argument("--wav", type=Path, default=Path("child-question-test.wav"))
    parser.add_argument("--result-json", type=Path, default=Path("child-question-test-result.json"))
    args = parser.parse_args()
    load_dotenv()
    run_child_question_substitution_test(
        AppConfig.from_environment(),
        args.image,
        args.question,
        args.pointed_object,
        args.reply_language,
        args.wav,
        args.result_json,
    )
