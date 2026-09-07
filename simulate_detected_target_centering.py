"""Locate a real-image target, then run its two-axis centering plan virtually.

This sends images to the configured vision model only when .env explicitly
enables cloud vision. Motor execution remains VirtualMotorBus-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lamp_core.cloud import OpenAIResponsesVisionClient, StaticStoryClient
from lamp_core.config import AppConfig, load_dotenv
from lamp_core.object_localization import ObjectLocalizationError, locate_target
from lamp_core.reading_demo import SimulatedBook, run_simulated_book_reading_flow
from simulate import JOINT_LIMITS


DEFAULT_TARGET = "the white-and-blue Crucial BX500 product package on the desk"


def _vision_client(config: AppConfig) -> OpenAIResponsesVisionClient:
    if not config.cloud_enabled or not config.api_key:
        raise SystemExit(
            "Cloud vision is not configured. Set OPENAI_API_KEY and ENABLE_CLOUD_VISION=true in .env."
        )
    return OpenAIResponsesVisionClient(
        config.api_key,
        config.model,
        base_url=config.api_base_url,
        stream=config.cloud_stream,
        reasoning_effort=config.cloud_reasoning_effort,
        max_output_tokens=300,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="JPEG files to localize")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="physical target the vision model should locate")
    args = parser.parse_args()

    load_dotenv()
    client = _vision_client(AppConfig.from_environment())
    for image_path in args.images:
        jpeg = image_path.read_bytes()
        try:
            detection = locate_target(jpeg, client, args.target, object_id=image_path.stem)
        except ObjectLocalizationError as error:
            print(f"{image_path.name}: INVALID_DETECTION {error}")
            continue
        if detection is None:
            print(f"{image_path.name}: NOT_FOUND")
            continue
        result = run_simulated_book_reading_flow(
            SimulatedBook(
                detection.object_id,
                detection.bbox_norm,
                snapshot_jpeg=jpeg,
                confidence=detection.confidence,
                label=detection.label,
            ),
            StaticStoryClient("已定位目标，正在进入阅读视角。"),
            JOINT_LIMITS,
        )
        print(
            f"{image_path.name}: FOUND label={detection.label!r} "
            f"bbox={tuple(round(value, 3) for value in detection.bbox_norm)} "
            f"initial_center=({detection.center_x:.3f},{detection.center_y:.3f}) "
            f"centered=({result.centered_book_x:.3f},{result.centered_book_y:.3f}) "
            f"servo_steps={result.visual_servo_steps} frames={result.motor_frame_count}"
        )


if __name__ == "__main__":
    main()
