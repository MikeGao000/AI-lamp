"""Run the offline Pi-local-detector simulation on real sample photographs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lamp_core.cloud import StaticStoryClient
from lamp_core.local_vision_fixtures import SIMULATED_PI_LOCAL_DETECTOR
from lamp_core.reading_demo import SimulatedBook, run_simulated_book_reading_flow
from simulate import JOINT_LIMITS


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="sample JPEG files")
    args = parser.parse_args()
    passed = 0
    for image_path in args.images:
        jpeg = image_path.read_bytes()
        detection = SIMULATED_PI_LOCAL_DETECTOR.locate(image_path.name, jpeg)
        if detection is None:
            print(f"{image_path.name}: LOCAL_DETECTOR_NOT_FOUND")
            continue
        result = run_simulated_book_reading_flow(
            SimulatedBook(
                detection.object_id,
                detection.bbox_norm,
                snapshot_jpeg=jpeg,
                confidence=detection.confidence,
                label=detection.label,
            ),
            StaticStoryClient("本地目标已定位，进入阅读视角。"),
            JOINT_LIMITS,
        )
        centered = abs(result.centered_book_x - 0.5) <= 0.04 and abs(result.centered_book_y - 0.5) <= 0.04
        status = "PASS" if centered else "FAIL"
        passed += int(centered)
        print(
            f"{image_path.name}: {status} LOCAL_SIM "
            f"initial=({detection.center_x:.3f},{detection.center_y:.3f}) "
            f"centered=({result.centered_book_x:.3f},{result.centered_book_y:.3f}) "
            f"servo_steps={result.visual_servo_steps} frames={result.motor_frame_count}"
        )
    print(f"SUMMARY {passed}/{len(args.images)} centering plans passed")


if __name__ == "__main__":
    main()
