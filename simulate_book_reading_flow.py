"""Run the automatic book-finding and reading flow with virtual motors."""

from __future__ import annotations

import argparse
import sys

from lamp_core.cloud import StaticStoryClient
from lamp_core.reading_demo import SimulatedBook, run_simulated_book_reading_flow
from simulate import JOINT_LIMITS


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-x", type=float, default=0.18, help="normalized book center x in [0, 1]")
    args = parser.parse_args()
    if not 0.1 <= args.book_x <= 0.9:
        raise SystemExit("--book-x must keep the simulated book inside the frame")
    book = SimulatedBook(
        object_id="book-1",
        bbox_norm=(args.book_x - 0.10, 0.20, args.book_x + 0.10, 0.88),
    )
    result = run_simulated_book_reading_flow(
        book,
        StaticStoryClient("我找到了这本绘本。让我们一起开始读吧。"),
        JOINT_LIMITS,
    )
    print("FLOW " + " -> ".join(result.flow_steps))
    print(f"BOOK_DIRECTION {result.book_direction.value}; GAZE {result.gaze_action_id}")
    print(
        f"CENTERED_BOOK_XY ({result.centered_book_x:.3f}, {result.centered_book_y:.3f}); "
        f"SERVO_STEPS {result.visual_servo_steps}"
    )
    print(f"MOTOR_FRAMES {result.motor_frame_count}")
    print("READING_POSE_RAD " + " ".join(f"{name}={value:.2f}" for name, value in result.final_positions_rad.items()))
    print("SPEECH " + result.recognition_text)


if __name__ == "__main__":
    main()
