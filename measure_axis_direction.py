"""Measure which way a positive J1 command actually moves the image.

Two calibrations of this axis have disagreed before and the direction flag was
flipped on that basis, so this measures it directly rather than trusting a
label.

It uses phase correlation between a reference frame and each stepped frame, so
the answer does not depend on recognising the same object twice. An earlier
attempt compared the centre of the detected text block and produced nonsense
(+20 and -20 both appeared to move the text right) because the detector simply
found a different block at each angle.

Reading the result:
  * content moves LEFT for a positive offset  -> the camera turned RIGHT
    -> ``--j1-positive-camera-direction right``
  * content moves RIGHT for a positive offset -> the camera turned LEFT
    -> ``--j1-positive-camera-direction left``
"""

from __future__ import annotations

import argparse
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offsets", default="0,8,-8,16,-16,0")
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--frames-per-step", type=int, default=4)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument("--j1-node-id", type=int, default=1)
    parser.add_argument("--j1-checksum", default="additive")
    parser.add_argument("--j1-gear-ratio", type=float, default=1.0)
    parser.add_argument("--j1-speed-rpm", type=int, default=12)
    parser.add_argument("--j1-acceleration", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import cv2
    import numpy as np
    from picamera2 import Picamera2

    from lamp_core.mks_can_protocol import (
        ChecksumMode,
        absolute_coordinate_move,
        set_bus_enabled,
        set_working_mode,
        set_zero_point,
    )
    from lamp_core.mks_single_axis import COUNTS_PER_REVOLUTION, MksSingleAxisProbe
    from run_mks_single_axis_motion import SocketCanTransport

    offsets = [float(value) for value in args.offsets.split(",") if value.strip()]
    mode = ChecksumMode(args.j1_checksum)
    node = args.j1_node_id
    counts_per_degree = COUNTS_PER_REVOLUTION * args.j1_gear_ratio / 360.0

    transport = SocketCanTransport(args.can_interface)
    transport.send(set_working_mode(node, 0x05, mode))
    transport.send(set_zero_point(node, mode))
    probe = MksSingleAxisProbe(transport, node, mode)
    time.sleep(0.2)
    anchor = probe.snapshot().encoder_counts
    transport.send(set_bus_enabled(node, True, mode))

    camera = Picamera2()
    camera.configure(
        camera.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={"FrameRate": args.fps},
        )
    )
    camera.start()
    time.sleep(2.0)

    window = cv2.createHanningWindow((args.width, args.height), cv2.CV_32F)

    def move_to(offset: float) -> bool:
        target = anchor + round(offset * counts_per_degree)
        transport.send(
            absolute_coordinate_move(
                node, args.j1_speed_rpm, args.j1_acceleration, target, mode
            )
        )
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            snapshot = probe.snapshot()
            if snapshot.rpm == 0 and abs(snapshot.encoder_counts - target) <= 24:
                break
            time.sleep(0.1)
        time.sleep(args.settle_seconds)
        return abs(probe.snapshot().encoder_counts - target) <= 40

    def capture_gray():
        """Median of several frames, which suppresses sensor noise for correlation."""
        stack = []
        for _ in range(args.frames_per_step):
            image = camera.capture_array()
            stack.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32))
            time.sleep(0.2)
        return np.median(np.stack(stack), axis=0)

    try:
        print("reference at offset 0 ...")
        if not move_to(0.0):
            print("  warning: J1 did not reach the reference position")
        reference = capture_gray()

        rows: list[tuple[float, float, float]] = []
        for offset in offsets:
            if not move_to(offset):
                print(f"  warning: J1 did not settle at {offset:+.1f} deg")
            frame = capture_gray()
            (dx, dy), response = cv2.phaseCorrelate(reference, frame, window)
            rows.append((offset, dx, dy))
            print(
                f"  offset {offset:+6.1f} deg -> content moved dx={dx:+8.1f}px "
                f"dy={dy:+7.1f}px  (response {response:.3f})"
            )
    finally:
        camera.stop()
        try:
            transport.send(set_bus_enabled(node, False, mode))
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass
        transport.close()

    usable = [row for row in rows if abs(row[0]) > 1e-6]
    if not usable:
        print("\nno stepped samples to compare")
        return
    print("\nsummary")
    positive = [(offset, dx) for offset, dx, _ in usable if offset > 0]
    negative = [(offset, dx) for offset, dx, _ in usable if offset < 0]
    forward = sum(dx for _, dx in positive) / len(positive) if positive else None
    backward = sum(dx for _, dx in negative) / len(negative) if negative else None
    if forward is not None:
        print(f"  mean dx for positive offsets: {forward:+8.1f}px")
    if backward is not None:
        print(f"  mean dx for negative offsets: {backward:+8.1f}px")
    if forward is not None and backward is not None:
        print(f"  positive/negative separation: {forward - backward:+8.1f}px")
    if forward is None:
        print("  only positive samples; cannot check symmetry")
        return
    if abs(forward) < 20:
        print("  the image barely moved; is the camera seeing a textured scene?")
        return
    # Gain, and the field of view it implies, come from slope through the origin
    # so unequal offsets on each side are handled correctly.
    numerator = sum(offset * dx for offset, dx in positive + negative)
    denominator = sum(offset * offset for offset, _ in positive + negative)
    gain = numerator / denominator if denominator else 0.0
    print(f"  gain: {gain:+.1f} px per degree")
    if gain < 0:
        print("  content moves LEFT for a positive offset -> camera turned RIGHT")
        print("  =>  --j1-positive-camera-direction right")
    else:
        print("  content moves RIGHT for a positive offset -> camera turned LEFT")
        print("  =>  --j1-positive-camera-direction left")
    if gain:
        print(f"  implied field of view: {args.width / abs(gain):.1f} deg across the frame")


if __name__ == "__main__":
    main()
