"""Photograph the picture book on this rig to fine-tune a local detector.

Why this exists: no downloadable weights fit "an open picture book on this desk,
under this lamp, from this camera". Measured evidence says so -- COCO's ``book``
class scores 0.37 alongside ``tv`` and ``cup`` on a sharp frame, and the book
models on Hugging Face all target spines, covers or scanned page layout. So the
data has to come from this rig.

Two things decide whether the frames are usable:

* **Motion blur.** Every frame is taken after the axis has stopped *and* reported
  itself settled, because blur is what defeats the on-device text detector.
* **Viewpoint variety.** One sweep of J1 gives many angles for one book position;
  move the book between passes for the rest.

The preview server must be stopped: the camera cannot be shared with it, and the
script drives the same J1 the follower would.

Example::

    python3 collect_book_dataset.py --note book-present --passes 2
    python3 collect_book_dataset.py --note desk-empty --passes 1 --no-motor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="defaults to datasets/book_<stamp>")
    parser.add_argument("--note", default="book", help="what this pass shows, e.g. book or desk-empty")
    parser.add_argument("--angles", default="-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30")
    parser.add_argument("--passes", type=int, default=1, help="full sweeps; move the book between them")
    parser.add_argument("--frames-per-angle", type=int, default=3)
    parser.add_argument("--frame-gap-seconds", type=float, default=0.6)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--settle-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--settle-tolerance-counts", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--no-motor", action="store_true", help="capture without moving J1")
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument("--j1-node-id", type=int, default=1)
    parser.add_argument("--j1-checksum", default="additive")
    parser.add_argument("--j1-gear-ratio", type=float, default=1.0)
    parser.add_argument("--j1-speed-rpm", type=int, default=12)
    parser.add_argument("--j1-acceleration", type=int, default=120)
    return parser


def default_output_dir(note: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in note)
    return os.path.join("datasets", f"book_{safe}_{stamp}")


class Camera:
    """Minimal picamera2 wrapper that yields the same array the server uses."""

    def __init__(self, width: int, height: int, fps: float, rotation: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as error:  # pragma: no cover - Pi only
            raise SystemExit("Picamera2 is missing; install python3-picamera2 on the Pi") from error
        self._camera = Picamera2()
        configuration = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self._camera.configure(configuration)
        self._camera.start()
        self._rotation = rotation
        # Let auto exposure and white balance settle before the first frame, or
        # the first sweep is systematically darker than the rest.
        time.sleep(2.0)

    def read(self):
        import cv2

        image = self._camera.capture_array()
        if self._rotation:
            codes = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            image = cv2.rotate(image, codes[self._rotation])
        return image

    def close(self) -> None:
        try:
            self._camera.stop()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass


class Axis:
    """Absolute-coordinate J1 moves, with a real settle check."""

    def __init__(self, args: argparse.Namespace) -> None:
        from lamp_core.mks_can_protocol import (
            ChecksumMode,
            absolute_coordinate_move,
            set_bus_enabled,
            set_working_mode,
            set_zero_point,
        )
        from lamp_core.mks_single_axis import COUNTS_PER_REVOLUTION, MksSingleAxisProbe
        from run_mks_single_axis_motion import SocketCanTransport

        self._absolute_move = absolute_coordinate_move
        self.mode = ChecksumMode(args.j1_checksum)
        self.node_id = args.j1_node_id
        self.gear_ratio = args.j1_gear_ratio
        self.counts_per_degree = COUNTS_PER_REVOLUTION * self.gear_ratio / 360.0
        self.transport = SocketCanTransport(args.can_interface)
        # Bus FOC position mode is mandatory before 0xF5 absolute moves.
        self.transport.send(set_working_mode(self.node_id, 0x05, self.mode))
        self.transport.send(set_zero_point(self.node_id, self.mode))
        self.probe = MksSingleAxisProbe(self.transport, self.node_id, self.mode)
        time.sleep(0.2)
        self.anchor_counts = self.probe.snapshot().encoder_counts
        self.transport.send(set_bus_enabled(self.node_id, True, self.mode))

    def move_to_degrees(self, degrees: float, speed_rpm: int, acceleration: int) -> int:
        target = self.anchor_counts + round(degrees * self.counts_per_degree)
        self.transport.send(
            self._absolute_move(self.node_id, speed_rpm, acceleration, target, self.mode)
        )
        return target

    def wait_until_settled(self, target: int, timeout: float, tolerance_counts: int) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.probe.snapshot()
            if snapshot.rpm == 0 and abs(snapshot.encoder_counts - target) <= tolerance_counts:
                return True
            time.sleep(0.1)
        return False

    def close(self) -> None:
        from lamp_core.mks_can_protocol import set_bus_enabled

        try:
            self.transport.send(set_bus_enabled(self.node_id, False, self.mode))
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass
        self.transport.close()


def main() -> None:
    args = build_parser().parse_args()
    import cv2

    output_dir = args.output_dir or default_output_dir(args.note)
    os.makedirs(output_dir, exist_ok=True)
    angles = [float(value) for value in args.angles.split(",") if value.strip()]

    axis = None
    if not args.no_motor:
        axis = Axis(args)

    camera = Camera(args.width, args.height, args.fps, args.rotation)
    manifest: list[dict[str, object]] = []
    written = 0
    try:
        for pass_index in range(1, args.passes + 1):
            if pass_index > 1:
                print(f"\n--- pass {pass_index} of {args.passes} ---")
                print("Move the book to a new position, then press Enter to continue.")
                try:
                    input()
                except EOFError:
                    print("(no console; continuing after a short pause)")
                    time.sleep(3.0)
            for angle in angles:
                target = None
                settled = True
                if axis is not None:
                    target = axis.move_to_degrees(angle, args.j1_speed_rpm, args.j1_acceleration)
                    settled = axis.wait_until_settled(
                        target, args.settle_timeout_seconds, args.settle_tolerance_counts
                    )
                    # A little extra quiet time after the encoder stops: the
                    # mechanism can still be ringing even at zero RPM.
                    time.sleep(args.settle_seconds)
                for frame_index in range(args.frames_per_angle):
                    image = camera.read()
                    ok, buffer = cv2.imencode(
                        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                    )
                    if not ok:
                        continue
                    name = f"p{pass_index:02d}_a{angle:+06.1f}_f{frame_index:02d}.jpg"
                    path = os.path.join(output_dir, name)
                    with open(path, "wb") as handle:
                        handle.write(buffer.tobytes())
                    manifest.append(
                        {
                            "file": name,
                            "pass": pass_index,
                            "angle_degrees": angle,
                            "frame": frame_index,
                            "target_counts": target,
                            "settled": settled,
                            "note": args.note,
                            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                    )
                    written += 1
                    time.sleep(args.frame_gap_seconds)
                print(f"  pass {pass_index} angle {angle:+6.1f} settled={settled} -> {written} frames")
    finally:
        camera.close()
        if axis is not None:
            axis.close()

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"note": args.note, "angles": angles, "frames": manifest}, handle, indent=2)
    print(f"\nWrote {written} frames to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
