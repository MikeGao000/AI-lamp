"""How long does "move and settle" really take? It dominates the photo-driven loop.

Every design estimate so far assumed 1.5-2 s per step without measuring it, and
`speed_rpm 12` should be ~72 deg/s while the logs showed ~10 deg/s. This measures
move+settle at several speed/acceleration settings, with symmetric moves so the
axis ends where it started, and also times a camera capture.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

from lamp_core.mks_can_protocol import (  # noqa: E402
    ChecksumMode,
    absolute_coordinate_move,
    set_bus_enabled,
    set_working_mode,
    set_zero_point,
)
from lamp_core.mks_single_axis import (  # noqa: E402
    COUNTS_PER_REVOLUTION,
    MksSingleAxisProbe,
)
from run_mks_single_axis_motion import SocketCanTransport  # noqa: E402

IFACE = "can0"
NODE = 1
MODE = ChecksumMode("additive")
COUNTS_PER_DEGREE = COUNTS_PER_REVOLUTION / 360.0
STEPS = [(12, 120), (30, 120), (60, 200), (120, 255)]
OFFSET = 20.0


def main() -> None:
    transport = SocketCanTransport(IFACE)
    transport.send(set_working_mode(NODE, 0x05, MODE))
    transport.send(set_zero_point(NODE, MODE))
    probe = MksSingleAxisProbe(transport, NODE, MODE)
    time.sleep(0.2)
    anchor = probe.snapshot().encoder_counts
    transport.send(set_bus_enabled(NODE, True, MODE))
    time.sleep(0.2)

    def move(offset: float, rpm: int, acceleration: int) -> tuple[float, float, int]:
        target = anchor + round(offset * COUNTS_PER_DEGREE)
        started = time.perf_counter()
        transport.send(
            absolute_coordinate_move(NODE, rpm, acceleration, target, MODE)
        )
        moved = None
        deadline = time.perf_counter() + 20.0
        while time.perf_counter() < deadline:
            snapshot = probe.snapshot()
            if snapshot.rpm == 0 and abs(snapshot.encoder_counts - target) <= 24:
                moved = time.perf_counter() - started
                break
            time.sleep(0.02)
        # Settle: how much longer until the encoder stops moving at all.
        settled = time.perf_counter()
        last = probe.snapshot().encoder_counts
        stable_since = time.perf_counter()
        while time.perf_counter() - settled < 5.0:
            now_counts = probe.snapshot().encoder_counts
            if abs(now_counts - last) > 1:
                stable_since = time.perf_counter()
            last = now_counts
            if time.perf_counter() - stable_since >= 0.20:
                break
            time.sleep(0.02)
        total = time.perf_counter() - started
        return (moved if moved is not None else float("nan"),
                total, abs(last - target))

    print(f"{'rpm':>4} {'accel':>6} {'move ms':>8} {'total ms':>9} {'err cnt':>8} "
          f"{'deg/s':>7}")
    for rpm, acceleration in STEPS:
        try:
            first = move(OFFSET, rpm, acceleration)
            second = move(-OFFSET, rpm, acceleration)
        except Exception as error:  # noqa: BLE001
            print(f"{rpm:>4} {acceleration:>6}  failed: {type(error).__name__}: {error}")
            continue
        move_ms = (first[0] + second[0]) / 2 * 1000.0
        total_ms = (first[1] + second[1]) / 2 * 1000.0
        error = max(first[2], second[2])
        print(f"{rpm:>4} {acceleration:>6} {move_ms:8.0f} {total_ms:9.0f} "
              f"{error:8d} {OFFSET / (total_ms / 1000.0):7.1f}")

    # Return to the starting pose.
    move(0.0, 60, 240)
    final = probe.snapshot().encoder_counts - anchor
    print(f"\nback at start, remaining offset {final / COUNTS_PER_DEGREE:+.2f} deg")

    try:
        from picamera2 import Picamera2

        camera = Picamera2()
        camera.configure(
            camera.create_video_configuration(
                main={"size": (960, 720), "format": "RGB888"},
                controls={"FrameRate": 10.0},
            )
        )
        camera.start()
        time.sleep(2.0)
        times = []
        for _ in range(6):
            start = time.perf_counter()
            camera.capture_array()
            times.append((time.perf_counter() - start) * 1000.0)
        times.sort()
        print(f"camera capture_array: median {times[3]:.0f} ms  min {times[0]:.0f}")
        camera.stop()
    except Exception as error:  # noqa: BLE001
        print(f"camera timing skipped: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
