"""Where does the remaining time in the capture loop go?

The preview reaches 1.72 fps even with detection reduced to one frame in three, so
roughly 0.26 s per frame is spent elsewhere. The follower reads the encoder over CAN
on every update, so this times that round trip directly.
"""

from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

from lamp_core.mks_can_protocol import ChecksumMode  # noqa: E402
from lamp_core.mks_single_axis import MksSingleAxisProbe  # noqa: E402
from run_mks_single_axis_motion import SocketCanTransport  # noqa: E402

transport = SocketCanTransport("can0")
probe = MksSingleAxisProbe(transport, 1, ChecksumMode("additive"))

times = []
for _ in range(25):
    started = time.perf_counter()
    probe.snapshot()
    times.append((time.perf_counter() - started) * 1000.0)
times.sort()
print(f"encoder snapshot (a CAN request/response): median {times[len(times)//2]:.1f} ms  "
      f"min {times[0]:.1f}  max {times[-1]:.1f}")

started = time.perf_counter()
for _ in range(10):
    transport.send(
        __import__("lamp_core.mks_can_protocol", fromlist=["read_motor_rpm"]).read_motor_rpm(
            1, ChecksumMode("additive")
        )
    )
print(f"send-only x10: {((time.perf_counter() - started) / 10) * 1000:.1f} ms each")

import cv2  # noqa: E402

image = None
try:
    import urllib.request

    import numpy as np

    raw = urllib.request.urlopen("http://127.0.0.1:8000/snapshot.jpg", timeout=5).read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
except Exception as error:  # noqa: BLE001
    print(f"snapshot fetch failed: {error}")

for label, work in (
    ("annotate-only draw", lambda: image.copy()),
    ("jpeg encode q80", lambda: cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])),
):
    if image is None:
        break
    started = time.perf_counter()
    for _ in range(5):
        work()
    print(f"{label:<22}: {((time.perf_counter() - started) / 5) * 1000:.1f} ms")
