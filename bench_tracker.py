"""Split the tracker cost into create / init / update, and compare KCF.

The 54-frame benchmark lumped create+init+update into one number and reported
4.3 s, which contradicts the service running at several frames per second. The
whole scan/acquire/track split depends on knowing which part is actually slow.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

FRAME = sys.argv[1] if len(sys.argv) > 1 else "/tmp/freshraw/fresh_0.jpg"

image = cv2.imread(FRAME)
height, width = image.shape[:2]
# A page-ish box, as the salient model would hand over.
box = (int(0.05 * width), int(0.15 * height), int(0.60 * width), int(0.60 * height))
print(f"{FRAME}  {width}x{height}  box {box}")

for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
    factory = getattr(cv2, name, None)
    if factory is None:
        print(f"{name}: not available")
        continue
    start = time.perf_counter()
    tracker = factory()
    create_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    tracker.init(image, box)
    init_ms = (time.perf_counter() - start) * 1000.0
    updates = []
    for _ in range(4):
        start = time.perf_counter()
        tracker.update(image)
        updates.append((time.perf_counter() - start) * 1000.0)
    print(f"{name:<20} create {create_ms:8.1f} ms   init {init_ms:8.1f} ms   "
          f"update median {sorted(updates)[len(updates) // 2]:8.1f} ms  "
          f"(min {min(updates):.1f})")

start = time.perf_counter()
for _ in range(5):
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"{'cvtColor':<20} {((time.perf_counter() - start) / 5) * 1000:8.1f} ms per call")
