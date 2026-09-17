"""Recompute the benchmark metrics with the two defects fixed.

Defect 1: the six fresh frames are consecutive shots of one static scene, but they
were grouped per filename, so each became a group of one and was skipped by the
stability measurement -- exactly the best sample available.

Defect 2: containment was tested against text_anchor_box's output, which is grown
by 15%; that mismatch already produced one false 21%-containment figure. The raw
detected text boxes are the honest reference.

The salient boxes are already on disk, so this only re-runs the cheap OCR pass.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

with open("/tmp/bench_u2netp.json", encoding="utf-8") as handle:
    records = json.load(handle)
detector = PpocrTextDetector(None, long_side=320)


def union(boxes):
    if not boxes:
        return None
    xs = [v for b in boxes for v in (b.bbox[0], b.bbox[2])]
    ys = [v for b in boxes for v in (b.bbox[1], b.bbox[3])]
    return (min(xs), min(ys), max(xs), max(ys))


def inside(outer, inner, slack=0.02):
    return (outer[0] <= inner[0] + slack and outer[1] <= inner[1] + slack
            and outer[2] >= inner[2] - slack and outer[3] >= inner[3] - slack)


def view_key(path: str) -> str:
    """Frames of one unchanged view. The fresh captures are one static scene."""

    directory = os.path.basename(os.path.dirname(path))
    if "freshraw" in directory:
        return "freshraw (one static scene)"
    match = re.search(r"_a([+-]\d+\.\d)_", os.path.basename(path))
    return f"{directory}/{match.group(1)}" if match else os.path.basename(path)


views: dict[str, list] = {}
raw_ok = padded_ok = both = 0
areas = []
for record in records:
    path = record["path"]
    image = cv2.imread(path)
    if image is None or record["box"] is None:
        continue
    boxes = detector.detect(image)
    raw = union(boxes)
    anchor = text_anchor_box(boxes)
    box = record["box"]
    areas.append((box[2] - box[0]) * (box[3] - box[1]))
    if raw is not None:
        both += 1
        raw_ok += int(inside(box, raw))
        padded_ok += int(bool(anchor) and inside(box, anchor))
    views.setdefault(view_key(path), []).append(box)

areas.sort()
print(f"frames with a box      : {len(areas)} / {len(records)}")
print(f"box area               : min {areas[0]:.3f}  median "
      f"{areas[len(areas) // 2]:.3f}  max {areas[-1]:.3f}")
print(f"contains RAW text union: {raw_ok} / {both}  "
      f"({100 * raw_ok / max(1, both):.0f}%)")
print(f"contains padded anchor : {padded_ok} / {both}  "
      f"({100 * padded_ok / max(1, both):.0f}%)")

print(f"\nstability per view ({len(views)} views):")
spreads = []
print(f"  {'view':<40} {'n':>3} {'spread':>8}")
for key, boxes in sorted(views.items()):
    spread = max(max(b[i] for b in boxes) - min(b[i] for b in boxes) for i in range(4))
    print(f"  {key:<40} {len(boxes):>3} {spread:8.4f}")
    if len(boxes) >= 2:
        spreads.append(spread)
spreads.sort()
print(f"\nviews with n>=2: {len(spreads)}   median spread "
      f"{spreads[len(spreads) // 2]:.4f}   worst {spreads[-1]:.4f}")
