"""Why does the transcript come back empty? Print the crop and profile stats."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from camera_preview import page_crop  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector  # noqa: E402
from lamp_core.text_recognition import line_ranges  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
NAME = "p01_a-015.0_f00.jpg"

with open("/tmp/bench_confirm.json", encoding="utf-8") as handle:
    boxes_by_name = {r["name"]: r["box"] for r in json.load(handle) if r["box"]}

image = cv2.imread(os.path.join(DIRECTORY, NAME))
crop = page_crop(image, cv2, tuple(boxes_by_name[NAME]), margin=0.02)
detector = PpocrTextDetector(None, long_side=320)
text_boxes = detector.detect(crop)
height, width = crop.shape[:2]
print(f"page crop {crop.shape}  boxes {len(text_boxes)}")
for box in text_boxes:
    x1, y1, x2, y2 = box.bbox
    left, top = max(0, int(x1 * width)), max(0, int(y1 * height))
    right, bottom = min(width, int(x2 * width)), min(height, int(y2 * height))
    block = crop[top:bottom, left:right]
    print(f"  box {[round(v, 2) for v in box.bbox]} conf {box.confidence:.2f} "
          f"-> block {block.shape}")
    if block.size == 0:
        continue
    grey = block if block.ndim == 2 else block.mean(axis=2)
    ink = 255.0 - np.asarray(grey, dtype="float32")
    profile = ink.mean(axis=1)
    print(f"     ink peak {profile.max():.1f}  mean {profile.mean():.1f}  "
          f"rows>floor {int((profile > profile.max() * 0.18).sum())}  "
          f"spans {line_ranges(profile.tolist())}")
    cv2.imwrite(f"/tmp/block_{int(x1 * 100)}.jpg", block)
