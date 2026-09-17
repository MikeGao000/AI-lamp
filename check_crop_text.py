"""Did reading text on the page crop actually help?

Compares, on the same 39 real frames with their saved page boxes:
  old: detect over the whole frame, then keep only text inside the page box
  new: detect over the page crop
and reports how often each finds the book's text, plus how much larger the page's own
text is rendered in the detector's fixed-size input.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from camera_preview import page_crop  # noqa: E402
from lamp_core.salient_page import saliency_text_anchor  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
with open("/tmp/bench_confirm.json", encoding="utf-8") as handle:
    records = [r for r in json.load(handle) if r["box"] is not None]

detector = PpocrTextDetector(None, long_side=320)
old_found = new_found = total = 0
old_boxes_total = new_boxes_total = 0
scales = []
print(f"{'frame':<24} {'page box':<24} {'old':>5} {'new':>5}  {'text scale x':>12}")
for record in records:
    path = os.path.join(DIRECTORY, record["name"])
    image = cv2.imread(path)
    if image is None:
        continue
    page = tuple(record["box"])
    height, width = image.shape[:2]
    total += 1

    whole = detector.detect(image)
    anchor = saliency_text_anchor(page, whole)
    old_ok = anchor is not None
    old_found += int(old_ok)
    old_boxes_total += len(whole)

    crop = page_crop(image, cv2, page, margin=0.02)
    cropped = detector.detect(crop)
    new_ok = bool(cropped)
    new_found += int(new_ok)
    new_boxes_total += len(cropped)

    frame_long = max(width, height)
    crop_long = max(crop.shape[0], crop.shape[1])
    scale = frame_long / max(1, crop_long)
    scales.append(scale)
    show_page = ",".join(f"{v:.2f}" for v in page)
    print(f"{record['name']:<24} {show_page:<24} {str(old_ok):>5} {str(new_ok):>5} "
          f"{scale:12.2f}")

scales.sort()
print(f"\nbook text found  OLD (whole frame + containment): {old_found} / {total}")
print(f"book text found  NEW (page crop)                : {new_found} / {total}")
print(f"text boxes kept  OLD {old_boxes_total}   NEW {new_boxes_total}")
print(f"page text rendered larger by: median {scales[len(scales)//2]:.2f}x  "
      f"max {scales[-1]:.2f}x")
