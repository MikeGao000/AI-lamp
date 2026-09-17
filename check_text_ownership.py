"""Does the page hypothesis plus text ownership find the book's text?

This is the measurement the whole approach rests on. Per frame: the salient page
box, every text box PP-OCR found, and which of those lie inside the page. If the
kept text tracks the framing across the sweep, the axis can be driven by the book's
own text instead of by a contaminated block.
"""

from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from lamp_core.salient_page import (  # noqa: E402
    SalientPageDetector,
    saliency_text_anchor,
    text_boxes_inside,
)
from lamp_core.text_detection import PpocrTextDetector  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"


def angle_of(name: str) -> float | None:
    match = re.search(r"_a([+-]\d+\.\d)_", name)
    return float(match.group(1)) if match else None


detector = SalientPageDetector("/tmp/u2netp.onnx")
text_detector = PpocrTextDetector(None, long_side=320)

rows = {}
paths = sorted(glob.glob(os.path.join(DIRECTORY, "*.jpg")))
for path in paths:
    image = cv2.imread(path)
    angle = angle_of(os.path.basename(path))
    if angle is None:
        continue
    page = detector.page_box(image)
    boxes = text_detector.detect(image)
    kept = text_boxes_inside(page, boxes) if page else []
    anchor = saliency_text_anchor(page, boxes) if page else None
    rows.setdefault(angle, []).append((page, len(boxes), len(kept), anchor))

print(f"{'angle':>6} {'page box':<26} {'text':>4} {'kept':>4} {'kept anchor':<26}")
page_ok = kept_ok = total = 0
track = []
for angle in sorted(rows):
    for page, count, kept_count, anchor in rows[angle][:1]:
        total += 1
        page_ok += int(page is not None)
        kept_ok += int(anchor is not None)
        show_page = "-" if page is None else ",".join(f"{v:.2f}" for v in page)
        show_anchor = "-" if anchor is None else ",".join(f"{v:.2f}" for v in anchor)
        print(f"{angle:6.1f} {show_page:<26} {count:>4} {kept_count:>4} {show_anchor:<26}")
        if anchor is not None:
            track.append((angle, (anchor[0] + anchor[2]) / 2))

print(f"\npage box from the salient model : {page_ok} / {total}")
print(f"book text found (kept >= 1 box) : {kept_ok} / {total}")
if len(track) >= 3:
    print("\ncentre of the BOOK's own text against angle:")
    for angle, centre in track:
        print(f"  {angle:+6.1f} -> {centre:.3f}")
