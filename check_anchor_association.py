"""Does the reference-frame text association break the page contour?

The live local candidate was a text band, not a page, which means
detect_document_bbox found no contour containing the text it was given. The live
path feeds it text_anchor_box(..., reference_bbox=<previous anchor>), which prefers
the previous frame's block; the earlier 6-of-6 page-contour result was measured
with the plain anchor. This measures both on the same 39 frames.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from camera_preview import detect_document_bbox  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
paths = sorted(glob.glob(os.path.join(DIRECTORY, "*.jpg")))
detector = PpocrTextDetector(None, long_side=320)


def to_pixels(anchor, image):
    height, width = image.shape[:2]
    x1, y1 = int(anchor[0] * width), int(anchor[1] * height)
    x2, y2 = int(anchor[2] * width), int(anchor[3] * height)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


plain_ok = ref_ok = both = 0
print(f"{'frame':<24} {'plain anchor':<26} {'-> page?':>8} {'ref anchor':<26} "
      f"{'-> page?':>8}")
previous = None
for path in paths:
    image = cv2.imread(path)
    boxes = detector.detect(image)
    plain = text_anchor_box(boxes)
    ref = text_anchor_box(boxes, reference_bbox=previous) if previous else plain
    plain_page = ref_page = None
    if plain is not None:
        plain_page = detect_document_bbox(
            image, cv2, text_evidence=to_pixels(plain, image)
        )
    if ref is not None:
        ref_page = detect_document_bbox(image, cv2, text_evidence=to_pixels(ref, image))
    both += 1
    plain_ok += int(plain_page is not None)
    ref_ok += int(ref_page is not None)
    show_plain = "-" if plain is None else f"{plain[0]:.2f},{plain[1]:.2f},{plain[2]:.2f},{plain[3]:.2f}"
    show_ref = "-" if ref is None else f"{ref[0]:.2f},{ref[1]:.2f},{ref[2]:.2f},{ref[3]:.2f}"
    print(f"{os.path.basename(path):<24} {show_plain:<26} "
          f"{'yes' if plain_page else 'NO':>8} {show_ref:<26} "
          f"{'yes' if ref_page else 'NO':>8}")
    if plain is not None:
        previous = plain

print(f"\npage contour from PLAIN anchor : {plain_ok} / {both}")
print(f"page contour from REFERENCE anchor: {ref_ok} / {both}")
