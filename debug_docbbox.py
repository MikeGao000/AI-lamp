"""Why does the contour page-upgrade find nothing on a high-contrast frame?

The current scene has a bright page on a dark mat (paper/desk luminance ratio 3.2)
and a clean page edge, yet detect_document_bbox still returns nothing even with
the edge and area gates relaxed. So this prints every candidate and the exact
reason it was rejected, and dumps the masks as images.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/freshraw/fresh_0.jpg"

image = cv2.imread(PATH)
height, width = image.shape[:2]
print(f"{PATH}  {width}x{height}")

detector = PpocrTextDetector(None, long_side=320)
anchor = text_anchor_box(detector.detect(image))
print(f"text anchor: {None if anchor is None else tuple(round(v, 3) for v in anchor)}")
if anchor is None:
    raise SystemExit("no text anchor")
tx1, ty1 = int(anchor[0] * width), int(anchor[1] * height)
tx2, ty2 = int(anchor[2] * width), int(anchor[3] * height)
text_box = (tx1, ty1, tx2 - tx1, ty2 - ty1)
text_area = float((tx2 - tx1) * (ty2 - ty1))
print(f"text box px: {text_box}  area {text_area:.0f} "
      f"({text_area / (width * height):.3f} of frame)")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 45, 140)
closed = cv2.morphologyEx(
    edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
)
otsu_value, bright = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
bright_regions = cv2.morphologyEx(
    bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
)
print(f"otsu threshold {otsu_value:.0f}  bright {bright.mean() / 255 * 100:.1f}% of pixels")
cv2.imwrite("/tmp/dbg_edges.jpg", edges)
cv2.imwrite("/tmp/dbg_closed.jpg", closed)
cv2.imwrite("/tmp/dbg_bright.jpg", bright)
cv2.imwrite("/tmp/dbg_bright_closed.jpg", bright_regions)

image_area = width * height
for mask_name, mask in (("edges", closed), ("bright", bright_regions)):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"--- {mask_name}: {len(contours)} contours")
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        area_ratio = box_area / image_area
        aspect = box_width / max(1, box_height)
        touched = sum((x <= 2, y <= 2, x + box_width >= width - 2,
                       y + box_height >= height - 2))
        covers = (x <= tx1 and y <= ty1
                  and x + box_width >= tx2 and y + box_height >= ty2)
        reason = "ok"
        if not 0.30 <= aspect <= 3.3:
            reason = f"aspect {aspect:.2f}"
        elif not covers:
            reason = "does not cover text"
        elif box_area < text_area * 1.3:
            reason = "too small vs text"
        elif not 0.02 <= area_ratio <= 0.98:
            reason = f"area {area_ratio:.3f}"
        print(f"    x[{x:4d},{x + box_width:4d}] y[{y:4d},{y + box_height:4d}] "
              f"area={area_ratio:.3f} aspect={aspect:5.2f} touched={touched} "
              f"covers={covers}  -> {reason}")
