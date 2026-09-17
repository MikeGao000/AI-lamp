"""Can the local recogniser read letters at all?

The detection model is detection-only (boxes, no characters), and the recogniser added
for the abandoned check produced nothing -- because the detector hands it whole regions
(text plus illustration), not lines. This feeds it hand-specified *line* boxes from the
same page crop, which isolates the recogniser from the detector's box quality.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from lamp_core.text_recognition import CrnnTextRecognizer  # noqa: E402

PAGE = "/tmp/reading_chain/page.jpg"

# Hand-specified lines on the 914x395 page crop: the two story lines on each page.
LINES = {
    "left page, top line": (0.00, 0.05, 0.30, 0.30),
    "left page, bottom line": (0.00, 0.28, 0.28, 0.58),
    "right page, top line": (0.30, 0.06, 0.72, 0.32),
    "right page, bottom line": (0.30, 0.30, 0.70, 0.58),
}

image = cv2.imread(PAGE)
recognizer = CrnnTextRecognizer(None)
print(f"available {recognizer.available}  model {recognizer.model_path}")
print(f"page crop {None if image is None else image.shape}")

if image is not None:
    height, width = image.shape[:2]
    for label, box in LINES.items():
        left, top = int(box[0] * width), int(box[1] * height)
        right, bottom = int(box[2] * width), int(box[3] * height)
        strip = image[top:bottom, left:right]
        text = recognizer.read_crop(strip)
        print(f"  {label:<24} {strip.shape}  -> {text!r}  "
              f"({recognizer.last_milliseconds:.0f} ms)")
        cv2.imwrite(f"/tmp/line_{label.split()[0]}_{label.split()[1]}.jpg", strip)
