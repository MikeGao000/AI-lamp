"""Are the "does not contain text" cases real, or the padded-anchor artefact?

The 54-frame benchmark tested containment against text_anchor_box's output, which
is grown by 15%. That exact mismatch already produced a false 21%-containment
figure once. So the ten failures are re-checked against the *raw* detected text
boxes, which is what is actually printed on the page.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

MODEL = "/tmp/u2netp.onnx"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 320
BASE = "/home/lamp/AI-lamp/datasets/book_book-present_20260915-140720/"
OTHER = "/home/lamp/AI-lamp/datasets/book_findbook_20260915-144441/"
NAMES = [
    (BASE, "p01_a+000.0_f02.jpg"), (BASE, "p01_a+025.0_f00.jpg"),
    (BASE, "p01_a+025.0_f01.jpg"), (BASE, "p01_a+030.0_f00.jpg"),
    (BASE, "p01_a+030.0_f01.jpg"), (BASE, "p01_a+030.0_f02.jpg"),
    (BASE, "p01_a-005.0_f00.jpg"), (BASE, "p01_a-005.0_f01.jpg"),
    (BASE, "p01_a-005.0_f02.jpg"), (BASE, "p01_a-015.0_f00.jpg"),
]
# The +030 frames in the prepared sweep are the laptop/window view; the wide sweep
# holds the genuinely bookless frames.
WIDE = [(OTHER, "p01_a+030.0_f00.jpg")]


def salient_box(net, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    blob = ((resized.astype(np.float32) / 255.0 - MEAN) / STD)
    blob = np.transpose(blob, (2, 0, 1))[None].astype(np.float32)
    net.setInput(blob)
    heat = np.asarray(net.forward()).reshape(SIZE, SIZE)
    peak = float(heat.max())
    mask = (heat / peak) > 0.5 if peak > 0 else heat > 1
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best, best_area = None, 0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 0.004 * SIZE * SIZE:
            continue
        if area > best_area:
            best_area, best = area, (x / SIZE, y / SIZE, (x + w) / SIZE, (y + h) / SIZE)
    return best


def union(boxes):
    if not boxes:
        return None
    xs = [v for b in boxes for v in (b.bbox[0], b.bbox[2])]
    ys = [v for b in boxes for v in (b.bbox[1], b.bbox[3])]
    return (min(xs), min(ys), max(xs), max(ys))


def inside(outer, inner, slack=0.02):
    return (outer[0] <= inner[0] + slack and outer[1] <= inner[1] + slack
            and outer[2] >= inner[2] - slack and outer[3] >= inner[3] - slack)


net = cv2.dnn.readNetFromONNX(MODEL)
detector = PpocrTextDetector(None, long_side=320)
print(f"{'frame':<24} {'salient box':<28} {'raw text union':<28} {'raw':>4} "
      f"{'padded':>7}")
raw_ok = padded_ok = total = 0
for directory, name in NAMES + WIDE:
    image = cv2.imread(directory + name)
    boxes = detector.detect(image)
    anchor = text_anchor_box(boxes)
    raw = union(boxes)
    box = salient_box(net, image)
    if box is None or raw is None:
        print(f"{name:<24} {'-':<28} {'-':<28} {'-':>4} {'-':>7}")
        continue
    total += 1
    a = inside(box, raw)
    b = inside(box, anchor) if anchor else None
    raw_ok += int(a)
    padded_ok += int(bool(b))
    show_raw = "[" + ",".join(f"{v:.2f}" for v in raw) + "]"
    show_box = "[" + ",".join(f"{v:.2f}" for v in box) + "]"
    print(f"{name:<24} {show_box:<28} {show_raw:<28} "
          f"{'yes' if a else 'no':>4} {'yes' if b else 'no':>7}")
print(f"\ncontains raw text union : {raw_ok} / {total}")
print(f"contains padded anchor  : {padded_ok} / {total}")
