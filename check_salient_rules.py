"""Does the numpy band rule agree with the connected-component box on real maps?

The earlier page-box numbers came from the largest connected component. The new
detector uses a purely numpy band rule so it can be tested without OpenCV. If the
two disagree, the new rule is the wrong foundation, so this compares them on every
frame of the current scene.
"""

from __future__ import annotations

import glob
import os
import statistics
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lamp_core.salient_page import INPUT_SIZE, SalientPageDetector, region_box  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"


def component_box(heat: np.ndarray) -> tuple[float, float, float, float] | None:
    """The rule used for the earlier measurements: largest connected component."""

    peak = float(heat.max())
    if peak <= 0:
        return None
    mask = ((heat / peak) > 0.5).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    size = heat.shape[0]
    best, best_area = None, 0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 0.004 * size * size:
            continue
        if area > best_area:
            best_area = area
            best = (x / size, y / size, (x + w) / size, (y + h) / size)
    return best


def iou(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


detector = SalientPageDetector("/tmp/u2netp.onnx")
text_detector = PpocrTextDetector(None, long_side=320)
paths = sorted(glob.glob(os.path.join(DIRECTORY, "*.jpg")))

agreements = []
band_ok = comp_ok = both = 0
band_holds = comp_holds = with_text = 0
print(f"{'frame':<24} {'band box':<26} {'component box':<26} {'IoU':>5}")
for path in paths:
    image = cv2.imread(path)
    heat = np.asarray(detector.saliency_map(image)).reshape(INPUT_SIZE, INPUT_SIZE)
    band = region_box(heat.tolist())
    comp = component_box(heat)
    value = iou(band, comp)
    if band is not None and comp is not None:
        agreements.append(value)
    band_ok += int(band is not None)
    comp_ok += int(comp is not None)
    anchor = text_anchor_box(text_detector.detect(image))
    if anchor is not None:
        both += 1
        from lamp_core.salient_page import box_contains

        band_holds += int(band is not None and box_contains(band, anchor))
        comp_holds += int(comp is not None and box_contains(comp, anchor))
    show_band = "-" if band is None else ",".join(f"{v:.2f}" for v in band)
    show_comp = "-" if comp is None else ",".join(f"{v:.2f}" for v in comp)
    print(f"{os.path.basename(path):<24} {show_band:<26} {show_comp:<26} {value:5.2f}")

print(f"\nband rule found a box     : {band_ok} / {len(paths)}")
print(f"component rule found a box: {comp_ok} / {len(paths)}")
if agreements:
    agreements.sort()
    print(f"IoU band vs component     : median {agreements[len(agreements)//2]:.2f}  "
          f"min {agreements[0]:.2f}  max {agreements[-1]:.2f}")
print(f"band contains text        : {band_holds} / {both}")
print(f"component contains text   : {comp_holds} / {both}")
