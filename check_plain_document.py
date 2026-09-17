"""The ready-made approach, retested on the current scene.

The classic document-scanner recipe -- grey, blur, threshold the bright page, close,
largest quadrilateral contour -- was rejected early using frames captured before the
lamp was repositioned, when page and desk differed by only 1.05-1.8x in luminance. The
current scene measures 3.2x, so it deserves a fresh test rather than a remembered
verdict.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"


def page_quad(image):
    """Largest bright, page-shaped region: the standard recipe, nothing bespoke."""

    height, width = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (7, 7), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(width * height)
    best = None
    best_area = 0.0
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < frame_area * 0.05 or area > frame_area * 0.98:
            continue
        aspect = box_width / max(1, box_height)
        if not 0.4 <= aspect <= 2.6:
            continue
        if area > best_area:
            best_area = area
            best = (x / width, y / height, (x + box_width) / width, (y + box_height) / height)
    return best


def main() -> None:
    with open("/tmp/bench_confirm.json", encoding="utf-8") as handle:
        salient = {r["name"]: r["box"] for r in json.load(handle) if r["box"]}

    paths = sorted(glob.glob(os.path.join(DIRECTORY, "*.jpg")))
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    found = 0
    areas = []
    print(f"{'frame':<26} {'plain box':<28} {'area':>5} {'salient':>8}")
    for index, path in enumerate(paths):
        image = cv2.imread(path)
        if image is None:
            continue
        box = page_quad(image)
        name = os.path.basename(path)
        reference = salient.get(name)
        if box is None:
            print(f"{name:<26} {'-':<28} {'-':>5} "
                  f"{'-' if reference is None else 'present':>8}")
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        found += 1
        areas.append(area)
        show = ",".join(f"{v:.2f}" for v in box)
        print(f"{name:<26} {show:<28} {area:5.2f} "
              f"{'-' if reference is None else 'present':>8}")
        if index < 3:
            marked = image.copy()
            height, width = image.shape[:2]
            cv2.rectangle(marked, (int(box[0] * width), int(box[1] * height)),
                          (int(box[2] * width), int(box[3] * height)), (0, 255, 0), 3)
            cv2.imwrite(f"/tmp/plain_{name}", marked)

    print(f"\nplain detector found a page on {found} / {len(paths)} frames")
    if areas:
        areas.sort()
        print(f"area: min {areas[0]:.2f}  median {areas[len(areas) // 2]:.2f}  "
              f"max {areas[-1]:.2f}")
    print("overlays for the first three -> /tmp/plain_*.jpg")


if __name__ == "__main__":
    main()
