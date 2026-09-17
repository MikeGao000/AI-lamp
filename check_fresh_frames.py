"""Fresh frames, measured now: is there page/desk contrast today, and does the
existing contour page-upgrade work on the current scene?

Every recent conclusion was drawn from frames captured at 14:07-14:45, before the
lamp was repositioned, so this captures new ones and re-measures instead of
arguing from stale data.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import camera_preview  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = "/tmp/fresh"
RAW_DIR = "/tmp/freshraw"


def capture(count: int):
    from picamera2 import Picamera2

    camera = Picamera2()
    camera.configure(
        camera.create_video_configuration(
            main={"size": (960, 720), "format": "RGB888"},
            controls={"FrameRate": 10.0},
        )
    )
    camera.start()
    time.sleep(2.0)
    frames = []
    try:
        for _ in range(count):
            frames.append(camera.capture_array().copy())
            time.sleep(0.5)
    finally:
        camera.stop()
    return frames


def luminance(gray, anchor):
    height, width = gray.shape
    if anchor is None:
        paper = float("nan")
    else:
        pad_x = (anchor[2] - anchor[0]) * 0.25
        pad_y = (anchor[3] - anchor[1]) * 0.25
        x1 = max(0, int((anchor[0] - pad_x) * width))
        x2 = min(width, int((anchor[2] + pad_x) * width))
        y1 = max(0, int((anchor[1] - pad_y) * height))
        y2 = min(height, int((anchor[3] + pad_y) * height))
        band = gray[y1:y2, x1:x2].reshape(-1).astype(np.float32)
        paper_pixels = band[band >= np.percentile(band, 60)] if band.size else band
        paper = float(paper_pixels.mean()) if paper_pixels.size else float("nan")
    desk = float(gray[int(height * 0.92):, :].mean())
    otsu, _ = cv2.threshold(gray.astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return paper, desk, float(otsu), float((gray > otsu).mean())


def main() -> None:
    import os

    os.makedirs(RAW_DIR, exist_ok=True)
    frames = capture(COUNT)
    detector = PpocrTextDetector(None, long_side=320)
    print(f"{'#':>2} {'boxes':>5} {'paper':>7} {'desk':>7} {'ratio':>6} {'otsu':>5} "
          f"{'above':>6} {'page box':>9} {'ratio':>6} {'covers':>7}")
    for index, frame in enumerate(frames):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        boxes = detector.detect(image)
        anchor = text_anchor_box(boxes)
        paper, desk, otsu, above = luminance(gray, anchor)

        page_box = None
        ratio = float("nan")
        covers = "-"
        if anchor is not None:
            tx1 = int(anchor[0] * width)
            ty1 = int(anchor[1] * height)
            tx2 = int(anchor[2] * width)
            ty2 = int(anchor[3] * height)
            text_box = (tx1, ty1, tx2 - tx1, ty2 - ty1)
            page_box = camera_preview.detect_document_bbox(
                image, cv2, text_evidence=text_box
            )
            text_area = max(1.0, float((tx2 - tx1) * (ty2 - ty1)))
            if page_box is not None:
                px, py, pw, ph = page_box
                ratio = (pw * ph) / text_area
                covers = "yes" if (px <= tx1 and py <= ty1
                                   and px + pw >= tx2 and py + ph >= ty2) else "no"
        shown = "-" if page_box is None else "found"
        print(f"{index:>2} {len(boxes):>5} {paper:7.1f} {desk:7.1f} "
              f"{paper / desk:6.3f} {otsu:5.0f} {above * 100:5.1f}% {shown:>9} "
              f"{ratio:6.1f} {covers:>7}")

        vis = image.copy()
        if anchor is not None:
            cv2.rectangle(vis, (int(anchor[0] * width), int(anchor[1] * height)),
                          (int(anchor[2] * width), int(anchor[3] * height)),
                          (0, 165, 255), 2)
        if page_box is not None:
            px, py, pw, ph = page_box
            cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 255, 0), 3)
        cv2.imwrite(f"{OUT}_{index}.jpg", vis)
        cv2.imwrite(f"{RAW_DIR}/fresh_{index}.jpg", image)
    print(f"\noverlays -> {OUT}_*.jpg   raw frames -> {RAW_DIR}/fresh_*.jpg")


if __name__ == "__main__":
    main()
