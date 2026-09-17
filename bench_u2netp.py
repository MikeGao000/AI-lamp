"""Measure the salient-object page anchor over every frame we have, plus timings.

Answers three things with numbers instead of impressions:

* how often the salient region contains the detected text (is it the page?),
* how stable its box is across consecutive frames of the same view,
* what each stage actually costs on this Pi, so the scan/acquire/track split can be
  designed against measured latency rather than guesses.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
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

DIRS = [
    "/home/lamp/AI-lamp/datasets/book_book-present_20260915-140720",
    "/home/lamp/AI-lamp/datasets/book_findbook_20260915-144441",
    "/tmp/freshraw",
]


def salient_box(net, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    blob = ((resized.astype(np.float32) / 255.0 - MEAN) / STD)
    blob = np.transpose(blob, (2, 0, 1))[None].astype(np.float32)
    net.setInput(blob)
    heat = np.asarray(net.forward()).reshape(SIZE, SIZE)
    peak = float(heat.max())
    if peak <= 0:
        return None, 0.0
    mask = (heat / peak) > 0.5
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best = None
    best_area = 0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 0.004 * SIZE * SIZE:
            continue
        if area > best_area:
            best_area = area
            best = (x / SIZE, y / SIZE, (x + w) / SIZE, (y + h) / SIZE)
    return best, best_area / float(SIZE * SIZE)


def main() -> None:
    net = cv2.dnn.readNetFromONNX(MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    detector = PpocrTextDetector(None, long_side=320)

    frames = []
    for directory in DIRS:
        for path in sorted(glob.glob(os.path.join(directory, "*.jpg"))):
            frames.append(path)
    print(f"{len(frames)} frames")

    records = []
    for path in frames:
        image = cv2.imread(path)
        if image is None:
            continue
        start = time.perf_counter()
        boxes = detector.detect(image)
        ms_ocr = (time.perf_counter() - start) * 1000.0
        anchor = text_anchor_box(boxes)
        start = time.perf_counter()
        box, share = salient_box(net, image)
        ms_sal = (time.perf_counter() - start) * 1000.0
        # A tracker update is what runs per frame in the loop.
        start = time.perf_counter()
        tracker = cv2.TrackerCSRT_create()
        height, width = image.shape[:2]
        if box is not None:
            x1, y1, x2, y2 = box
            tracker.init(image, (
                int(x1 * width), int(y1 * height),
                max(8, int((x2 - x1) * width)), max(8, int((y2 - y1) * height)),
            ))
            tracker.update(image)
        ms_track = (time.perf_counter() - start) * 1000.0
        contains = None
        if box is not None and anchor is not None:
            contains = bool(box[0] <= anchor[0] + 0.02 and box[1] <= anchor[1] + 0.02
                            and box[2] >= anchor[2] - 0.02 and box[3] >= anchor[3] - 0.02)
        records.append({
            "path": path,
            "name": os.path.basename(path),
            "group": os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)[:11],
            "box": box,
            "share": round(share, 4),
            "text_area": None if anchor is None else round((anchor[2] - anchor[0]) * (anchor[3] - anchor[1]), 4),
            "contains": contains,
            "ms_ocr": round(ms_ocr),
            "ms_sal": round(ms_sal),
            "ms_track": round(ms_track, 1),
        })

    with open("/tmp/bench_u2netp.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    found = [r for r in records if r["box"] is not None]
    with_text = [r for r in found if r["contains"] is not None]
    ok = [r for r in with_text if r["contains"]]
    areas = [(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) for r in found]
    print(f"\nbox found            : {len(found)} / {len(records)}")
    print(f"frames with text     : {len(with_text)}")
    print(f"  box contains text  : {len(ok)} / {len(with_text)}"
          f"  ({100 * len(ok) / max(1, len(with_text)):.0f}%)")
    if areas:
        areas.sort()
        print(f"box area             : min {areas[0]:.3f}  median "
              f"{areas[len(areas) // 2]:.3f}  max {areas[-1]:.3f}")
    for key in ("ms_ocr", "ms_sal", "ms_track"):
        values = sorted(r[key] for r in records)
        print(f"{key:<20} : median {values[len(values) // 2]:.0f}  "
              f"min {values[0]:.0f}  max {values[-1]:.0f}")

    groups: dict[str, list] = {}
    for record in found:
        groups.setdefault(record["group"], []).append(record["box"])
    spreads = []
    for group in groups.values():
        if len(group) < 2:
            continue
        spread = max(
            max(b[i] for b in group) - min(b[i] for b in group) for i in range(4)
        )
        spreads.append(spread)
    if spreads:
        spreads.sort()
        print(f"\nstability within a view (n={len(spreads)} views): "
              f"median max-coordinate spread {spreads[len(spreads) // 2]:.4f}  "
              f"worst {spreads[-1]:.4f}")
    print("records -> /tmp/bench_u2netp.json")


if __name__ == "__main__":
    main()
