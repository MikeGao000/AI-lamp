"""One consolidated, self-checking measurement on the current scene.

Everything is measured in a single run so the numbers cannot come from different
conditions: page-anchor quality over the fresh 39-frame sweep, the cost of every
stage, and a repeatability pass that re-runs the model on a subset and compares the
boxes exactly. Derived budgets are printed as explicit arithmetic from the medians.
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

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
MODEL = "/tmp/u2netp.onnx"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 320
REPEAT_FRAMES = 6


def salient(net, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    blob = ((resized.astype(np.float32) / 255.0 - MEAN) / STD)
    blob = np.transpose(blob, (2, 0, 1))[None].astype(np.float32)
    net.setInput(blob)
    heat = np.asarray(net.forward()).reshape(SIZE, SIZE)
    peak = float(heat.max())
    if peak <= 0:
        return None
    mask = (heat / peak) > 0.5
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
    xs = [v for b in boxes for v in (b.bbox[0], b.bbox[2])]
    ys = [v for b in boxes for v in (b.bbox[1], b.bbox[3])]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def inside(outer, inner, slack=0.02):
    return (outer[0] <= inner[0] + slack and outer[1] <= inner[1] + slack
            and outer[2] >= inner[2] - slack and outer[3] >= inner[3] - slack)


def area_of(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def angle_of(name):
    import re
    match = re.search(r"_a([+-]\d+\.\d)_", name)
    return float(match.group(1)) if match else None


def main() -> None:
    net = cv2.dnn.readNetFromONNX(MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    detector = PpocrTextDetector(None, long_side=320)
    paths = sorted(glob.glob(os.path.join(DIRECTORY, "*.jpg")))
    print(f"{len(paths)} frames from {os.path.basename(DIRECTORY)}\n")

    records = []
    for path in paths:
        image = cv2.imread(path)
        start = time.perf_counter()
        boxes = detector.detect(image)
        ms_ocr = (time.perf_counter() - start) * 1000.0
        anchor = text_anchor_box(boxes)
        raw = union(boxes)
        start = time.perf_counter()
        box = salient(net, image)
        ms_sal = (time.perf_counter() - start) * 1000.0
        records.append({
            "name": os.path.basename(path), "angle": angle_of(os.path.basename(path)),
            "box": box, "raw": raw, "anchor": anchor,
            "ms_ocr": ms_ocr, "ms_sal": ms_sal,
        })

    # ---- page-anchor quality
    with_box = [r for r in records if r["box"]]
    areas = sorted(area_of(r["box"]) for r in with_box)
    print("=== page anchor on this scene ===")
    print(f"box found            : {len(with_box)} / {len(records)}")
    print(f"area                 : min {areas[0]:.3f}  p25 {areas[len(areas)//4]:.3f}  "
          f"median {areas[len(areas)//2]:.3f}  max {areas[-1]:.3f}")
    raw_ok = sum(1 for r in with_box if r["raw"] and inside(r["box"], r["raw"]))
    anc_ok = sum(1 for r in with_box if r["anchor"] and inside(r["box"], r["anchor"]))
    n_raw = sum(1 for r in with_box if r["raw"])
    n_anc = sum(1 for r in with_box if r["anchor"])
    print(f"contains raw text    : {raw_ok}/{n_raw}"
          + (f" ({100*raw_ok/n_raw:.0f}%)" if n_raw else ""))
    print(f"contains text anchor : {anc_ok}/{n_anc}"
          + (f" ({100*anc_ok/n_anc:.0f}%)" if n_anc else ""))

    # ---- stability, three shots per angle
    groups: dict[float, list] = {}
    for record in with_box:
        if record["angle"] is not None:
            groups.setdefault(record["angle"], []).append(record)
    print(f"\n=== stability (3 consecutive shots per angle) ===")
    print(f"{'angle':>6} {'area':>6} {'width':>6} {'centre':>7} {'spread':>8}")
    spreads = []
    for angle in sorted(groups):
        boxes = [r["box"] for r in groups[angle]]
        spread = max(max(b[i] for b in boxes) - min(b[i] for b in boxes)
                     for i in range(4))
        spreads.append(spread)
        mean_area = statistics.mean(area_of(b) for b in boxes)
        mean_width = statistics.mean(b[2] - b[0] for b in boxes)
        mean_centre = statistics.mean((b[0] + b[2]) / 2 for b in boxes)
        print(f"{angle:6.1f} {mean_area:6.3f} {mean_width:6.3f} {mean_centre:7.3f} "
              f"{spread:8.4f}")
    spreads.sort()
    print(f"median spread {spreads[len(spreads)//2]:.4f}   worst {spreads[-1]:.4f}")

    # ---- repeatability: identical input must give an identical box
    print(f"\n=== repeatability ({REPEAT_FRAMES} frames re-run) ===")
    mismatches = 0
    for record in records[:REPEAT_FRAMES]:
        image = cv2.imread(os.path.join(DIRECTORY, record["name"]))
        again = salient(net, image)
        same = (again is not None and record["box"] is not None
                and all(abs(a - b) < 1e-9 for a, b in zip(again, record["box"])))
        mismatches += int(not same)
        print(f"  {record['name']:<24} {'identical' if same else 'DIFFERENT'}")
    print(f"mismatches: {mismatches} / {REPEAT_FRAMES}")

    # ---- per-stage cost
    print("\n=== per-stage cost (ms) ===")
    for key, label in (("ms_ocr", "PP-OCR"), ("ms_sal", "u2netp")):
        values = sorted(r[key] for r in records)
        print(f"{label:<8} median {values[len(values)//2]:6.0f}  min {values[0]:6.0f}  "
              f"max {values[-1]:6.0f}")

    image = cv2.imread(os.path.join(DIRECTORY, records[0]["name"]))
    height, width = image.shape[:2]
    box = records[0]["box"] or (0.1, 0.1, 0.6, 0.6)
    x1, y1, x2, y2 = box
    rect = (int(x1 * width), int(y1 * height),
            max(8, int((x2 - x1) * width)), max(8, int((y2 - y1) * height)))
    for name in ("TrackerCSRT_create", "TrackerMIL_create"):
        factory = getattr(cv2, name, None)
        if factory is None:
            continue
        start = time.perf_counter()
        tracker = factory()
        create_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        tracker.init(image, rect)
        init_ms = (time.perf_counter() - start) * 1000.0
        updates = []
        for _ in range(3):
            start = time.perf_counter()
            tracker.update(image)
            updates.append((time.perf_counter() - start) * 1000.0)
        print(f"{name.replace('Tracker','').replace('_create',''):<8} "
              f"create {create_ms:6.1f}  init {init_ms:7.1f}  "
              f"update median {statistics.median(updates):7.1f}")

    captures = []
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        camera.configure(camera.create_video_configuration(
            main={"size": (960, 720), "format": "RGB888"},
            controls={"FrameRate": 10.0}))
        camera.start()
        time.sleep(2.0)
        for _ in range(6):
            start = time.perf_counter()
            camera.capture_array()
            captures.append((time.perf_counter() - start) * 1000.0)
        camera.stop()
        print(f"{'capture':<8} median {statistics.median(captures):6.0f}  "
              f"min {min(captures):6.0f}  max {max(captures):6.0f}")
    except Exception as error:  # noqa: BLE001
        print(f"capture timing skipped: {type(error).__name__}")

    with open("/tmp/bench_confirm.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print("\nrecords -> /tmp/bench_confirm.json")


if __name__ == "__main__":
    main()
