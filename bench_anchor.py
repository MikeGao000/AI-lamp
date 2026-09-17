"""Benchmark the page anchor on one directory, saving everything needed offline.

Parameterised version of bench_u2netp.py: it records the salient box, the RAW text
union and the padded anchor per frame, plus per-stage timings, so the metrics and
the closed-loop simulation can be recomputed without re-running a 5 s model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", default="/tmp/bench_new.json")
    parser.add_argument("--model", default="/tmp/u2netp.onnx")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import cv2
    import numpy as np

    from lamp_core.text_detection import PpocrTextDetector, text_anchor_box

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    size = 320
    net = cv2.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    detector = PpocrTextDetector(None, long_side=320)

    def salient(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        blob = ((resized.astype(np.float32) / 255.0 - mean) / std)
        blob = np.transpose(blob, (2, 0, 1))[None].astype(np.float32)
        net.setInput(blob)
        heat = np.asarray(net.forward()).reshape(size, size)
        peak = float(heat.max())
        if peak <= 0:
            return None
        mask = (heat / peak) > 0.5
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        best, best_area = None, 0
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if area < 0.004 * size * size:
                continue
            if area > best_area:
                best_area = area
                best = (x / size, y / size, (x + w) / size, (y + h) / size)
        return best

    paths = sorted(glob.glob(os.path.join(args.directory, "*.jpg")))
    records = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        start = time.perf_counter()
        boxes = detector.detect(image)
        ms_ocr = (time.perf_counter() - start) * 1000.0
        anchor = text_anchor_box(boxes)
        xs = [v for b in boxes for v in (b.bbox[0], b.bbox[2])]
        ys = [v for b in boxes for v in (b.bbox[1], b.bbox[3])]
        raw = (min(xs), min(ys), max(xs), max(ys)) if xs else None
        start = time.perf_counter()
        box = salient(image)
        ms_sal = (time.perf_counter() - start) * 1000.0
        records.append({
            "path": path,
            "name": os.path.basename(path),
            "box": box,
            "raw_text": list(raw) if raw else None,
            "anchor": list(anchor) if anchor else None,
            "ms_ocr": round(ms_ocr),
            "ms_sal": round(ms_sal),
        })
        print(f"{os.path.basename(path):<26} "
              f"{'-' if box is None else '[' + ','.join(f'{v:.2f}' for v in box) + ']':<26} "
              f"{ms_ocr:5.0f} {ms_sal:5.0f}")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    sal = [r["ms_sal"] for r in records]
    ocr = [r["ms_ocr"] for r in records]
    sal.sort()
    ocr.sort()
    print(f"\n{len(records)} frames")
    print(f"u2netp  median {sal[len(sal) // 2]:.0f} ms")
    print(f"PP-OCR  median {ocr[len(ocr) // 2]:.0f} ms")
    print(f"records -> {args.output}")


if __name__ == "__main__":
    main()
