"""Can a salient-object model pull the cartoon illustration off the page?

The user's idea, and it targets the right thing: hand-tuned features (colour,
brightness, edges) all failed to express "this is a cartoon", because that is a
*learned* concept. u2netp is a 4.6 MB salient-object detector, small enough for
the Pi, trained to segment the prominent object in a scene.

The question this answers is narrow and measurable: does its mask land on the
cartoon, and is the resulting box (a) page-like in size and (b) more central than
the text block? The masks are written out so they can be looked at.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MODEL = "/tmp/u2netp.onnx"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 320

from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402


def load_net():
    net = cv2.dnn.readNetFromONNX(MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def saliency(net, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    blob = ((resized.astype(np.float32) / 255.0 - MEAN) / STD)
    blob = np.transpose(blob, (2, 0, 1))[None].astype(np.float32)
    net.setInput(blob)
    out = net.forward()
    return np.asarray(out).reshape(SIZE, SIZE)


def largest_blob(mask, minimum=0.004):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    best = None
    best_area = 0
    height, width = mask.shape
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < minimum * height * width:
            continue
        if area > best_area:
            best_area = area
            best = (x / width, y / height, (x + w) / width, (y + h) / height)
    return best, best_area / float(height * width)


def main() -> None:
    net = load_net()
    detector = PpocrTextDetector(None, long_side=320)
    print(f"{'frame':<22} {'peak':>5} {'blob%':>6} {'box area':>8} {'blob box':<26} "
          f"{'text area':>9} {'holds':>5} {'ms':>5}")
    for path in sys.argv[1:]:
        image = cv2.imread(path)
        if image is None:
            print(f"cannot read {path}")
            continue
        name = path.split("/")[-1]
        start = time.perf_counter()
        heat = saliency(net, image)
        elapsed = (time.perf_counter() - start) * 1000.0
        peak = float(heat.max())

        # Normalise for display and threshold at half the peak: u2net output is
        # already sigmoid-like, but the peak varies with how salient the scene is.
        normalised = heat / peak if peak > 0 else heat
        mask = normalised > 0.5
        box, share = largest_blob(mask)

        anchor = text_anchor_box(detector.detect(image))
        text_area = 0.0
        holds = "-"
        if anchor is not None:
            text_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
        if box is not None and anchor is not None:
            holds = "yes" if (box[0] <= anchor[0] + 0.02 and box[1] <= anchor[1] + 0.02
                              and box[2] >= anchor[2] - 0.02
                              and box[3] >= anchor[3] - 0.02) else "no"
        box_area = 0.0 if box is None else (box[2] - box[0]) * (box[3] - box[1])
        shown = "-" if box is None else "[" + ",".join(f"{v:.2f}" for v in box) + "]"
        print(f"{name:<22} {peak:5.2f} {share * 100:5.1f}% {box_area:8.3f} "
              f"{shown:<26} {text_area:9.3f} {holds:>5} {elapsed:5.0f}")

        vis = image.copy()
        height, width = image.shape[:2]
        heat_up = cv2.resize(normalised, (width, height), interpolation=cv2.INTER_LINEAR)
        overlay = np.zeros_like(vis)
        overlay[:, :, 1] = (heat_up * 255).clip(0, 255).astype(np.uint8)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        if anchor is not None:
            cv2.rectangle(vis, (int(anchor[0] * width), int(anchor[1] * height)),
                          (int(anchor[2] * width), int(anchor[3] * height)),
                          (0, 165, 255), 2)
        if box is not None:
            cv2.rectangle(vis, (int(box[0] * width), int(box[1] * height)),
                          (int(box[2] * width), int(box[3] * height)),
                          (0, 255, 0), 3)
        cv2.imwrite("/tmp/sal_" + name, vis)
    print("\noverlays -> /tmp/sal_*.jpg")


if __name__ == "__main__":
    main()
