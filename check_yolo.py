"""Does COCO YOLO find our book, on the current scene?

Tested once before on a single frame and judged weak (book 0.37, tied with tv and cup),
and the model had since been wiped by a reboot. This reruns it over every current-scene
frame with the zoo's own YOLOX contract: 640x640 input, ImageNet normalisation, grid
decode, batched NMS.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MODEL = "/home/lamp/AI-lamp/models/yolox.onnx"
DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
#: COCO's 80 classes, in order; 73 is 'book'.
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
BOOK = COCO.index("book")
SIZE = 640
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(1, 1, 3)


def anchors(strides=(8, 16, 32)):
    grids, expanded = [], []
    for stride in strides:
        size = SIZE // stride
        xv, yv = np.meshgrid(np.arange(size), np.arange(size))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded.append(np.full((1, grid.shape[1], 1), stride))
    return np.concatenate(grids, 1), np.concatenate(expanded, 1)


def detect(net, grids, strides, image, *, conf=0.25):
    height, width = image.shape[:2]
    resized = cv2.resize(image, (SIZE, SIZE))
    blob = ((resized[:, :, ::-1].astype("float32") / 255.0 - MEAN) / STD)
    blob = np.transpose(blob, (2, 0, 1))[None].astype("float32")
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    dets = np.asarray(outs[0]).reshape(-1, 85).copy()
    dets[:, :2] = (dets[:, :2] + grids) * strides
    dets[:, 2:4] = np.exp(dets[:, 2:4]) * strides
    boxes = dets[:, :4]
    xywh = np.ones_like(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 2]
    xywh[:, 3] = boxes[:, 3]
    scores = dets[:, 4:5] * dets[:, 5:]
    best = np.amax(scores, axis=1)
    index = np.argmax(scores, axis=1)
    keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), best.tolist(), index.tolist(), conf, 0.5)
    out = []
    for position in np.asarray(keep).reshape(-1):
        x, y, w, h = xywh[position]
        out.append((
            COCO[index[position]] if index[position] < len(COCO) else "?",
            float(best[position]),
            (x / SIZE, y / SIZE, (x + w) / SIZE, (y + h) / SIZE),
        ))
    return out


def main() -> None:
    net = cv2.dnn.readNet(MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    grids, strides = anchors()

    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(DIRECTORY, "*.jpg")))
    hits = 0
    book_scores = []
    print(f"{'frame':<26} {'best book':>9} {'book box':<28} other classes")
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        found = detect(net, grids, strides, image)
        book = [d for d in found if d[0] == "book"]
        others = [f"{d[0]}:{d[1]:.2f}" for d in found if d[0] != "book"][:4]
        name = os.path.basename(path)
        if book:
            hits += 1
            best = max(book, key=lambda d: d[1])
            book_scores.append(best[1])
            show = ",".join(f"{v:.2f}" for v in best[2])
            print(f"{name:<26} {best[1]:9.2f} {show:<28} {' '.join(others)}")
        else:
            print(f"{name:<26} {'none':>9} {'-':<28} {' '.join(others)}")
    print(f"\nbook detected on {hits} / {len(paths)} frames")
    if book_scores:
        book_scores.sort()
        print(f"book score: min {book_scores[0]:.2f}  "
              f"median {book_scores[len(book_scores) // 2]:.2f}  "
              f"max {book_scores[-1]:.2f}")


if __name__ == "__main__":
    main()
