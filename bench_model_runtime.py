"""Is cv2.dnn the slow way to run the page model?

Measured on this Pi, u2netp takes 5.5 s for one 320x320 forward pass, which is about
0.24 GFLOP/s -- low for four Cortex-A53 cores, and it suggests the OpenCV importer is
not using efficient kernels for this graph. onnxruntime is already installed here, so
this times the same model, same input, both runtimes.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

MODEL = "/tmp/u2netp.onnx"
FRAME = "/home/lamp/AI-lamp/datasets/bench_current/p01_a-015.0_f00.jpg"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 320

image = cv2.imread(FRAME)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
blob = ((resized.astype(np.float32) / 255.0 - MEAN) / STD)
blob = np.transpose(blob, (2, 0, 1))[None].astype(np.float32)

net = cv2.dnn.readNetFromONNX(MODEL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setInput(blob)
times = []
for _ in range(3):
    start = time.perf_counter()
    out = net.forward()
    times.append((time.perf_counter() - start) * 1000.0)
times.sort()
print(f"cv2.dnn      median {times[1]:7.0f} ms   (all {[round(t) for t in times]})")
cv_heat = np.asarray(out).reshape(SIZE, SIZE)

try:
    import onnxruntime as ort

    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    session.run(None, {name: blob})          # warm up
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = session.run(None, {name: blob})
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    print(f"onnxruntime  median {times[1]:7.0f} ms   (all {[round(t) for t in times]})")
    ort_heat = np.asarray(result[0]).reshape(SIZE, SIZE)
    print(f"max |cv2.dnn - onnxruntime| = {np.abs(cv_heat - ort_heat).max():.5f}")
except Exception as error:  # noqa: BLE001
    print(f"onnxruntime run failed: {type(error).__name__}: {error}")
