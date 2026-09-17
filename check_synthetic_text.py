"""Is the CRNN pipeline right, or is the imagery the problem?

Render known text into an image with OpenCV, feed it through the same path, and see
whether it reads back. If synthetic text fails too, the charset or preprocessing is
wrong; if it succeeds, the photos are the limit.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lamp_core.text_recognition import CHARSET_CH_94, CrnnTextRecognizer  # noqa: E402

recognizer = CrnnTextRecognizer(None)
recognizer._load()
print(f"charset size {len(recognizer._charset)}  first {recognizer._charset[:12]}")

for text in ("hello", "abc", "kagen", "Plet"):
    canvas = np.full((80, 420, 3), 255, dtype="uint8")
    cv2.putText(canvas, text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3,
                cv2.LINE_AA)
    read = recognizer.read_crop(canvas)
    print(f"  rendered {text!r:<10} -> {read!r}")

# The zoo's own charset order, printed so it can be compared with the header file.
print("charset[10:36] =", "".join(CHARSET_CH_94[10:36]))
print("charset[36:62] =", "".join(CHARSET_CH_94[36:62]))
print("charset[62:]   =", "".join(CHARSET_CH_94[62:]))
