"""Does local CRNN read the page, and how well does the cloud agree with it?

Two questions with one run: whether the recogniser produces real words on this rig, and
what overlap a *correct* cloud reading scores -- the threshold for refusing a
hallucination has to come from data, not from a guess.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from camera_preview import page_crop  # noqa: E402
from lamp_core.config import AppConfig, load_dotenv  # noqa: E402
from lamp_core.reading_check import transcript_overlap, words  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector  # noqa: E402
from lamp_core.text_recognition import CrnnTextRecognizer  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
FRAMES = ["p01_a-015.0_f00.jpg", "p01_a-020.0_f00.jpg", "p01_a-010.0_f00.jpg"]

with open("/tmp/bench_confirm.json", encoding="utf-8") as handle:
    boxes_by_name = {r["name"]: r["box"] for r in json.load(handle) if r["box"]}

detector = PpocrTextDetector(None, long_side=320)
recognizer = CrnnTextRecognizer(None)
print(f"recogniser available: {recognizer.available}  model {recognizer.model_path}")

load_dotenv()
config = AppConfig.from_environment()
client = None
if config.cloud_enabled and config.api_key:
    from app_main import cloud_client

    client = cloud_client(config)

import app_main  # noqa: E402
from lamp_core.reading_prompt import build_picture_book_prompt  # noqa: E402

for name in FRAMES:
    page_box = boxes_by_name.get(name)
    if page_box is None:
        continue
    image = cv2.imread(os.path.join(DIRECTORY, name))
    if image is None:
        continue
    crop = page_crop(image, cv2, tuple(page_box), margin=0.02)
    text_boxes = detector.detect(crop)

    started = time.perf_counter()
    local = recognizer.transcript(crop, text_boxes)
    local_ms = (time.perf_counter() - started) * 1000.0

    print(f"\n--- {name}   page box {[round(v, 2) for v in page_box]}   "
          f"boxes {len(text_boxes)}")
    print(f"    LOCAL  ({local_ms:6.0f} ms, {len(words(local))} words): {local!r}")

    if client is None:
        continue
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        continue
    raw = client.describe_page(
        buffer.tobytes(),
        prompt=build_picture_book_prompt("Danish"),
        system_instructions=app_main.PICTURE_BOOK_SYSTEM_INSTRUCTIONS,
    )
    payload = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else {}
    cloud = str(payload.get("visible_text") or "")
    print(f"    CLOUD: {cloud!r}")
    print(f"    overlap of local words found in the cloud reading: "
          f"{transcript_overlap(cloud, local):.2f}")
