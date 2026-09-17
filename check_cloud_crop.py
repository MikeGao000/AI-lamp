"""Two checks for the "send the cloud a clear page photo" goal.

1. Does the relative sharpness gate reject a blurred frame and accept a sharp one?
   No absolute threshold is used, so this must hold without knowing the scene.
2. Does handing the cloud a tight crop of the page read better than handing it the
   whole desk photo? The cloud reads text well but localises badly (measured: its
   box centre was the image centre every time), so this decides whether the crop
   should be what gets sent.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from camera_preview import SharpnessGate, page_crop, sharpness  # noqa: E402
from lamp_core.config import AppConfig, load_dotenv  # noqa: E402
from lamp_core.object_localization import locate_page  # noqa: E402
from camera_preview import SemanticBookTracker  # noqa: E402

with open("/tmp/bench_confirm.json", encoding="utf-8") as handle:
    records = [r for r in json.load(handle) if r["box"] is not None]
DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
for record in records:
    record["path"] = os.path.join(DIRECTORY, record["name"])

print("=== sharpness gate (relative, no constant) ===")
gate = SharpnessGate(fraction=0.5)
for record in records[:6]:
    image = cv2.imread(record["path"])
    sharp = sharpness(image, cv2)
    blurred = cv2.GaussianBlur(image, (31, 31), 0)
    soft = sharpness(blurred, cv2)
    print(f"  {record['name']:<24} sharp {sharp:8.1f} -> gate "
          f"{'PASS' if gate.observe(sharp) else 'REJECT'}   "
          f"blurred {soft:8.1f} -> "
          f"{'PASS' if gate.observe(soft) else 'REJECT'}  "
          f"(reference {gate.reference:.1f})")

print("\n=== cloud reading: whole frame vs page crop ===")
load_dotenv()
config = AppConfig.from_environment()
if not config.cloud_enabled or not config.api_key:
    print("cloud not configured")
    raise SystemExit(0)
from app_main import cloud_client

client = cloud_client(config)
description = SemanticBookTracker.TARGET_DESCRIPTION
PICK = ["p01_a-015.0_f00.jpg", "p01_a-020.0_f00.jpg", "p01_a-010.0_f00.jpg"]
by_name = {r["name"]: r for r in records}
for name in PICK:
    record = by_name.get(name)
    if record is None:
        continue
    image = cv2.imread(record["path"])
    crop = page_crop(image, cv2, tuple(record["box"]), margin=0.05)
    for label, picture in (("full frame", image), ("page crop", crop)):
        ok, buffer = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            continue
        reading = locate_page(buffer.tobytes(), client, description)
        text = "" if reading is None else reading.visible_text
        print(f"  {name:<22} {label:<11} {len(buffer)/1024:5.1f} KB  "
              f"chars {len(text):3d}  {text[:52]!r}")
