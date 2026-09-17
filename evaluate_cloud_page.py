"""Measure the cloud's page box against the local text anchor on real frames.

Plan B rests on one assumption: that locate_page returns a page-shaped box on this
rig. Two examples seen in earlier logs were page-shaped, which is not enough to
build on, so this runs it over a spread of real frames and reports the box, its
area, whether it covers the text the local detector found, and the latency.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from camera_preview import SemanticBookTracker  # noqa: E402
from lamp_core.config import AppConfig, load_dotenv  # noqa: E402
from lamp_core.object_localization import locate_page  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

CENTRED = "/home/lamp/AI-lamp/datasets/book_book-present_20260915-140720"
WIDE = "/home/lamp/AI-lamp/datasets/book_findbook_20260915-144441"
FRAMES = [
    f"{CENTRED}/p01_a+000.0_f00.jpg",
    f"{CENTRED}/p01_a+000.0_f01.jpg",
    f"{CENTRED}/p01_a-010.0_f00.jpg",
    f"{CENTRED}/p01_a-020.0_f00.jpg",
    f"{CENTRED}/p01_a+015.0_f00.jpg",
    f"{WIDE}/p01_a-030.0_f00.jpg",
    f"{WIDE}/p01_a-040.0_f00.jpg",
    f"{WIDE}/p01_a+030.0_f00.jpg",
]


def main() -> None:
    load_dotenv()
    config = AppConfig.from_environment()
    if not config.cloud_enabled or not config.api_key:
        print("cloud vision is not configured", file=sys.stderr)
        return
    from app_main import cloud_client

    client = cloud_client(config)
    detector = PpocrTextDetector(None, long_side=320)
    description = SemanticBookTracker.TARGET_DESCRIPTION

    covers = 0
    both = 0
    areas: list[float] = []
    record: list[dict[str, object]] = []
    print(f"{'frame':<24} {'text area':>9} {'page area':>9} {'ratio':>6} "
          f"{'covers':>6} {'ms':>6}")
    for path in FRAMES:
        image = cv2.imread(path)
        if image is None:
            continue
        height, width = image.shape[:2]
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        start = time.perf_counter()
        reading = locate_page(buffer.tobytes(), client, description)
        elapsed = (time.perf_counter() - start) * 1000.0
        name = path.split("/")[-1]
        anchor = text_anchor_box(detector.detect(image))
        text_area = 0.0
        if anchor is not None:
            text_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])

        vis = image.copy()
        if anchor is not None:
            cv2.rectangle(vis,
                          (int(anchor[0] * width), int(anchor[1] * height)),
                          (int(anchor[2] * width), int(anchor[3] * height)),
                          (0, 165, 255), 2)
        entry: dict[str, object] = {
            "frame": name,
            "text_anchor": list(anchor) if anchor else None,
            "cloud_box": None,
            "visible_text": None,
            "latency_ms": round(elapsed),
        }
        if reading is None or reading.bbox_norm is None:
            print(f"{name:<24} {text_area:9.3f} {'none':>9} {'-':>6} {'-':>6} "
                  f"{elapsed:6.0f}")
            record.append(entry)
            cv2.imwrite("/tmp/cloudpage_" + name, vis)
            continue
        x1, y1, x2, y2 = reading.bbox_norm
        entry["cloud_box"] = [x1, y1, x2, y2]
        entry["visible_text"] = reading.visible_text
        page_area = (x2 - x1) * (y2 - y1)
        areas.append(page_area)
        flag = "-"
        if anchor is not None:
            both += 1
            holds = (x1 <= anchor[0] and y1 <= anchor[1]
                     and x2 >= anchor[2] and y2 >= anchor[3])
            covers += int(holds)
            flag = "yes" if holds else "no"
        cv2.rectangle(vis, (int(x1 * width), int(y1 * height)),
                      (int(x2 * width), int(y2 * height)), (0, 255, 0), 3)
        cv2.putText(vis, "cloud page", (int(x1 * width), max(16, int(y1 * height) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite("/tmp/cloudpage_" + name, vis)
        record.append(entry)
        print(f"{name:<24} {text_area:9.3f} {page_area:9.3f} "
              f"{page_area / max(1e-6, text_area):6.1f} {flag:>6} {elapsed:6.0f}"
              f"   text={reading.visible_text[:34]!r}")
    with open("/tmp/cloudpage_boxes.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print()
    print("boxes also saved to /tmp/cloudpage_boxes.json and /tmp/cloudpage_*.jpg")
    if areas:
        areas.sort()
        print(f"cloud page area: min {areas[0]:.3f}  median "
              f"{areas[len(areas) // 2]:.3f}  max {areas[-1]:.3f}")
    print(f"cloud box covers the local text anchor: {covers} / {both}")


if __name__ == "__main__":
    main()
