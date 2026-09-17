"""Two cheap questions that decide what to do next.

1. Is the cloud better at *pointing* at the book than at boxing it? A VLM that
   returns a wandering box may still return a stable point, and J1 only needs a
   horizontal coordinate. Measured by calling each prompt twice per frame and
   comparing the spread.
2. Is the page actually *brighter* than the desk under this lamp? Every local
   method has failed because page and desk look the same, so if there is a real
   luminance difference a plain threshold could segment the page with no model at
   all. Measured directly, no cloud calls.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from camera_preview import SemanticBookTracker  # noqa: E402
from lamp_core.config import AppConfig, load_dotenv  # noqa: E402
from lamp_core.object_localization import _json_object, locate_page  # noqa: E402
from lamp_core.text_detection import PpocrTextDetector, text_anchor_box  # noqa: E402

CENTRED = "/home/lamp/AI-lamp/datasets/book_book-present_20260915-140720"
FRAMES = [
    f"{CENTRED}/p01_a+000.0_f00.jpg",
    f"{CENTRED}/p01_a-010.0_f00.jpg",
    f"{CENTRED}/p01_a+015.0_f00.jpg",
]

POINT_INSTRUCTIONS = """You point at one children's picture book page in a single photo.
Return only one json object. Never include Markdown or explanatory prose.
If any portion of an open children's picture book page is visible, return exactly:
{"found":true,"centre_norm":[x,y],"confidence":0.0}
centre_norm is one point in normalized coordinates, x growing left to right and y growing top to
bottom, at the centre of the visible book page area. If no picture book page is visible, return
exactly: {"found":false}
Exclude computer screens, product packaging, loose unrelated papers, cables, motors, furniture
and background objects."""

POINT_PROMPT = (
    "Point at the centre of the children's picture book page in this photo. "
    "Reply with the point as json."
)


def ask_point(client, jpeg):
    response = client.describe_page(
        jpeg, prompt=POINT_PROMPT, system_instructions=POINT_INSTRUCTIONS
    )
    payload = _json_object(response)
    if payload.get("found") is not True:
        return None
    point = payload.get("centre_norm")
    if not isinstance(point, list) or len(point) != 2:
        return None
    return float(point[0]), float(point[1])


def luminance(image, anchor):
    """Mean luminance of the paper around the text, versus the desk below it."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    if anchor is None:
        paper = float("nan")
    else:
        pad_x = (anchor[2] - anchor[0]) * 0.25
        pad_y = (anchor[3] - anchor[1]) * 0.25
        x1 = max(0, int((anchor[0] - pad_x) * width))
        x2 = min(width, int((anchor[2] + pad_x) * width))
        y1 = max(0, int((anchor[1] - pad_y) * height))
        y2 = min(height, int((anchor[3] + pad_y) * height))
        band = gray[y1:y2, x1:x2].reshape(-1)
        # Keep the brighter part of the sample: text strokes are ink, and this is
        # meant to estimate the paper they are printed on.
        paper_pixels = band[band >= np.percentile(band, 60)] if band.size else band
        paper = float(paper_pixels.mean()) if paper_pixels.size else float("nan")
    desk_band = gray[int(height * 0.92):, :]
    desk = float(desk_band.mean())
    otsu, _ = cv2.threshold(gray.astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    above = float((gray > otsu).mean())
    return paper, desk, float(otsu), above


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

    print("=== local: is the page brighter than the desk? ===")
    print(f"{'frame':<24} {'paper':>7} {'desk':>7} {'ratio':>6} {'otsu':>6} {'above':>6}")
    for path in FRAMES:
        image = cv2.imread(path)
        anchor = text_anchor_box(detector.detect(image))
        paper, desk, otsu, above = luminance(image, anchor)
        name = path.split("/")[-1]
        print(f"{name:<24} {paper:7.1f} {desk:7.1f} {paper / desk:6.3f} "
              f"{otsu:6.0f} {above * 100:5.1f}%")

    print()
    print("=== cloud: box vs point repeatability (2 calls each) ===")
    for path in FRAMES:
        image = cv2.imread(path)
        if image is None:
            continue
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        jpeg = buffer.tobytes()
        name = path.split("/")[-1]

        centres = []
        areas = []
        for _ in range(2):
            reading = locate_page(jpeg, client, description)
            if reading is None:
                continue
            x1, y1, x2, y2 = reading.bbox_norm
            centres.append(((x1 + x2) / 2, (y1 + y2) / 2))
            areas.append((x2 - x1) * (y2 - y1))
        points = [p for p in (ask_point(client, jpeg) for _ in range(2)) if p]

        print(f"--- {name}")
        if areas:
            print(f"    box  areas   : {[round(a, 3) for a in areas]}"
                  f"   spread {max(areas) - min(areas):.3f}")
            print(f"    box  centres : {[(round(c[0], 3), round(c[1], 3)) for c in centres]}")
            if len(centres) == 2:
                print(f"    box  centre drift: "
                      f"dx {abs(centres[0][0] - centres[1][0]):.3f}  "
                      f"dy {abs(centres[0][1] - centres[1][1]):.3f}")
        else:
            print("    box  : none returned")
        if points:
            print(f"    point centres: {[(round(p[0], 3), round(p[1], 3)) for p in points]}")
            if len(points) == 2:
                print(f"    point drift  : dx {abs(points[0][0] - points[1][0]):.3f}  "
                      f"dy {abs(points[0][1] - points[1][1]):.3f}")
        else:
            print("    point: none returned")


if __name__ == "__main__":
    main()
