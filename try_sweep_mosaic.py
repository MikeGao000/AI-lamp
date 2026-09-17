"""Sweep photos as one wide image: can the cloud say where the book is?

No single frame contains the whole book, so asking the cloud about one frame was
always asking about a cropped view. The sweep already has photos at known angles,
so their union does contain the whole book, and each horizontal position maps back
to a motor angle through the measured 44.5 degrees per frame width.

Two variants, because the cloud is good at judging and bad at precise boxes:
  A stitched panorama, asking for a bounding box
  B contact sheet of angle-labelled tiles, asking which tile is best
Ground truth: in this sweep the page is most fully visible at -15 deg.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lamp_core.config import AppConfig, load_dotenv  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
DEGREES_PER_FRAME_WIDTH = 44.5
TILE_WIDTH = 640
GROUND_TRUTH_ANGLE = -15.0

MOSAIC_PROMPT = (
    "This image is a panorama stitched by panning a camera across a desk. Find the "
    "children's picture book and reply with json only: "
    '{"found":true,"bbox_norm":[x1,y1,x2,y2],"confidence":0.0} where the box covers '
    "every visible pixel of the book's pages. If no picture book is visible reply "
    '{"found":false}. Exclude computer screens, packaging, cables and loose paper.'
)

SHEET_INSTRUCTIONS = """You judge photos of a desk taken at different pan angles.
Return only one json object, never Markdown or prose.
The image is a grid of numbered tiles, read left to right then top to bottom, each tile
labelled with its pan angle. If a children's picture book is visible in any tile, return
exactly: {"found":true,"best_tile":<tile number>,"confidence":0.0}
best_tile is the tile where the book's page is most fully and clearly visible, with the
page inside the frame rather than cut off at an edge. Exclude computer screens, product
packaging, loose unrelated papers, cables and furniture."""

SHEET_PROMPT = (
    "Which numbered tile shows the children's picture book page most completely? "
    "Reply as json."
)


def angle_of(name: str) -> float | None:
    match = re.search(r"_a([+-]\d+\.\d)_", name)
    return float(match.group(1)) if match else None


def load_frames():
    frames = []
    for path in sorted(glob.glob(os.path.join(DIRECTORY, "*.jpg"))):
        angle = angle_of(os.path.basename(path))
        if angle is None:
            continue
        image = cv2.imread(path)
        if image is not None:
            frames.append((angle, image))
    frames.sort(key=lambda item: item[0])
    # One frame per angle keeps the panorama linear and the sheet small.
    unique: dict[float, object] = {}
    for angle, image in frames:
        unique.setdefault(angle, image)
    return sorted(unique.items())


def build_mosaic(pairs):
    height, width = pairs[0][1].shape[:2]
    scale = TILE_WIDTH / width
    tile_h = int(height * scale)
    pixels_per_degree = TILE_WIDTH / DEGREES_PER_FRAME_WIDTH
    offsets = {angle: int(round(angle * pixels_per_degree)) for angle, _ in pairs}
    left = min(offsets.values())
    right = max(offsets.values()) + TILE_WIDTH
    canvas_w = right - left
    # Feather every tile instead of overwriting. With 5 deg steps the offset is only
    # ~7 px per degree, so writing tiles in order left only a 36 px sliver of each
    # frame and buried the book entirely -- the model rightly answered "no book".
    # Far-field content aligns at the offset and stays sharp; near-field content
    # (the book) smears into a band, and that band is itself the localisation.
    ramp = np.hanning(TILE_WIDTH).astype(np.float32)
    ramp = np.maximum(ramp, 1e-3)
    accum = np.zeros((tile_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((tile_h, canvas_w), dtype=np.float32)
    for angle, image in pairs:
        tile = cv2.resize(image, (TILE_WIDTH, tile_h), interpolation=cv2.INTER_AREA)
        tile = tile.astype(np.float32)
        x = offsets[angle] - left
        accum[:, x:x + TILE_WIDTH] += tile * ramp[None, :, None]
        weight[:, x:x + TILE_WIDTH] += ramp[None, :]
    canvas = (accum / np.maximum(weight, 1e-6)[:, :, None]).clip(0, 255)
    canvas = canvas.astype(np.uint8)
    for angle, _ in pairs:
        x = offsets[angle] - left
        cv2.line(canvas, (x, 0), (x, tile_h), (0, 0, 255), 1)
        cv2.putText(canvas, f"{angle:+.0f}", (x + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas, offsets, left


def build_sheet(pairs, columns=5):
    tile_h = int(pairs[0][1].shape[0] * TILE_WIDTH / pairs[0][1].shape[1])
    rows = (len(pairs) + columns - 1) // columns
    sheet = np.zeros((rows * tile_h, columns * TILE_WIDTH, 3), dtype=np.uint8)
    index = 0
    for row in range(rows):
        for column in range(columns):
            if index >= len(pairs):
                break
            angle, image = pairs[index]
            tile = cv2.resize(image, (TILE_WIDTH, tile_h), interpolation=cv2.INTER_AREA)
            y, x = row * tile_h, column * TILE_WIDTH
            sheet[y:y + tile_h, x:x + TILE_WIDTH] = tile
            cv2.rectangle(sheet, (x, y), (x + TILE_WIDTH - 1, y + tile_h - 1),
                          (255, 255, 255), 2)
            cv2.putText(sheet, f"#{index + 1} {angle:+.0f}", (x + 8, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            index += 1
    return sheet


def main() -> None:
    import app_main
    from lamp_core.object_localization import _json_object

    load_dotenv()
    config = AppConfig.from_environment()
    client = app_main.cloud_client(config)
    pairs = load_frames()
    print(f"{len(pairs)} angles: {[a for a, _ in pairs]}")

    mosaic, offsets, left = build_mosaic(pairs)
    cv2.imwrite("/tmp/mosaic.jpg", mosaic)
    print(f"mosaic {mosaic.shape} -> /tmp/mosaic.jpg")
    ok, buffer = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, 86])
    response = client.describe_page(buffer.tobytes(), prompt=MOSAIC_PROMPT)
    payload = _json_object(response) if isinstance(response, str) else response
    print(f"A mosaic answer: {payload}")
    if isinstance(payload, dict) and payload.get("found") is True:
        box = payload.get("bbox_norm")
        if isinstance(box, list) and len(box) == 4:
            centre_px = (float(box[0]) + float(box[2])) / 2 * mosaic.shape[1]
            degree = (centre_px + left) / (TILE_WIDTH / DEGREES_PER_FRAME_WIDTH)
            print(f"   -> centre maps to {degree:+.1f} deg "
                  f"(ground truth {GROUND_TRUTH_ANGLE:+.1f})")

    sheet = build_sheet(pairs)
    cv2.imwrite("/tmp/sheet.jpg", sheet)
    print(f"sheet {sheet.shape} -> /tmp/sheet.jpg")
    ok, buffer = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
    response = client.describe_page(
        buffer.tobytes(), prompt=SHEET_PROMPT, system_instructions=SHEET_INSTRUCTIONS
    )
    payload = _json_object(response) if isinstance(response, str) else response
    print(f"B sheet answer: {payload}")
    if isinstance(payload, dict) and payload.get("found") is True:
        tile = payload.get("best_tile")
        if isinstance(tile, int) and 1 <= tile <= len(pairs):
            print(f"   -> tile {tile} is angle {pairs[tile - 1][0]:+.1f} "
                  f"(ground truth {GROUND_TRUTH_ANGLE:+.1f})")


if __name__ == "__main__":
    main()
