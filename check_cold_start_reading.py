"""Cold start: does reading several sweeps photos beat reading one?

The user's design is two-mode. On a cold start no single frame holds the whole
spread, so the content must come from several photos aggregated; once the page is
centred, one photo is enough. This tests the first half: read the frame at 0 deg,
where the left page is off-frame, against a side-by-side composite of several
angles. No geometric stitching is involved, so there is no parallax problem -- the
tiles are simply placed next to each other and the model is asked to merge them.

Ground truth hint: the spread reads "Nu kan vi begynde med at lave dejen." with a
speech bubble "Vi er allerede i gang."
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lamp_core.config import AppConfig, load_dotenv  # noqa: E402

DIRECTORY = "/home/lamp/AI-lamp/datasets/bench_current"
TILE_WIDTH = 480

INSTRUCTIONS = """You read children's picture book pages from photos of a desk.
Return only one json object, never Markdown or prose.
The image may contain several photos of the same book, taken at different pan angles,
placed side by side and separated by white lines. Read every legible printed word in
all of them and merge the result into one complete reading of the book page, without
repeating words that appear in more than one photo. Return exactly:
{"page_language":"<code>","visible_text":"<every legible printed word, merged>",
"spoken_reading":"<the same content as one natural sentence for reading aloud>"}
If no printed words are legible at all, return exactly: {"visible_text":""}"""

PROMPT = (
    "Read the printed words in this image and reply as json. If it shows the same "
    "book page more than once, merge the words into one complete reading."
)


def load(angle: float):
    for suffix in ("f00", "f01"):
        path = os.path.join(DIRECTORY, f"p01_a{angle:+06.1f}_{suffix}.jpg")
        if os.path.isfile(path):
            image = cv2.imread(path)
            if image is not None:
                return image
    return None


def to_jpeg(image, quality=88):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else None


def composite(images):
    tiles = []
    for image in images:
        height, width = image.shape[:2]
        tiles.append(
            cv2.resize(image, (TILE_WIDTH, int(height * TILE_WIDTH / width)),
                       interpolation=cv2.INTER_AREA)
        )
    height = max(tile.shape[0] for tile in tiles)
    gap = 6
    total = sum(tile.shape[1] for tile in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((height, total, 3), 255, dtype=np.uint8)
    x = 0
    for tile in tiles:
        canvas[:tile.shape[0], x:x + tile.shape[1]] = tile
        x += tile.shape[1] + gap
    return canvas


def main() -> None:
    import app_main
    from lamp_core.object_localization import _json_object

    load_dotenv()
    config = AppConfig.from_environment()
    client = app_main.cloud_client(config)

    def read(jpeg, label):
        response = client.describe_page(
            jpeg, prompt=PROMPT, system_instructions=INSTRUCTIONS
        )
        payload = _json_object(response) if isinstance(response, str) else response
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("visible_text") or "")
        print(f"{label:<34} {len(text):4d} chars  {text[:150]!r}")
        return text

    single = load(0.0)
    if single is not None:
        read(to_jpeg(single), "single frame at 0 deg")

    angles = [-25.0, -15.0, -5.0, 5.0]
    images = [image for image in (load(a) for a in angles) if image is not None]
    if images:
        sheet = composite(images)
        cv2.imwrite("/tmp/read_composite.jpg", sheet)
        print(f"composite {sheet.shape} -> /tmp/read_composite.jpg")
        read(to_jpeg(sheet), f"composite of {len(images)} angles")


if __name__ == "__main__":
    main()
