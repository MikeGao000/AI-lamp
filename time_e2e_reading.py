"""End-to-end timing on the real flow, measured at the moments that matter.

Uses the running service's own snapshot and the page box it is tracking, so the
crop is exactly what the lamp would send. Streaming speech is measured by when
playback actually starts (is_playing), not by when a batch synthesis returns --
the user is right that generation is incremental, so "time to first sound" is the
number that describes what a child experiences.

Never prints the audio payload.
"""

from __future__ import annotations

import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SERVICE = "http://127.0.0.1:8000"


def get(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=10).read()


def main() -> None:
    import json

    import app_main
    from lamp_core.config import AppConfig, load_dotenv
    from lamp_core.reading_prompt import build_picture_book_prompt

    load_dotenv()
    config = AppConfig.from_environment()

    started = time.perf_counter()
    raw_frame = get(f"{SERVICE}/snapshot.jpg")
    alignment = json.loads(get(f"{SERVICE}/alignment.json"))
    t_fetch = time.perf_counter() - started
    box = alignment.get("bbox_norm") or [0.05, 0.10, 0.70, 0.90]
    print(f"frame {len(raw_frame) / 1024:.0f} KB   tracked box "
          f"{[round(v, 3) for v in box]}   fetch {t_fetch * 1000:.0f} ms")

    started = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(raw_frame, np.uint8), cv2.IMREAD_COLOR)
    import camera_preview

    crop = camera_preview.page_crop(image, cv2, tuple(box), margin=0.05)
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    t_crop = time.perf_counter() - started
    jpeg = buffer.tobytes()
    print(f"crop {crop.shape} -> {len(jpeg) / 1024:.0f} KB   prepare {t_crop * 1000:.0f} ms")

    client = app_main.cloud_client(config)
    started = time.perf_counter()
    raw = client.describe_page(
        jpeg,
        prompt=build_picture_book_prompt("Danish"),
        system_instructions=app_main.PICTURE_BOOK_SYSTEM_INSTRUCTIONS,
    )
    t_vision = time.perf_counter() - started
    payload = json.loads(raw) if isinstance(raw, str) else raw
    spoken = ""
    if isinstance(payload, dict):
        for key in ("narration", "spoken_reading", "story", "text", "visible_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                spoken = value.strip()
                break
    print(f"vision {t_vision * 1000:.0f} ms   spoken text {len(spoken)} chars: "
          f"{spoken[:120]!r}")

    speaker = app_main.create_production_speaker(config)
    inner = getattr(speaker, "client", None)
    print(f"speaker {type(speaker).__name__}  inner {type(inner).__name__}")

    marks: dict[str, float] = {}
    started = time.perf_counter()
    done = threading.Event()

    def run() -> None:
        try:
            speaker.speak(spoken)
        finally:
            marks["done"] = time.perf_counter() - started
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while not done.is_set() and time.perf_counter() - started < 90.0:
        if "first_audio" not in marks and getattr(speaker, "is_playing", False):
            marks["first_audio"] = time.perf_counter() - started
        time.sleep(0.02)
    thread.join(timeout=5.0)

    first = marks.get("first_audio")
    total = marks.get("done")
    print()
    print(f"speech: first audio {first * 1000:.0f} ms" if first
          else "speech: playback never reported is_playing")
    if total:
        print(f"speech: finished     {total * 1000:.0f} ms")
    if first:
        print(f"\nEND TO END photo -> first sound : "
              f"{(t_fetch + t_crop + t_vision + first):.2f} s")
    if total:
        print(f"END TO END photo -> narration done: "
              f"{(t_fetch + t_crop + t_vision + total):.2f} s")


if __name__ == "__main__":
    main()
