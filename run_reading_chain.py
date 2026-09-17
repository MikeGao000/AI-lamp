"""Whole reading chain on the page the lamp is looking at right now.

Snapshot (with the box it is tracking) -> page crop -> the project's own production
reading path (cloud vision + TTS) -> WAV. Uses app_main_test's substitution runner so
the flow under measurement is the product's, not a reimplementation.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import app_main_test  # noqa: E402
from camera_preview import page_crop  # noqa: E402
from lamp_core.config import AppConfig, load_dotenv  # noqa: E402

SERVICE = "http://127.0.0.1:8000"
OUT = Path("/tmp/reading_chain")


def fetch(path: str) -> bytes:
    return urllib.request.urlopen(f"{SERVICE}{path}", timeout=15).read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw = fetch("/snapshot.jpg")
    alignment = json.loads(fetch("/alignment.json"))
    box = alignment.get("bbox_norm") or [0, 0, 0, 0]
    width_ratio = box[2] - box[0]
    if width_ratio < 0.05:
        print(f"nothing usable is being tracked: box {box}", file=sys.stderr)
        return 1
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    crop = page_crop(image, cv2, tuple(box), margin=0.05)
    page_path = OUT / "page.jpg"
    cv2.imwrite(str(page_path), crop)
    prepared = time.perf_counter() - started
    print(f"tracked box {[round(v, 3) for v in box]}  crop {crop.shape}  "
          f"{page_path} ({page_path.stat().st_size // 1024} KB)  prepared {prepared:.2f}s")

    load_dotenv()
    config = AppConfig.from_environment()
    wav_path = OUT / "reading.wav"
    result_path = OUT / "reading.json"
    started = time.perf_counter()
    app_main_test.run_image_hardware_substitution_test(
        config, page_path, wav_path, result_path
    )
    chain = time.perf_counter() - started

    print(f"\nchain (cloud vision + TTS) {chain:.2f} s")
    if wav_path.is_file():
        with wave.open(str(wav_path)) as handle:
            duration = handle.getnframes() / max(1, handle.getframerate())
            print(f"WAV {wav_path}  {wav_path.stat().st_size // 1024} KB  "
                  f"{duration:.1f} s  {handle.getframerate()} Hz  "
                  f"{handle.getnchannels()} ch")
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"recognized page: {payload.get('recognized_page')!r}")
        print(f"text sent to TTS: {payload.get('text_sent_to_tts')!r}")
        print(f"timing recorded by the app: {payload.get('timing_s')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
