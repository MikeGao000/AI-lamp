"""Time the reading chain: page photo -> cloud vision -> text -> speech audio.

Stages are timed separately so the total can be attributed. The script prints the
public methods of whatever the speech factory returns before calling it, because
guessing a method name is how several of today's mistakes started.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

import cv2  # noqa: E402

from lamp_core.config import AppConfig, load_dotenv  # noqa: E402

FRAME = "/home/lamp/AI-lamp/datasets/bench_current/p01_a-015.0_f00.jpg"
BOX = (0.043, 0.10, 0.72, 0.88)   # page box measured for this frame
OUT = "/tmp/reading_chain.wav"


def main() -> None:
    load_dotenv()
    config = AppConfig.from_environment()
    print(f"cloud_enabled={config.cloud_enabled} tts_voice={getattr(config, 'tts_voice', '?')}")

    import app_main
    from lamp_core.reading_prompt import build_picture_book_prompt

    client = app_main.cloud_client(config)

    # ---- stage 0: the page crop that the loop would hand over
    started = time.perf_counter()
    image = cv2.imread(FRAME)
    crop = app_main.__dict__.get("page_crop")
    if crop is None:
        import camera_preview
        crop = camera_preview.page_crop
    picture = crop(image, cv2, BOX, margin=0.05)
    ok, buffer = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    jpeg = buffer.tobytes()
    print(f"stage 0 crop+encode   {(time.perf_counter() - started) * 1000:7.0f} ms  "
          f"{len(jpeg) / 1024:6.1f} KB  shape {picture.shape}")

    # ---- stage 1: cloud vision -> story text
    started = time.perf_counter()
    raw = client.describe_page(
        jpeg,
        prompt=build_picture_book_prompt("Danish"),
        system_instructions=app_main.PICTURE_BOOK_SYSTEM_INSTRUCTIONS,
    )
    vision_ms = (time.perf_counter() - started) * 1000.0
    print(f"stage 1 cloud vision  {vision_ms:7.0f} ms")
    print(f"        raw type {type(raw).__name__}, {len(raw) if hasattr(raw, '__len__') else '?'} chars")
    print(f"        {' '.join(str(raw).split())[:220]}")

    payload = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else raw
    spoken = None
    if isinstance(payload, dict):
        for key in ("story", "text", "story_text", "page_text"):
            if isinstance(payload.get(key), str) and payload[key].strip():
                spoken = payload[key].strip()
                break
        if spoken is None:
            spoken = app_main.story_extension_for_tts(payload)
    else:
        spoken = str(payload)
    print(f"stage 1b spoken text   {len(spoken)} chars: {spoken[:160]!r}")

    # ---- stage 2: speech audio
    speaker = app_main.create_production_speaker(config)
    print(f"stage 2 speaker {type(speaker).__name__}; public API: "
          f"{[m for m in dir(speaker) if not m.startswith('_')][:12]}")
    started = time.perf_counter()
    produced = None
    used = None
    for name in ("synthesize", "write", "speak", "say", "play"):
        method = getattr(speaker, name, None)
        if method is None:
            continue
        try:
            produced = method(spoken)
            used = name
            print(f"        called {name}()")
            break
        except Exception as error:  # noqa: BLE001
            print(f"        {name}() failed: {type(error).__name__}: {error}")
    speech_ms = (time.perf_counter() - started) * 1000.0
    print(f"stage 2 speech        {speech_ms:7.0f} ms   via {used}()  returned {produced!r}")
    # speak() may include playback time; time it separately now that any cache is warm.
    play_ms = None
    if used != "speak" and hasattr(speaker, "speak"):
        started = time.perf_counter()
        try:
            speaker.speak(spoken)
            play_ms = (time.perf_counter() - started) * 1000.0
            print(f"stage 2b speak()      {play_ms:7.0f} ms  (synthesis + playback)")
        except Exception as error:  # noqa: BLE001
            print(f"stage 2b speak() failed: {type(error).__name__}: {error}")
    print(f"\nTOTAL (vision + generation) {vision_ms + speech_ms:7.0f} ms")
    if play_ms is not None:
        print(f"TOTAL (vision + playback)   {vision_ms + play_ms:7.0f} ms")
    print(f"audio file: {OUT} exists={os.path.exists(OUT)} "
          f"size={os.path.getsize(OUT) if os.path.exists(OUT) else 0}")


if __name__ == "__main__":
    main()
