"""Camera adapters and a deliberately simple, testable stillness gate."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass
class StillnessGate:
    """Declare a page stable only after continuous low-motion observations."""

    required_seconds: float = 1.5
    threshold: float = 8.0
    still_since: float | None = None

    def observe(self, motion_score: float, now: float | None = None) -> bool:
        now = monotonic() if now is None else now
        if motion_score > self.threshold:
            self.still_since = None
            return False
        if self.still_since is None:
            self.still_since = now
            return False
        return now - self.still_since >= self.required_seconds

    def reset(self) -> None:
        self.still_since = None


class Picamera2FrameSource:
    """Optional Raspberry Pi CSI-camera adapter; imported only on the Pi."""

    def __init__(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("Picamera2 is required on Raspberry Pi; install python3-picamera2") from error
        self._camera = Picamera2()
        self._camera.configure(self._camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
        self._camera.start()
        self._previous_gray = None

    def capture_jpeg_and_motion(self) -> tuple[bytes, float]:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for the motion gate") from error
        frame = self._camera.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        motion = 0.0 if self._previous_gray is None else float(cv2.absdiff(gray, self._previous_gray).mean())
        self._previous_gray = gray
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("camera JPEG encoding failed")
        return encoded.tobytes(), motion

    def close(self) -> None:
        self._camera.stop()
