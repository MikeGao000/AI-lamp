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
    """Pi camera with a low-resolution tracking stream and high-resolution stills."""

    def __init__(
        self,
        *,
        tracking_size: tuple[int, int] = (320, 240),
        still_size: tuple[int, int] = (1600, 1200),
        frame_rate: float = 12.0,
    ) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("Picamera2 is required on Raspberry Pi; install python3-picamera2") from error
        self._camera = Picamera2()
        self._tracking_width, self._tracking_height = tracking_size
        self._camera.configure(
            self._camera.create_video_configuration(
                main={"size": still_size, "format": "RGB888"},
                lores={"size": tracking_size, "format": "YUV420"},
                controls={"FrameRate": frame_rate},
                buffer_count=3,
            )
        )
        self._camera.start()
        self._previous_gray = None

    def capture_jpeg_and_motion(self) -> tuple[bytes, float]:
        """Return one inexpensive tracking JPEG and its frame-to-frame motion."""

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for the motion gate") from error
        yuv = self._camera.capture_array("lores")
        gray = yuv[: self._tracking_height, : self._tracking_width]
        motion = 0.0 if self._previous_gray is None else float(cv2.absdiff(gray, self._previous_gray).mean())
        self._previous_gray = gray.copy()
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("camera JPEG encoding failed")
        return encoded.tobytes(), motion

    def capture_high_resolution_jpeg(self) -> bytes:
        """Capture and encode the configured high-resolution main stream once."""

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for camera JPEG encoding") from error
        # Picamera2 RGB888 is already laid out in OpenCV's expected BGR order.
        # An extra RGB->BGR conversion would swap red and blue.
        frame = self._camera.capture_array("main")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("high-resolution camera JPEG encoding failed")
        return encoded.tobytes()

    def close(self) -> None:
        self._camera.stop()
