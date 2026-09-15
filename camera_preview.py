"""Serve the Raspberry Pi camera as a small MJPEG preview web page.

Run this on the Pi, then open ``http://<pi-ip>:8000`` on another device on
the same local network. Picamera2 is imported only when capture starts.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lamp_core.alignment import (
    HorizontalDirection,
    ResponsiveJ1Trajectory,
    mapped_j1_tracking_degrees,
    observe_bbox,
    profiled_motor_speed_rpm,
    smooth_bounded_scan_degrees,
)


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Lamp Camera</title>
  <style>
    body{margin:0;background:#151515;color:#fff;font-family:system-ui,sans-serif;text-align:center}
    main{max-width:1100px;margin:auto;padding:16px} h1{font-size:1.25rem}
    img{display:block;width:100%;height:auto;background:#000;border-radius:12px}
    p{color:#bbb} a{color:#9dd7ff}
    button{border:0;border-radius:8px;padding:10px 18px;background:#b83232;color:white;font-weight:700;cursor:pointer}
  </style>
</head>
<body><main>
  <h1>AI Lamp &#23454;&#26102;&#25668;&#20687;&#22836;&#23545;&#40784;&#25351;&#31034;</h1>
  <img src="/stream.mjpg" alt="camera live preview">
  <p id="status">Waiting for camera...</p>
  <p><a href="/snapshot.jpg" target="_blank">&#25171;&#24320;&#24403;&#21069;&#21333;&#24103;&#29031;&#29255;</a></p>
  <p><button id="stop">&#20572;&#27490;&#25668;&#20687;&#22836;&#21644; J1 &#36319;&#38543;</button></p>
  <script>
    setInterval(async()=>{try{const r=await fetch('/alignment.json',{cache:'no-store'});
      const s=await r.json();document.getElementById('status').textContent=s.message;}catch(e){}},300);
    document.getElementById('stop').onclick=async()=>{
      if(!confirm('Stop camera preview and J1 tracking?'))return;
      await fetch('/shutdown',{method:'POST'});
      document.getElementById('status').textContent='Program stopped. The motor holds its last position.';
      document.getElementById('stop').disabled=true;
    };
  </script>
</main></body></html>""".encode("utf-8")


def detect_document_bbox(image: object, cv2: object) -> tuple[int, int, int, int] | None:
    """Find the strongest page/box-like region without a cloud request."""

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    bright_regions = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, region_kernel)
    contours = []
    for mask in (closed_edges, bright_regions):
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    image_area = width * height
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        area_ratio = box_area / image_area
        aspect = box_width / max(1, box_height)
        touched_edges = sum(
            (
                x <= 2,
                y <= 2,
                x + box_width >= width - 2,
                y + box_height >= height - 2,
            )
        )
        if not 0.06 <= area_ratio <= 0.92 or not 0.30 <= aspect <= 3.3:
            continue
        if touched_edges >= 2:
            continue
        contour_area = max(1.0, float(cv2.contourArea(contour)))
        rectangularity = min(1.0, contour_area / box_area)
        # Pages and packages are usually large, portrait/landscape rectangles.
        # Rectangularity helps but is not mandatory because hands and lamp bars
        # can interrupt an otherwise valid boundary.
        candidates.append((area_ratio * (0.75 + rectangularity), (x, y, box_width, box_height)))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def detect_text_bbox(image: object, cv2: object) -> tuple[int, int, int, int] | None:
    """Group horizontal stroke-rich regions; text outranks generic shapes."""

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient = cv2.convertScaleAbs(gradient)
    _, strokes = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, line_kernel)
    contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_ratio = box_width * box_height / (width * height)
        if (
            box_width >= max(18, round(width * 0.025))
            and 3 <= box_height <= round(height * 0.16)
            and box_width / max(1, box_height) >= 1.25
            and 0.00015 <= area_ratio <= 0.12
        ):
            boxes.append((x, y, box_width, box_height))
    if not boxes:
        return None

    def interval_gap(a1: int, a2: int, b1: int, b2: int) -> int:
        return max(0, max(a1, b1) - min(a2, b2))

    # Keep unrelated text in the room (monitor, loose papers, labels) from
    # becoming one frame-wide target. Connected text lines form one cluster.
    parent = list(range(len(boxes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(boxes)):
        ax, ay, aw, ah = boxes[first]
        for second in range(first + 1, len(boxes)):
            bx, by, bw, bh = boxes[second]
            x_gap = interval_gap(ax, ax + aw, bx, bx + bw)
            y_gap = interval_gap(ay, ay + ah, by, by + bh)
            if x_gap <= width * 0.07 and y_gap <= height * 0.10:
                join(first, second)

    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for index, box in enumerate(boxes):
        groups.setdefault(root(index), []).append(box)
    selected = max(
        groups.values(),
        key=lambda group: sum(box[2] * box[3] for box in group) * (1 + 0.12 * len(group)),
    )
    total_line_width = sum(box[2] for box in selected)
    if len(selected) < 2 and total_line_width < width * 0.10:
        return None
    x1 = min(box[0] for box in selected)
    y1 = min(box[1] for box in selected)
    x2 = max(box[0] + box[2] for box in selected)
    y2 = max(box[1] + box[3] for box in selected)
    cluster_width = x2 - x1
    cluster_height = y2 - y1
    cluster_area_ratio = cluster_width * cluster_height / (width * height)
    # Floor grain, carpet and cables can create stroke-like contours across
    # most of the frame. Real text blocks should remain spatially compact.
    if cluster_area_ratio > 0.45 or cluster_height > height * 0.58:
        return None
    return x1, y1, x2 - x1, y2 - y1


def detect_priority_target(image: object, cv2: object) -> tuple[tuple[int, int, int, int] | None, str]:
    """Prefer dense text, then fall back to a large page-like rectangle."""

    text_bbox = detect_text_bbox(image, cv2)
    if text_bbox is not None:
        return text_bbox, "text_dense"
    document_bbox = detect_document_bbox(image, cv2)
    if document_bbox is not None:
        return document_bbox, "page_rectangle"
    return None, "search"


def annotate_alignment(image: object, cv2: object) -> tuple[object, dict[str, object]]:
    """Overlay camera centre, detected target centre and the suggested J1 direction."""

    bbox, priority = detect_priority_target(image, cv2)
    return annotate_bbox(image, cv2, bbox, priority)


def annotate_bbox(
    image: object,
    cv2: object,
    bbox: tuple[int, int, int, int] | None,
    priority: str,
) -> tuple[object, dict[str, object]]:
    """Draw one selected target, regardless of whether local or semantic."""

    height, width = image.shape[:2]
    camera_center = (width // 2, height // 2)
    cv2.drawMarker(image, camera_center, (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    if bbox is None:
        message = f"{priority.upper()}: no confirmed book | J1: SEARCH"
        cv2.putText(image, message, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        return image, {"found": False, "priority": priority, "direction": "search", "message": message}

    x, y, box_width, box_height = bbox
    normalized = (x / width, y / height, (x + box_width) / width, (y + box_height) / height)
    observation = observe_bbox(normalized)
    target_center = (round(observation.target_x * width), round(observation.target_y * height))
    colour = (0, 220, 0) if observation.centered else (0, 180, 255)
    cv2.rectangle(image, (x, y), (x + box_width, y + box_height), colour, 2)
    cv2.circle(image, target_center, 7, (255, 80, 0), -1)
    cv2.arrowedLine(image, camera_center, target_center, colour, 3, tipLength=0.08)
    direction = observation.horizontal.value.upper()
    message = f"{priority.upper()} {observation.error_x * 100:+.1f}% | J1: {direction}"
    cv2.putText(image, message, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return image, {
        "found": True,
        "priority": priority,
        "bbox_norm": [round(value, 4) for value in normalized],
        "error_x": round(observation.error_x, 4),
        "error_y": round(observation.error_y, 4),
        "direction": observation.horizontal.value,
        "centered": observation.centered,
        "message": message,
    }


class SemanticBookTracker:
    """Use the cloud once to select the book, then track it locally with CSRT."""

    TARGET_DESCRIPTION = (
        "the complete visible picture-book page intended for a child, containing story text and/or "
        "an illustration. Exclude computer screens, SSD/product packaging, loose unrelated papers, "
        "cables, motors, furniture and background objects"
    )

    def __init__(
        self,
        client: object,
        *,
        reacquire_seconds: float = 8.0,
        tracker_name: str = "kcf",
    ) -> None:
        if tracker_name not in {"kcf", "csrt"}:
            raise ValueError("OpenCV tracker must be kcf or csrt")
        self.client = client
        self.reacquire_seconds = reacquire_seconds
        self.tracker_name = tracker_name
        self._lock = threading.Lock()
        self._pending_bbox_norm: tuple[float, float, float, float] | None = None
        self._locating = False
        self._last_request_at = 0.0
        self._tracker = None
        self.api_miss_count = 0
        self.help_requested = False
        self.status = "waiting_for_semantic_book"

    def update(
        self,
        image: object,
        cv2: object,
        *,
        allow_reacquire: bool = True,
    ) -> tuple[int, int, int, int] | None:
        if self._tracker is not None:
            ok, tracked = self._tracker.update(image)
            if ok:
                x, y, width, height = (round(value) for value in tracked)
                if width > 8 and height > 8:
                    self.status = f"opencv_{self.tracker_name}_tracking"
                    return x, y, width, height
            self._tracker = None
            self.status = "opencv_tracker_lost"

        with self._lock:
            pending = self._pending_bbox_norm
            self._pending_bbox_norm = None
        if pending is not None:
            image_height, image_width = image.shape[:2]
            x1, y1, x2, y2 = pending
            bbox = (
                round(x1 * image_width),
                round(y1 * image_height),
                max(1, round((x2 - x1) * image_width)),
                max(1, round((y2 - y1) * image_height)),
            )
            tracker_factory = (
                cv2.TrackerKCF_create if self.tracker_name == "kcf" else cv2.TrackerCSRT_create
            )
            tracker = tracker_factory()
            tracker.init(image, bbox)
            self._tracker = tracker
            self.status = f"semantic_book_locked_{self.tracker_name}"
            return bbox

        now = time.monotonic()
        with self._lock:
            should_request = (
                allow_reacquire
                and not self._locating
                and now - self._last_request_at >= self.reacquire_seconds
            )
            if should_request:
                self._locating = True
                self._last_request_at = now
        if should_request:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                threading.Thread(
                    target=self._locate,
                    args=(encoded.tobytes(),),
                    name="semantic-book-locator",
                    daemon=True,
                ).start()
            else:
                with self._lock:
                    self._locating = False
        return None

    def _locate(self, jpeg: bytes) -> None:
        try:
            from lamp_core.object_localization import locate_target

            detection = locate_target(jpeg, self.client, self.TARGET_DESCRIPTION, object_id="book-page")
            with self._lock:
                if detection is not None and detection.confidence >= 0.55:
                    self._pending_bbox_norm = detection.bbox_norm
                    self.api_miss_count = 0
                    self.help_requested = False
                    self.status = f"semantic_found_{detection.confidence:.2f}"
                else:
                    was_help_requested = self.help_requested
                    self.api_miss_count = min(3, self.api_miss_count + 1)
                    self.help_requested = self.api_miss_count >= 3
                    if self.help_requested:
                        self.status = "please_move_book"
                        if not was_help_requested:
                            print("PLEASE_MOVE_BOOK: three semantic searches found no book")
                    else:
                        self.status = f"semantic_book_not_found_{self.api_miss_count}_of_3"
        except Exception as error:
            detail = " ".join(str(error).split())[:120] or "no detail"
            self.status = f"semantic_error_{type(error).__name__}: {detail}"
        finally:
            with self._lock:
                self._locating = False

    def resume_after_stable_local_candidate(self) -> None:
        """Resume cloud confirmation after the user has moved a target into view."""

        with self._lock:
            if not self.help_requested:
                return
            self.api_miss_count = 0
            self.help_requested = False
            self._last_request_at = 0.0
            self.status = "local_candidate_ready_for_semantic_check"


class RealtimeJ1Follower:
    """Map live image X to a bounded absolute J1 target around startup."""

    def __init__(
        self,
        *,
        interface: str,
        node_id: int,
        checksum: str,
        gear_ratio: float,
        positive_camera_direction: HorizontalDirection,
        envelope_degrees: float,
        speed_rpm: int,
        acceleration: int,
        update_hz: float = 8.0,
    ) -> None:
        from lamp_core.mks_can_protocol import ChecksumMode, set_bus_enabled
        from lamp_core.mks_single_axis import MksSingleAxisProbe
        from run_mks_single_axis_motion import SocketCanTransport

        if gear_ratio < 1:
            raise ValueError("J1 gear ratio must be at least 1")
        maximum_speed = 360 if gear_ratio > 1 else 180
        if not 1 <= speed_rpm <= maximum_speed:
            raise ValueError(f"J1 speed must be within 1..{maximum_speed} RPM")
        if not 0 <= acceleration <= 255:
            raise ValueError("J1 acceleration must be within 0..255")
        self.gear_ratio = gear_ratio
        self.positive_camera_direction = positive_camera_direction
        self.envelope_degrees = envelope_degrees
        self.speed_rpm = speed_rpm
        self.acceleration = acceleration
        self.minimum_interval_s = 1.0 / update_hz
        self._last_send_at = 0.0
        self._last_trajectory_at = time.monotonic()
        self._last_target_counts: int | None = None
        self._search_started_at = time.monotonic()
        maximum_output_velocity = speed_rpm / gear_ratio * 6.0
        maximum_output_acceleration = maximum_output_velocity / 0.30
        self._trajectory = ResponsiveJ1Trajectory(
            max_velocity_degrees_s=maximum_output_velocity,
            max_acceleration_degrees_s2=maximum_output_acceleration,
            max_jerk_degrees_s3=maximum_output_acceleration / 0.18,
        )
        self._transport = SocketCanTransport(interface)
        mode = ChecksumMode(checksum)
        self._mode = mode
        probe = MksSingleAxisProbe(self._transport, node_id, mode)
        snapshot = probe.snapshot()
        if abs(snapshot.rpm) > 1:
            self._transport.close()
            raise RuntimeError(f"J1 is already moving at {snapshot.rpm} RPM")
        self.anchor_counts = snapshot.encoder_counts
        self.node_id = node_id
        self._transport.send(set_bus_enabled(node_id, True, mode))

    def update(self, alignment: dict[str, object]) -> None:
        from lamp_core.mks_can_protocol import absolute_coordinate_move
        from lamp_core.mks_single_axis import COUNTS_PER_REVOLUTION

        now = time.monotonic()
        if not alignment.get("found"):
            # A sine path naturally slows to zero at both turnarounds. It stays
            # inside the startup envelope and never accumulates position.
            desired_degrees = smooth_bounded_scan_degrees(
                now - self._search_started_at,
                self.envelope_degrees,
            )
            alignment["tracking_mode"] = "bounded_search"
        else:
            bbox = tuple(float(value) for value in alignment["bbox_norm"])
            observation = observe_bbox(bbox)
            desired_degrees = mapped_j1_tracking_degrees(
                observation,
                positive_camera_direction=self.positive_camera_direction,
                envelope_degrees=self.envelope_degrees,
            )
            self._search_started_at = now
            alignment["tracking_mode"] = (
                "book_follow"
                if alignment.get("priority") in ("semantic_book", "page_rectangle", "page_lock")
                else "text_follow"
            )
        raw_target_degrees = desired_degrees
        trajectory_dt = max(0.001, min(now - self._last_trajectory_at, 0.15))
        self._last_trajectory_at = now
        desired_degrees = self._trajectory.update(raw_target_degrees, trajectory_dt)
        offset_counts = round(
            desired_degrees / 360 * COUNTS_PER_REVOLUTION * self.gear_ratio
        )
        target_counts = self.anchor_counts + offset_counts
        alignment["j1_raw_target_degrees"] = round(raw_target_degrees, 2)
        alignment["j1_target_degrees"] = round(desired_degrees, 2)
        alignment["j1_velocity_degrees_s"] = round(self._trajectory.velocity_degrees_s, 2)
        alignment["j1_target_counts"] = target_counts
        alignment["message"] += f" | motor {desired_degrees:+.1f} deg"
        if (
            self._last_target_counts is not None
            and abs(target_counts - self._last_target_counts) < 32
        ) or now - self._last_send_at < self.minimum_interval_s:
            return
        # Keep the configured top speed. Match each F5 command to the current
        # S-curve velocity so a near-centre correction does not hit the target
        # with the same aggressiveness as a large deliberate movement.
        profiled_speed_rpm = profiled_motor_speed_rpm(
            self._trajectory.velocity_degrees_s,
            self.gear_ratio,
            self.speed_rpm,
        )
        alignment["j1_command_speed_rpm"] = profiled_speed_rpm
        self._transport.send(
            absolute_coordinate_move(
                self.node_id,
                speed_rpm=profiled_speed_rpm,
                acceleration=self.acceleration,
                coordinate=target_counts,
                mode=self._mode,
            )
        )
        self._last_target_counts = target_counts
        self._last_send_at = now

    def close(self) -> None:
        self._transport.close()


class CameraStream:
    """Continuously capture the newest JPEG and notify streaming clients."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        quality: int,
        rotation: int,
        j1_follower: RealtimeJ1Follower | None = None,
        semantic_tracker: SemanticBookTracker | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.rotation = rotation
        self.j1_follower = j1_follower
        self.semantic_tracker = semantic_tracker
        self.condition = threading.Condition()
        self.frame: bytes | None = None
        self.alignment: dict[str, object] = {
            "found": False,
            "direction": "hold",
            "message": "Waiting for camera...",
        }
        self.error: str | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera = None
        self._locked_page_alignment: dict[str, object] | None = None
        self._locked_page_at = 0.0
        self._local_candidate_since = 0.0

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("Picamera2 is missing; install python3-picamera2 on the Pi") from error

        self._camera = Picamera2()
        configuration = self._camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
        )
        self._camera.configure(configuration)
        self._camera.start()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-preview", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            import cv2

            rotate_codes = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            delay = 1.0 / self.fps
            while not self._stopping.is_set():
                started = time.monotonic()
                image = self._camera.capture_array()
                if self.rotation:
                    image = cv2.rotate(image, rotate_codes[self.rotation])
                if self.semantic_tracker is not None:
                    local_bbox, local_priority = detect_priority_target(image, cv2)
                    now = time.monotonic()
                    if local_bbox is None:
                        self._local_candidate_since = 0.0
                    elif self.semantic_tracker.help_requested:
                        if self._local_candidate_since == 0.0:
                            self._local_candidate_since = now
                        elif now - self._local_candidate_since >= 1.0:
                            self.semantic_tracker.resume_after_stable_local_candidate()
                            self._local_candidate_since = 0.0
                    semantic_bbox = self.semantic_tracker.update(
                        image,
                        cv2,
                        allow_reacquire=not self.semantic_tracker.help_requested,
                    )
                    if local_bbox is not None:
                        selected_bbox, priority = local_bbox, local_priority
                    elif semantic_bbox is not None:
                        selected_bbox, priority = semantic_bbox, "semantic_book"
                    else:
                        selected_bbox, priority = None, self.semantic_tracker.status
                    image, alignment = annotate_bbox(image, cv2, selected_bbox, priority)
                    alignment["semantic_status"] = self.semantic_tracker.status
                    alignment["api_miss_count"] = self.semantic_tracker.api_miss_count
                    if self.semantic_tracker.help_requested:
                        alignment["needs_help"] = True
                        if selected_bbox is None:
                            alignment["message"] = "请把书移动一下，我找不到它了。"
                        else:
                            alignment["message"] += " | 请把书移动一下，我找不到它了。"
                else:
                    image, alignment = annotate_alignment(image, cv2)
                    priority = alignment.get("priority")
                    now = time.monotonic()
                    if priority in ("text_dense", "page_rectangle"):
                        self._locked_page_alignment = dict(alignment)
                        self._locked_page_at = now
                    elif (
                        self._locked_page_alignment is not None
                        and now - self._locked_page_at <= 1.5
                    ):
                        alignment = dict(self._locked_page_alignment)
                        alignment["priority"] = "page_lock"
                        alignment["message"] = (
                            f"PAGE_LOCK {float(alignment['error_x']) * 100:+.1f}% | "
                            f"J1: {str(alignment['direction']).upper()}"
                        )
                if self.j1_follower is not None:
                    self.j1_follower.update(alignment)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise RuntimeError("camera JPEG encoding failed")
                with self.condition:
                    self.frame = encoded.tobytes()
                    self.alignment = alignment
                    self.condition.notify_all()
                self._stopping.wait(max(0.0, delay - (time.monotonic() - started)))
        except Exception as error:  # Surface capture failures to browser clients.
            with self.condition:
                self.error = str(error)
                self.condition.notify_all()

    def wait_for_frame(self, previous: bytes | None = None, timeout: float = 5.0) -> bytes | None:
        with self.condition:
            self.condition.wait_for(
                lambda: (self.frame is not None and self.frame is not previous) or self.error is not None,
                timeout=timeout,
            )
            return self.frame

    def stop(self) -> None:
        self._stopping.set()
        with self.condition:
            self.condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._camera is not None:
            self._camera.stop()


def make_handler(stream: CameraStream) -> type[BaseHTTPRequestHandler]:
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", INDEX_HTML)
            elif path == "/health":
                status = HTTPStatus.OK if stream.frame is not None and stream.error is None else HTTPStatus.SERVICE_UNAVAILABLE
                self._send_bytes(status, "text/plain; charset=utf-8", (stream.error or "ok").encode())
            elif path == "/snapshot.jpg":
                frame = stream.wait_for_frame(timeout=5.0)
                if frame is None:
                    self._send_bytes(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain", b"camera frame unavailable")
                else:
                    self._send_bytes(HTTPStatus.OK, "image/jpeg", frame)
            elif path == "/alignment.json":
                with stream.condition:
                    body = json.dumps(stream.alignment, ensure_ascii=False).encode("utf-8")
                self._send_bytes(HTTPStatus.OK, "application/json; charset=utf-8", body)
            elif path == "/stream.mjpg":
                self._send_stream()
            else:
                self._send_bytes(HTTPStatus.NOT_FOUND, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.split("?", 1)[0] != "/shutdown":
                self._send_bytes(HTTPStatus.NOT_FOUND, "text/plain", b"not found")
                return
            self._send_bytes(HTTPStatus.OK, "application/json", b'{"stopping":true}')
            threading.Thread(
                target=self.server.shutdown,
                name="preview-shutdown",
                daemon=True,
            ).start()

        def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            previous = None
            try:
                while stream.error is None:
                    frame = stream.wait_for_frame(previous)
                    if frame is None or frame is previous:
                        continue
                    previous = frame
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, message: str, *args: object) -> None:
            print(f"{self.client_address[0]} - {message % args}")

    return PreviewHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show the Pi camera in a computer web browser")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--quality", type=int, choices=range(40, 96), default=80, metavar="40-95")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--execute-j1", action="store_true", help="enable bounded live J1 tracking on SocketCAN")
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument("--j1-node-id", type=int, default=1)
    parser.add_argument("--j1-checksum", choices=("additive", "xor"), default="additive")
    parser.add_argument("--j1-gear-ratio", type=float)
    parser.add_argument("--j1-positive-camera-direction", choices=("left", "right"))
    parser.add_argument("--j1-envelope-degrees", type=float, default=10.0)
    parser.add_argument("--j1-speed-rpm", type=int)
    parser.add_argument("--j1-acceleration", type=int, default=40)
    parser.add_argument(
        "--semantic-book-priority",
        action="store_true",
        help="use configured cloud vision once to select a picture-book page, then track locally",
    )
    parser.add_argument("--opencv-tracker", choices=("kcf", "csrt"), default="kcf")
    parser.add_argument("--semantic-reacquire-seconds", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height and fps must be positive")

    follower = None
    if args.execute_j1:
        if args.j1_gear_ratio is None or args.j1_positive_camera_direction is None or args.j1_speed_rpm is None:
            raise SystemExit(
                "--execute-j1 requires --j1-gear-ratio, --j1-positive-camera-direction and --j1-speed-rpm"
            )
        follower = RealtimeJ1Follower(
            interface=args.can_interface,
            node_id=args.j1_node_id,
            checksum=args.j1_checksum,
            gear_ratio=args.j1_gear_ratio,
            positive_camera_direction=HorizontalDirection(args.j1_positive_camera_direction),
            envelope_degrees=args.j1_envelope_degrees,
            speed_rpm=args.j1_speed_rpm,
            acceleration=args.j1_acceleration,
        )

    semantic_tracker = None
    if args.semantic_book_priority:
        from app_main import cloud_client
        from lamp_core.config import AppConfig, load_dotenv

        load_dotenv()
        config = AppConfig.from_environment()
        if not config.cloud_enabled or not config.api_key:
            raise SystemExit("semantic book priority requires ENABLE_CLOUD_VISION=true and OPENAI_API_KEY")
        semantic_tracker = SemanticBookTracker(
            cloud_client(config),
            reacquire_seconds=args.semantic_reacquire_seconds,
            tracker_name=args.opencv_tracker,
        )

    stream = CameraStream(
        args.width,
        args.height,
        args.fps,
        args.quality,
        args.rotation,
        follower,
        semantic_tracker,
    )
    try:
        stream.start()
    except Exception as error:
        if follower is not None:
            follower.close()
        raise SystemExit(f"Cannot start camera: {error}. Stop app_main.py or other camera programs first.") from error

    server = ThreadingHTTPServer((args.host, args.port), make_handler(stream))
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f"Camera preview: http://<raspberry-pi-ip>:{args.port}")
    print("Press Ctrl+C to stop. The camera cannot be shared with app_main.py.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stream.stop()
        if follower is not None:
            follower.close()


if __name__ == "__main__":
    main()
