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
    AlignmentObservation,
    CoarseToFineAim,
    HorizontalDirection,
    NarrowingScan,
    VisualServoPid,
    observe_bbox,
    page_score,
    saliency_thirds,
    profiled_motor_speed_rpm,
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
  <p><button id="restart" style="background:#2e7d32">重新测试（当前位置为 0）</button></p>
  <p><button id="stop">&#20572;&#27490;&#25668;&#20687;&#22836;&#21644; J1 &#36319;&#38543;</button></p>
  <script>
    setInterval(async()=>{try{const r=await fetch('/alignment.json',{cache:'no-store'});
      const s=await r.json();document.getElementById('status').textContent=s.message;}catch(e){}},300);
    document.getElementById('restart').onclick=async()=>{
      await fetch('/restart',{method:'POST'});
      document.getElementById('status').textContent='已重置：当前位置为 0，重新开始 ±10° 搜索。';
    };
    document.getElementById('stop').onclick=async()=>{
      if(!confirm('Stop camera preview and J1 tracking?'))return;
      await fetch('/shutdown',{method:'POST'});
      document.getElementById('status').textContent='Program stopped. The motor holds its last position.';
      document.getElementById('stop').disabled=true;
    };
  </script>
</main></body></html>""".encode("utf-8")


def detect_document_bbox(
    image: object,
    cv2: object,
    *,
    text_evidence: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Find a page-like region, preferring one that contains detected text."""

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
        contains_text = False
        if text_evidence is not None:
            tx, ty, tw, th = text_evidence
            text_center = (tx + tw / 2, ty + th / 2)
            contains_text = (
                x <= text_center[0] <= x + box_width
                and y <= text_center[1] <= y + box_height
                and box_area >= tw * th * 1.5
            )
        # A close book can legitimately touch two image edges. Only accept that
        # relaxed case when the candidate surrounds real text evidence.
        if touched_edges >= 2 and not (contains_text and touched_edges == 2):
            continue
        contour_area = max(1.0, float(cv2.contourArea(contour)))
        rectangularity = min(1.0, contour_area / box_area)
        # Pages and packages are usually large, portrait/landscape rectangles.
        # Rectangularity helps but is not mandatory because hands and lamp bars
        # can interrupt an otherwise valid boundary.
        evidence_bonus = 3.0 if contains_text else 1.0
        candidates.append(
            (area_ratio * (0.75 + rectangularity) * evidence_bonus, (x, y, box_width, box_height))
        )
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
    """Use text as book evidence, but follow the containing page rather than its text block."""

    text_bbox = detect_text_bbox(image, cv2)
    document_bbox = detect_document_bbox(image, cv2, text_evidence=text_bbox)
    if text_bbox is not None and document_bbox is not None:
        tx, ty, tw, th = text_bbox
        dx, dy, dw, dh = document_bbox
        text_center_x, text_center_y = tx + tw / 2, ty + th / 2
        if (
            dx <= text_center_x <= dx + dw
            and dy <= text_center_y <= dy + dh
            and dw * dh >= tw * th * 1.5
        ):
            return document_bbox, "book_page_with_text"
    if text_bbox is not None:
        return text_bbox, "text_fallback"
    if document_bbox is not None:
        return document_bbox, "page_rectangle"
    return None, "search"


def choose_tracking_target(
    local_bbox: tuple[int, int, int, int] | None,
    local_priority: str,
    semantic_bbox: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Only a model-confirmed whole book may become a motor target.

    Local text/page candidates are still used to decide when semantic
    reacquisition is worthwhile, but cannot steer J1 by themselves.
    """

    if semantic_bbox is not None:
        return semantic_bbox, "semantic_book"
    return None, f"awaiting_semantic_{local_priority}"


def detect_candidate_boxes(
    image: object,
    cv2: object,
    *,
    max_candidates: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Return several plausible page/book rectangles, best guess first.

    The local heuristics are cheap but cannot tell a book from a monitor, a
    packaging box or a bright strip. Offering all of them, numbered, lets the
    cloud model pick the right one as a classification instead of being asked
    for a pixel box it is not good at.
    """

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
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
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
        if not 0.04 <= area_ratio <= 0.95 or not 0.25 <= aspect <= 3.6:
            continue
        if touched_edges >= 3:
            continue
        contour_area = max(1.0, float(cv2.contourArea(contour)))
        rectangularity = min(1.0, contour_area / box_area)
        scored.append((area_ratio * (0.75 + rectangularity), (x, y, box_width, box_height)))

    text_bbox = detect_text_bbox(image, cv2)
    if text_bbox is not None:
        scored.append((0.5, text_bbox))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for _, box in scored:
        if all(bbox_intersection_over_union(box, kept) < 0.30 for kept in selected):
            selected.append(box)
        if len(selected) >= max_candidates:
            break
    return selected


def annotate_candidate_boxes(
    image: object,
    cv2: object,
    boxes: list[tuple[int, int, int, int]],
) -> object:
    """Draw numbered rectangles so the model can refer to them by number."""

    annotated = image.copy()
    for index, (x, y, box_width, box_height) in enumerate(boxes, start=1):
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (0, 200, 255), 3)
        label_y = min(y + 46, annotated.shape[0] - 12)
        cv2.putText(
            annotated, str(index), (x + 8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7
        )
        cv2.putText(
            annotated, str(index), (x + 8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3
        )
    return annotated


def local_candidate_is_stable(
    first_seen_at: float,
    now: float,
    *,
    required_seconds: float = 0.6,
) -> bool:
    """Delay cloud localization until a candidate survives camera motion."""

    if first_seen_at <= 0 or now < first_seen_at or required_seconds <= 0:
        return False
    return now - first_seen_at >= required_seconds - 1e-9


def bbox_intersection_over_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Measure whether two consecutive local candidates describe the same page."""

    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def bbox_touches_image_edge(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    *,
    margin_ratio: float = 0.04,
) -> bool:
    """Report a locked box that reaches the frame edge (a partial book view).

    A tracked box touching the image boundary usually means only part of the
    book is visible. That is the signal to periodically re-confirm the book
    with the semantic model so the box can expand as the rest of the page pans
    into view, instead of staying locked to the first small region KCF saw.
    """

    if min(image_width, image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 <= margin_ratio < 0.25:
        raise ValueError("edge margin ratio must be within 0..0.25")
    x, y, box_width, box_height = bbox
    margin_x = image_width * margin_ratio
    margin_y = image_height * margin_ratio
    return (
        x <= margin_x
        or y <= margin_y
        or x + box_width >= image_width - margin_x
        or y + box_height >= image_height - margin_y
    )


def annotate_alignment(image: object, cv2: object) -> tuple[object, dict[str, object]]:
    """Overlay camera centre, detected target centre and the suggested J1 direction."""

    bbox, priority = detect_priority_target(image, cv2)
    return annotate_bbox(image, cv2, bbox, priority)


def annotate_bbox(
    image: object,
    cv2: object,
    bbox: tuple[int, int, int, int] | None,
    priority: str,
    horizontal_deadband: float = 0.08,
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
    observation = observe_bbox(normalized, horizontal_deadband=horizontal_deadband)
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
    """Find the picture-book page locally first, then track its OpenCV geometry.

    Two detectors can locate the page. The on-device PP-OCR text detector is
    tried first: it costs no tokens, needs no network and runs in roughly 0.3 s
    on the Pi, and picture-book story text is what the lamp is there to look at.
    The cloud model is kept for the pages the text detector cannot serve -- an
    illustration-only spread, a closed book, or an ONNX model that is simply not
    installed -- and it answers the binary "is any of the book visible" question
    far more reliably than local rectangle heuristics can.

    Either way the tracking box is then maintained frame by frame by OpenCV, so a
    partially visible book is accepted naturally and non-book objects stay gated
    behind a semantic confirmation.
    """

    TARGET_DESCRIPTION = (
        "any visible portion of an open children's picture book or picture-book page spread, including "
        "story text and illustrations. Return found true when an obvious book is cropped by the image edge, "
        "rotated, or partly occluded; bound all visible pixels belonging to that physical book/page spread. "
        "Never bound only a text paragraph or one illustration when more of the book is visible. Exclude "
        "computer screens, SSD/product packaging, loose unrelated papers, cables, motors, furniture and "
        "background objects"
    )

    # Consecutive refreshes a source may miss before another source may take over.
    SOURCE_MISS_LIMIT = 3

    #: How long a tracking box may go unconfirmed before it is dropped. Must sit
    #: comfortably above the settled refresh cadence, or healthy tracking would
    #: invalidate itself just before its own sanity refresh arrives.
    CONFIRMATION_TIMEOUT_SECONDS = 18.0

    #: A valid refresh may take longer than the normal trust window on a slow
    #: network.  While exactly one generation-matched request is in flight, keep
    #: the current tracker for this bounded grace period rather than dropping it
    #: milliseconds before a legitimate answer arrives.
    CONFIRMATION_REQUEST_GRACE_SECONDS = 30.0

    #: An aim error this small means the servo is already where it wants to be.
    #: Re-anchoring then would only swap a settled box for a noisier fresh one and
    #: make the loop chase its own measurement -- the end-game sluggishness. Kept
    #: at or above the visual servo deadband, because inside that band the servo
    #: does not act at all, so re-anchoring there is pure disturbance.
    SETTLED_ERROR_TOLERANCE = 0.08

    #: A settled target is still re-checked this rarely, to catch slow drift or a
    #: book that has been swapped, without disturbing a loop that has come to rest.
    SETTLED_SANITY_SECONDS = 12.0

    #: A fresh anchor whose centre lands this close to the accepted one is taken
    #: to be the same target. Measured on the real rig, successive anchors move
    #: the box edge by ~0.07-0.17 of the frame, so this threshold is what stops
    #: that jitter from reaching the servo.
    ANCHOR_CONSISTENT_ERROR = 0.05

    #: How much of a fresh accepted anchor replaces the old one. Blending halves
    #: the size of the step the servo sees, which is what lets the loop rest.
    ANCHOR_SMOOTHING = 0.5

    #: Agreeing detections needed before a jumped anchor is believed rather than
    #: dismissed as a one-frame outlier. A book that genuinely moved keeps
    #: producing the new position, so it wins after this many detections.
    ANCHOR_CONFIRMATIONS = 2

    def __init__(
        self,
        client: object | None,
        *,
        reacquire_seconds: float = 8.0,
        tracker_name: str = "kcf",
        text_detector: object | None = None,
    ) -> None:
        if tracker_name not in {"kcf", "csrt"}:
            raise ValueError("OpenCV tracker must be kcf or csrt")
        self.client = client
        self.text_detector = text_detector
        self.reacquire_seconds = reacquire_seconds
        self.tracker_name = tracker_name
        self._lock = threading.Lock()
        self._pending_bbox: tuple[int, int, int, int] | None = None
        self._locating = False
        self._last_request_at = 0.0
        self._tracker = None
        self._tracked_bbox_norm: tuple[float, float, float, float] | None = None
        # Text read by the same call that located the page, so the reading flow
        # can reuse it instead of paying to describe the frame a second time.
        self.page_text = ""
        # Latest page score (visible area, edge-cropped pages penalised) and a
        # counter so the scan can tell one measurement from the next.
        self.page_score = 0.0
        self.pick_count = 0
        self.api_miss_count = 0
        self.help_requested = False
        # Which detector owns the current target, and how many consecutive
        # refreshes it has missed. A text-line box and a whole-page box do not
        # share a centre, so switching between them every refresh would move the
        # aim point and make the servo hunt.
        self._anchor_source: str | None = None
        self._source_miss_count = 0
        # Last anchor actually handed to the tracker, plus a jumped candidate
        # waiting to prove it repeats before it is believed.
        self._anchor_bbox: tuple[float, float, float, float] | None = None
        self._anchor_candidate: tuple[float, float, float, float] | None = None
        self._anchor_candidate_count = 0
        # When a detector last confirmed the target, so an unconfirmed tracker box
        # cannot be followed for ever.
        self._last_confirmed_at = time.monotonic()
        self._generation = 0
        self.status = "waiting_for_semantic_book"

    def _confirmation_expired(self, now: float) -> bool:
        """Whether the running tracker has gone too long without confirmation."""

        with self._lock:
            request_in_flight = self._locating
            request_age = now - self._last_request_at
        if request_in_flight and request_age <= self.CONFIRMATION_REQUEST_GRACE_SECONDS:
            return False
        return now - self._last_confirmed_at > self.CONFIRMATION_TIMEOUT_SECONDS

    def update(
        self,
        image: object,
        cv2: object,
        *,
        local_bbox: tuple[int, int, int, int] | None = None,
        allow_reacquire: bool = True,
    ) -> tuple[int, int, int, int] | None:
        # The model both locates the book and self-validates the resulting crop.
        # This replaces the old reliance on an OpenCV candidate box, which on a
        # real desk picked bright rectangles or text strips that were not books.
        with self._lock:
            pending = self._pending_bbox
            self._pending_bbox = None
        now = time.monotonic()
        if pending is not None:
            self._init_tracker(cv2, image, pending)
            frame_height, frame_width = image.shape[:2]
            self._tracked_bbox_norm = (
                pending[0] / frame_width,
                pending[1] / frame_height,
                (pending[0] + pending[2]) / frame_width,
                (pending[1] + pending[3]) / frame_height,
            )
            # Start the refresh cooldown from this fresh lock, not from startup.
            self._last_request_at = now
            self._last_confirmed_at = now
            self.api_miss_count = 0
            self.help_requested = False
            self.status = f"semantic_book_locked_{self.tracker_name}"
            return pending
        if self._tracker is not None and self._confirmation_expired(now):
            # Nothing has confirmed this box for a long time. The local tracker
            # will happily keep returning *something* -- measured on the rig it
            # drifted onto a laptop and the loop went on reporting "centred" with
            # the book nowhere in frame -- so an unconfirmed box is dropped rather
            # than followed. Dropping it hands the situation back to the search,
            # which is what can still find a book from a bad angle.
            self._tracker = None
            self._tracked_bbox_norm = None
            with self._lock:
                self._anchor_bbox = None
                self._anchor_source = None
                self._source_miss_count = 0
            self.status = "target_unconfirmed_searching"

        if self._tracker is not None:
            ok, tracked = self._tracker.update(image)
            if ok:
                x, y, width, height = (round(value) for value in tracked)
                # A tracker may drift outside the frame; clamp it so downstream
                # normalized coordinates stay valid instead of crashing the loop.
                image_height, image_width = image.shape[:2]
                x = max(0, min(x, image_width - 1))
                y = max(0, min(y, image_height - 1))
                width = max(1, min(width, image_width - x))
                height = max(1, min(height, image_height - y))
                if width > 8 and height > 8:
                    with self._lock:
                        self._tracked_bbox_norm = (
                            x / image_width,
                            y / image_height,
                            (x + width) / image_width,
                            (y + height) / image_height,
                        )
                    self.status = f"opencv_{self.tracker_name}_tracking"
                    centre_x = (x + width / 2.0) / image_width
                    self._maybe_refresh_target(
                        image,
                        cv2,
                        at_edge=bbox_touches_image_edge(
                            (x, y, width, height),
                            image_width,
                            image_height,
                        ),
                        settled=abs(centre_x - 0.5) <= self.SETTLED_ERROR_TOLERANCE,
                    )
                    return x, y, width, height
            self._tracker = None
            self._tracked_bbox_norm = None
            self.status = "opencv_tracker_lost"

        now = time.monotonic()
        with self._lock:
            # Fast path: a stable local candidate makes a fresh cloud look worth
            # it. Slow path: keep looking periodically anyway, because the local
            # detector may simply never find the book on a cluttered desk.
            candidate_ready = allow_reacquire and now - self._last_request_at >= 2.0
            due_anyway = now - self._last_request_at >= self.reacquire_seconds
            should_request = (candidate_ready or due_anyway) and not self._locating
            if should_request:
                self._locating = True
                self._last_request_at = now
        if should_request:
            self._request_confirm(image, cv2)
        return None

    def _init_tracker(self, cv2: object, image: object, bbox: tuple[int, int, int, int]) -> None:
        tracker_factory = (
            cv2.TrackerKCF_create if self.tracker_name == "kcf" else cv2.TrackerCSRT_create
        )
        tracker = tracker_factory()
        tracker.init(image, bbox)
        self._tracker = tracker

    @property
    def awaiting_confirmation(self) -> bool:
        """True while a crop of one candidate region is out for validation."""

        with self._lock:
            return self._locating

    @property
    def anchor_source(self) -> str | None:
        """Which detector owns the aim point: "local_text", "cloud" or None."""

        with self._lock:
            return self._anchor_source

    def _request_confirm(self, image: object, cv2: object) -> None:
        """Number the local candidate rectangles and let the model pick one.

        Choosing among numbered candidates is a classification, which the vision
        model does far more reliably than emitting a precise pixel box, and far
        more reliably than the local rectangle heuristics can tell a book from a
        monitor or a bright strip.
        """

        frame = image.copy()
        with self._lock:
            generation = self._generation
        threading.Thread(
            target=self._pick_candidate,
            args=(frame, cv2, generation),
            name="semantic-book-picker",
            daemon=True,
        ).start()

    def _maybe_refresh_target(
        self,
        image: object,
        cv2: object,
        *,
        at_edge: bool,
        settled: bool = False,
    ) -> None:
        """Periodically re-pick the page while tracking, and urgently at an edge.

        CSRT and KCF slowly shrink or drift onto one illustration or text block,
        after which the box no longer covers the whole page: its centre can sit
        near the middle of the frame and the servo then reports "centred" while
        the book is still cropped. Refreshing on a timer restores a box that
        covers everything visible; a box already touching the frame edge means the
        page is cropped, so that refresh is taken sooner.

        A *settled* target is deliberately left alone. Measured on the real rig,
        successive anchors move the box edge by ~0.17 of the frame, so refreshing
        every couple of seconds while the aim is already good kept injecting a
        fresh step for the servo to chase and the loop never came to rest.
        Shrinkage is self-correcting here: it moves the box centre, the error
        grows, and the refresh comes back on its own.
        """

        now = time.monotonic()
        if at_edge:
            interval = min(self.reacquire_seconds, 2.0)
        elif settled:
            interval = max(self.SETTLED_SANITY_SECONDS, self.reacquire_seconds)
        else:
            interval = self.reacquire_seconds
        with self._lock:
            should_request = (
                not self._locating and now - self._last_request_at >= interval
            )
            if should_request:
                self._locating = True
                self._last_request_at = now
        if should_request:
            self._request_confirm(image, cv2)

    def _record_miss(self) -> None:
        was_help_requested = self.help_requested
        self.api_miss_count = min(3, self.api_miss_count + 1)
        self.help_requested = self.api_miss_count >= 3
        if self.help_requested:
            self.status = "please_move_book"
            if not was_help_requested:
                print("PLEASE_MOVE_BOOK: three semantic searches found no book")
        else:
            self.status = f"semantic_book_not_found_{self.api_miss_count}_of_3"

    @staticmethod
    def _box_from_norm(
        image: object,
        bbox_norm: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int] | None:
        frame_height, frame_width = image.shape[:2]
        x1, y1, x2, y2 = bbox_norm
        px1 = max(0, min(frame_width, int(x1 * frame_width)))
        py1 = max(0, min(frame_height, int(y1 * frame_height)))
        px2 = max(0, min(frame_width, int(x2 * frame_width)))
        py2 = max(0, min(frame_height, int(y2 * frame_height)))
        if px2 - px1 < 48 or py2 - py1 < 48:
            return None
        return px1, py1, px2 - px1, py2 - py1

    @staticmethod
    def _crop_tracked_region(
        image: object,
        bbox_norm: tuple[float, float, float, float],
        *,
        padding: float = 0.10,
    ) -> object:
        """Crop the current tracked target with context for presence validation."""

        frame_height, frame_width = image.shape[:2]
        x1, y1, x2, y2 = bbox_norm
        width, height = x2 - x1, y2 - y1
        px1 = max(0, int((x1 - width * padding) * frame_width))
        py1 = max(0, int((y1 - height * padding) * frame_height))
        px2 = min(frame_width, int((x2 + width * padding) * frame_width))
        py2 = min(frame_height, int((y2 + height * padding) * frame_height))
        return image[py1:py2, px1:px2]

    @staticmethod
    def _centre_distance(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        """Largest per-axis gap between two box centres, in normalized units."""

        return max(
            abs((first[0] + first[2]) / 2.0 - (second[0] + second[2]) / 2.0),
            abs((first[1] + first[3]) / 2.0 - (second[1] + second[3]) / 2.0),
        )

    def _accept_anchor(
        self,
        candidate: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float] | None:
        """Screen a fresh anchor, blending it in or holding an outlier out.

        Returns the box to hand the tracker, or None to leave the running tracker
        alone. Callers must hold ``self._lock``.
        """

        previous = self._anchor_bbox
        # Outlier screening exists to protect a running tracker from being yanked
        # onto a one-frame mistake. With nothing tracking there is nothing to
        # protect, and refusing the candidate would simply never acquire a target.
        if previous is None or self._tracker is None:
            self._anchor_bbox = candidate
            self._anchor_candidate = None
            self._anchor_candidate_count = 0
            return candidate
        if self._centre_distance(previous, candidate) <= self.ANCHOR_CONSISTENT_ERROR:
            blended = tuple(
                previous[index]
                + (candidate[index] - previous[index]) * self.ANCHOR_SMOOTHING
                for index in range(4)
            )
            self._anchor_bbox = blended
            self._anchor_candidate = None
            self._anchor_candidate_count = 0
            return blended
        pending = self._anchor_candidate
        if pending is not None and self._centre_distance(pending, candidate) <= self.ANCHOR_CONSISTENT_ERROR:
            self._anchor_candidate_count += 1
        else:
            self._anchor_candidate = candidate
            self._anchor_candidate_count = 1
        if self._anchor_candidate_count >= self.ANCHOR_CONFIRMATIONS:
            self._anchor_bbox = candidate
            self._anchor_candidate = None
            self._anchor_candidate_count = 0
            return candidate
        return None

    def _pick_candidate(
        self,
        image: object,
        cv2: object,
        generation: int | None = None,
    ) -> None:
        """Locate the page, with the free on-device detector tried first.

        The PP-OCR pass costs about 0.3 s and no tokens, so it runs before the
        cloud. When it finds story text the page box comes from that text cluster
        and no request is made at all; J1 pans horizontally only, so the text
        cluster is a dependable anchor even though it is not the whole page. The
        cloud call stays for pages with no detectable text, and locating and
        reading in one request is what keeps its token cost down.
        """

        try:
            from lamp_core.object_localization import (
                choose_candidate_index,
                confirm_target_present,
                locate_page,
            )
            from lamp_core.text_detection import text_anchor_box

            box: tuple[int, int, int, int] | None = None
            source: str | None = None

            detector = self.text_detector
            local_ready = detector is not None and detector.available
            # Stick to whichever source already owns the target. Only three
            # consecutive misses release it, so one blurred frame cannot discard
            # a good target and cannot re-aim onto a differently shaped box.
            with self._lock:
                if generation is None:
                    generation = self._generation
                if generation != self._generation:
                    return
                anchor_source = self._anchor_source
                reference_anchor = self._anchor_bbox if anchor_source == "local_text" else None
                tracked_anchor = self._tracked_bbox_norm
                tracker_active = self._tracker is not None

            # Once a cloud-confirmed page is being tracked, periodic cloud work
            # is a sanity check, never a second localization.  Re-localizing the
            # whole frame twice picked the same wrong region on the real rig and
            # drove J1 from +4 to the +40 degree limit.  Validate a padded crop of
            # the live CSRT box and keep its geometry unchanged.
            if (
                self.client is not None
                and anchor_source == "cloud"
                and tracker_active
                and tracked_anchor is not None
            ):
                crop = self._crop_tracked_region(image, tracked_anchor)
                ok, encoded_crop = cv2.imencode(
                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if not ok:
                    raise RuntimeError("tracked book crop JPEG encoding failed")
                confirmed = confirm_target_present(
                    encoded_crop.tobytes(),
                    self.client,
                    self.TARGET_DESCRIPTION,
                    cropped_region=True,
                )
                with self._lock:
                    if generation != self._generation:
                        return
                    self.pick_count += 1
                    if confirmed:
                        self._last_confirmed_at = time.monotonic()
                        self._source_miss_count = 0
                        self.api_miss_count = 0
                        self.help_requested = False
                        self.page_score = page_score(tracked_anchor)
                        self.status = "semantic_book_confirmed_in_place"
                    else:
                        self.page_score = 0.0
                        self._source_miss_count += 1
                        self._record_miss()
                        if self._source_miss_count >= self.SOURCE_MISS_LIMIT:
                            self._tracker = None
                            self._tracked_bbox_norm = None
                            self._anchor_source = None
                            self._anchor_bbox = None
                            self._anchor_candidate = None
                            self._anchor_candidate_count = 0
                return
            owns_local = anchor_source != "cloud"
            if local_ready and owns_local:
                anchor = text_anchor_box(
                    detector.detect(image),
                    reference_bbox=reference_anchor,
                )
                if anchor is not None:
                    text_box = self._box_from_norm(image, anchor)
                    if text_box is not None:
                        # OCR tells us where printed content is; whenever a page
                        # contour actually surrounds it, follow that page centre
                        # instead of the paragraph centre.  This prevents a page
                        # with left- or right-aligned text from being deliberately
                        # framed off-centre.
                        page_box = (
                            detect_document_bbox(
                                image,
                                cv2,
                                text_evidence=text_box,
                            )
                            if hasattr(cv2, "cvtColor")
                            else None
                        )
                        box = page_box or text_box
                        source = "local_text"

            # With a semantic client available, OCR is evidence that it is worth
            # checking this frame, not proof that the text belongs to a book.
            # This closes the path where a laptop screen continually refreshed
            # its own trust timestamp and was followed forever.  Offline mode
            # deliberately keeps the text-only fallback.
            if box is not None and self.client is not None and anchor_source is None:
                box = None
                source = None

            if box is None and self.client is not None and anchor_source != "local_text":
                ok, full = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    reading = locate_page(full.tobytes(), self.client, self.TARGET_DESCRIPTION)
                    if reading is not None:
                        with self._lock:
                            self.page_text = reading.visible_text
                        box = self._box_from_norm(image, reading.bbox_norm)
                        if box is not None:
                            source = "cloud"

            if box is None and self.client is not None and anchor_source != "local_text":
                candidates = detect_candidate_boxes(image, cv2)
                if candidates:
                    annotated = annotate_candidate_boxes(image, cv2, candidates)
                    ok, buffer = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82]
                    )
                    if ok:
                        choice = choose_candidate_index(
                            buffer.tobytes(),
                            self.client,
                            self.TARGET_DESCRIPTION,
                            len(candidates),
                        )
                        if 1 <= choice <= len(candidates):
                            box = candidates[choice - 1]
                            source = "cloud"
            frame_height, frame_width = image.shape[:2]
            accepted = None
            candidate_norm = None
            if box is not None:
                candidate_norm = (
                    box[0] / frame_width,
                    box[1] / frame_height,
                    (box[0] + box[2]) / frame_width,
                    (box[1] + box[3]) / frame_height,
                )
                # Only anchors the tracker could actually hold are worth accepting,
                # so a too-small box never becomes the committed anchor.
            with self._lock:
                if generation != self._generation:
                    return
                if (
                    candidate_norm is not None
                    and self._box_from_norm(image, candidate_norm) is not None
                ):
                    accepted = self._accept_anchor(candidate_norm)
                self.pick_count += 1
                if accepted is not None:
                    box = self._box_from_norm(image, accepted)
                if box is not None and accepted is not None:
                    self._pending_bbox = box
                    if source == "local_text":
                        # Detection only, so no text was read this frame. Clearing
                        # it keeps the reading flow from reusing stale text.
                        self.page_text = ""
                    self.page_score = page_score(accepted)
                    self._anchor_source = source
                    self._source_miss_count = 0
                    self._last_confirmed_at = time.monotonic()
                    self.api_miss_count = 0
                    self.help_requested = False
                    self.status = (
                        "local_text_book_candidate"
                        if source == "local_text"
                        else "semantic_book_candidate_picked"
                    )
                elif box is not None:
                    # A source did see something, but the anchor jumped away from
                    # the accepted one on a single frame. Keep the running tracker
                    # instead of re-aiming at an outlier; the source is not at
                    # fault, so this is not counted as a miss.
                    self.status = "anchor_outlier_held"
                else:
                    self.page_score = 0.0
                    if self._anchor_source is not None:
                        self._source_miss_count += 1
                        if self._source_miss_count < self.SOURCE_MISS_LIMIT:
                            # A hiccup, not an empty scene: keep the existing
                            # tracker and do not escalate to the user.
                            self.status = (
                                f"{self._anchor_source}_soft_miss"
                                f"_{self._source_miss_count}_of_{self.SOURCE_MISS_LIMIT}"
                            )
                            return
                        self._anchor_source = None
                        self._source_miss_count = 0
                        # The source has given up, so the accepted anchor is stale
                        # and must not screen the next source's box as an outlier.
                        self._anchor_bbox = None
                        self._anchor_candidate = None
                        self._anchor_candidate_count = 0
                    self._record_miss()
        except Exception as error:
            detail = " ".join(str(error).split())[:120] or "no detail"
            self.status = f"semantic_error_{type(error).__name__}: {detail}"
        finally:
            with self._lock:
                if generation is None or generation == self._generation:
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

    def reset(self) -> None:
        """Forget any lock and restart the confirm-then-track flow."""

        with self._lock:
            self._generation += 1
            self._tracker = None
            self._tracked_bbox_norm = None
            self._pending_bbox = None
            self._locating = False
            self._last_request_at = 0.0
            self.page_text = ""
            self.page_score = 0.0
            self.pick_count = 0
            self.api_miss_count = 0
            self.help_requested = False
            self._anchor_source = None
            self._source_miss_count = 0
            self._anchor_bbox = None
            self._anchor_candidate = None
            self._anchor_candidate_count = 0
            self._last_confirmed_at = time.monotonic()
            self.status = "waiting_for_semantic_book"


class RealtimeJ1Follower:
    """Map live image X to a bounded absolute J1 target around startup."""

    #: How much the measured aim error must move before direct aim will correct
    #: the same target twice. Below this the reading is treated as not yet
    #: reflecting the previous move.
    AIM_VERIFY_DELTA = 0.01

    #: Improvement in |aim error| that counts as progress while pulling a clipped
    #: target back into frame.
    EDGE_PROGRESS_DELTA = 0.02

    #: Moves without progress before the edge attempt is abandoned. Measured on
    #: the rig, the edge path once drove 40 deg to the travel stop while the
    #: target stayed clipped the whole way -- pushing was never going to work.
    EDGE_STALL_LIMIT = 3

    #: How long the axis will hold still for an in-flight detection. Cloud calls
    #: take over a second and are re-issued every couple of seconds, so an
    #: unbounded hold froze the axis for 36 seconds straight while the book sat
    #: clipped at the frame edge and nothing else could run.
    VALIDATE_HOLD_SECONDS = 2.0

    #: How many extra phase-shifted sweeps to run when a whole sweep scored zero.
    SEARCH_RESTART_LIMIT = 2

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
        search_degrees: float | None = None,
        travel_degrees: float = 60.0,
        servo_kp: float = 180.0,
        servo_ki: float = 80.0,
        servo_kd: float = 8.0,
        servo_deadband: float = 0.03,
        direct_aim: bool = True,
        aim_settle_seconds: float = 0.8,
        aim_max_step_degrees: float = 8.0,
        degrees_per_error: float = 44.5,
        scan_step_degrees: float = 10.0,
        scan_dwell_seconds: float = 2.5,
        hold_seconds: float = 4.0,
        update_hz: float = 8.0,
    ) -> None:
        from lamp_core.mks_can_protocol import ChecksumMode, set_bus_enabled, set_working_mode
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
        # The find-the-book sweep covers the whole reachable arc. It used to
        # default to twice the envelope, and because the opening scan round only
        # reached two steps either side of centre, a book 40 degrees off was
        # simply unreachable no matter what the envelope was set to.
        requested_search = search_degrees
        self.travel_degrees = max(
            travel_degrees, envelope_degrees * 2.0, requested_search or 0.0
        )
        self.search_degrees = (
            requested_search if requested_search is not None else self.travel_degrees
        )
        # Measured horizontal field of view (44.5 deg on this rig) sets kp so a
        # full-scale error is corrected in about a quarter of a second.
        self.servo_kp = servo_kp
        self.servo_ki = servo_ki
        self.servo_kd = servo_kd
        # Aiming tolerance. With anchor smoothing holding the measured box to
        # about +-0.01 this can sit close to the noise floor, so "centred" now
        # means about 1.3 deg instead of the old 3.6 deg.
        self.servo_deadband = servo_deadband
        self.direct_aim = direct_aim
        self.aim_settle_seconds = aim_settle_seconds
        # A large fast move blurs the frame and the tracker lags, so the measured
        # error can stay put for seconds. Capping one move and insisting the
        # reading has actually changed before moving again is what stops those
        # stale readings being corrected repeatedly until the axis hits its stop.
        self.aim_max_step_degrees = aim_max_step_degrees
        # Measured horizontal field of view (calibrated on this rig): the axis
        # turns this many degrees to move a target from the image edge to centre.
        # Also the scale of the camera's own motion in the frame, which the servo
        # subtracts so it never chases its own rotation.
        self.degrees_per_error = degrees_per_error
        self.scan_step_degrees = scan_step_degrees
        self.scan_dwell_seconds = scan_dwell_seconds
        self.hold_seconds = hold_seconds
        self.speed_rpm = speed_rpm
        self.acceleration = acceleration
        self.minimum_interval_s = 1.0 / update_hz
        self._last_send_at = 0.0
        self._last_step_at = time.monotonic()
        self._last_target_counts: int | None = None
        self._search_started_at = time.monotonic()
        self._last_found_at = 0.0
        # When the current run of edge-clipped sightings began, so pulling a
        # clipped target back into frame cannot become an endless tug of war.
        self._edge_started_at: float | None = None
        # Best |error| seen this episode, and how many moves have failed to beat it.
        self._edge_best_error: float | None = None
        self._edge_stall_count = 0
        # When the current wait-for-a-detection began, so it cannot freeze the
        # axis indefinitely while detections keep being issued.
        self._validate_hold_started: float | None = None
        # Phase-shifted sweeps already run since the last real sighting.
        self._search_restarts = 0
        self._last_good_degrees = 0.0
        self._offset_degrees = 0.0
        self._scan: NarrowingScan | None = None
        self._scan_target: float | None = None
        self._scan_seen_pick = 0
        self._scan_dwell_started = 0.0
        self._scan_arrival_not_before = 0.0
        maximum_output_velocity = speed_rpm / gear_ratio * 6.0
        # Velocity-form PID on the image error (image-based visual servoing).
        # This replaces the old low-pass + second-order position smoothing that
        # made centring crawl: a large error now commands a fast velocity, and a
        # centred target commands zero velocity so the axis simply holds.
        self._servo = VisualServoPid(
            kp=servo_kp,
            ki=servo_ki,
            kd=servo_kd,
            max_velocity_degrees_s=maximum_output_velocity,
            deadband=servo_deadband,
        )
        self._coarse_aim = CoarseToFineAim(
            degrees_per_error=degrees_per_error,
            deadband=servo_deadband,
            maximum_step_degrees=aim_max_step_degrees,
        )
        # Direct aim issues one computed move per settle period, so it must not
        # start another until the image has caught up with the last one.
        self._aim_hold_until = 0.0
        self._last_aim_delta = 0.0
        self._aim_error_at_move: float | None = None
        self._transport = SocketCanTransport(interface)
        mode = ChecksumMode(checksum)
        self._mode = mode
        # Verify the physical state before changing modes.  The accumulated
        # encoder becomes this session's origin; ordinary starts must never
        # rewrite the motor's persistent zero coordinate.
        probe = MksSingleAxisProbe(self._transport, node_id, mode)
        snapshot = probe.snapshot()
        if abs(snapshot.rpm) > 1:
            self._transport.close()
            raise RuntimeError(f"J1 is already moving at {snapshot.rpm} RPM")
        self.anchor_counts = snapshot.encoder_counts
        self.node_id = node_id
        self._transport.send(set_working_mode(node_id, 0x05, mode))
        self._transport.send(set_bus_enabled(node_id, True, mode))

    def update(self, alignment: dict[str, object]) -> None:
        from lamp_core.mks_can_protocol import absolute_coordinate_move
        from lamp_core.mks_single_axis import COUNTS_PER_REVOLUTION

        now = time.monotonic()
        dt = max(0.001, min(now - self._last_step_at, 0.25))
        self._last_step_at = now
        # The wait budget belongs to one detection, so it restarts once none is
        # in flight.
        if not alignment.get("awaiting_confirmation"):
            self._validate_hold_started = None
        velocity = 0.0
        bbox = (
            tuple(float(value) for value in alignment["bbox_norm"])
            if alignment.get("found")
            else None
        )
        fully_visible = bbox is not None and not self._bbox_is_cropped(bbox)
        if fully_visible:
            # Only an unclipped sighting counts as a found pose. Recording a
            # clipped box here kept the target inside the hold window below for
            # ever, so a book half out of frame froze the axis and the search
            # below could never run.
            self._last_found_at = now
        if fully_visible:
            # The whole visible page fits in frame, so follow it normally and
            # forget any scan that was still narrowing.
            self._edge_started_at = None
            self._edge_best_error = None
            self._edge_stall_count = 0
            self._search_restarts = 0
            self._scan = None
            observation = observe_bbox(bbox, horizontal_deadband=self.servo_deadband)
            velocity = self._aim_block(observation, now, dt)
            self._last_good_degrees = self._offset_degrees
            self._search_started_at = now
            alignment["tracking_mode"] = (
                "book_follow"
                if alignment.get("priority") in ("semantic_book", "page_rectangle", "page_lock")
                else "text_follow"
            )
            alignment["aim_delta_degrees"] = round(self._last_aim_delta, 2)
        elif (
            bbox is not None
            and alignment.get("awaiting_confirmation")
            and self._validate_hold_active(now)
        ):
            # A detection is out for this angle and there is a candidate to
            # validate. Hold perfectly still so it runs on a sharp frame, and
            # freeze the dwell clock while waiting. The wait is bounded, and it
            # deliberately does not apply when nothing has been found: detections
            # are issued back to back, so an unbounded hold starved every other
            # branch -- once freezing the axis for 36 seconds straight while the
            # book sat clipped at the frame edge.
            self._servo.reset()
            self._search_started_at += dt
            if self._scan is not None:
                self._scan_dwell_started += dt
            alignment["tracking_mode"] = "validate_hold"
        elif bbox is not None and self._edge_recentre_active(
            now, abs(self._error_x_of(bbox))
        ):
            # Visible but clipped by a frame edge. The visible centre of a clipped
            # page sits toward that edge, so nulling that error is exactly what
            # drags the missing part back into frame, and it self-corrects in
            # either direction. It runs through the same bounded, verified aim as
            # normal following: an unbounded velocity here once drove 40 deg to
            # the travel stop in three seconds. Never recorded as a good centred
            # pose, so a clip that refuses to clear still reaches the search.
            observation = observe_bbox(bbox, horizontal_deadband=self.servo_deadband)
            velocity = self._aim_block(observation, now, dt)
            if self._last_aim_delta:
                self._record_edge_move(abs(observation.error_x))
            alignment["tracking_mode"] = "edge_recentre"
            alignment["aim_delta_degrees"] = round(self._last_aim_delta, 2)
        elif bbox is not None:
            # A trusted page is still visible, but repeated horizontal edge
            # corrections have stopped improving it.  Never turn that into a
            # blind sweep: doing so made a valid lock walk -10, -20, +20 degrees
            # across the desk.  Hold the best visible pose until a refreshed
            # box either becomes actionable again or the trust gate drops it.
            self._servo.reset()
            self._last_good_degrees = self._offset_degrees
            self._last_found_at = now
            alignment["tracking_mode"] = "visible_hold"
            alignment["aim_delta_degrees"] = 0.0
        elif now - self._last_found_at < self.hold_seconds:
            # Hold the offset that worked instead of sweeping away from it.
            self._servo.reset()
            self._offset_degrees = self._last_good_degrees
            alignment["tracking_mode"] = "centred_hold"
        else:
            self._servo.reset()
            before_scan = self._offset_degrees
            self._advance_narrowing_scan(alignment, now)
            scan_delta = self._offset_degrees - before_scan
            if abs(scan_delta) > 1e-6:
                maximum_output_speed = self.speed_rpm / self.gear_ratio * 6.0
                velocity = (1.0 if scan_delta > 0 else -1.0) * maximum_output_speed
        desired_degrees = self._offset_degrees
        offset_counts = round(desired_degrees / 360 * COUNTS_PER_REVOLUTION * self.gear_ratio)
        target_counts = self.anchor_counts + offset_counts
        alignment["j1_raw_target_degrees"] = round(desired_degrees, 2)
        alignment["j1_target_degrees"] = round(desired_degrees, 2)
        alignment["j1_velocity_degrees_s"] = round(velocity, 2)
        alignment["j1_target_counts"] = target_counts
        alignment["message"] += f" | motor {desired_degrees:+.1f} deg"
        if (
            self._last_target_counts is not None
            and abs(target_counts - self._last_target_counts) < 16
            and abs(velocity) < 1.0
        ) or now - self._last_send_at < self.minimum_interval_s:
            return
        # Command the F5 speed from the PID velocity so a near-centre
        # correction does not hit the target as hard as a large move.
        profiled_speed_rpm = profiled_motor_speed_rpm(
            abs(velocity),
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

    def _clamp_travel(self, degrees: float) -> float:
        return max(-self.travel_degrees, min(self.travel_degrees, degrees))

    @staticmethod
    def _bbox_is_cropped(
        bbox_norm: tuple[float, float, float, float],
        margin: float = 0.02,
    ) -> bool:
        """True when the page is clipped on an edge that J1 can correct.

        J1 is a horizontal axis.  A page touching the top or bottom of the image
        cannot be repaired by yawing left or right, and treating that as a lost
        horizontal view caused a centred book to fall into a wide search.
        """

        x1, _, x2, _ = bbox_norm
        return x1 <= margin or x2 >= 1.0 - margin

    def _edge_recentre_active(self, now: float, error_abs: float | None = None) -> bool:
        """Whether a clipped target still deserves servo time.

        Returns True and starts the clock on the first clipped sighting, then
        True until the budget runs out or pushing stops helping. After that the
        caller falls through to the narrowing search, which is what a target
        wider than the field of view -- or a detection that is not really the
        book -- genuinely needs.
        """

        if self._edge_stall_count >= self.EDGE_STALL_LIMIT:
            return False
        if self._edge_started_at is None:
            self._edge_started_at = now
        return now - self._edge_started_at < max(self.hold_seconds * 2.0, 1.0)

    def _record_edge_move(self, error_abs: float) -> None:
        """Count actual correction attempts, never raw camera frames."""

        if (
            self._edge_best_error is None
            or error_abs < self._edge_best_error - self.EDGE_PROGRESS_DELTA
        ):
            self._edge_best_error = error_abs
            self._edge_stall_count = 0
        else:
            self._edge_stall_count += 1

    @staticmethod
    def _error_x_of(bbox_norm: tuple[float, float, float, float]) -> float:
        """Horizontal aim error straight from the box, without range validation."""

        return (bbox_norm[0] + bbox_norm[2]) / 2.0 - 0.5

    def _validate_hold_active(self, now: float) -> bool:
        """Whether waiting for an in-flight detection still earns a frozen axis.

        Holding still gives the detection a sharp frame, which is worth about as
        long as one detection takes. Beyond that the wait is not buying anything
        and only stops the axis from doing what the situation needs.
        """

        if self._validate_hold_started is None:
            self._validate_hold_started = now
        return now - self._validate_hold_started <= self.VALIDATE_HOLD_SECONDS

    def _aim_delta_degrees(self, error_x: float) -> float:
        """Joint change that should null this image error outright.

        The calibrated field of view says one unit of normalized image error is
        worth ``degrees_per_error`` of joint rotation, so the correction can be
        computed instead of integrated toward. The sign goes through the same
        verified positive-command direction the velocity servo uses.
        """

        delta = error_x * self.degrees_per_error
        if self.positive_camera_direction is HorizontalDirection.RIGHT:
            return delta
        return -delta

    def _aim_block(self, observation: AlignmentObservation, now: float, dt: float) -> float:
        """Move the axis onto the target, returning the velocity to command at.

        Direct aim places the target with one computed move per settle period
        instead of integrating a velocity toward it. The settle gap matters: the
        image lags the command, so measuring again immediately would correct a
        position the axis has not reached yet and overshoot. Between moves the
        axis is told to hold still.
        """

        self._last_aim_delta = 0.0
        if not self.direct_aim:
            velocity = self._servo_velocity(observation, dt)
            self._offset_degrees = self._clamp_travel(self._offset_degrees + velocity * dt)
            return velocity
        if abs(observation.error_x) <= self.servo_deadband or now < self._aim_hold_until:
            return 0.0
        previous = self._aim_error_at_move
        if (
            previous is not None
            and abs(observation.error_x - previous) <= self.AIM_VERIFY_DELTA
        ):
            # The same reading we already corrected for. The measurement has not
            # caught up with the last move yet, so acting on it again would drive
            # the axis past the target for an error that is already being fixed.
            return 0.0
        coarse_aim = getattr(self, "_coarse_aim", None)
        if coarse_aim is None:
            coarse_aim = CoarseToFineAim(
                degrees_per_error=self.degrees_per_error,
                deadband=self.servo_deadband,
                maximum_step_degrees=self.aim_max_step_degrees,
            )
            self._coarse_aim = coarse_aim
        logical_delta = coarse_aim.update(observation.error_x)
        delta = (
            logical_delta
            if self.positive_camera_direction is HorizontalDirection.RIGHT
            else -logical_delta
        )
        if not delta:
            return 0.0
        self._offset_degrees = self._clamp_travel(self._offset_degrees + delta)
        maximum_output_speed = self.speed_rpm / self.gear_ratio * 6.0
        estimated_travel = abs(delta) / max(maximum_output_speed, 1e-3)
        self._aim_hold_until = now + max(
            self.aim_settle_seconds,
            estimated_travel + 0.15,
        )
        self._aim_error_at_move = observation.error_x
        self._last_aim_delta = delta
        # Speed the firmware should use for this one move.
        return delta / max(self.aim_settle_seconds, 1e-3)

    def _servo_velocity(self, observation: AlignmentObservation, dt: float) -> float:
        """PID velocity for this frame, in output degrees per second.

        The goal is the page centred *in the image*, so the measured frame error
        is exactly the signal to null: the shift the axis itself produces is the
        control effect, not noise to be removed. The sign is mapped through the
        verified positive-command direction.
        """

        velocity = self._servo.update(observation.error_x, dt)
        if self.positive_camera_direction is HorizontalDirection.RIGHT:
            return velocity
        return -velocity

    def _advance_narrowing_scan(self, alignment: dict[str, object], now: float) -> None:
        """Dwell on each preset angle, score it, and halve the step onto the best.

        Sampling every angle from a settled camera gives each detection a sharp
        frame, and comparing the scores is what finds the angle where the book is
        most fully in view rather than stopping at the first angle that happened
        to see it.
        """

        pick_count = int(alignment.get("pick_count", 0) or 0)
        if self._scan is not None and now < self._scan_arrival_not_before:
            # A result returned while travelling belongs to an image captured at
            # the previous angle.  Consume its sequence number but never score it
            # against the new target.
            self._scan_seen_pick = pick_count
            alignment["scan_phase"] = "moving"
            self._publish_scan(alignment)
            return
        if self._scan is not None and self._scan.finished:
            # A finished sweep must never be scored again -- record() raises once
            # the last round is complete -- so the next pick is where the sweep is
            # either repeated or settled. Nothing scoring above zero means the book
            # was not in view at any sampled angle, so parking on an angle that
            # scored zero would just sit there; sweep again with the samples
            # shifted by half a step to cover the angles in between. After a
            # couple of passes the miss escalation asks the user to move the book.
            if self._scan.best_score <= 0.0 and self._search_restarts < self.SEARCH_RESTART_LIMIT:
                self._search_restarts += 1
                self._scan = self._new_scan(alignment)
                self._scan_seen_pick = pick_count
                self._set_scan_target(self._scan.next_target(), now)
                alignment["scan_restarts"] = self._search_restarts
            elif self._scan_target is None:
                self._set_scan_target(self._scan.best_degrees, now)
            self._publish_scan(alignment)
            return
        if self._scan is None:
            self._scan = self._new_scan(alignment)
            self._scan_seen_pick = pick_count
            self._set_scan_target(self._scan.next_target(), now)
        elif pick_count != self._scan_seen_pick:
            self._scan_seen_pick = pick_count
            self._scan.record(float(alignment.get("page_score", 0.0) or 0.0))
            self._set_scan_target(self._scan.next_target(), now)
        elif now - self._scan_dwell_started > self.scan_dwell_seconds * 3.0:
            # Nothing came back for this angle; score it zero and move on so a
            # silent angle cannot stall the whole scan.
            self._scan.record(0.0)
            self._set_scan_target(self._scan.next_target(), now)
        if self._scan_target is None:
            self._set_scan_target(self._scan.best_degrees, now)
        self._publish_scan(alignment)

    def _set_scan_target(self, target: float | None, now: float) -> None:
        """Adopt one scan pose and delay scoring until it can physically arrive."""

        previous = self._offset_degrees
        self._scan_target = target
        if target is None:
            self._scan_arrival_not_before = now
            self._scan_dwell_started = now
            return
        maximum_output_speed = self.speed_rpm / self.gear_ratio * 6.0
        estimated_travel = abs(target - previous) / max(maximum_output_speed, 1e-3)
        self._scan_arrival_not_before = now + estimated_travel + 0.15
        self._scan_dwell_started = self._scan_arrival_not_before

    def _publish_scan(self, alignment: dict[str, object]) -> None:
        self._offset_degrees = self._clamp_travel(self._scan_target)
        alignment["tracking_mode"] = "narrowing_scan"
        alignment["scan_best_degrees"] = round(self._scan.best_degrees, 1)
        alignment["scan_best_score"] = round(self._scan.best_score, 4)
        alignment["scan_finished"] = self._scan.finished

    def _salient_first_degrees(self, alignment: dict[str, object]) -> float | None:
        """Angle worth sampling first, from the cheap whole-frame saliency hint."""

        third = alignment.get("salient_third")
        if third is None:
            return None
        try:
            centre = (int(third) + 0.5) / 3.0
        except (TypeError, ValueError):
            return None
        # The third's centre sits at 1/6, 1/2 or 5/6 of the frame, so aiming it at
        # the middle is an ordinary aim correction for that error.
        return self._aim_delta_degrees(centre - 0.5)

    def _new_scan(self, alignment: dict[str, object]) -> NarrowingScan:
        """Build the next sweep, biased by saliency and phase-shifted on retries."""

        first = self._salient_first_degrees(alignment) if self._search_restarts == 0 else None
        if first is None and self._search_restarts:
            half_step = self.scan_step_degrees / 2.0
            first = half_step if self._search_restarts % 2 else -half_step
        return NarrowingScan(
            envelope_degrees=self.search_degrees,
            step_degrees=self.scan_step_degrees,
            minimum_step_degrees=max(1.0, self.scan_step_degrees / 4.0),
            first_degrees=first,
        )

    def reanchor(self) -> None:
        """Treat the current encoder position as the new zero for the search."""

        from lamp_core.mks_single_axis import MksSingleAxisProbe

        if hasattr(self._transport, "receive"):
            snapshot = MksSingleAxisProbe(self._transport, self.node_id, self._mode).snapshot()
            if abs(snapshot.rpm) > 1:
                raise RuntimeError(f"J1 is still moving at {snapshot.rpm} RPM")
            self.anchor_counts = snapshot.encoder_counts
        self._last_target_counts = None
        self._last_send_at = 0.0
        self._search_started_at = time.monotonic()
        self._last_step_at = time.monotonic()
        self._offset_degrees = 0.0
        self._last_good_degrees = 0.0
        self._aim_hold_until = 0.0
        self._last_aim_delta = 0.0
        self._aim_error_at_move = None
        self._edge_started_at = None
        self._edge_best_error = None
        self._edge_stall_count = 0
        self._validate_hold_started = None
        self._search_restarts = 0
        self._scan = None
        self._scan_target = None
        self._scan_arrival_not_before = 0.0
        self._servo.reset()
        coarse_aim = getattr(self, "_coarse_aim", None)
        if coarse_aim is not None:
            coarse_aim.reset()

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
        self._last_local_bbox: tuple[int, int, int, int] | None = None
        self._reset_requested = threading.Event()

    def request_reset(self) -> None:
        """Ask the capture loop to restart tracking from the current position."""
        self._reset_requested.set()

    def _apply_reset(self) -> None:
        self._reset_requested.clear()
        if self.semantic_tracker is not None:
            self.semantic_tracker.reset()
        if self.j1_follower is not None:
            self.j1_follower.reanchor()
        self._local_candidate_since = 0.0
        self._last_local_bbox = None
        self._locked_page_alignment = None

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
                if self._reset_requested.is_set():
                    self._apply_reset()
                started = time.monotonic()
                image = self._camera.capture_array()
                if self.rotation:
                    image = cv2.rotate(image, rotate_codes[self.rotation])
                if self.semantic_tracker is not None:
                    local_bbox, local_priority = detect_priority_target(image, cv2)
                    now = time.monotonic()
                    if local_bbox is None:
                        self._local_candidate_since = 0.0
                        self._last_local_bbox = None
                    else:
                        same_candidate = (
                            self._last_local_bbox is not None
                            and bbox_intersection_over_union(self._last_local_bbox, local_bbox) >= 0.55
                        )
                        if not same_candidate or self._local_candidate_since == 0.0:
                            self._local_candidate_since = now
                        self._last_local_bbox = local_bbox
                    candidate_stable = local_candidate_is_stable(
                        self._local_candidate_since,
                        now,
                    )
                    if self.semantic_tracker.help_requested and candidate_stable:
                        self.semantic_tracker.resume_after_stable_local_candidate()
                        self._local_candidate_since = 0.0
                    semantic_bbox = self.semantic_tracker.update(
                        image,
                        cv2,
                        local_bbox=local_bbox,
                        allow_reacquire=(
                            candidate_stable and not self.semantic_tracker.help_requested
                        ),
                    )
                    selected_bbox, priority = choose_tracking_target(
                        local_bbox,
                        local_priority,
                        semantic_bbox,
                    )
                    if selected_bbox is None:
                        priority = self.semantic_tracker.status
                    image, alignment = annotate_bbox(
                        image,
                        cv2,
                        selected_bbox,
                        priority,
                        # A label that says centred while the axis is still moving
                        # would be worse than useless, so match the servo.
                        horizontal_deadband=getattr(self.j1_follower, "servo_deadband", 0.08),
                    )
                    if local_bbox is not None:
                        frame_height, frame_width = image.shape[:2]
                        bx, by, box_width, box_height = local_bbox
                        alignment["local_candidate_norm"] = (
                            round(bx / frame_width, 4),
                            round(by / frame_height, 4),
                            round((bx + box_width) / frame_width, 4),
                            round((by + box_height) / frame_height, 4),
                        )
                        alignment["local_candidate_stable"] = candidate_stable
                    alignment["awaiting_confirmation"] = self.semantic_tracker.awaiting_confirmation
                    alignment["page_score"] = round(self.semantic_tracker.page_score, 4)
                    alignment["pick_count"] = self.semantic_tracker.pick_count
                    alignment["semantic_status"] = self.semantic_tracker.status
                    if self.semantic_tracker.page_text:
                        alignment["page_text"] = self.semantic_tracker.page_text
                    alignment["api_miss_count"] = self.semantic_tracker.api_miss_count
                    # Which detector owns the aim point right now. Without this the
                    # logs cannot tell a source switch from ordinary tracker drift.
                    if self.semantic_tracker.anchor_source is not None:
                        alignment["anchor_source"] = self.semantic_tracker.anchor_source
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
                if not alignment.get("found"):
                    # Nothing found, so a sweep is about to run. Computing the hint
                    # here, before the follower looks at the frame, is what lets it
                    # order the sweep; a frame later would be too late, because the
                    # scan is created on the first frame it is needed.
                    #
                    # A cheap whole-frame saliency profile only orders that sweep --
                    # it never counts as a detection, because a monitor showing a
                    # picture scores just as high.
                    profile = saliency_thirds(image, cv2)
                    if max(profile) > 0.0:
                        alignment["saliency"] = tuple(round(value, 3) for value in profile)
                        alignment["salient_third"] = profile.index(max(profile))
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
            # Wake MJPEG clients and make their streaming loops exit before the
            # HTTP server waits for handler threads.
            if self.error is None:
                self.error = "preview stopped"
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
            path = self.path.split("?", 1)[0]
            if path == "/restart":
                stream.request_reset()
                self._send_bytes(HTTPStatus.OK, "application/json", b'{"restarting":true}')
                return
            if path != "/shutdown":
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
    parser.add_argument("--j1-checksum", choices=("additive", "fixed_6b"), default="additive")
    parser.add_argument("--j1-gear-ratio", type=float)
    parser.add_argument("--j1-positive-camera-direction", choices=("left", "right"))
    parser.add_argument("--j1-envelope-degrees", type=float, default=10.0)
    parser.add_argument(
        "--j1-search-degrees",
        type=float,
        help="initial find-the-book sweep width; defaults to twice the centring envelope",
    )
    parser.add_argument(
        "--j1-travel-degrees",
        type=float,
        default=60.0,
        help="total travel allowed each side of the origin while following a found book",
    )
    parser.add_argument(
        "--j1-servo-kp",
        type=float,
        default=180.0,
        help="image-error -> deg/s proportional gain (180 ~= 44.5 deg FOV / 0.25 s)",
    )
    parser.add_argument(
        "--j1-servo-ki",
        type=float,
        default=80.0,
        help="PID integral gain; removes the final steady-state offset (anti-windup clamped)",
    )
    parser.add_argument(
        "--j1-servo-kd",
        type=float,
        default=8.0,
        help="PID derivative gain; low-pass filtered to reject box jitter",
    )
    parser.add_argument(
        "--j1-degrees-per-error",
        type=float,
        default=44.5,
        help="calibrated horizontal FOV; also scales the camera self-motion removed by the servo",
    )
    parser.add_argument(
        "--j1-scan-step-degrees",
        type=float,
        default=10.0,
        help="patrol scan step between dwell positions",
    )
    parser.add_argument(
        "--j1-scan-dwell-seconds",
        type=float,
        default=2.5,
        help="how long to hold still at each patrol position so detection sees a sharp frame",
    )
    parser.add_argument(
        "--j1-servo-deadband",
        type=float,
        default=0.03,
        help="aim tolerance in normalized image error; 0.03 is about 1.3 deg on this rig",
    )
    parser.add_argument(
        "--j1-direct-aim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="place the target with one computed move per settle period instead of a velocity PID",
    )
    parser.add_argument(
        "--j1-aim-settle-seconds",
        type=float,
        default=0.8,
        help="wait this long after a direct-aim move before measuring again",
    )
    parser.add_argument(
        "--j1-aim-max-step-degrees",
        type=float,
        default=8.0,
        help="largest single direct-aim move; bounds the damage a lagging measurement can do",
    )
    parser.add_argument("--j1-speed-rpm", type=int)
    parser.add_argument("--j1-acceleration", type=int, default=40)
    parser.add_argument(
        "--semantic-book-priority",
        action="store_true",
        help="use configured cloud vision once to select a picture-book page, then track locally",
    )
    parser.add_argument("--opencv-tracker", choices=("kcf", "csrt"), default="kcf")
    parser.add_argument("--semantic-reacquire-seconds", type=float, default=8.0)
    parser.add_argument(
        "--local-text-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="locate the page with the on-device PP-OCR detector before calling the cloud",
    )
    parser.add_argument(
        "--text-model-path",
        help="path to text_detection_en_ppocrv3_2023may.onnx; searched for by default",
    )
    parser.add_argument(
        "--text-detect-long-side",
        type=int,
        default=320,
        help="PP-OCR network long side; 320 is about 0.3 s per frame on a Pi 3B",
    )
    parser.add_argument("--text-score-threshold", type=float, default=0.5)
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
            search_degrees=args.j1_search_degrees,
            travel_degrees=args.j1_travel_degrees,
            servo_kp=args.j1_servo_kp,
            servo_ki=args.j1_servo_ki,
            servo_kd=args.j1_servo_kd,
            servo_deadband=args.j1_servo_deadband,
            direct_aim=args.j1_direct_aim,
            aim_settle_seconds=args.j1_aim_settle_seconds,
            aim_max_step_degrees=args.j1_aim_max_step_degrees,
            degrees_per_error=args.j1_degrees_per_error,
            scan_step_degrees=args.j1_scan_step_degrees,
            scan_dwell_seconds=args.j1_scan_dwell_seconds,
            speed_rpm=args.j1_speed_rpm,
            acceleration=args.j1_acceleration,
        )

    semantic_tracker = None
    if args.semantic_book_priority:
        from lamp_core.config import AppConfig, load_dotenv

        load_dotenv()
        config = AppConfig.from_environment()

        text_detector = None
        if args.local_text_detection:
            from lamp_core.text_detection import PpocrTextDetector

            text_detector = PpocrTextDetector(
                args.text_model_path or None,
                long_side=args.text_detect_long_side,
                score_threshold=args.text_score_threshold,
            )
            if text_detector.available:
                print(f"Local text detector: {text_detector.model_path}")
            else:
                print("Local text detector: no ONNX model found, using the cloud only")

        client = None
        if config.cloud_enabled and config.api_key:
            from app_main import cloud_client

            client = cloud_client(config)

        if client is None and (text_detector is None or not text_detector.available):
            raise SystemExit(
                "semantic book priority needs either the local text detector "
                "(models/text_detection_en_ppocrv3_2023may.onnx) or "
                "ENABLE_CLOUD_VISION=true with OPENAI_API_KEY"
            )
        semantic_tracker = SemanticBookTracker(
            client,
            reacquire_seconds=args.semantic_reacquire_seconds,
            tracker_name=args.opencv_tracker,
            text_detector=text_detector,
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
    # A browser may keep /stream.mjpg open indefinitely. Such clients must not
    # prevent SIGTERM or the Stop button from ending the process.
    server.daemon_threads = True
    server.block_on_close = False
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f"Camera preview: http://<raspberry-pi-ip>:{args.port}")
    print("Press Ctrl+C to stop. The camera cannot be shared with app_main.py.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        server.server_close()
        if follower is not None:
            follower.close()


if __name__ == "__main__":
    main()
