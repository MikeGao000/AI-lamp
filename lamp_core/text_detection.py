"""On-device picture-book anchoring via PP-OCRv3 text detection.

The cloud model locates a picture-book page reliably, but every acquisition costs
a network round trip and tokens. A DBNet text detector trained on real scene text
finds the story text on a page locally, for free, in roughly a third of a second
on the Pi -- and that text cluster is a dependable anchor for the J1 axis, which
pans horizontally only. Measured 0.98-0.99 confidence on real frames, framing the
story text exactly.

Model: ``text_detection_en_ppocrv3_2023may.onnx`` (2.4 MB) from OpenCV Zoo's
``models/text_detection_ppocr``, driven through ``cv2.dnn.TextDetectionModel_DB``.

Measured on the Raspberry Pi 3B (long side 320 / 480 / 640 px): about
0.30 s / 0.60 s / 1.15 s per frame.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

DEFAULT_MODEL_FILENAME = "text_detection_en_ppocrv3_2023may.onnx"

# PaddleOCR normalisation, byte for byte what opencv_zoo's ppocr_det.py applies.
# Leaving it out is the difference between 0.98 confidence and near-noise: the DB
# network expects (pixel - mean) * scale, not raw 0..255 pixels.
INPUT_MEAN = (123.675, 116.28, 103.53)
INPUT_SCALE = (1.0 / 255.0 / 0.229, 1.0 / 255.0 / 0.224, 1.0 / 255.0 / 0.225)

# DBNet downsamples by 32 three times, so both sides must be a multiple of 32.
_SIZE_MULTIPLE = 32
_MIN_SIDE = 32


@dataclass(frozen=True)
class TextBox:
    """One detected text line, normalised to the 0..1 frame."""

    bbox: tuple[float, float, float, float]
    confidence: float


def candidate_model_paths(filename: str = DEFAULT_MODEL_FILENAME) -> tuple[str, ...]:
    """Places the ONNX detector may live, in priority order."""

    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(package_dir)
    return (
        os.path.join(project_dir, "models", filename),
        os.path.join(package_dir, "models", filename),
        os.path.join(project_dir, filename),
        os.path.join(os.getcwd(), filename),
    )


def find_model_path(
    filename: str = DEFAULT_MODEL_FILENAME,
    explicit: str | None = None,
) -> str | None:
    """Return the detector ONNX path, or None when it is not installed."""

    if explicit:
        expanded = os.path.expanduser(explicit)
        return expanded if os.path.isfile(expanded) else None
    for path in candidate_model_paths(filename):
        if os.path.isfile(path):
            return path
    return None


def _axis_aligned(rectangle: object) -> tuple[float, float, float, float] | None:
    """Reduce a detected text polygon to (x1, y1, x2, y2)."""

    xs: list[float] = []
    ys: list[float] = []
    for point in rectangle:  # type: ignore[union-attr]
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            # Detector output is third-party data; a shape we cannot read must
            # drop the box, never abort the whole frame.
            return None
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


class PpocrTextDetector:
    """Find text lines on a picture-book page with PP-OCRv3's DBNet head."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        long_side: int = 320,
        score_threshold: float = 0.5,
        binary_threshold: float = 0.3,
        polygon_threshold: float = 0.5,
        unclip_ratio: float = 2.0,
        max_candidates: int = 200,
        cv2_module: object | None = None,
    ) -> None:
        self.model_path = find_model_path(explicit=model_path)
        self.long_side = max(_MIN_SIDE, int(long_side))
        self.score_threshold = float(score_threshold)
        self.binary_threshold = float(binary_threshold)
        self.polygon_threshold = float(polygon_threshold)
        self.unclip_ratio = float(unclip_ratio)
        self.max_candidates = int(max_candidates)
        self.last_detect_seconds = 0.0
        self.last_box_count = 0
        self._cv2 = cv2_module
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True when the ONNX detector is present on disk."""

        return self.model_path is not None

    def _cv(self) -> object:
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        cv2 = self._cv()
        model = cv2.dnn.TextDetectionModel_DB(cv2.dnn.readNet(self.model_path))
        model.setBinaryThreshold(self.binary_threshold)
        model.setPolygonThreshold(self.polygon_threshold)
        model.setMaxCandidates(self.max_candidates)
        model.setUnclipRatio(self.unclip_ratio)
        model.setInputMean(INPUT_MEAN)
        model.setInputScale(INPUT_SCALE)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._model = model
        return model

    @staticmethod
    def fitted_size(width: int, height: int, long_side: int) -> tuple[int, int]:
        """Scale the long side to ``long_side``, both sides multiples of 32."""

        scale = long_side / float(max(width, height))
        fitted_width = max(
            _MIN_SIDE, int(round(width * scale / _SIZE_MULTIPLE)) * _SIZE_MULTIPLE
        )
        fitted_height = max(
            _MIN_SIDE, int(round(height * scale / _SIZE_MULTIPLE)) * _SIZE_MULTIPLE
        )
        return fitted_width, fitted_height

    def detect(self, image: object) -> list[TextBox]:
        """Return normalised text boxes for a frame, most confident first."""

        if self.model_path is None:
            return []
        frame_height, frame_width = image.shape[:2]  # type: ignore[union-attr]
        if frame_height <= 0 or frame_width <= 0:
            return []
        cv2 = self._cv()
        fitted_width, fitted_height = self.fitted_size(
            frame_width, frame_height, self.long_side
        )
        frame = cv2.resize(image, (fitted_width, fitted_height))
        with self._lock:
            model = self._ensure_model()
            model.setInputSize(fitted_width, fitted_height)
            started = time.perf_counter()
            rectangles, confidences = model.detect(frame)
            self.last_detect_seconds = time.perf_counter() - started
        boxes: list[TextBox] = []
        for rectangle, confidence in zip(rectangles, confidences):
            score = float(confidence)
            if score < self.score_threshold:
                continue
            axis_aligned = _axis_aligned(rectangle)
            if axis_aligned is None:
                continue
            x1, y1, x2, y2 = axis_aligned
            boxes.append(
                TextBox(
                    (
                        max(0.0, min(1.0, x1 / fitted_width)),
                        max(0.0, min(1.0, y1 / fitted_height)),
                        max(0.0, min(1.0, x2 / fitted_width)),
                        max(0.0, min(1.0, y2 / fitted_height)),
                    ),
                    score,
                )
            )
        boxes.sort(key=lambda box: box.confidence, reverse=True)
        self.last_box_count = len(boxes)
        return boxes


def text_anchor_box(
    boxes: list[TextBox],
    *,
    reference_bbox: tuple[float, float, float, float] | None = None,
    margin: float = 0.15,
    top_edge_margin: float = 0.05,
    block_span: float = 1.5,
    minimum_width: float = 0.10,
    minimum_area: float = 0.01,
) -> tuple[float, float, float, float] | None:
    """Bounding box of the dominant text block, grown by ``margin``, or None.

    Anchoring on the union of *every* detected box let the aim point jump around:
    which lines a detector finds varies from frame to frame, so one stray box
    elsewhere on the page moved the centre by a third of the frame and the servo
    hunted. The largest box therefore picks the block, and only boxes centred
    over it take part -- which also drops a lone false positive off to one side.

    Boxes touching the top edge are dropped as monitor or window chrome above the
    working area rather than book text, and text clipped by the frame edge is not
    a stable anchor anyway.

    Small boxes are dropped too. A page's story text is large -- measured boxes
    on this rig are 0.60-0.68 of the frame wide scoring 0.15-0.31 -- whereas a
    stray word is far smaller: one 0.07x0.08 fragment scoring 0.004 captured the
    anchor and parked the axis against its travel stop.
    """

    candidates = text_anchor_candidates(
        boxes,
        margin=margin,
        top_edge_margin=top_edge_margin,
        block_span=block_span,
        minimum_width=minimum_width,
        minimum_area=minimum_area,
    )
    if not candidates:
        return None
    if reference_bbox is None:
        return candidates[0]

    # Prefer the block that continues the previously accepted block.  The old
    # implementation selected the largest block afresh on every refresh, so a
    # detector alternating between two paragraphs moved the aim point by several
    # degrees even though the book itself had not moved.
    associated = sorted(
        candidates,
        key=lambda candidate: (
            _bbox_iou(candidate, reference_bbox),
            -_bbox_centre_distance(candidate, reference_bbox),
            _bbox_area(candidate),
        ),
        reverse=True,
    )
    best = associated[0]
    if _bbox_iou(best, reference_bbox) >= 0.05 or _bbox_centre_distance(
        best, reference_bbox
    ) <= 0.18:
        return best
    return candidates[0]


def text_anchor_candidates(
    boxes: list[TextBox],
    *,
    margin: float = 0.15,
    top_edge_margin: float = 0.05,
    block_span: float = 1.5,
    minimum_width: float = 0.10,
    minimum_area: float = 0.01,
) -> list[tuple[float, float, float, float]]:
    """Return stable text-block anchors, largest first.

    Keeping all plausible blocks lets the caller associate detections across
    frames instead of repeatedly jumping to whichever paragraph happened to be
    largest in one detector pass.
    """

    usable = [
        box.bbox
        for box in boxes
        if box.bbox[1] > top_edge_margin
        and (box.bbox[2] - box.bbox[0]) >= minimum_width
        and (box.bbox[2] - box.bbox[0]) * (box.bbox[3] - box.bbox[1]) >= minimum_area
    ]
    if not usable:
        return []

    remaining = list(usable)
    anchors: list[tuple[float, float, float, float]] = []
    while remaining:
        primary = max(remaining, key=_bbox_area)
        primary_height = max(1e-6, primary[3] - primary[1])
        members = [
            box
            for box in remaining
            if primary[0] <= (box[0] + box[2]) / 2.0 <= primary[2]
            and box[1] <= primary[3] + primary_height * block_span
            and box[3] >= primary[1] - primary_height * block_span
        ]
        for member in members:
            remaining.remove(member)
        x1 = min(box[0] for box in members)
        y1 = min(box[1] for box in members)
        x2 = max(box[2] for box in members)
        y2 = max(box[3] for box in members)
        width = x2 - x1
        height = y2 - y1
        anchors.append(
            (
                max(0.0, x1 - width * margin),
                max(0.0, y1 - height * margin),
                min(1.0, x2 + width * margin),
                min(1.0, y2 + height * margin),
            )
        )
    anchors.sort(key=_bbox_area, reverse=True)
    return anchors


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_centre_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    return max(
        abs((first[0] + first[2] - second[0] - second[2]) / 2.0),
        abs((first[1] + first[3] - second[1] - second[3]) / 2.0),
    )
