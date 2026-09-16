"""Structured visual target localization for the reading and gaze pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from lamp_core.autonomous_contracts import ContractError, VisualObject
from lamp_core.cloud import StoryClient


class ObjectLocalizationError(ValueError):
    pass


@dataclass(frozen=True)
class TargetDetection:
    """A model-detected target in normalized camera coordinates."""

    object_id: str
    label: str
    bbox_norm: tuple[float, float, float, float]
    confidence: float

    @property
    def center_x(self) -> float:
        return (self.bbox_norm[0] + self.bbox_norm[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox_norm[1] + self.bbox_norm[3]) / 2.0

    def validate(self) -> None:
        try:
            VisualObject(self.object_id, self.label, self.bbox_norm, self.confidence).validate()
        except ContractError as error:
            raise ObjectLocalizationError(str(error)) from error


@dataclass(frozen=True)
class SimulatedLocalTargetDetector:
    """Offline stand-in for a Pi detector while camera/model hardware is absent.

    Its boxes are explicitly annotated fixtures, not a claim that a trained
    model has inferred them.  The adapter exists to exercise exactly the same
    target-detection-to-motion interface that a TFLite/OpenNI implementation
    will provide on the Pi.
    """

    detections_by_image_name: Mapping[str, TargetDetection | None]

    def locate(self, image_name: str, jpeg: bytes) -> TargetDetection | None:
        if not image_name.strip() or not jpeg:
            raise ObjectLocalizationError("image name and bytes must not be empty")
        detection = self.detections_by_image_name.get(image_name)
        if detection is not None:
            detection.validate()
        return detection


def localization_instructions() -> str:
    return """You locate a user-requested physical target in one image.
Return only one json object. Never include Markdown or explanatory prose.
Use normalized image coordinates in the unrotated input image: x grows from
left to right and y grows from top to bottom.
If the target is visible, return exactly:
{"found":true,"label":"short target label","bbox_norm":[x1,y1,x2,y2],"confidence":0.0}
If it is not visible, return exactly: {"found":false}
The bounding box must tightly cover the requested target, not nearby objects."""


def localization_prompt(target_description: str) -> str:
    if not target_description.strip():
        raise ObjectLocalizationError("target_description must not be empty")
    return f"Find this target: {target_description.strip()} Reply as json."


def confirmation_instructions() -> str:
    """Binary presence question; far more reliable than pixel-perfect boxes."""

    return """You decide whether a requested target is present in one image.
Return only one json object. Never include Markdown or explanatory prose.
If any portion of the target is visible (including a page cropped by the image
edge, rotated, or partly occluded), return exactly: {"found":true}
If none of the target is visible, return exactly: {"found":false}"""


def confirmation_prompt(target_description: str, *, cropped_region: bool = False) -> str:
    if not target_description.strip():
        raise ObjectLocalizationError("target_description must not be empty")
    description = target_description.strip()
    if cropped_region:
        return (
            "This image is a crop of exactly one region taken from a larger photo. "
            f"Does this cropped region itself show {description}? "
            "Judge only this crop, not whatever else might exist in the original photo. "
            "Reply as json."
        )
    return f"Is any portion of this target visible? {description} Reply as json."


def confirm_target_present(
    jpeg: bytes,
    client: StoryClient,
    target_description: str,
    *,
    cropped_region: bool = False,
) -> bool:
    """Ask only whether the target is present; no bounding box is required.

    With ``cropped_region`` the question is scoped to one local candidate crop,
    so a region that is *not* a book is rejected even when a book happens to be
    visible somewhere else in the frame. Without it the question is global.
    """

    if not jpeg:
        raise ObjectLocalizationError("image must not be empty")
    response = client.describe_page(
        jpeg,
        prompt=confirmation_prompt(target_description, cropped_region=cropped_region),
        system_instructions=confirmation_instructions(),
    )
    payload = _json_object(response)
    if payload.get("found") is True:
        return True
    if payload.get("found") is False:
        return False
    raise ObjectLocalizationError("vision response must declare found true or false")


def selection_instructions() -> str:
    """Pick which numbered rectangle contains the target (classification, not regression)."""

    return """You choose which numbered rectangle drawn on one image contains a target.
Return only one json object. Never include Markdown or explanatory prose.
If exactly one numbered rectangle contains the target, return exactly: {"choice": N}
where N is that rectangle's number.
If no numbered rectangle contains the target, return exactly: {"choice": 0}"""


def selection_prompt(target_description: str, count: int) -> str:
    if not target_description.strip():
        raise ObjectLocalizationError("target_description must not be empty")
    if count < 1:
        raise ObjectLocalizationError("at least one candidate rectangle is required")
    return (
        f"The image has {count} numbered rectangles drawn on it. Which single numbered "
        f"rectangle contains {target_description.strip()}? Consider only the pixels inside "
        "each rectangle. Reply as json."
    )


def choose_candidate_index(
    jpeg: bytes,
    client: StoryClient,
    target_description: str,
    count: int,
) -> int:
    """Return the 1-based number of the rectangle the model chose, or 0 for none.

    Choosing among a handful of local candidates is a classification, which the
    vision model does far more reliably than emitting a precise pixel box.
    """

    if not jpeg:
        raise ObjectLocalizationError("image must not be empty")
    response = client.describe_page(
        jpeg,
        prompt=selection_prompt(target_description, count),
        system_instructions=selection_instructions(),
    )
    payload = _json_object(response)
    choice = payload.get("choice")
    if isinstance(choice, bool) or not isinstance(choice, int):
        raise ObjectLocalizationError("vision response must contain an integer choice")
    if not 0 <= choice <= count:
        raise ObjectLocalizationError("vision response choice is outside the offered rectangles")
    return choice


@dataclass(frozen=True)
class PageReading:
    """One combined locate-and-read answer for a picture-book page."""

    bbox_norm: tuple[float, float, float, float]
    visible_text: str
    confidence: float


def page_instructions() -> str:
    return """You find and read one children's picture book page in a single photo.
Return only one json object. Never include Markdown or explanatory prose.
If any portion of an open children's picture book page is visible, return exactly:
{"found":true,"bbox_norm":[x1,y1,x2,y2],"visible_text":"printed words you can read","confidence":0.0}
bbox_norm must cover every visible pixel of that book page in normalized coordinates, where
x grows left to right and y grows top to bottom, so a page cropped by the image edge is still
allowed. visible_text must quote only printed words that are genuinely legible, and must be an
empty string when there are none.
If no picture book page is visible, return exactly: {"found":false}
Exclude computer screens, product packaging, loose unrelated papers, cables, motors, furniture
and background objects."""


def page_prompt(target_description: str) -> str:
    if not target_description.strip():
        raise ObjectLocalizationError("target_description must not be empty")
    return (
        f"Find and read {target_description.strip()} in this photo. Return both the bounding "
        "box of the visible page and any legible printed text. Reply as json."
    )


def locate_page(
    jpeg: bytes,
    client: StoryClient,
    target_description: str,
) -> PageReading | None:
    """Locate the page and read its text in one request, so the caller pays once.

    The tracker needs the box to steer and the reading flow needs the words; both
    come back from this single call instead of a locate call followed by a second
    recognition call on the same frame.
    """

    if not jpeg:
        raise ObjectLocalizationError("image must not be empty")
    response = client.describe_page(
        jpeg,
        prompt=page_prompt(target_description),
        system_instructions=page_instructions(),
    )
    payload = _json_object(response)
    if payload.get("found") is False:
        return None
    if payload.get("found") is not True:
        raise ObjectLocalizationError("vision response must declare found true or false")
    raw_box = payload.get("bbox_norm")
    if (
        not isinstance(raw_box, list)
        or len(raw_box) != 4
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_box)
    ):
        raise ObjectLocalizationError("bbox_norm must be four numeric values")
    raw_text = payload.get("visible_text", "")
    raw_confidence = payload.get("confidence", 0.0)
    reading = PageReading(
        tuple(float(value) for value in raw_box),
        raw_text.strip() if isinstance(raw_text, str) else "",
        float(raw_confidence)
        if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool)
        else 0.0,
    )
    TargetDetection(
        "book-page",
        "book page",
        reading.bbox_norm,
        reading.confidence,
    ).validate()
    return reading


def _json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ObjectLocalizationError("vision response was not one JSON object") from error
    if not isinstance(payload, dict):
        raise ObjectLocalizationError("vision response must be a JSON object")
    return payload


def locate_target(
    jpeg: bytes,
    client: StoryClient,
    target_description: str,
    *,
    object_id: str = "target-1",
) -> TargetDetection | None:
    """Ask a vision client for a target box and validate its factual output."""

    if not jpeg:
        raise ObjectLocalizationError("image must not be empty")
    response = client.describe_page(
        jpeg,
        prompt=localization_prompt(target_description),
        system_instructions=localization_instructions(),
    )
    payload = _json_object(response)
    if payload.get("found") is False:
        return None
    if payload.get("found") is not True:
        raise ObjectLocalizationError("vision response must declare found true or false")
    try:
        label = payload["label"]
        raw_box = payload["bbox_norm"]
        confidence = payload["confidence"]
    except KeyError as error:
        raise ObjectLocalizationError(f"vision response is missing {error.args[0]}") from error
    if not isinstance(label, str) or not label.strip():
        raise ObjectLocalizationError("vision response label must be a non-empty string")
    if not isinstance(raw_box, list) or len(raw_box) != 4 or any(not isinstance(value, (int, float)) for value in raw_box):
        raise ObjectLocalizationError("bbox_norm must be four numeric values")
    if not isinstance(confidence, (int, float)):
        raise ObjectLocalizationError("confidence must be numeric")
    detection = TargetDetection(object_id, label.strip(), tuple(float(value) for value in raw_box), float(confidence))
    detection.validate()
    return detection
