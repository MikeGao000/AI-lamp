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
Return only one JSON object. Never include Markdown or explanatory prose.
Use normalized image coordinates in the unrotated input image: x grows from
left to right and y grows from top to bottom.
If the target is visible, return exactly:
{"found":true,"label":"short target label","bbox_norm":[x1,y1,x2,y2],"confidence":0.0}
If it is not visible, return exactly: {"found":false}
The bounding box must tightly cover the requested target, not nearby objects."""


def localization_prompt(target_description: str) -> str:
    if not target_description.strip():
        raise ObjectLocalizationError("target_description must not be empty")
    return f"Find this target: {target_description.strip()}"


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
