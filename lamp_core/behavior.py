"""Boundary between model behavior proposals and the local action catalogue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from lamp_core.action_catalog import ActionCatalog, ActionDefinition, DEFAULT_ACTION_CATALOG, TargetRequirement


@dataclass(frozen=True)
class ScreenTarget:
    """A local detector target; safety attributes are never model-supplied."""

    object_id: str
    label: str
    center_x: float
    center_y: float
    confidence: float
    reachable_for_touch: bool = False
    touch_safe: bool = False

    def validate(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if not 0.0 <= self.center_x <= 1.0 or not 0.0 <= self.center_y <= 1.0:
            raise ValueError("target center must be normalized to [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("target confidence must be normalized to [0, 1]")


@dataclass(frozen=True)
class BehaviorProposal:
    """A model can select one audited action_id, never a motor primitive."""

    action_id: str
    speech: str = ""
    intensity: float = 0.0
    target_ref: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class ApprovedBehavior:
    action_id: str
    motion_key: str | None
    target: ScreenTarget | None
    speech: str
    intensity: float
    reason: str
    verification_level: str


class BehaviorValidationError(ValueError):
    pass


def ai_behavior_tool_schema(catalog: ActionCatalog = DEFAULT_ACTION_CATALOG) -> dict:
    """Schema supplied to the model; its enum comes from the audited catalogue."""

    return {
        "type": "function",
        "name": "propose_behavior",
        "description": "Select one audited local action, never motor commands.",
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": {"type": "string", "enum": list(catalog.selectable_action_ids())},
                "speech": {"type": "string", "maxLength": 240},
                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "target_ref": {"type": "string"},
                "reason": {"type": "string", "maxLength": 240},
            },
            "required": ["action_id"],
            "additionalProperties": False,
        },
    }


def _validated_target(
    definition: ActionDefinition,
    target_ref: str | None,
    visible_objects: Mapping[str, ScreenTarget],
    minimum_confidence: float,
) -> ScreenTarget | None:
    requirement = definition.target_requirement
    if target_ref is None:
        if requirement in {TargetRequirement.REQUIRED_VISIBLE, TargetRequirement.REQUIRED_SAFE_REACHABLE}:
            raise BehaviorValidationError(f"{definition.action_id} requires a target_ref")
        return None
    if requirement is TargetRequirement.NONE:
        raise BehaviorValidationError("this action does not accept a target_ref")
    if target_ref not in visible_objects:
        raise BehaviorValidationError("target_ref is not currently visible")
    target = visible_objects[target_ref]
    target.validate()
    if target.confidence < minimum_confidence:
        raise BehaviorValidationError("target confidence is too low")
    if requirement is TargetRequirement.REQUIRED_SAFE_REACHABLE and not (
        target.reachable_for_touch and target.touch_safe
    ):
        raise BehaviorValidationError("target is not locally verified as reachable and safe")
    return target


def approve_behavior(
    proposal: BehaviorProposal,
    visible_objects: Mapping[str, ScreenTarget],
    *,
    safety_ready: bool,
    minimum_confidence: float = 0.65,
    catalog: ActionCatalog = DEFAULT_ACTION_CATALOG,
) -> ApprovedBehavior:
    """Resolve and validate a catalogue action before a local compiler sees it."""

    try:
        definition = catalog.resolve_for_model(proposal.action_id)
    except ValueError as error:
        raise BehaviorValidationError(str(error)) from error
    if not isinstance(proposal.intensity, (int, float)) or not (
        definition.minimum_intensity <= proposal.intensity <= definition.maximum_intensity
    ):
        raise BehaviorValidationError(
            f"intensity must be within [{definition.minimum_intensity}, {definition.maximum_intensity}]"
        )
    if len(proposal.speech) > 240 or len(proposal.reason) > 240:
        raise BehaviorValidationError("speech and reason must be at most 240 characters")
    if proposal.action_id != "STOP" and not safety_ready:
        raise BehaviorValidationError("motion is not safe/ready")
    target = _validated_target(definition, proposal.target_ref, visible_objects, minimum_confidence)
    return ApprovedBehavior(
        definition.action_id,
        definition.motion_key,
        target,
        proposal.speech,
        float(proposal.intensity),
        proposal.reason,
        definition.verification_level,
    )
