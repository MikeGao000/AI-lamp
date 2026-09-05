"""Audited local action catalogue for Autonomous behavior proposals.

Entries describe *what* an expression is, never how a motor must move.  The
motion key is resolved only by a local, verified MotionCompiler in a later
stage.  This lets the model choose rich LeLamp-inspired expressions without
ever receiving joint angles, CAN frames, or a way to invent an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class TargetRequirement(str, Enum):
    NONE = "none"
    OPTIONAL_VISIBLE = "optional_visible"
    REQUIRED_VISIBLE = "required_visible"
    REQUIRED_SAFE_REACHABLE = "required_safe_reachable"


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    motion_key: str | None
    source: str
    verification_level: str
    minimum_intensity: float = 0.0
    maximum_intensity: float = 1.0
    cooldown_ms: int = 0
    target_requirement: TargetRequirement = TargetRequirement.NONE
    enabled: bool = True
    model_selectable: bool = True

    def validate(self) -> None:
        if not self.action_id or not self.action_id.replace("_", "").isalnum() or self.action_id != self.action_id.upper():
            raise ValueError("action_id must be uppercase alphanumeric with underscores")
        if not self.source.strip() or self.verification_level not in {"L0", "L1", "L2", "L3", "L4", "L5"}:
            raise ValueError("action definition requires a source and known verification level")
        if not 0.0 <= self.minimum_intensity <= self.maximum_intensity <= 1.0:
            raise ValueError("action intensity range must be within [0, 1]")
        if self.cooldown_ms < 0:
            raise ValueError("action cooldown must be non-negative")
        if self.model_selectable and not self.enabled:
            raise ValueError("disabled action cannot be model selectable")


class ActionCatalog:
    def __init__(self, definitions: Iterable[ActionDefinition]) -> None:
        supplied = tuple(definitions)
        self._definitions = {definition.action_id: definition for definition in supplied}
        if not self._definitions:
            raise ValueError("action catalogue must not be empty")
        if len(self._definitions) != len(supplied):
            raise ValueError("action catalogue action_ids must be unique")
        for definition in self._definitions.values():
            definition.validate()

    def resolve_for_model(self, action_id: str) -> ActionDefinition:
        try:
            definition = self._definitions[action_id]
        except KeyError as error:
            raise ValueError(f"unknown action_id: {action_id}") from error
        if not definition.enabled or not definition.model_selectable:
            raise ValueError(f"action_id is not available to the model: {action_id}")
        return definition

    def selectable_action_ids(self) -> tuple[str, ...]:
        return tuple(
            definition.action_id
            for definition in self._definitions.values()
            if definition.enabled and definition.model_selectable
        )


DEFAULT_ACTION_CATALOG = ActionCatalog(
    (
        ActionDefinition("LOOK_LEFT", "look_left", "Autonomous v0.2", "L1"),
        ActionDefinition("LOOK_CENTER", "look_center", "Autonomous v0.2", "L1"),
        ActionDefinition("LOOK_RIGHT", "look_right", "Autonomous v0.2", "L1"),
        ActionDefinition("READING_POSTURE", "reading_posture", "Autonomous v0.2", "L1", target_requirement=TargetRequirement.OPTIONAL_VISIBLE),
        ActionDefinition("GENTLE_NOD", "gentle_nod", "Autonomous v0.2", "L1", maximum_intensity=0.6, cooldown_ms=1_000),
        ActionDefinition("LISTENING_POSE", "listening_pose", "Autonomous v0.2", "L1"),
        ActionDefinition("SET_LIGHT_SCENE", None, "Autonomous v0.2", "L0"),
        ActionDefinition("STOP", None, "Autonomous v0.2", "L0"),
        ActionDefinition("NONE", None, "Autonomous v0.2", "L0"),
        # The following are expressive catalogue entries inspired by LeLamp's
        # interaction vocabulary. They remain symbolic until a local motion
        # library validates and registers the matching motion_key.
        ActionDefinition("LELAMP_GREET_SMALL", "lelamp_greet_small", "LeLamp-inspired + local L1 pose composition", "L1", maximum_intensity=0.5, cooldown_ms=2_000),
        ActionDefinition("LELAMP_CURIOUS_TILT", "lelamp_curious_tilt", "LeLamp-inspired + local L1 pose composition", "L1", maximum_intensity=0.4, cooldown_ms=1_500),
        ActionDefinition("LELAMP_ACKNOWLEDGE", "lelamp_acknowledge", "LeLamp-inspired + local L1 pose composition", "L1", maximum_intensity=0.5, cooldown_ms=800),
        ActionDefinition("LELAMP_HEAD_SHAKE", "lelamp_head_shake", "LeLamp-inspired + local L1 pose composition", "L1", maximum_intensity=0.4, cooldown_ms=1_500),
        # Kept in the library for a future L4 implementation, but never
        # exposed to a model until reachability and real-hardware checks exist.
        ActionDefinition("LELAMP_TOUCH_TARGET", "lelamp_touch_target", "LeLamp-inspired", "L4", target_requirement=TargetRequirement.REQUIRED_SAFE_REACHABLE, enabled=False, model_selectable=False),
    )
)
