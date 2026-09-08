"""Compile audited expression IDs into existing local pose transitions.

This is the project-specific adaptation layer: LeLamp-inspired expression
names choose compositions, while the actual five-axis values and minimum-jerk
planner remain local. It borrows the time-parameterized, multi-axis control
structure from MechanicalArm_Code_V2 and never imports its geometry, gains,
or CAN protocol. SmallRobotArm's useful contribution here is the explicit
separation between a named pose library and its later actuator conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from lamp_core.action_catalog import ActionCatalog, DEFAULT_ACTION_CATALOG
from lamp_core.motion import JointLimit, TrajectoryPoint, plan_synchronised_minimum_jerk
from lamp_core.pose_library import IDLE_POSE, POSE_LIBRARY


class ActionMotionError(ValueError):
    pass


@dataclass(frozen=True)
class CompiledAction:
    action_id: str
    motion_key: str | None
    pose_names: tuple[str, ...]
    segments: tuple[tuple[TrajectoryPoint, ...], ...]


# Each expression is a composition of locally calibrated poses.  The
# compositions borrow LeLamp's expressive joint *relationships* while keeping
# this lamp's different J1--J5 axes and limits authoritative.
ACTION_RECIPES: Mapping[str, tuple[str, ...]] = {
    "LOOK_LEFT": ("look_left",),
    "LOOK_CENTER": (),
    "LOOK_RIGHT": ("look_right",),
    "READING_POSTURE": ("reading_pose",),
    "GENTLE_NOD": ("nod_down", "nod_up", "idle"),
    "LISTENING_POSE": ("listening_pose",),
    "LELAMP_GREET_SMALL": ("nod_up", "nod_down", "idle"),
    "LELAMP_CURIOUS_TILT": ("curious_left", "curious_right", "idle"),
    "LELAMP_ACKNOWLEDGE": ("nod_down", "nod_up", "idle"),
    "LELAMP_HEAD_SHAKE": ("head_shake_left", "head_shake_right", "idle"),
}


def compile_action_to_ideal_segments(
    action_id: str,
    start_rad: Mapping[str, float],
    limits: Mapping[str, JointLimit],
    *,
    catalog: ActionCatalog = DEFAULT_ACTION_CATALOG,
    sample_period_s: float = 0.02,
) -> CompiledAction:
    """Compile one model-selectable L1 expression for the ideal simulator."""

    try:
        definition = catalog.resolve_for_model(action_id)
    except ValueError as error:
        raise ActionMotionError(str(error)) from error
    if definition.motion_key is None:
        return CompiledAction(action_id, None, (), ())
    if definition.verification_level != "L1":
        raise ActionMotionError(f"{action_id} has no L1 motion recipe")
    try:
        pose_names = ACTION_RECIPES[action_id]
    except KeyError as error:
        raise ActionMotionError(f"no local motion recipe is registered for {action_id}") from error

    current = dict(start_rad)
    segments: list[tuple[TrajectoryPoint, ...]] = []
    for pose_name in pose_names:
        target = IDLE_POSE if pose_name == "idle" else POSE_LIBRARY[pose_name]
        frames = tuple(plan_synchronised_minimum_jerk(current, target, limits, sample_period_s))
        segments.append(frames)
        current = dict(target)
    return CompiledAction(action_id, definition.motion_key, pose_names, tuple(segments))
