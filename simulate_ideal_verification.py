"""Repeatable ideal-only verification of configured five-axis motion profiles.

This script assumes perfect tracking.  It is deliberately unable to prove
torque, temperature, CAN latency, structural stiffness, collision avoidance,
or fault-recovery behavior on real hardware.
"""

from __future__ import annotations

import argparse
from itertools import product
from typing import Mapping

from lamp_core.action_catalog import DEFAULT_ACTION_CATALOG
from lamp_core.action_motion import compile_action_to_ideal_segments
from lamp_core.ideal_plant import IdealTrajectoryReport, verify_ideal_trajectory
from lamp_core.motion import JointLimit, plan_synchronised_minimum_jerk
from lamp_core.pose_library import IDLE_POSE, POSE_LIBRARY
from simulate import JOINT_LIMITS


IDLE = IDLE_POSE
PROFILES = POSE_LIBRARY


def _format_axis_values(values: Mapping[str, float]) -> str:
    return " ".join(f"{name}={value:.3f}" for name, value in values.items())


def _plan_and_verify(
    start: Mapping[str, float], target: Mapping[str, float], limits: Mapping[str, JointLimit]
) -> IdealTrajectoryReport:
    frames = plan_synchronised_minimum_jerk(start, target, limits, sample_period_s=0.02)
    return verify_ideal_trajectory(frames, limits)


def run_curated_profiles() -> list[tuple[str, IdealTrajectoryReport]]:
    """Exercise representative posture motions from idle in a perfect plant."""

    return [(name, _plan_and_verify(IDLE, target, JOINT_LIMITS)) for name, target in PROFILES.items()]


def _corners(limits: Mapping[str, JointLimit]) -> list[dict[str, float]]:
    names = tuple(limits)
    return [
        dict(zip(names, values, strict=True))
        for values in product(*((limits[name].minimum_rad, limits[name].maximum_rad) for name in names))
    ]


def run_exhaustive_corner_transitions() -> tuple[int, IdealTrajectoryReport]:
    """Exercise every configured min/max joint-space corner transition.

    This is exhaustive over the configured 32 endpoint corners, not over real
    mechanical dynamics or every continuous pose between them.
    """

    corners = _corners(JOINT_LIMITS)
    count = 0
    longest: IdealTrajectoryReport | None = None
    for start in corners:
        for target in corners:
            if start == target:
                continue
            report = _plan_and_verify(start, target, JOINT_LIMITS)
            count += 1
            if longest is None or report.duration_s > longest.duration_s:
                longest = report
    if longest is None:
        raise RuntimeError("corner verification unexpectedly had no transitions")
    return count, longest


def run_catalogue_actions() -> int:
    """Compile every L1 catalogue action into its local pose composition."""

    segment_count = 0
    for action_id in DEFAULT_ACTION_CATALOG.selectable_action_ids():
        definition = DEFAULT_ACTION_CATALOG.resolve_for_model(action_id)
        if definition.verification_level != "L1":
            continue
        compiled = compile_action_to_ideal_segments(action_id, IDLE, JOINT_LIMITS)
        for segment in compiled.segments:
            verify_ideal_trajectory(segment, JOINT_LIMITS)
            segment_count += 1
        pose_names = ",".join(compiled.pose_names) or "no motion"
        print(f"CATALOGUE {action_id}: {pose_names}")
    return segment_count


def run(exhaustive: bool, catalogue: bool) -> None:
    print("IDEAL-ONLY VERIFICATION: zero lag, zero error, no gravity, no collision model")
    for name, report in run_curated_profiles():
        print(f"PROFILE {name}: duration_s={report.duration_s:.2f} frames={report.frame_count}")
        print("  max_speed_rad_s " + _format_axis_values(report.max_abs_speed_rad_s))
        print("  max_accel_rad_s2 " + _format_axis_values(report.max_abs_acceleration_rad_s2))
        print(f"  tracking_error_rad={report.max_tracking_error_rad:.1f}")

    if exhaustive:
        transitions, longest = run_exhaustive_corner_transitions()
        print(
            "CORNER_SWEEP "
            f"transitions={transitions} longest_duration_s={longest.duration_s:.2f} "
            f"longest_frames={longest.frame_count}"
        )

    if catalogue:
        segments = run_catalogue_actions()
        print(f"CATALOGUE_SWEEP segments={segments}")

    print("PASS: configured position and discrete speed limits held in the ideal model")
    print("NOT VERIFIED: acceleration limits, torque, gravity, flex, backlash, CAN timing, heat, collision")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="also verify every min/max joint-space corner transition",
    )
    parser.add_argument(
        "--catalogue",
        action="store_true",
        help="compile and verify every L1 ActionCatalog expression",
    )
    args = parser.parse_args()
    run(args.exhaustive, args.catalogue)
