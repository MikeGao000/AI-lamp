"""Offline, no-hardware simulation of a single 5-axis lamp motion."""

from __future__ import annotations

import argparse

from lamp_core.motion import JointLimit, plan_synchronised_minimum_jerk
from lamp_core.safety import MotionSafetyCheck, SafetyController


JOINT_LIMITS = {
    "j1_base_yaw": JointLimit(-1.57, 1.57, 0.80),
    "j2_shoulder": JointLimit(-0.78, 0.78, 0.45),
    "j3_elbow": JointLimit(-0.95, 0.95, 0.55),
    "j4_neck_pitch": JointLimit(-0.70, 0.70, 0.75),
    "j5_head_yaw": JointLimit(-1.05, 1.05, 0.90),
}


def run(sample_period_s: float) -> None:
    safety = SafetyController()
    print(f"STATE {safety.state.name}")
    safety.begin_homing()
    print(f"STATE {safety.state.name}: simulated limit switches reached")
    safety.complete_homing()
    safety.enable_drives()
    print(f"STATE {safety.state.name}: drives enabled in SIMULATION ONLY")

    start = {name: 0.0 for name in JOINT_LIMITS}
    target = {
        "j1_base_yaw": 0.60,
        "j2_shoulder": -0.35,
        "j3_elbow": 0.45,
        "j4_neck_pitch": 0.22,
        "j5_head_yaw": -0.50,
    }
    trajectory = plan_synchronised_minimum_jerk(
        start, target, JOINT_LIMITS, sample_period_s=sample_period_s
    )
    safety.begin_motion(MotionSafetyCheck.simulated_clear())
    print(f"STATE {safety.state.name}: {len(trajectory)} synchronised frames")
    print("time_s,j1_base_yaw,j2_shoulder,j3_elbow,j4_neck_pitch,j5_head_yaw")
    for point in trajectory:
        values = ",".join(f"{point.positions_rad[name]:.4f}" for name in JOINT_LIMITS)
        print(f"{point.time_s:.3f},{values}")
    safety.finish_motion()
    print(f"STATE {safety.state.name}: target reached; no hardware was commanded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-period", type=float, default=0.10)
    args = parser.parse_args()
    run(args.sample_period)
