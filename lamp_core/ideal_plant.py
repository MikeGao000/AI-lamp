"""Ideal trajectory verification only; it deliberately contains no physics model.

The plant represented here has zero tracking error, zero CAN latency, no
backlash, no gravity, and infinite available torque.  Passing this verifier
means only that a requested trajectory is internally consistent with the
configured software position and speed limits.  It is not evidence that the
physical lamp can execute or hold the motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from lamp_core.motion import JointLimit, MotionError, TrajectoryPoint


@dataclass(frozen=True)
class IdealTrajectoryReport:
    """Measurements from a perfect-tracking execution of one trajectory."""

    duration_s: float
    frame_count: int
    max_abs_speed_rad_s: Mapping[str, float]
    max_abs_acceleration_rad_s2: Mapping[str, float]
    max_tracking_error_rad: float = 0.0


def verify_ideal_trajectory(
    frames: Sequence[TrajectoryPoint], limits: Mapping[str, JointLimit]
) -> IdealTrajectoryReport:
    """Verify a planned joint trajectory against the ideal software envelope.

    Each frame is treated as both the command and the measured position.  The
    discrete velocity is checked against each configured speed limit.  The
    acceleration is reported for later comparison with measured hardware data;
    no acceleration limit exists in the current configuration yet.
    """

    if len(frames) < 2:
        raise MotionError("ideal verification requires at least two trajectory frames")

    names = tuple(limits)
    if not names:
        raise MotionError("at least one joint limit is required")

    max_speed = {name: 0.0 for name in names}
    max_acceleration = {name: 0.0 for name in names}
    previous_velocity: dict[str, float] | None = None

    for index, frame in enumerate(frames):
        if set(frame.positions_rad) != set(names):
            raise MotionError("trajectory frame does not specify every configured joint")
        for name, position in frame.positions_rad.items():
            limits[name].validate(position)

        if index == 0:
            continue

        previous = frames[index - 1]
        dt = frame.time_s - previous.time_s
        if dt <= 0:
            raise MotionError("trajectory timestamps must be strictly increasing")

        velocity = {
            name: (frame.positions_rad[name] - previous.positions_rad[name]) / dt
            for name in names
        }
        for name, value in velocity.items():
            absolute_speed = abs(value)
            max_speed[name] = max(max_speed[name], absolute_speed)
            if absolute_speed > limits[name].maximum_speed_rad_s + 1e-9:
                raise MotionError(
                    f"{name} ideal speed {absolute_speed:.6f} exceeds "
                    f"configured limit {limits[name].maximum_speed_rad_s:.6f}"
                )

        if previous_velocity is not None:
            for name in names:
                acceleration = abs((velocity[name] - previous_velocity[name]) / dt)
                max_acceleration[name] = max(max_acceleration[name], acceleration)
        previous_velocity = velocity

    return IdealTrajectoryReport(
        duration_s=frames[-1].time_s - frames[0].time_s,
        frame_count=len(frames),
        max_abs_speed_rad_s=max_speed,
        max_abs_acceleration_rad_s2=max_acceleration,
    )
