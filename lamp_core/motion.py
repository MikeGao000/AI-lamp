from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Mapping


class MotionError(ValueError):
    pass


@dataclass(frozen=True)
class JointLimit:
    minimum_rad: float
    maximum_rad: float
    maximum_speed_rad_s: float
    maximum_acceleration_rad_s2: float | None = None

    def validate(self, value: float) -> None:
        if not self.minimum_rad <= value <= self.maximum_rad:
            raise MotionError(
                f"target {value:.3f} rad outside [{self.minimum_rad:.3f}, {self.maximum_rad:.3f}]"
            )


@dataclass(frozen=True)
class TrajectoryPoint:
    time_s: float
    positions_rad: dict[str, float]


def plan_synchronised_minimum_jerk(
    start_rad: Mapping[str, float],
    target_rad: Mapping[str, float],
    limits: Mapping[str, JointLimit],
    sample_period_s: float = 0.02,
) -> list[TrajectoryPoint]:
    """Return a time-synchronised, zero-end-velocity joint trajectory.

    The 10t^3-15t^4+6t^5 profile is minimum-jerk. Duration is selected from
    its peak velocity (1.875 * distance / duration) and peak acceleration
    (10 / sqrt(3) * distance / duration²), so no axis exceeds configured
    limits. This is planning, not a replacement for a motor controller's
    current/position loop.
    """
    if set(start_rad) != set(target_rad) or set(start_rad) != set(limits):
        raise MotionError("start, target and limits must contain the same joint names")
    if sample_period_s <= 0:
        raise MotionError("sample period must be positive")

    duration_s = sample_period_s
    for name, target in target_rad.items():
        limit = limits[name]
        limit.validate(start_rad[name])
        limit.validate(target)
        if limit.maximum_speed_rad_s <= 0:
            raise MotionError(f"{name} maximum speed must be positive")
        duration_s = max(
            duration_s,
            1.875 * abs(target - start_rad[name]) / limit.maximum_speed_rad_s,
        )
        if limit.maximum_acceleration_rad_s2 is not None:
            if limit.maximum_acceleration_rad_s2 <= 0:
                raise MotionError(f"{name} maximum acceleration must be positive")
            duration_s = max(
                duration_s,
                sqrt(
                    (10 / sqrt(3))
                    * abs(target - start_rad[name])
                    / limit.maximum_acceleration_rad_s2
                ),
            )

    intervals = max(1, ceil(duration_s / sample_period_s))
    duration_s = intervals * sample_period_s
    points: list[TrajectoryPoint] = []
    for index in range(intervals + 1):
        u = index / intervals
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        positions = {
            name: start_rad[name] + (target_rad[name] - start_rad[name]) * blend
            for name in start_rad
        }
        points.append(TrajectoryPoint(index * sample_period_s, positions))
    return points
