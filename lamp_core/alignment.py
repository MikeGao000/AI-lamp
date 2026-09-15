"""Camera/target alignment math independent of camera and motor hardware."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class HorizontalDirection(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass(frozen=True)
class AlignmentObservation:
    """Normalized target location relative to the camera optical centre."""

    bbox_norm: tuple[float, float, float, float]
    target_x: float
    target_y: float
    error_x: float
    error_y: float
    horizontal: HorizontalDirection

    @property
    def centered(self) -> bool:
        return self.horizontal is HorizontalDirection.CENTER


@dataclass
class ResponsiveJ1Trajectory:
    """Fast adaptive target filtering followed by a jerk-limited S-curve.

    Small camera jitter is strongly suppressed while a deliberate large target
    movement gets a much higher filter gain. Position, velocity and
    acceleration remain continuous, so reversals do not create a mechanical
    snap. Limits are output-joint units, independent of the MKS inner FOC loop.
    """

    max_velocity_degrees_s: float
    max_acceleration_degrees_s2: float
    max_jerk_degrees_s3: float
    position_degrees: float = 0.0
    velocity_degrees_s: float = 0.0
    acceleration_degrees_s2: float = 0.0
    filtered_target_degrees: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.max_velocity_degrees_s,
            self.max_acceleration_degrees_s2,
            self.max_jerk_degrees_s3,
        ) <= 0:
            raise ValueError("trajectory velocity, acceleration and jerk limits must be positive")

    def update(self, target_degrees: float, dt_s: float) -> float:
        if dt_s <= 0:
            raise ValueError("trajectory time step must be positive")
        dt_s = min(dt_s, 0.25)

        target_delta = target_degrees - self.filtered_target_degrees
        # One-Euro-like adaptive gain: quiet around a stable centre, responsive
        # when the child moves the book by a meaningful distance.
        movement = min(1.0, abs(target_delta) / 3.0)
        target_gain = 0.10 + 0.75 * movement
        self.filtered_target_degrees += target_gain * target_delta

        error = self.filtered_target_degrees - self.position_degrees
        # Critically damped second-order tracking reaches a moved target quickly
        # without the repeated overshoot of a raw position step. The subsequent
        # jerk limiter makes the commanded acceleration itself continuous.
        natural_frequency = 4.5
        desired_acceleration = max(
            -self.max_acceleration_degrees_s2,
            min(
                self.max_acceleration_degrees_s2,
                natural_frequency**2 * error
                - 2.0 * natural_frequency * self.velocity_degrees_s,
            ),
        )
        acceleration_step = max(
            -self.max_jerk_degrees_s3 * dt_s,
            min(
                self.max_jerk_degrees_s3 * dt_s,
                desired_acceleration - self.acceleration_degrees_s2,
            ),
        )
        self.acceleration_degrees_s2 += acceleration_step
        self.velocity_degrees_s += self.acceleration_degrees_s2 * dt_s
        self.velocity_degrees_s = max(
            -self.max_velocity_degrees_s,
            min(self.max_velocity_degrees_s, self.velocity_degrees_s),
        )
        next_position = self.position_degrees + self.velocity_degrees_s * dt_s
        self.position_degrees = next_position
        return self.position_degrees


def smooth_bounded_scan_degrees(
    elapsed_s: float,
    envelope_degrees: float,
    *,
    period_s: float = 6.0,
) -> float:
    """A sinusoidal search path with zero velocity at both turnarounds."""

    if elapsed_s < 0 or envelope_degrees <= 0 or period_s <= 0:
        raise ValueError("scan elapsed time must be nonnegative and limits positive")
    return math.sin(2.0 * math.pi * elapsed_s / period_s) * envelope_degrees


def profiled_motor_speed_rpm(
    output_velocity_degrees_s: float,
    gear_ratio: float,
    maximum_motor_rpm: int,
) -> int:
    """Convert S-curve velocity to an F5 speed while preserving its configured cap."""

    if gear_ratio < 1 or maximum_motor_rpm < 1:
        raise ValueError("gear ratio and maximum motor RPM must be positive")
    requested = round(abs(output_velocity_degrees_s) * gear_ratio / 6.0 * 1.15)
    return max(1, min(maximum_motor_rpm, requested))


def observe_bbox(
    bbox_norm: tuple[float, float, float, float],
    *,
    horizontal_deadband: float = 0.08,
) -> AlignmentObservation:
    """Describe a normalized bounding box relative to the image centre."""

    if not 0 <= horizontal_deadband < 0.5:
        raise ValueError("horizontal deadband must be within 0..0.5")
    x1, y1, x2, y2 = bbox_norm
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("bbox must be normalized x1,y1,x2,y2 with positive area")
    target_x = (x1 + x2) / 2
    target_y = (y1 + y2) / 2
    error_x = target_x - 0.5
    error_y = target_y - 0.5
    if error_x < -horizontal_deadband:
        direction = HorizontalDirection.LEFT
    elif error_x > horizontal_deadband:
        direction = HorizontalDirection.RIGHT
    else:
        direction = HorizontalDirection.CENTER
    return AlignmentObservation(bbox_norm, target_x, target_y, error_x, error_y, direction)


def bounded_j1_test_degrees(
    observation: AlignmentObservation,
    *,
    positive_camera_direction: HorizontalDirection,
    maximum_degrees: float = 10.0,
    minimum_degrees: float = 2.0,
) -> float:
    """Return one bounded J1 test move; never creates a repeated control loop.

    ``positive_camera_direction`` records the physically verified direction in
    which a positive J1 command would turn an attached camera. The current
    disconnected-camera test uses the value only to spin the motor once in the
    direction suggested by the image.
    """

    if positive_camera_direction is HorizontalDirection.CENTER:
        raise ValueError("positive camera direction must be left or right")
    if not 0 < minimum_degrees <= maximum_degrees <= 15:
        raise ValueError("J1 test bounds must satisfy 0 < minimum <= maximum <= 15 degrees")
    if observation.centered:
        return 0.0
    magnitude = max(minimum_degrees, min(maximum_degrees, abs(observation.error_x) * 2 * maximum_degrees))
    same_direction = observation.horizontal is positive_camera_direction
    return magnitude if same_direction else -magnitude


def mapped_j1_tracking_degrees(
    observation: AlignmentObservation,
    *,
    positive_camera_direction: HorizontalDirection,
    envelope_degrees: float = 10.0,
) -> float:
    """Map live image position to an absolute offset around the startup anchor.

    This is safe for a mechanically disconnected camera test because persistent
    image error cannot accumulate into unbounded motor travel.
    """

    if positive_camera_direction is HorizontalDirection.CENTER:
        raise ValueError("positive camera direction must be left or right")
    if not 0 < envelope_degrees <= 15:
        raise ValueError("J1 tracking envelope must be within 0..15 degrees")
    if observation.centered:
        return 0.0
    logical_camera_degrees = max(
        -envelope_degrees,
        min(envelope_degrees, observation.error_x * 2 * envelope_degrees),
    )
    return (
        logical_camera_degrees
        if positive_camera_direction is HorizontalDirection.RIGHT
        else -logical_camera_degrees
    )
