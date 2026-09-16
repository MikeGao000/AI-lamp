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
        # One-Euro-like adaptive gain: small corrections stay damped but still
        # converge quickly, while a deliberate large move stays responsive.
        movement = min(1.0, abs(target_delta) / 3.0)
        target_gain = 0.30 + 0.55 * movement
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


@dataclass
class CoarseToFineAim:
    """Turn image error into a decisive move that becomes gentler near centre.

    This is deliberately not a PID loop.  A large, stable displacement receives
    one almost complete correction; the residual is corrected with a smaller
    gain.  Adaptive filtering rejects small box jitter, a Schmitt deadband keeps
    a centred page centred, and a small reversal must repeat before it can send
    the motor back across the target.
    """

    degrees_per_error: float
    deadband: float
    maximum_step_degrees: float
    reverse_confirmations: int = 2
    filtered_error: float = 0.0
    settled: bool = False
    last_move_sign: int = 0
    pending_reverse_sign: int = 0
    pending_reverse_count: int = 0
    initialized: bool = False

    def __post_init__(self) -> None:
        if self.degrees_per_error <= 0 or self.maximum_step_degrees <= 0:
            raise ValueError("aim scale and maximum step must be positive")
        if not 0 < self.deadband < 0.25:
            raise ValueError("aim deadband must be within 0..0.25")
        if self.reverse_confirmations < 1:
            raise ValueError("reverse confirmations must be positive")

    def update(self, error_x: float) -> float:
        """Return the next correction in camera-positive output degrees."""

        magnitude = min(1.0, abs(error_x) / 0.20)
        # Approximately One-Euro behaviour without depending on frame rate:
        # large changes pass promptly, while near-centre box noise is damped.
        alpha = 0.18 + 0.72 * magnitude
        if not self.initialized:
            # The first sighting is not jitter; delaying it makes acquisition
            # needlessly timid.  Filtering starts with the following frame.
            self.filtered_error = error_x
            self.initialized = True
        else:
            self.filtered_error += alpha * (error_x - self.filtered_error)
        filtered_abs = abs(self.filtered_error)

        # Hysteresis prevents a box fluctuating around the deadband boundary from
        # repeatedly waking the axis.  It takes a clearly new displacement to
        # leave the settled state.
        if self.settled:
            if filtered_abs <= self.deadband * 1.6:
                return 0.0
            self.settled = False
        if filtered_abs <= self.deadband:
            self.settled = True
            self.pending_reverse_sign = 0
            self.pending_reverse_count = 0
            return 0.0

        gain = 0.55 + 0.45 * min(1.0, magnitude * 2.0)
        correction = self.filtered_error * self.degrees_per_error * gain
        correction = max(
            -self.maximum_step_degrees,
            min(self.maximum_step_degrees, correction),
        )
        sign = 1 if correction > 0 else -1

        # A genuine large reversal should remain responsive.  Only a small
        # near-centre reversal is held for one extra observation, which removes
        # the characteristic left-right-left finishing wobble.
        if self.last_move_sign and sign != self.last_move_sign and filtered_abs < 0.12:
            if sign == self.pending_reverse_sign:
                self.pending_reverse_count += 1
            else:
                self.pending_reverse_sign = sign
                self.pending_reverse_count = 1
            if self.pending_reverse_count < self.reverse_confirmations:
                return 0.0
        self.last_move_sign = sign
        self.pending_reverse_sign = 0
        self.pending_reverse_count = 0
        self.initialized = False
        return correction

    def reset(self) -> None:
        self.filtered_error = 0.0
        self.settled = False
        self.last_move_sign = 0
        self.pending_reverse_sign = 0
        self.pending_reverse_count = 0
        self.initialized = False


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


def stepped_scan_degrees(
    elapsed_s: float,
    envelope_degrees: float,
    *,
    step_degrees: float = 10.0,
    dwell_seconds: float = 2.5,
) -> float:
    """Discrete stop-and-stare scan: dwell at each angle, then step to the next.

    Continuous sweeping blurs the frame exactly while the model is looking at it,
    which is why detection on a moving camera is unreliable. Holding still at a
    small set of preset angles is the classic PTZ patrol pattern, and it gives the
    detector a sharp image at a known angle. With a 20 deg envelope and a 10 deg
    step the order is 0, +10, -10, +20, -20.
    """

    if elapsed_s < 0 or envelope_degrees <= 0 or step_degrees <= 0 or dwell_seconds <= 0:
        raise ValueError("scan elapsed time must be nonnegative and limits positive")
    positions = [0.0]
    for index in range(1, int(envelope_degrees // step_degrees) + 1):
        positions.append(index * step_degrees)
        positions.append(-index * step_degrees)
    return positions[int(elapsed_s / dwell_seconds) % len(positions)]


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
    if not 0 < envelope_degrees <= 30:
        raise ValueError("J1 tracking envelope must be within 0..30 degrees")
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


def visual_servo_degrees(
    observation: AlignmentObservation,
    *,
    positive_camera_direction: HorizontalDirection,
    degrees_per_error: float,
    maximum_degrees: float,
) -> float:
    """Calibrated image-error -> camera-angle correction.

    ``degrees_per_error`` is the measured horizontal field of view: how many
    degrees the axis must turn to move a target from the image edge to the
    centre. It comes from a measurement (turn a known angle, measure the pixel
    shift) instead of an assumed envelope, so a correction centres the target in
    about one step and does not undershoot into a slow crawl.
    """

    if positive_camera_direction is HorizontalDirection.CENTER:
        raise ValueError("positive camera direction must be left or right")
    if degrees_per_error <= 0:
        raise ValueError("degrees_per_error must be positive")
    if maximum_degrees <= 0:
        raise ValueError("maximum_degrees must be positive")
    if observation.centered:
        return 0.0
    logical_degrees = max(
        -maximum_degrees,
        min(maximum_degrees, observation.error_x * degrees_per_error),
    )
    return (
        logical_degrees
        if positive_camera_direction is HorizontalDirection.RIGHT
        else -logical_degrees
    )


def motion_compensated_error(
    error_x: float,
    commanded_offset_degrees: float,
    degrees_per_error: float,
) -> float:
    """Remove the axis's own rotation from a measured image error.

    Turning the camera moves the whole frame, so the measured error contains the
    commanded offset as well as the target's real displacement. Dividing the
    commanded offset by the measured field of view converts it back into image
    units and subtracts it, which is what stops a servo from chasing its own
    motion until it hits the travel limit.
    """

    if degrees_per_error <= 0:
        raise ValueError("degrees_per_error must be positive")
    return error_x - commanded_offset_degrees / degrees_per_error


def page_score(
    bbox_norm: tuple[float, float, float, float] | None,
    *,
    edge_margin: float = 0.02,
) -> float:
    """Score how completely a page is inside the frame; higher is better.

    The visible page area grows as more of the book comes into view, so
    maximising it finds the angle that frames the book best. A box that reaches
    the frame edge is being cropped, which is penalised so the scan prefers an
    angle where the whole visible page fits.
    """

    if bbox_norm is None:
        return 0.0
    x1, y1, x2, y2 = bbox_norm
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (x2 - x1) * (y2 - y1)
    touched = (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= 1.0 - edge_margin
        or y2 >= 1.0 - edge_margin
    )
    return area * (0.5 if touched else 1.0)


def saliency_thirds(
    image: object,
    cv2_module: object,
    *,
    sample_width: int = 96,
    sample_height: int = 72,
) -> tuple[float, float, float]:
    """Coarse "there is something here" profile for the left, centre and right third.

    Deliberately built on *colour variability* rather than absolute colour. On
    this rig the page and the desk are the same pale pink under the lamp, which is
    why rectangle heuristics kept failing, but a page's illustrations are far more
    saturated than either. Saturation alone would miss a text-only page, where
    black print on white is completely unsaturated, so local edge energy is folded
    in as well.

    The result only ever *orders* a search. A monitor showing a picture scores
    just as high, so it must never be treated as a detection.

    Returns three non-negative scores summing to 1, or three zeros when there is
    no usable signal.
    """

    import numpy as np

    height, width = image.shape[:2]  # type: ignore[union-attr]
    if height <= 0 or width <= 0:
        return (0.0, 0.0, 0.0)
    small = cv2_module.resize(image, (sample_width, sample_height))
    hsv = cv2_module.cvtColor(small, cv2_module.COLOR_BGR2HSV)
    gray = cv2_module.cvtColor(small, cv2_module.COLOR_BGR2GRAY)
    saturation = np.asarray(hsv)[:, :, 1].astype(np.float32) / 255.0
    energy = np.abs(np.asarray(cv2_module.Laplacian(gray, cv2_module.CV_32F)))
    energy = energy / (float(energy.max()) or 1.0)

    scores: list[float] = []
    for index in range(3):
        start = index * sample_width // 3
        stop = (index + 1) * sample_width // 3
        saturation_patch = saturation[:, start:stop]
        energy_patch = energy[:, start:stop]
        # Spread as well as level: a flat pale desk and a flat pale page look the
        # same on average, but only one of them has structure on it.
        scores.append(
            float(saturation_patch.mean())
            + float(saturation_patch.std())
            + float(energy_patch.mean())
        )
    total = sum(scores)
    if total <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (scores[0] / total, scores[1] / total, scores[2] / total)


class NarrowingScan:
    """Discrete hill-climb that narrows onto the angle with the best page score.

    The first round patrols the whole envelope from the centre outward. Every
    later round samples the current best angle and its two neighbours at half the
    previous step, and the scan ends once the step falls below
    ``minimum_step_degrees``. Sampling from a settled camera is what makes each
    detection sharp, and comparing scores is what finds the angle where the book
    is most fully in view instead of just stopping at the first angle that
    happened to see it.

    The opening round used to reach only two steps either side of centre, so the
    real search radius was ``2 * step_degrees`` no matter how wide the envelope
    was configured -- a book 40 degrees off could never be reached from a 40
    degree travel. It now covers the envelope it is given.

    ``first_degrees`` is sampled ahead of the rest, which lets a cheap whole-frame
    hint (see :func:`saliency_thirds`) try the likeliest side first instead of
    starting blind at the middle.
    """

    def __init__(
        self,
        envelope_degrees: float = 20.0,
        step_degrees: float = 10.0,
        minimum_step_degrees: float = 2.5,
        first_degrees: float | None = None,
    ) -> None:
        if envelope_degrees <= 0 or step_degrees <= 0 or minimum_step_degrees <= 0:
            raise ValueError("scan limits must be positive")
        self.envelope_degrees = envelope_degrees
        self.step_degrees = step_degrees
        self.minimum_step_degrees = minimum_step_degrees
        self._first_degrees = first_degrees
        self.best_degrees = 0.0
        self.best_score = -1.0
        self.finished = False
        self._round: list[float] = []
        self._index = 0
        self._step = step_degrees
        self._round_best = 0.0
        self._round_best_score = -1.0
        self._start_round(0.0, step_degrees, wide=True)

    def _clamp(self, degrees: float) -> float:
        return max(-self.envelope_degrees, min(self.envelope_degrees, degrees))

    def _start_round(self, centre: float, step: float, *, wide: bool = False) -> None:
        offsets: list[float] = []
        if wide and self._first_degrees is not None:
            offsets.append(self._clamp(self._first_degrees))
        offsets.append(self._clamp(centre))
        offsets = list(dict.fromkeys(offsets))
        if wide:
            # Patrol outward until the envelope is covered, so the reach is the
            # envelope rather than a fixed two steps.
            reach = int(self.envelope_degrees // step)
            for index in range(1, reach + 1):
                for candidate in (centre - index * step, centre + index * step):
                    clamped = self._clamp(candidate)
                    if clamped not in offsets:
                        offsets.append(clamped)
        else:
            for candidate in (centre - step, centre + step):
                clamped = self._clamp(candidate)
                if clamped not in offsets:
                    offsets.append(clamped)
        self._round = offsets
        self._index = 0
        self._step = step
        self._round_best = self._clamp(centre)
        self._round_best_score = -1.0
        self.finished = False

    def next_target(self) -> float | None:
        """Angle to dwell at now, or None once the scan has converged."""

        if self._index >= len(self._round):
            return None
        return self._round[self._index]

    def record(self, score: float) -> None:
        """Record the score measured at the angle last returned by ``next_target``."""

        if self._index >= len(self._round):
            raise ValueError("no scan target is awaiting a score")
        target = self._round[self._index]
        self._index += 1
        if score > self._round_best_score:
            self._round_best_score = score
            self._round_best = target
        if score > self.best_score:
            self.best_score = score
            self.best_degrees = target
        if self._index >= len(self._round):
            next_step = self._step / 2.0
            if next_step < self.minimum_step_degrees:
                self.finished = True
            else:
                self._start_round(self._round_best, next_step)


@dataclass
class VisualServoPid:
    """Velocity-form PID on normalized image error (image-based visual servoing).

    Follows the common PTZ / IBVS control law: the image error drives an angular
    *velocity* command rather than a smoothed position target. A large error
    therefore produces a fast move, and a centred target produces zero velocity
    so the axis simply holds. The integral removes the final steady-state crawl,
    its clamp prevents wind-up while the target is occluded, and a first-order
    filter damps the derivative noise that bounding-box jitter would amplify.
    """

    kp: float
    ki: float
    kd: float
    max_velocity_degrees_s: float
    integral_limit: float = 1.5
    derivative_smoothing: float = 0.35
    deadband: float = 0.08
    integral: float = 0.0
    derivative: float = 0.0
    previous_error: float | None = None

    def __post_init__(self) -> None:
        if self.max_velocity_degrees_s <= 0:
            raise ValueError("maximum servo velocity must be positive")
        if not 0 < self.derivative_smoothing <= 1:
            raise ValueError("derivative smoothing must be within 0..1")
        if not 0 <= self.deadband < 0.5:
            raise ValueError("deadband must be within 0..0.5")

    def reset(self) -> None:
        self.integral = 0.0
        self.derivative = 0.0
        self.previous_error = None

    def update(self, error_x: float, dt_s: float) -> float:
        """Return the desired output velocity in degrees per second."""

        if dt_s <= 0:
            raise ValueError("servo time step must be positive")
        error = 0.0 if abs(error_x) < self.deadband else error_x
        self.integral = max(
            -self.integral_limit, min(self.integral_limit, self.integral + error * dt_s)
        )
        raw_derivative = (
            0.0 if self.previous_error is None else (error - self.previous_error) / dt_s
        )
        self.derivative = (
            self.derivative_smoothing * raw_derivative
            + (1.0 - self.derivative_smoothing) * self.derivative
        )
        self.previous_error = error
        velocity = self.kp * error + self.ki * self.integral + self.kd * self.derivative
        return max(-self.max_velocity_degrees_s, min(self.max_velocity_degrees_s, velocity))
