"""Protocol-independent feedback safety checks for closed-loop motor nodes.

Smart motors close their own current/position loop, but the host still must
refuse new motion if an encoder report is stale, faulted, or too far from the
commanded reference.  This is deliberately independent of any MKS CAN frame
format; the model-specific adapter will populate ``JointFeedback`` later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class FeedbackError(RuntimeError):
    """A condition that must fault/stop host-level motion."""


@dataclass(frozen=True)
class JointFeedback:
    position_rad: float
    received_at_s: float
    drive_fault: bool = False


@dataclass(frozen=True)
class FeedbackLimits:
    max_age_s: float
    max_tracking_error_rad: float

    def validate(self) -> None:
        if self.max_age_s <= 0:
            raise FeedbackError("feedback max age must be positive")
        if self.max_tracking_error_rad <= 0:
            raise FeedbackError("tracking-error limit must be positive")


def verify_feedback(
    expected_positions_rad: Mapping[str, float],
    feedback: Mapping[str, JointFeedback],
    limits: FeedbackLimits,
    now_s: float,
) -> None:
    """Raise before another move segment when feedback is unsafe to trust.

    This is a host-level watchdog, not a high-bandwidth PID controller. It is
    checked at segment boundaries and by a future CAN health task; MKS remains
    responsible for FOC and its internal encoder loop.
    """
    limits.validate()
    if set(expected_positions_rad) != set(feedback):
        raise FeedbackError("feedback joints do not match the expected joint set")
    for joint, expected in expected_positions_rad.items():
        report = feedback[joint]
        if report.drive_fault:
            raise FeedbackError(f"{joint} reports a motor fault")
        age_s = now_s - report.received_at_s
        if age_s < 0 or age_s > limits.max_age_s:
            raise FeedbackError(f"{joint} feedback is stale ({age_s:.3f}s)")
        error_rad = abs(expected - report.position_rad)
        if error_rad > limits.max_tracking_error_rad:
            raise FeedbackError(
                f"{joint} tracking error {error_rad:.3f} rad exceeds "
                f"{limits.max_tracking_error_rad:.3f} rad"
            )
