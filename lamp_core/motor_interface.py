"""Real motor I/O is intentionally blocked until a driver protocol is verified."""

from __future__ import annotations

from lamp_core.motion import TrajectoryPoint


class UnconfiguredMotorDriver:
    """Fail closed: never emit guessed serial frames to a physical actuator."""

    @property
    def positions_rad(self) -> dict[str, float]:
        raise RuntimeError("physical motor driver is not configured")

    def execute(self, frames: list[TrajectoryPoint]) -> None:
        raise RuntimeError("physical motor driver is not configured; simulation only")
