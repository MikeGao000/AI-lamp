"""Fail-closed local safety state and hardware-neutral motion readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SafetyState(Enum):
    SAFE_DISABLED = auto()
    HOMING_REQUIRED = auto()
    READY_HOLD = auto()
    MOVING = auto()
    PAUSED = auto()
    FAULT_LATCHED = auto()


class SafetyError(RuntimeError):
    """Raised when a command is invalid for the present safety state."""


@dataclass(frozen=True)
class MotionSafetyCheck:
    """Facts required before a new optional motion can start.

    These values come from a future motor-safety service or from the explicit
    virtual-hardware test fixture.  Unknown hardware data must be represented
    as false rather than silently assumed safe.
    """

    calibrated: bool
    can_healthy: bool
    heartbeats_fresh: bool
    positions_readable: bool
    limits_healthy: bool
    temperature_healthy: bool
    target_valid: bool
    trajectory_valid: bool
    action_budget_available: bool

    @classmethod
    def simulated_clear(cls) -> "MotionSafetyCheck":
        """An explicit all-clear fixture for offline virtual-bus tests only."""

        return cls(True, True, True, True, True, True, True, True, True)

    def rejected_reasons(self) -> tuple[str, ...]:
        values = {
            "calibration": self.calibrated,
            "can": self.can_healthy,
            "heartbeat": self.heartbeats_fresh,
            "position": self.positions_readable,
            "limit": self.limits_healthy,
            "temperature": self.temperature_healthy,
            "target": self.target_valid,
            "trajectory": self.trajectory_valid,
            "action_budget": self.action_budget_available,
        }
        return tuple(name for name, accepted in values.items() if not accepted)


@dataclass
class SafetyController:
    """Deterministic interlock; the physical E-stop remains mandatory."""

    state: SafetyState = SafetyState.SAFE_DISABLED
    homed: bool = False
    drives_enabled: bool = False
    fault_reason: str | None = None
    _estop_latched: bool = False

    def begin_homing(self) -> None:
        if self.state is not SafetyState.SAFE_DISABLED or self._estop_latched:
            raise SafetyError(f"cannot home from {self.state.name}")
        self.state = SafetyState.HOMING_REQUIRED
        self.homed = False
        self.drives_enabled = False

    def complete_homing(self) -> None:
        if self.state is not SafetyState.HOMING_REQUIRED:
            raise SafetyError("homing was not started")
        self.homed = True
        self.state = SafetyState.READY_HOLD

    def enable_drives(self) -> None:
        if self.state is not SafetyState.READY_HOLD or not self.homed:
            raise SafetyError("drives may only be enabled after successful homing")
        self.drives_enabled = True

    def begin_motion(self, safety_check: MotionSafetyCheck) -> None:
        if self.state is not SafetyState.READY_HOLD or not self.drives_enabled:
            raise SafetyError("motion requires READY_HOLD state and enabled drives")
        rejected = safety_check.rejected_reasons()
        if rejected:
            raise SafetyError(f"motion rejected by safety checks: {', '.join(rejected)}")
        self.state = SafetyState.MOVING

    def finish_motion(self) -> None:
        if self.state is not SafetyState.MOVING:
            raise SafetyError("no motion is active")
        self.state = SafetyState.READY_HOLD

    def pause_motion(self) -> None:
        if self.state is not SafetyState.MOVING:
            raise SafetyError("only active motion can be paused")
        self.state = SafetyState.PAUSED

    def resume_motion(self, safety_check: MotionSafetyCheck) -> None:
        if self.state is not SafetyState.PAUSED:
            raise SafetyError("no paused motion is active")
        rejected = safety_check.rejected_reasons()
        if rejected:
            raise SafetyError(f"motion resume rejected by safety checks: {', '.join(rejected)}")
        self.state = SafetyState.MOVING

    def cancel_motion(self) -> None:
        if self.state not in (SafetyState.MOVING, SafetyState.PAUSED):
            raise SafetyError("no motion is active")
        self.state = SafetyState.READY_HOLD

    def fault(self, reason: str) -> None:
        self.fault_reason = reason
        self.drives_enabled = False
        self.state = SafetyState.FAULT_LATCHED

    def acknowledge_fault(self) -> None:
        if self.state is not SafetyState.FAULT_LATCHED:
            raise SafetyError("system has no latched fault")
        self.state = SafetyState.SAFE_DISABLED
        self.homed = False
        self.fault_reason = None

    def emergency_stop(self, reason: str = "emergency stop") -> None:
        self.fault_reason = reason
        self.drives_enabled = False
        self.homed = False
        self._estop_latched = True
        self.state = SafetyState.SAFE_DISABLED

    def reset_estop(self) -> None:
        if not self._estop_latched:
            raise SafetyError("system is not in an emergency stop")
        self._estop_latched = False
        self.state = SafetyState.SAFE_DISABLED
        self.homed = False
        self.fault_reason = None
