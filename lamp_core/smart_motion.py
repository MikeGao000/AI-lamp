"""Protocol-independent SEQ / INT / TRJ arbitration for smart motor nodes.

This module deliberately has no guessed MKS serial frames.  A model-specific
adapter will translate these calls once the exact MKS model and manual are
available.  All angular values are in joint-output radians.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol

from lamp_core.motion import JointLimit


class MotionMode(str, Enum):
    SEQ = "SEQ"
    INT = "INT"
    TRJ = "TRJ"


class SmartMotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class JointMove:
    targets_rad: Mapping[str, float]
    duration_s: float
    label: str = ""

    def validate(self, limits: Mapping[str, JointLimit]) -> None:
        if set(self.targets_rad) != set(limits):
            raise SmartMotionError("move must specify every configured joint exactly once")
        if self.duration_s <= 0:
            raise SmartMotionError("move duration must be positive")
        for name, target in self.targets_rad.items():
            limits[name].validate(target)


class SmartMotorBackend(Protocol):
    """Capabilities exposed by a verified MKS model-specific adapter."""

    def prepare_move(self, move: JointMove) -> None: ...

    def start_prepared_move(self) -> None: ...

    def cancel_motion(self) -> None: ...


class SmartMotionDispatcher:
    """Owns the three command channels and their safety-oriented priority."""

    def __init__(self, limits: Mapping[str, JointLimit], backend: SmartMotorBackend) -> None:
        self.limits = limits
        self.backend = backend
        self._sequence: deque[JointMove] = deque()
        self._trajectory: deque[JointMove] = deque()
        self.active_mode: MotionMode | None = None

    @property
    def queued_sequence_count(self) -> int:
        return len(self._sequence)

    @property
    def queued_trajectory_count(self) -> int:
        return len(self._trajectory)

    def submit_sequence(self, move: JointMove) -> None:
        move.validate(self.limits)
        self._sequence.append(move)

    def submit_interrupt(self, move: JointMove) -> None:
        """Interrupt wins over all soft motion. The caller still handles E-stop."""
        move.validate(self.limits)
        self._sequence.clear()
        self._trajectory.clear()
        self.backend.cancel_motion()
        self.backend.prepare_move(move)
        self.backend.start_prepared_move()
        self.active_mode = MotionMode.INT

    def submit_trajectory_segment(self, move: JointMove) -> None:
        move.validate(self.limits)
        self._trajectory.append(move)

    def tick(self, motion_finished: bool) -> None:
        """Start at most one next segment after verified completion feedback."""
        if not motion_finished:
            return
        if self._sequence:
            move = self._sequence.popleft()
            self.backend.prepare_move(move)
            self.backend.start_prepared_move()
            self.active_mode = MotionMode.SEQ
            return
        if self._trajectory:
            move = self._trajectory.popleft()
            self.backend.prepare_move(move)
            self.backend.start_prepared_move()
            self.active_mode = MotionMode.TRJ
            return
        self.active_mode = None

    def clear_soft_motion(self) -> None:
        """Called on fault/E-stop before hardware enable is removed."""
        self._sequence.clear()
        self._trajectory.clear()
        self.backend.cancel_motion()
        self.active_mode = None
