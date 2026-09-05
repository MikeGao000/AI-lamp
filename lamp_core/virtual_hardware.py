"""Deterministic virtual devices used before any motor hardware is connected."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from lamp_core.motion import JointLimit, TrajectoryPoint


class MotorBusTimeoutError(ConnectionError):
    pass


class HardwareLimitError(RuntimeError):
    pass


@dataclass
class VirtualMotorBus:
    """A no-I/O motor bus that records accepted trajectory frames."""

    limits: Mapping[str, JointLimit]
    connected: bool = True
    limit_triggered: str | None = None
    positions_rad: dict[str, float] = field(init=False)
    received_frames: list[TrajectoryPoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.positions_rad = {name: 0.0 for name in self.limits}

    def execute(self, frames: Iterable[TrajectoryPoint]) -> None:
        if not self.connected:
            raise MotorBusTimeoutError("virtual RS-485 bus timed out")
        if self.limit_triggered is not None:
            raise HardwareLimitError(f"hardware limit triggered: {self.limit_triggered}")
        for frame in frames:
            if set(frame.positions_rad) != set(self.limits):
                raise ValueError("trajectory frame does not match configured joints")
            for name, value in frame.positions_rad.items():
                self.limits[name].validate(value)
            self.positions_rad = dict(frame.positions_rad)
            self.received_frames.append(frame)

    def disconnect(self) -> None:
        self.connected = False

    def reconnect(self) -> None:
        self.connected = True

    def trigger_limit(self, joint_name: str) -> None:
        if joint_name not in self.limits:
            raise KeyError(joint_name)
        self.limit_triggered = joint_name

    def clear_limit(self) -> None:
        self.limit_triggered = None


@dataclass
class SpeechStub:
    """Records spoken text; replace this later with a real local TTS adapter."""

    messages: list[str] = field(default_factory=list)

    def speak(self, text: str) -> None:
        self.messages.append(text)

