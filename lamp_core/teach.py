"""Safe manual teach, edit, and replay primitives for the five MKS joints.

Recording/replay has no physical I/O here. A verified MKS 42D adapter supplies
encoder samples and consumes the resulting JointMove segments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lamp_core.motion import JointLimit
from lamp_core.smart_motion import JointMove, SmartMotionDispatcher

FORMAT_VERSION = 1


@dataclass(frozen=True)
class TeachSample:
    time_s: float
    positions_rad: dict[str, float]


@dataclass(frozen=True)
class TeachRecording:
    name: str
    joint_names: tuple[str, ...]
    samples: tuple[TeachSample, ...]

    def validate(self, limits: Mapping[str, JointLimit]) -> None:
        if not self.name.strip():
            raise ValueError("recording name must not be empty")
        if tuple(limits) != self.joint_names:
            raise ValueError("recording joint order does not match configured robot")
        if len(self.samples) < 2:
            raise ValueError("recording needs at least two samples")
        previous_time = -1.0
        for sample in self.samples:
            if sample.time_s <= previous_time:
                raise ValueError("sample times must be strictly increasing")
            if set(sample.positions_rad) != set(limits):
                raise ValueError("sample joints do not match configured robot")
            for joint, angle in sample.positions_rad.items():
                limits[joint].validate(angle)
            previous_time = sample.time_s


class TeachingRecorder:
    """Collect slow, manually moved encoder samples after motors are made safe."""

    def __init__(self, name: str, limits: Mapping[str, JointLimit]) -> None:
        self.name = name
        self.limits = limits
        self._samples: list[TeachSample] = []

    def append(self, time_s: float, positions_rad: Mapping[str, float]) -> None:
        sample = TeachSample(time_s, dict(positions_rad))
        # A single early sample is valid enough to check joint names/limits.
        if self._samples and time_s <= self._samples[-1].time_s:
            raise ValueError("teach samples must have increasing timestamps")
        if set(sample.positions_rad) != set(self.limits):
            raise ValueError("teach sample joints do not match configured robot")
        for joint, angle in sample.positions_rad.items():
            self.limits[joint].validate(angle)
        self._samples.append(sample)

    def finish(self) -> TeachRecording:
        recording = TeachRecording(self.name, tuple(self.limits), tuple(self._samples))
        recording.validate(self.limits)
        return recording


def smooth(recording: TeachRecording, limits: Mapping[str, JointLimit], radius: int = 1) -> TeachRecording:
    """Centered moving-average smoothing; endpoints remain unchanged."""
    recording.validate(limits)
    if radius < 1:
        return recording
    samples = list(recording.samples)
    output: list[TeachSample] = []
    for index, sample in enumerate(samples):
        if index < radius or index >= len(samples) - radius:
            output.append(sample)
            continue
        window = samples[index - radius : index + radius + 1]
        positions = {
            joint: sum(item.positions_rad[joint] for item in window) / len(window)
            for joint in recording.joint_names
        }
        output.append(TeachSample(sample.time_s, positions))
    result = TeachRecording(recording.name, recording.joint_names, tuple(output))
    result.validate(limits)
    return result


def save(recording: TeachRecording, limits: Mapping[str, JointLimit], path: str | Path) -> None:
    recording.validate(limits)
    payload = {
        "format_version": FORMAT_VERSION,
        "name": recording.name,
        "joint_names": list(recording.joint_names),
        "samples": [
            {"time_s": sample.time_s, "positions_rad": sample.positions_rad}
            for sample in recording.samples
        ],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: str | Path, limits: Mapping[str, JointLimit]) -> TeachRecording:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported teach recording format")
    result = TeachRecording(
        name=str(payload["name"]),
        joint_names=tuple(payload["joint_names"]),
        samples=tuple(
            TeachSample(float(row["time_s"]), {key: float(value) for key, value in row["positions_rad"].items()})
            for row in payload["samples"]
        ),
    )
    result.validate(limits)
    return result


def queue_safe_replay(
    recording: TeachRecording,
    limits: Mapping[str, JointLimit],
    dispatcher: SmartMotionDispatcher,
    playback_rate: float = 0.35,
) -> int:
    """Queue a deliberately slowed trajectory; real hardware requires homing first."""
    recording.validate(limits)
    if not 0 < playback_rate <= 1:
        raise ValueError("playback_rate must be in (0, 1]")
    queued = 0
    for before, after in zip(recording.samples, recording.samples[1:]):
        original_duration = after.time_s - before.time_s
        duration = original_duration / playback_rate
        # Never exceed per-joint speed limits, even for a bad recording.
        duration = max(
            duration,
            *(
                abs(after.positions_rad[name] - before.positions_rad[name])
                / limits[name].maximum_speed_rad_s
                for name in limits
            ),
        )
        dispatcher.submit_trajectory_segment(
            JointMove(dict(after.positions_rad), duration, label=f"teach:{recording.name}:{queued}")
        )
        queued += 1
    return queued
