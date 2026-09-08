"""Offline imitation learning from the lamp's audited pose library.

The learner fits the time curve from generated pose-library trajectories.  It
does not replace the existing motion planner: it gives MuJoCo training a small,
reproducible policy that can replay an arbitrary library start/target pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from lamp_core.motion import JointLimit, plan_synchronised_minimum_jerk
from lamp_core.pose_library import IDLE_POSE, POSE_LIBRARY


JOINT_NAMES = tuple(IDLE_POSE)


@dataclass(frozen=True)
class PoseImitationSample:
    transition_index: int
    start_name: str
    target_name: str
    progress: float
    start_rad: dict[str, float]
    target_rad: dict[str, float]
    observed_rad: dict[str, float]


@dataclass(frozen=True)
class MinimumJerkImitationPolicy:
    """A fitted fifth-order pose interpolation policy."""

    coefficients: tuple[float, float, float]

    def predict(
        self,
        start_rad: Mapping[str, float],
        target_rad: Mapping[str, float],
        progress: float,
    ) -> dict[str, float]:
        progress = min(1.0, max(0.0, progress))
        blend = sum(
            coefficient * progress**power
            for coefficient, power in zip(self.coefficients, (3, 4, 5))
        )
        return {
            name: start_rad[name] + (target_rad[name] - start_rad[name]) * blend
            for name in JOINT_NAMES
        }


def library_poses() -> dict[str, dict[str, float]]:
    """Return a copy of the one source-of-truth pose library, including idle."""
    return {"idle": dict(IDLE_POSE), **{name: dict(pose) for name, pose in POSE_LIBRARY.items()}}


def generate_pose_imitation_samples(
    limits: Mapping[str, JointLimit], *, sample_period_s: float = 0.02
) -> tuple[PoseImitationSample, ...]:
    """Import every ordered pair of audited poses as a labelled trajectory."""
    poses = library_poses()
    pose_names = tuple(poses)
    samples: list[PoseImitationSample] = []
    transition_index = 0
    for start_name in pose_names:
        for target_name in pose_names:
            if start_name == target_name:
                continue
            start = poses[start_name]
            target = poses[target_name]
            frames = plan_synchronised_minimum_jerk(start, target, limits, sample_period_s)
            final_time = frames[-1].time_s
            for frame in frames:
                samples.append(
                    PoseImitationSample(
                        transition_index=transition_index,
                        start_name=start_name,
                        target_name=target_name,
                        progress=frame.time_s / final_time,
                        start_rad=dict(start),
                        target_rad=dict(target),
                        observed_rad=dict(frame.positions_rad),
                    )
                )
            transition_index += 1
    return tuple(samples)


def split_by_transition(
    samples: tuple[PoseImitationSample, ...], *, validation_every: int = 5
) -> tuple[tuple[PoseImitationSample, ...], tuple[PoseImitationSample, ...]]:
    if validation_every < 2:
        raise ValueError("validation_every must be at least 2")
    training = tuple(sample for sample in samples if sample.transition_index % validation_every)
    validation = tuple(sample for sample in samples if not sample.transition_index % validation_every)
    if not training or not validation:
        raise ValueError("split produced an empty training or validation set")
    return training, validation


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for pivot_column in range(3):
        pivot_row = max(range(pivot_column, 3), key=lambda row: abs(augmented[row][pivot_column]))
        if abs(augmented[pivot_row][pivot_column]) < 1e-12:
            raise ValueError("pose training matrix is singular")
        augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]
        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot for value in augmented[pivot_column]]
        for row in range(3):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[pivot_column])
            ]
    return tuple(augmented[row][3] for row in range(3))  # type: ignore[return-value]


def fit_minimum_jerk_imitation(
    samples: tuple[PoseImitationSample, ...],
) -> MinimumJerkImitationPolicy:
    """Fit the fifth-order blend coefficients from library demonstration frames."""
    normal = [[0.0 for _ in range(3)] for _ in range(3)]
    rhs = [0.0, 0.0, 0.0]
    observations = 0
    for sample in samples:
        features = (sample.progress**3, sample.progress**4, sample.progress**5)
        for name in JOINT_NAMES:
            delta = sample.target_rad[name] - sample.start_rad[name]
            if abs(delta) < 1e-12:
                continue
            observed_blend = (sample.observed_rad[name] - sample.start_rad[name]) / delta
            for row in range(3):
                rhs[row] += features[row] * observed_blend
                for column in range(3):
                    normal[row][column] += features[row] * features[column]
            observations += 1
    if observations == 0:
        raise ValueError("pose training needs at least one moving joint")
    return MinimumJerkImitationPolicy(_solve_3x3(normal, rhs))


def maximum_angle_error(
    policy: MinimumJerkImitationPolicy, samples: tuple[PoseImitationSample, ...]
) -> float:
    maximum = 0.0
    for sample in samples:
        predicted = policy.predict(sample.start_rad, sample.target_rad, sample.progress)
        maximum = max(
            maximum,
            *(abs(predicted[name] - sample.observed_rad[name]) for name in JOINT_NAMES),
        )
    return maximum
