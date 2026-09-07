"""Camera-frame centering through a calibrated multi-joint projection model.

The default model is only an ideal-simulation fixture.  A physical lamp must
replace it with measured image displacement per joint before hardware use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class CameraKinematicsError(ValueError):
    pass


@dataclass(frozen=True)
class CameraProjectionModel:
    """First-order mapping from joint radians to normalized image displacement."""

    joint_names: tuple[str, ...]
    horizontal_gain_per_rad: Mapping[str, float]
    vertical_gain_per_rad: Mapping[str, float]

    def validate(self) -> None:
        names = set(self.joint_names)
        if not self.joint_names or set(self.horizontal_gain_per_rad) != names or set(self.vertical_gain_per_rad) != names:
            raise CameraKinematicsError("camera gains must specify every camera joint exactly once")
        if not any(value != 0.0 for value in self.horizontal_gain_per_rad.values()) or not any(
            value != 0.0 for value in self.vertical_gain_per_rad.values()
        ):
            raise CameraKinematicsError("camera model must respond in both image axes")

    def image_offset(self, joint_positions_rad: Mapping[str, float]) -> tuple[float, float]:
        self.validate()
        if set(joint_positions_rad) != set(self.joint_names):
            raise CameraKinematicsError("camera positions must match calibrated joints")
        return (
            -sum(self.horizontal_gain_per_rad[name] * joint_positions_rad[name] for name in self.joint_names),
            -sum(self.vertical_gain_per_rad[name] * joint_positions_rad[name] for name in self.joint_names),
        )

    def minimum_norm_correction(self, image_error_x: float, image_error_y: float) -> dict[str, float]:
        """Distribute an image error across every calibrated joint.

        The closed-form two-row pseudoinverse avoids a numerical dependency on
        the Pi while keeping the controller compatible with an arbitrary number
        of camera-carrying joints.
        """

        self.validate()
        horizontal_square = sum(value * value for value in self.horizontal_gain_per_rad.values())
        vertical_square = sum(value * value for value in self.vertical_gain_per_rad.values())
        cross = sum(
            self.horizontal_gain_per_rad[name] * self.vertical_gain_per_rad[name] for name in self.joint_names
        )
        determinant = horizontal_square * vertical_square - cross * cross
        if determinant <= 1e-12:
            raise CameraKinematicsError("camera model cannot independently center horizontal and vertical errors")
        horizontal_component = (vertical_square * image_error_x - cross * image_error_y) / determinant
        vertical_component = (horizontal_square * image_error_y - cross * image_error_x) / determinant
        return {
            name: self.horizontal_gain_per_rad[name] * horizontal_component
            + self.vertical_gain_per_rad[name] * vertical_component
            for name in self.joint_names
        }


DEFAULT_IDEAL_CAMERA_MODEL = CameraProjectionModel(
    joint_names=("j1_base_yaw", "j2_shoulder", "j3_elbow", "j4_neck_pitch", "j5_head_yaw"),
    horizontal_gain_per_rad={
        "j1_base_yaw": 0.50,
        "j2_shoulder": 0.03,
        "j3_elbow": 0.02,
        "j4_neck_pitch": 0.02,
        "j5_head_yaw": 0.10,
    },
    vertical_gain_per_rad={
        "j1_base_yaw": 0.02,
        "j2_shoulder": 0.12,
        "j3_elbow": 0.18,
        "j4_neck_pitch": 0.38,
        "j5_head_yaw": 0.03,
    },
)
