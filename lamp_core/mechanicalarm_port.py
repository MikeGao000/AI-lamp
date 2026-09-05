"""Python port of MechanicalArm_Code_V2's *control structure*.

This module keeps the source project's control topology while deliberately
leaving out its CAN packet bytes, motor ratios, DH table, and STM32 details:

    Cartesian target -> sine S-curve samples -> IK ->
    feed-forward joint velocity + P(position error) -> motor network

The concrete MKS adapter must implement ``ClosedLoopMotorNetwork`` using the
manual supplied with the purchased motor.  The module runs at 100 Hz and is
small enough for a Raspberry Pi 3B; it uses only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Callable, Mapping, Protocol


class MechanicalArmPortError(RuntimeError):
    pass


@dataclass(frozen=True)
class CartesianPose:
    """Position plus ZYZ-like orientation values, in the project's units."""

    x: float
    y: float
    z: float
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0


@dataclass(frozen=True)
class CartesianReference:
    time_s: float
    pose: CartesianPose
    path_speed: float


@dataclass(frozen=True)
class MechanicalArmProfileConfig:
    """Defaults intentionally match MechanicalArm_Code_V2's planner."""

    max_acceleration: float = 500.0
    max_speed: float = 300.0
    control_period_s: float = 0.01

    def validate(self) -> None:
        if self.max_acceleration <= 0 or self.max_speed <= 0 or self.control_period_s <= 0:
            raise MechanicalArmPortError("profile acceleration, speed, and period must be positive")


def _shortest_angle_delta(target: float, current: float) -> float:
    delta = target - current
    while delta > pi:
        delta -= 2.0 * pi
    while delta < -pi:
        delta += 2.0 * pi
    return delta


def _interpolate_pose(target: CartesianPose, current: CartesianPose, fraction: float) -> CartesianPose:
    fraction = min(1.0, max(0.0, fraction))
    return CartesianPose(
        x=current.x + fraction * (target.x - current.x),
        y=current.y + fraction * (target.y - current.y),
        z=current.z + fraction * (target.z - current.z),
        alpha=current.alpha + fraction * _shortest_angle_delta(target.alpha, current.alpha),
        beta=current.beta + fraction * _shortest_angle_delta(target.beta, current.beta),
        gamma=current.gamma + fraction * _shortest_angle_delta(target.gamma, current.gamma),
    )


def plan_mechanicalarm_s_curve(
    target: CartesianPose,
    current: CartesianPose,
    config: MechanicalArmProfileConfig = MechanicalArmProfileConfig(),
) -> list[CartesianReference]:
    """Port the source project's sine acceleration / constant speed profile.

    The original plans translation distance and linearly interpolates pose;
    this preserves that behaviour.  The zero-translation case is explicitly
    handled here, rather than allowing the original ``s / S`` division.
    """
    config.validate()
    distance = sqrt((target.x - current.x) ** 2 + (target.y - current.y) ** 2 + (target.z - current.z) ** 2)
    if distance <= 1e-9:
        return [CartesianReference(0.0, target, 0.0)]

    omega = 2.0 * config.max_acceleration / config.max_speed
    distance_for_accel = pi * config.max_speed / omega
    t1_limit = 2.0 * pi / omega
    if distance >= distance_for_accel:
        vmax = config.max_speed
        t1 = t1_limit
        t0 = (distance - distance_for_accel) / vmax
        total_time = t0 + t1
    else:
        vmax = distance * omega / pi
        t1 = t1_limit
        t0 = 0.0
        total_time = t1
    half_v_over_omega = vmax / (2.0 * omega)

    step_count = max(1, round(total_time / config.control_period_s))
    references: list[CartesianReference] = []
    for index in range(step_count):
        time_s = min((index + 1) * config.control_period_s, total_time)
        if distance >= distance_for_accel:
            if time_s <= t1 / 2.0:
                velocity = (vmax / 2.0) * sin(omega * time_s - pi / 2.0) + vmax / 2.0
                traveled = vmax * time_s / 2.0 - half_v_over_omega * cos(omega * time_s - pi / 2.0)
            elif time_s <= t1 / 2.0 + t0:
                velocity = vmax
                traveled = distance_for_accel / 2.0 + vmax * (time_s - t1 / 2.0)
            else:
                decel_time = time_s - t0
                velocity = (vmax / 2.0) * sin(omega * decel_time - pi / 2.0) + vmax / 2.0
                traveled = (
                    vmax * decel_time / 2.0
                    - half_v_over_omega * cos(omega * decel_time - pi / 2.0)
                    + vmax * t0
                )
        else:
            velocity = (vmax / 2.0) * sin(omega * time_s - pi / 2.0) + vmax / 2.0
            traveled = vmax * time_s / 2.0 - half_v_over_omega * cos(omega * time_s - pi / 2.0)
        fraction = min(1.0, max(0.0, traveled / distance))
        references.append(CartesianReference(time_s, _interpolate_pose(target, current, fraction), velocity))

    # Guarantee the exact target despite sample-time rounding.
    if references[-1].pose != target:
        references[-1] = CartesianReference(total_time, target, 0.0)
    return references


class ClosedLoopMotorNetwork(Protocol):
    """The exact seam where the verified MKS CAN adapter plugs in."""

    def read_joint_positions_rad(self) -> Mapping[str, float]: ...

    def command_joint_velocities_rad_s(self, velocities_rad_s: Mapping[str, float]) -> None: ...


InverseKinematics = Callable[[CartesianPose, Mapping[str, float]], Mapping[str, float]]


class MechanicalArmControlLoop:
    """100 Hz port of the source project's outer-loop coordination method."""

    def __init__(
        self,
        network: ClosedLoopMotorNetwork,
        inverse_kinematics: InverseKinematics,
        position_gains: Mapping[str, float],
        config: MechanicalArmProfileConfig = MechanicalArmProfileConfig(),
    ) -> None:
        config.validate()
        if not position_gains or any(gain < 0 for gain in position_gains.values()):
            raise MechanicalArmPortError("provide a non-negative P gain for every joint")
        self.network = network
        self.inverse_kinematics = inverse_kinematics
        self.position_gains = dict(position_gains)
        self.config = config
        self._references: list[CartesianReference] = []
        self._index = 0
        self._last_target_rad: dict[str, float] | None = None

    @property
    def active(self) -> bool:
        return self._index < len(self._references)

    def start(self, target: CartesianPose, current: CartesianPose) -> None:
        self._references = plan_mechanicalarm_s_curve(target, current, self.config)
        self._index = 0
        self._last_target_rad = None

    def tick(self) -> Mapping[str, float]:
        """Run one source-equivalent 10 ms outer-loop iteration."""
        if not self.active:
            raise MechanicalArmPortError("no active Cartesian trajectory")
        actual = dict(self.network.read_joint_positions_rad())
        if set(actual) != set(self.position_gains):
            raise MechanicalArmPortError("motor feedback joints do not match configured P gains")
        target = dict(self.inverse_kinematics(self._references[self._index].pose, actual))
        if set(target) != set(actual):
            raise MechanicalArmPortError("inverse kinematics joints do not match motor feedback")

        if self._last_target_rad is None:
            feedforward = {joint: 0.0 for joint in target}
        else:
            feedforward = {
                joint: (target[joint] - self._last_target_rad[joint]) / self.config.control_period_s
                for joint in target
            }
        command = {
            joint: feedforward[joint] + self.position_gains[joint] * (target[joint] - actual[joint])
            for joint in target
        }
        self.network.command_joint_velocities_rad_s(command)
        self._last_target_rad = target
        self._index += 1
        return command

    def stop(self) -> None:
        """Use the source project's outer-loop stop behaviour: command zero velocity."""
        positions = self.network.read_joint_positions_rad()
        self.network.command_joint_velocities_rad_s({joint: 0.0 for joint in positions})
        self._references = []
        self._index = 0
        self._last_target_rad = None
