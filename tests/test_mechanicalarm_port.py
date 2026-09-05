import unittest

from lamp_core.mechanicalarm_port import (
    CartesianPose,
    MechanicalArmControlLoop,
    MechanicalArmProfileConfig,
    plan_mechanicalarm_s_curve,
)


class NetworkStub:
    def __init__(self) -> None:
        self.positions = {"j1": 0.0, "j2": 0.0}
        self.commands: list[dict[str, float]] = []

    def read_joint_positions_rad(self):
        return self.positions

    def command_joint_velocities_rad_s(self, velocities_rad_s):
        self.commands.append(dict(velocities_rad_s))


def simple_ik(pose, current):
    return {"j1": pose.x / 100.0, "j2": pose.y / 100.0}


class MechanicalArmPortTests(unittest.TestCase):
    def test_s_curve_reaches_exact_target(self):
        current = CartesianPose(0, 0, 0)
        target = CartesianPose(100, 20, 0, alpha=0.2)
        profile = plan_mechanicalarm_s_curve(target, current)
        self.assertGreater(len(profile), 1)
        self.assertEqual(target, profile[-1].pose)
        self.assertGreater(profile[-1].time_s, 0.0)

    def test_zero_translation_is_single_reference_without_division(self):
        target = CartesianPose(0, 0, 0, alpha=0.5)
        profile = plan_mechanicalarm_s_curve(target, CartesianPose(0, 0, 0))
        self.assertEqual([profile[0].pose], [target])

    def test_ported_loop_uses_feedback_feedforward_and_p_error(self):
        network = NetworkStub()
        loop = MechanicalArmControlLoop(
            network,
            simple_ik,
            {"j1": 0.65, "j2": 0.5},
            MechanicalArmProfileConfig(max_acceleration=500, max_speed=300, control_period_s=0.01),
        )
        loop.start(CartesianPose(100, 0, 0), CartesianPose(0, 0, 0))
        first = loop.tick()
        second = loop.tick()
        self.assertGreater(first["j1"], 0.0)  # P correction from current encoder position.
        self.assertGreater(second["j1"], first["j1"])  # Feed-forward joins after tick one.
        self.assertEqual(2, len(network.commands))
        loop.stop()
        self.assertEqual({"j1": 0.0, "j2": 0.0}, network.commands[-1])
