import unittest

from lamp_core.motion import JointLimit, MotionError, plan_synchronised_minimum_jerk


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.limits = {
            "pan": JointLimit(-1.0, 1.0, 0.5),
            "tilt": JointLimit(-0.5, 0.5, 1.0),
        }

    def test_hits_targets_together_and_respects_limits(self):
        points = plan_synchronised_minimum_jerk(
            {"pan": 0.0, "tilt": 0.0}, {"pan": 0.5, "tilt": -0.25}, self.limits
        )
        self.assertEqual(points[0].positions_rad, {"pan": 0.0, "tilt": 0.0})
        self.assertEqual(points[-1].positions_rad, {"pan": 0.5, "tilt": -0.25})
        self.assertGreater(len(points), 2)

    def test_rejects_out_of_range_target(self):
        with self.assertRaises(MotionError):
            plan_synchronised_minimum_jerk(
                {"pan": 0.0, "tilt": 0.0}, {"pan": 2.0, "tilt": 0.0}, self.limits
            )

    def test_acceleration_limit_extends_duration(self):
        limits = {
            "pan": JointLimit(-1.0, 1.0, 100.0, maximum_acceleration_rad_s2=1.0),
            "tilt": JointLimit(-1.0, 1.0, 100.0, maximum_acceleration_rad_s2=100.0),
        }
        points = plan_synchronised_minimum_jerk(
            {"pan": 0.0, "tilt": 0.0}, {"pan": 0.5, "tilt": 0.0}, limits
        )
        self.assertGreaterEqual(points[-1].time_s, 1.69)

    def test_rejects_invalid_acceleration_limit(self):
        limits = {
            "pan": JointLimit(-1.0, 1.0, 1.0, maximum_acceleration_rad_s2=0.0),
            "tilt": JointLimit(-1.0, 1.0, 1.0),
        }
        with self.assertRaises(MotionError):
            plan_synchronised_minimum_jerk(
                {"pan": 0.0, "tilt": 0.0}, {"pan": 0.1, "tilt": 0.0}, limits
            )
