import unittest

from lamp_core.ideal_plant import verify_ideal_trajectory
from lamp_core.motion import JointLimit, plan_synchronised_minimum_jerk
from simulate_ideal_verification import PROFILES, run_curated_profiles


class IdealPlantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = {
            "pan": JointLimit(-1.0, 1.0, 0.5),
            "tilt": JointLimit(-0.5, 0.5, 1.0),
        }

    def test_ideal_execution_reports_zero_tracking_error_and_bounded_speed(self) -> None:
        frames = plan_synchronised_minimum_jerk(
            {"pan": -1.0, "tilt": 0.5},
            {"pan": 1.0, "tilt": -0.5},
            self.limits,
            sample_period_s=0.02,
        )
        report = verify_ideal_trajectory(frames, self.limits)
        self.assertEqual(0.0, report.max_tracking_error_rad)
        self.assertLessEqual(report.max_abs_speed_rad_s["pan"], self.limits["pan"].maximum_speed_rad_s)
        self.assertLessEqual(report.max_abs_speed_rad_s["tilt"], self.limits["tilt"].maximum_speed_rad_s)
        self.assertGreater(report.max_abs_acceleration_rad_s2["pan"], 0.0)

    def test_all_curated_five_axis_profiles_pass_the_ideal_verifier(self) -> None:
        reports = dict(run_curated_profiles())
        self.assertEqual(set(PROFILES), set(reports))
        for report in reports.values():
            self.assertGreater(report.frame_count, 1)
            self.assertGreater(report.duration_s, 0.0)
            self.assertEqual(0.0, report.max_tracking_error_rad)
