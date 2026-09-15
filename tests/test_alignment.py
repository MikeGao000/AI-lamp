import unittest

from lamp_core.alignment import (
    HorizontalDirection,
    ResponsiveJ1Trajectory,
    bounded_j1_test_degrees,
    mapped_j1_tracking_degrees,
    observe_bbox,
    profiled_motor_speed_rpm,
    smooth_bounded_scan_degrees,
)


class AlignmentTests(unittest.TestCase):
    def test_sine_search_starts_centered_and_eases_through_turnarounds(self):
        self.assertAlmostEqual(0.0, smooth_bounded_scan_degrees(0.0, 10.0))
        self.assertAlmostEqual(10.0, smooth_bounded_scan_degrees(1.5, 10.0))
        self.assertAlmostEqual(0.0, smooth_bounded_scan_degrees(3.0, 10.0), places=6)
        before = smooth_bounded_scan_degrees(1.49, 10.0)
        after = smooth_bounded_scan_degrees(1.51, 10.0)
        self.assertAlmostEqual(before, after, places=5)

    def test_responsive_trajectory_is_fast_but_jerk_limited(self):
        profile = ResponsiveJ1Trajectory(70.0, 230.0, 1280.0)
        previous_acceleration = 0.0
        positions = []
        for _ in range(6):
            positions.append(profile.update(10.0, 0.1))
            self.assertLessEqual(abs(profile.velocity_degrees_s), 70.0)
            self.assertLessEqual(
                abs(profile.acceleration_degrees_s2 - previous_acceleration),
                128.0 + 1e-9,
            )
            previous_acceleration = profile.acceleration_degrees_s2
        self.assertGreater(positions[0], 0.0)
        self.assertLess(positions[0], 2.0)
        self.assertGreater(positions[3], 5.0)
        settled = [profile.update(10.0, 0.1) for _ in range(24)]
        self.assertLess(max(positions + settled), 10.01)
        self.assertGreater(settled[-1], 9.9)

    def test_responsive_trajectory_reverses_without_secondary_wobble(self):
        profile = ResponsiveJ1Trajectory(70.0, 230.0, 1280.0)
        for _ in range(15):
            profile.update(10.0, 0.1)
        reversed_positions = [profile.update(-10.0, 0.1) for _ in range(20)]
        self.assertTrue(
            all(first >= second for first, second in zip(reversed_positions, reversed_positions[1:]))
        )
        self.assertGreater(min(reversed_positions), -10.01)

    def test_responsive_trajectory_suppresses_small_visual_jitter(self):
        profile = ResponsiveJ1Trajectory(70.0, 230.0, 1280.0)
        outputs = [profile.update(target, 0.1) for target in (0.15, -0.12, 0.10, -0.08, 0.05)]
        self.assertLess(max(abs(value) for value in outputs), 0.15)

    def test_motor_speed_follows_trajectory_without_lowering_the_cap(self):
        self.assertEqual(1, profiled_motor_speed_rpm(0.0, 13.7, 160))
        self.assertLess(
            profiled_motor_speed_rpm(2.0, 13.7, 160),
            profiled_motor_speed_rpm(20.0, 13.7, 160),
        )
        self.assertEqual(160, profiled_motor_speed_rpm(1000.0, 13.7, 160))

    def test_reports_target_to_right_of_camera(self):
        result = observe_bbox((0.60, 0.20, 0.90, 0.80))
        self.assertEqual(HorizontalDirection.RIGHT, result.horizontal)
        self.assertAlmostEqual(0.25, result.error_x)

    def test_deadband_prevents_motor_hunting(self):
        result = observe_bbox((0.42, 0.20, 0.62, 0.80))
        self.assertTrue(result.centered)
        self.assertEqual(0.0, bounded_j1_test_degrees(result, positive_camera_direction=HorizontalDirection.RIGHT))

    def test_one_shot_move_respects_calibrated_positive_direction(self):
        result = observe_bbox((0.65, 0.20, 0.95, 0.80))
        self.assertGreater(
            bounded_j1_test_degrees(result, positive_camera_direction=HorizontalDirection.RIGHT), 0
        )
        self.assertLess(
            bounded_j1_test_degrees(result, positive_camera_direction=HorizontalDirection.LEFT), 0
        )

    def test_test_move_is_capped_at_fifteen_degrees(self):
        result = observe_bbox((0.90, 0.20, 1.00, 0.80))
        with self.assertRaises(ValueError):
            bounded_j1_test_degrees(
                result,
                positive_camera_direction=HorizontalDirection.RIGHT,
                maximum_degrees=16,
            )

    def test_realtime_mapping_is_absolute_not_accumulating(self):
        result = observe_bbox((0.65, 0.20, 0.95, 0.80))
        first = mapped_j1_tracking_degrees(
            result,
            positive_camera_direction=HorizontalDirection.RIGHT,
            envelope_degrees=10,
        )
        second = mapped_j1_tracking_degrees(
            result,
            positive_camera_direction=HorizontalDirection.RIGHT,
            envelope_degrees=10,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(6.0, first)

    def test_realtime_mapping_reverses_for_opposite_calibration(self):
        result = observe_bbox((0.65, 0.20, 0.95, 0.80))
        self.assertAlmostEqual(
            -6.0,
            mapped_j1_tracking_degrees(
                result,
                positive_camera_direction=HorizontalDirection.LEFT,
                envelope_degrees=10,
            ),
        )


if __name__ == "__main__":
    unittest.main()
