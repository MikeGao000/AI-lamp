import unittest

import numpy as np

from lamp_core.alignment import (
    CoarseToFineAim,
    HorizontalDirection,
    NarrowingScan,
    ResponsiveJ1Trajectory,
    VisualServoPid,
    bounded_j1_test_degrees,
    mapped_j1_tracking_degrees,
    motion_compensated_error,
    observe_bbox,
    page_score,
    profiled_motor_speed_rpm,
    saliency_thirds,
    smooth_bounded_scan_degrees,
    stepped_scan_degrees,
    visual_servo_degrees,
)


class CoarseToFineAimTests(unittest.TestCase):
    def test_large_first_error_gets_a_decisive_correction(self):
        aim = CoarseToFineAim(44.5, 0.03, 8.0)
        self.assertAlmostEqual(6.675, aim.update(0.15), places=6)

    def test_settled_hysteresis_ignores_boundary_jitter(self):
        aim = CoarseToFineAim(44.5, 0.03, 8.0)
        self.assertEqual(0.0, aim.update(0.02))
        self.assertEqual(0.0, aim.update(0.04))

    def test_small_reversal_must_repeat(self):
        aim = CoarseToFineAim(44.5, 0.03, 8.0)
        self.assertGreater(aim.update(0.08), 0.0)
        # Repeated negative observations first pull the adaptive filter through
        # zero, then must confirm the actual reverse command.
        outputs = [aim.update(-0.08) for _ in range(3)]
        self.assertEqual(0.0, outputs[0])
        self.assertLess(outputs[1], 0.0)

    def test_reset_discards_the_old_filtered_measurement(self):
        aim = CoarseToFineAim(44.5, 0.03, 8.0)
        aim.update(0.08)
        aim.update(-0.08)
        aim.reset()
        self.assertAlmostEqual(-3.240, aim.update(-0.08), places=3)


class AlignmentTests(unittest.TestCase):
    def test_sine_search_starts_centered_and_eases_through_turnarounds(self):
        self.assertAlmostEqual(0.0, smooth_bounded_scan_degrees(0.0, 10.0))
        self.assertAlmostEqual(10.0, smooth_bounded_scan_degrees(1.5, 10.0))
        self.assertAlmostEqual(0.0, smooth_bounded_scan_degrees(3.0, 10.0), places=6)
        before = smooth_bounded_scan_degrees(1.49, 10.0)
        after = smooth_bounded_scan_degrees(1.51, 10.0)
        self.assertAlmostEqual(before, after, places=5)

    def test_page_score_rewards_a_larger_uncropped_page(self):
        self.assertEqual(0.0, page_score(None))
        self.assertAlmostEqual(0.16, page_score((0.3, 0.3, 0.7, 0.7)))
        self.assertAlmostEqual(0.08, page_score((0.0, 0.3, 0.4, 0.7)))

    def test_narrowing_scan_narrows_onto_the_best_angle(self):
        scan = NarrowingScan(envelope_degrees=20.0, step_degrees=10.0, minimum_step_degrees=5.0)
        targets = []
        while True:
            target = scan.next_target()
            if target is None:
                break
            targets.append(target)
            scan.record(1.0 - abs(target - 10.0) / 20.0)
        self.assertEqual([0.0, -10.0, 10.0, -20.0, 20.0, 10.0, 5.0, 15.0], targets)
        self.assertTrue(scan.finished)
        self.assertAlmostEqual(10.0, scan.best_degrees)

    def test_narrowing_scan_rejects_a_score_after_it_finished(self):
        scan = NarrowingScan(envelope_degrees=10.0, step_degrees=10.0, minimum_step_degrees=8.0)
        while scan.next_target() is not None:
            scan.record(1.0)
        self.assertTrue(scan.finished)
        with self.assertRaises(ValueError):
            scan.record(1.0)

    def test_stepped_scan_dwells_then_steps_through_preset_angles(self):
        # 20 deg envelope with a 10 deg step gives 0, +10, -10, +20, -20.
        self.assertEqual(0.0, stepped_scan_degrees(0.0, 20.0))
        self.assertEqual(0.0, stepped_scan_degrees(2.4, 20.0))
        self.assertEqual(10.0, stepped_scan_degrees(2.5, 20.0))
        self.assertEqual(-10.0, stepped_scan_degrees(5.0, 20.0))
        self.assertEqual(20.0, stepped_scan_degrees(7.5, 20.0))
        self.assertEqual(-20.0, stepped_scan_degrees(10.0, 20.0))
        self.assertEqual(0.0, stepped_scan_degrees(12.5, 20.0))
        with self.assertRaises(ValueError):
            stepped_scan_degrees(1.0, 20.0, dwell_seconds=0.0)

    def test_motion_compensation_removes_the_cameras_own_rotation(self):
        # A frame shift of 10/44.5 is exactly the 10 deg command at the measured
        # 44.5 deg field of view, so the compensated error is zero.
        self.assertAlmostEqual(
            0.0, motion_compensated_error(10.0 / 44.5, 10.0, 44.5), places=9
        )
        self.assertAlmostEqual(-0.1, motion_compensated_error(0.2, 13.35, 44.5), places=4)
        with self.assertRaises(ValueError):
            motion_compensated_error(0.0, 10.0, 0.0)

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

    def test_visual_servo_uses_the_calibrated_horizontal_fov(self):
        # error_x = +0.25, measured FOV 44.5 deg -> 0.25 * 44.5 = 11.125 deg.
        result = observe_bbox((0.60, 0.20, 0.90, 0.80))
        self.assertAlmostEqual(
            11.125,
            visual_servo_degrees(
                result,
                positive_camera_direction=HorizontalDirection.RIGHT,
                degrees_per_error=44.5,
                maximum_degrees=60.0,
            ),
        )
        self.assertAlmostEqual(
            -11.125,
            visual_servo_degrees(
                result,
                positive_camera_direction=HorizontalDirection.LEFT,
                degrees_per_error=44.5,
                maximum_degrees=60.0,
            ),
        )

    def test_visual_servo_holds_centred_and_clamps_to_travel(self):
        centred = observe_bbox((0.42, 0.20, 0.62, 0.80))
        self.assertEqual(
            0.0,
            visual_servo_degrees(
                centred,
                positive_camera_direction=HorizontalDirection.RIGHT,
                degrees_per_error=44.5,
                maximum_degrees=60.0,
            ),
        )
        far_left = observe_bbox((0.00, 0.20, 0.10, 0.80))  # error_x = -0.45
        self.assertEqual(
            -15.0,
            visual_servo_degrees(
                far_left,
                positive_camera_direction=HorizontalDirection.RIGHT,
                degrees_per_error=44.5,
                maximum_degrees=15.0,
            ),
        )

    def test_visual_servo_pid_turns_error_into_velocity(self):
        servo = VisualServoPid(kp=180.0, ki=80.0, kd=8.0, max_velocity_degrees_s=72.0)
        velocity = servo.update(0.25, 0.1)
        self.assertGreater(velocity, 40.0)
        self.assertLessEqual(velocity, 72.0)

    def test_visual_servo_pid_holds_when_centred_and_clamps_to_cap(self):
        servo = VisualServoPid(kp=180.0, ki=80.0, kd=8.0, max_velocity_degrees_s=72.0)
        self.assertEqual(0.0, servo.update(0.05, 0.1))
        self.assertEqual(72.0, servo.update(0.5, 0.1))

    def test_visual_servo_pid_clamps_integral_and_resets(self):
        servo = VisualServoPid(
            kp=0.0, ki=100.0, kd=0.0, max_velocity_degrees_s=1000.0, integral_limit=1.0
        )
        for _ in range(200):
            servo.update(0.4, 0.1)
        self.assertAlmostEqual(100.0, servo.update(0.4, 0.1), places=3)
        servo.reset()
        self.assertEqual(0.0, servo.integral)
        self.assertIsNone(servo.previous_error)


class _NumpyCV2:
    """Just enough OpenCV for saliency_thirds, backed by numpy."""

    COLOR_BGR2HSV = 0
    COLOR_BGR2GRAY = 1
    CV_32F = 5

    @staticmethod
    def resize(image, size):
        height, width = image.shape[:2]
        rows = np.clip(np.arange(size[1]) * height // size[1], 0, height - 1)
        columns = np.clip(np.arange(size[0]) * width // size[0], 0, width - 1)
        return image[rows][:, columns]

    @staticmethod
    def cvtColor(image, code):
        if code == _NumpyCV2.COLOR_BGR2GRAY:
            return image[..., :3].astype(np.float32).mean(axis=2)
        channels = image[..., :3].astype(np.float32)
        maximum = channels.max(axis=2)
        minimum = channels.min(axis=2)
        saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
        hsv = np.zeros(image.shape, dtype=np.float32)
        hsv[..., 1] = saturation * 255.0
        return hsv

    @staticmethod
    def Laplacian(image, dtype):
        return (
            np.roll(image, 1, axis=0)
            + np.roll(image, -1, axis=0)
            + np.roll(image, 1, axis=1)
            + np.roll(image, -1, axis=1)
            - 4.0 * image
        )


class SaliencyThirdsTests(unittest.TestCase):
    """The hint only orders a sweep, so it must be cheap and honest about flatness."""

    def test_saturated_content_marks_its_own_third(self):
        image = np.full((60, 90, 3), 128, dtype=np.uint8)
        image[10:50, 0:28] = (0, 0, 255)  # saturated red inside the left third
        profile = saliency_thirds(image, _NumpyCV2())
        self.assertEqual(0, profile.index(max(profile)))
        self.assertAlmostEqual(1.0, sum(profile), places=6)

    def test_content_on_the_right_marks_the_right_third(self):
        image = np.full((60, 90, 3), 128, dtype=np.uint8)
        image[10:50, 62:90] = (0, 0, 255)
        profile = saliency_thirds(image, _NumpyCV2())
        self.assertEqual(2, profile.index(max(profile)))

    def test_a_flat_frame_reports_no_signal(self):
        image = np.full((60, 90, 3), 128, dtype=np.uint8)
        self.assertEqual((0.0, 0.0, 0.0), saliency_thirds(image, _NumpyCV2()))

    def test_unsaturated_print_on_white_still_counts(self):
        # Black print on a white page has no saturation at all, so edge energy
        # has to carry it or a text-only page would look empty.
        image = np.full((60, 90, 3), 255, dtype=np.uint8)
        image[20:40, 30:35] = 0
        image[20:40, 45:50] = 0
        profile = saliency_thirds(image, _NumpyCV2())
        self.assertEqual(1, profile.index(max(profile)))
        self.assertGreater(max(profile), 0.0)


class NarrowingScanReachTests(unittest.TestCase):
    def _sweep(self, scan):
        targets = []
        while True:
            target = scan.next_target()
            if target is None:
                break
            targets.append(target)
            scan.record(0.0)
        return targets

    def test_opening_round_reaches_the_whole_envelope(self):
        # The reach used to be two steps either side of centre whatever the
        # envelope was, which made a book 40 degrees off unreachable.
        scan = NarrowingScan(
            envelope_degrees=40.0, step_degrees=10.0, minimum_step_degrees=2.5
        )
        targets = self._sweep(scan)
        for expected in (-40.0, -30.0, 30.0, 40.0):
            self.assertIn(expected, targets)

    def test_the_envelope_is_still_respected(self):
        scan = NarrowingScan(
            envelope_degrees=20.0, step_degrees=10.0, minimum_step_degrees=10.0
        )
        self.assertLessEqual(max(abs(target) for target in self._sweep(scan)), 20.0)

    def test_a_salient_angle_is_sampled_first(self):
        scan = NarrowingScan(
            envelope_degrees=40.0,
            step_degrees=10.0,
            minimum_step_degrees=5.0,
            first_degrees=14.8,
        )
        self.assertAlmostEqual(14.8, scan.next_target(), places=6)

    def test_a_first_angle_outside_the_envelope_is_clamped(self):
        scan = NarrowingScan(
            envelope_degrees=20.0,
            step_degrees=10.0,
            minimum_step_degrees=5.0,
            first_degrees=99.0,
        )
        self.assertAlmostEqual(20.0, scan.next_target(), places=6)


if __name__ == "__main__":
    unittest.main()
