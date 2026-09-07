import unittest

from lamp_core.camera_kinematics import DEFAULT_IDEAL_CAMERA_MODEL


class CameraKinematicsTests(unittest.TestCase):
    def test_two_axis_error_is_distributed_across_all_camera_joints(self):
        correction = DEFAULT_IDEAL_CAMERA_MODEL.minimum_norm_correction(-0.2, 0.3)
        self.assertEqual(set(DEFAULT_IDEAL_CAMERA_MODEL.joint_names), set(correction))
        self.assertTrue(all(abs(value) > 1e-6 for value in correction.values()))

    def test_correction_cancels_the_requested_image_error(self):
        correction = DEFAULT_IDEAL_CAMERA_MODEL.minimum_norm_correction(0.17, -0.11)
        horizontal = sum(
            DEFAULT_IDEAL_CAMERA_MODEL.horizontal_gain_per_rad[name] * correction[name]
            for name in correction
        )
        vertical = sum(
            DEFAULT_IDEAL_CAMERA_MODEL.vertical_gain_per_rad[name] * correction[name]
            for name in correction
        )
        self.assertAlmostEqual(0.17, horizontal)
        self.assertAlmostEqual(-0.11, vertical)
