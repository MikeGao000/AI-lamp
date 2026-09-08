import unittest

from lamp_core.pose_imitation import (
    fit_minimum_jerk_imitation,
    generate_pose_imitation_samples,
    library_poses,
    maximum_angle_error,
    split_by_transition,
)
from simulate import JOINT_LIMITS


class PoseImitationTests(unittest.TestCase):
    def test_imports_every_ordered_pair_of_library_poses(self):
        samples = generate_pose_imitation_samples(JOINT_LIMITS)
        transition_ids = {sample.transition_index for sample in samples}
        pose_count = len(library_poses())
        self.assertEqual(pose_count * (pose_count - 1), len(transition_ids))
        self.assertEqual(30, len(transition_ids))

    def test_policy_learns_held_out_minimum_jerk_transitions(self):
        samples = generate_pose_imitation_samples(JOINT_LIMITS)
        training, validation = split_by_transition(samples)
        policy = fit_minimum_jerk_imitation(training)
        self.assertAlmostEqual(10.0, policy.coefficients[0], places=7)
        self.assertAlmostEqual(-15.0, policy.coefficients[1], places=7)
        self.assertAlmostEqual(6.0, policy.coefficients[2], places=7)
        self.assertLess(maximum_angle_error(policy, validation), 1e-8)


if __name__ == "__main__":
    unittest.main()
