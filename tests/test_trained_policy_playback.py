import unittest

from lamp_core.motion import plan_synchronised_minimum_jerk
from lamp_core.pose_imitation import (
    fit_minimum_jerk_imitation,
    generate_pose_imitation_samples,
    library_poses,
    split_by_transition,
)
from simulate import JOINT_LIMITS


class TrainedPolicyPlaybackTests(unittest.TestCase):
    def test_trained_policy_replays_a_visualized_pose_transition(self):
        samples = generate_pose_imitation_samples(JOINT_LIMITS)
        training, _ = split_by_transition(samples)
        policy = fit_minimum_jerk_imitation(training)
        poses = library_poses()
        reference = plan_synchronised_minimum_jerk(poses["idle"], poses["reading_pose"], JOINT_LIMITS)
        last_index = len(reference) - 1
        for index, frame in enumerate(reference):
            predicted = policy.predict(poses["idle"], poses["reading_pose"], index / last_index)
            for joint_name, value in frame.positions_rad.items():
                self.assertAlmostEqual(value, predicted[joint_name], places=8)


if __name__ == "__main__":
    unittest.main()
