import unittest

from lamp_core.action_motion import ACTION_RECIPES, ActionMotionError, compile_action_to_ideal_segments
from lamp_core.action_catalog import DEFAULT_ACTION_CATALOG
from lamp_core.ideal_plant import verify_ideal_trajectory
from lamp_core.pose_library import IDLE_POSE
from simulate import JOINT_LIMITS


class ActionMotionTests(unittest.TestCase):
    def test_every_l1_catalogue_expression_compiles_to_existing_pose_transitions(self):
        l1_action_ids = [
            action_id
            for action_id in DEFAULT_ACTION_CATALOG.selectable_action_ids()
            if DEFAULT_ACTION_CATALOG.resolve_for_model(action_id).verification_level == "L1"
        ]
        self.assertEqual(set(ACTION_RECIPES), set(l1_action_ids))
        for action_id in l1_action_ids:
            compiled = compile_action_to_ideal_segments(action_id, IDLE_POSE, JOINT_LIMITS)
            self.assertEqual(action_id, compiled.action_id)
            self.assertTrue(compiled.motion_key)
            for segment in compiled.segments:
                report = verify_ideal_trajectory(segment, JOINT_LIMITS)
                self.assertEqual(0.0, report.max_tracking_error_rad)

    def test_non_motion_or_unknown_actions_do_not_invent_a_trajectory(self):
        scene = compile_action_to_ideal_segments("SET_LIGHT_SCENE", IDLE_POSE, JOINT_LIMITS)
        self.assertEqual((), scene.segments)
        with self.assertRaisesRegex(ActionMotionError, "unknown"):
            compile_action_to_ideal_segments("INVENT_NEW_MOVE", IDLE_POSE, JOINT_LIMITS)
