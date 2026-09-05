import unittest

from lamp_core.action_catalog import DEFAULT_ACTION_CATALOG
from lamp_core.behavior import (
    BehaviorProposal,
    BehaviorValidationError,
    ScreenTarget,
    ai_behavior_tool_schema,
    approve_behavior,
)


BOOK = ScreenTarget("book-1", "book", 0.25, 0.55, 0.91)


class BehaviorTests(unittest.TestCase):
    def test_ai_can_select_a_visible_book_by_id(self):
        approved = approve_behavior(
            BehaviorProposal("READING_POSTURE", target_ref="book-1", speech="我们开始读吧。", intensity=0.3),
            {"book-1": BOOK},
            safety_ready=True,
        )
        self.assertEqual("READING_POSTURE", approved.action_id)
        self.assertEqual("reading_posture", approved.motion_key)
        self.assertEqual(BOOK, approved.target)

    def test_lelamp_expression_is_selectable_but_remains_a_symbolic_motion_key(self):
        approved = approve_behavior(
            BehaviorProposal("LELAMP_GREET_SMALL", intensity=0.3), {}, safety_ready=True
        )
        self.assertEqual("LELAMP_GREET_SMALL", approved.action_id)
        self.assertEqual("lelamp_greet_small", approved.motion_key)
        self.assertEqual("L1", approved.verification_level)

    def test_unknown_disabled_or_invalid_intensity_actions_are_rejected(self):
        with self.assertRaisesRegex(BehaviorValidationError, "unknown"):
            approve_behavior(BehaviorProposal("INVENT_NEW_MOVE"), {}, safety_ready=True)
        with self.assertRaisesRegex(BehaviorValidationError, "not available"):
            approve_behavior(BehaviorProposal("LELAMP_TOUCH_TARGET"), {}, safety_ready=True)
        with self.assertRaisesRegex(BehaviorValidationError, "intensity"):
            approve_behavior(BehaviorProposal("GENTLE_NOD", intensity=0.9), {}, safety_ready=True)

    def test_target_must_be_visible_when_an_action_accepts_it(self):
        with self.assertRaisesRegex(BehaviorValidationError, "not currently visible"):
            approve_behavior(
                BehaviorProposal("READING_POSTURE", target_ref="book-9"), {}, safety_ready=True
            )
        with self.assertRaisesRegex(BehaviorValidationError, "does not accept"):
            approve_behavior(
                BehaviorProposal("GENTLE_NOD", target_ref="book-1"), {"book-1": BOOK}, safety_ready=True
            )

    def test_stop_is_allowed_after_a_fault_but_motion_is_not(self):
        approved = approve_behavior(BehaviorProposal("STOP"), {}, safety_ready=False)
        self.assertEqual("STOP", approved.action_id)
        with self.assertRaisesRegex(BehaviorValidationError, "not safe"):
            approve_behavior(BehaviorProposal("GENTLE_NOD"), {}, safety_ready=False)

    def test_tool_schema_exposes_only_enabled_model_catalogue_entries(self):
        actions = set(ai_behavior_tool_schema()["parameters"]["properties"]["action_id"]["enum"])
        self.assertEqual(set(DEFAULT_ACTION_CATALOG.selectable_action_ids()), actions)
        self.assertIn("LELAMP_GREET_SMALL", actions)
        self.assertNotIn("LELAMP_TOUCH_TARGET", actions)
        self.assertNotIn("drive_can_motor", actions)
