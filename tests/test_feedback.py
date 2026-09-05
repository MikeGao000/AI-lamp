import unittest

from lamp_core.feedback import FeedbackError, FeedbackLimits, JointFeedback, verify_feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"j1": 0.1, "j2": -0.2}
        self.limits = FeedbackLimits(max_age_s=0.2, max_tracking_error_rad=0.05)

    def test_accepts_fresh_feedback_near_reference(self):
        verify_feedback(
            self.expected,
            {
                "j1": JointFeedback(0.12, 10.0),
                "j2": JointFeedback(-0.18, 9.9),
            },
            self.limits,
            now_s=10.1,
        )

    def test_rejects_stale_or_faulted_feedback(self):
        with self.assertRaises(FeedbackError):
            verify_feedback(
                self.expected,
                {
                    "j1": JointFeedback(0.1, 9.0),
                    "j2": JointFeedback(-0.2, 10.0),
                },
                self.limits,
                now_s=10.0,
            )
        with self.assertRaises(FeedbackError):
            verify_feedback(
                self.expected,
                {
                    "j1": JointFeedback(0.1, 10.0, drive_fault=True),
                    "j2": JointFeedback(-0.2, 10.0),
                },
                self.limits,
                now_s=10.0,
            )

    def test_rejects_large_tracking_error(self):
        with self.assertRaises(FeedbackError):
            verify_feedback(
                self.expected,
                {
                    "j1": JointFeedback(0.3, 10.0),
                    "j2": JointFeedback(-0.2, 10.0),
                },
                self.limits,
                now_s=10.0,
            )
