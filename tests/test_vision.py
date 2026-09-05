import unittest

from lamp_core.vision import StillnessGate


class StillnessGateTests(unittest.TestCase):
    def test_requires_continuous_stillness(self):
        gate = StillnessGate(required_seconds=1.5, threshold=8.0)
        self.assertFalse(gate.observe(0.0, now=0.0))
        self.assertFalse(gate.observe(7.9, now=1.49))
        self.assertTrue(gate.observe(7.9, now=1.50))

    def test_motion_resets_timer(self):
        gate = StillnessGate(required_seconds=1.0, threshold=8.0)
        gate.observe(0.0, now=0.0)
        self.assertFalse(gate.observe(9.0, now=0.5))
        self.assertFalse(gate.observe(0.0, now=1.0))
