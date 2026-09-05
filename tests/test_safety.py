import unittest

from lamp_core.safety import MotionSafetyCheck, SafetyController, SafetyError, SafetyState


class SafetyTests(unittest.TestCase):
    def test_motion_requires_homing_enable_and_explicit_safety_facts(self):
        safety = SafetyController()
        with self.assertRaises(SafetyError):
            safety.begin_motion(MotionSafetyCheck.simulated_clear())
        safety.begin_homing()
        safety.complete_homing()
        safety.enable_drives()
        with self.assertRaisesRegex(SafetyError, "heartbeat"):
            safety.begin_motion(
                MotionSafetyCheck(
                    calibrated=True,
                    can_healthy=True,
                    heartbeats_fresh=False,
                    positions_readable=True,
                    limits_healthy=True,
                    temperature_healthy=True,
                    target_valid=True,
                    trajectory_valid=True,
                    action_budget_available=True,
                )
            )
        safety.begin_motion(MotionSafetyCheck.simulated_clear())
        self.assertEqual(safety.state, SafetyState.MOVING)

    def test_pause_fault_and_estop_are_latched_or_explicitly_reset(self):
        safety = SafetyController()
        safety.begin_homing()
        safety.complete_homing()
        safety.enable_drives()
        safety.begin_motion(MotionSafetyCheck.simulated_clear())
        safety.pause_motion()
        self.assertEqual(safety.state, SafetyState.PAUSED)
        safety.fault("simulated limit")
        self.assertEqual(safety.state, SafetyState.FAULT_LATCHED)
        safety.acknowledge_fault()
        self.assertEqual(safety.state, SafetyState.SAFE_DISABLED)
        safety.emergency_stop()
        with self.assertRaises(SafetyError):
            safety.begin_homing()
        safety.reset_estop()
        self.assertEqual(safety.state, SafetyState.SAFE_DISABLED)
