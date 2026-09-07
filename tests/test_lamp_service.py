import json
import unittest

from lamp_core.autonomous_contracts import EventKind, EventSource
from lamp_core.event_router import RouteKind
from lamp_core.safety import SafetyError, SafetyState
from lamp_service import LampService, SessionState


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class LampServiceTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.logs: list[str] = []
        self.cancel_count = 0

        def cancel() -> None:
            self.cancel_count += 1

        self.service = LampService(
            clock=self.clock,
            sink=self.logs.append,
            awake_session_seconds=60.0,
            cancel_soft_effects=cancel,
        )

    def test_boot_enables_observation_without_touching_motor_safety(self):
        self.assertEqual(SessionState.BOOTING, self.service.session_state)
        self.service.boot()
        self.assertEqual(SessionState.IDLE_OBSERVING, self.service.session_state)
        self.assertEqual(SafetyState.SAFE_DISABLED, self.service.runtime.safety.state)
        self.service.runtime.safety.begin_homing()

    def test_event_before_boot_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.service.push_event(EventKind.WAKE, source=EventSource.VOICE)

    def test_wake_and_read_are_local_and_refresh_the_session_deadline(self):
        self.service.boot()
        wake = self.service.push_voice_transcript("小灯")
        self.assertIsNotNone(wake)
        self.assertEqual(RouteKind.LOCAL_INTENT, wake.routing.route)
        self.assertIsNone(wake.model_route)
        self.assertEqual(SessionState.AWAKE_SESSION, self.service.session_state)

        self.clock.advance(50.0)
        read = self.service.push_voice_transcript("开始读")
        self.assertIsNotNone(read)
        self.assertEqual(RouteKind.LOCAL_INTENT, read.routing.route)
        self.clock.advance(11.0)
        self.service.tick()
        self.assertEqual(SessionState.AWAKE_SESSION, self.service.session_state)
        self.clock.advance(50.0)
        self.service.tick()
        self.assertEqual(SessionState.IDLE_OBSERVING, self.service.session_state)

    def test_soft_stop_cancels_interaction_without_latching_estop(self):
        self.service.boot()
        self.service.push_voice_transcript("小灯")
        decision = self.service.push_voice_transcript("停止")
        self.assertIsNotNone(decision)
        self.assertEqual(RouteKind.LOCAL_INTENT, decision.routing.route)
        self.assertEqual(SessionState.IDLE_OBSERVING, self.service.session_state)
        self.assertEqual(1, self.cancel_count)
        self.service.runtime.safety.begin_homing()

    def test_explicit_estop_is_latched_and_cancels_soft_effects(self):
        self.service.boot()
        self.service.push_voice_transcript("小灯")
        decision = self.service.push_voice_transcript("急停")
        self.assertIsNotNone(decision)
        self.assertEqual(RouteKind.EMERGENCY, decision.routing.route)
        self.assertEqual(SessionState.IDLE_OBSERVING, self.service.session_state)
        self.assertEqual(SafetyState.SAFE_DISABLED, self.service.runtime.safety.state)
        self.assertEqual(1, self.cancel_count)
        with self.assertRaises(SafetyError):
            self.service.runtime.safety.begin_homing()
        self.service.runtime.safety.reset_estop()
        self.service.runtime.safety.begin_homing()

    def test_teach_stop_is_reachable_but_ignored_until_teaching_is_wired(self):
        self.service.boot()
        decision = self.service.push_voice_transcript("停止录制")
        self.assertIsNone(decision)
        self.assertEqual(SessionState.IDLE_OBSERVING, self.service.session_state)

    def test_unknown_voice_is_fail_closed(self):
        self.service.boot()
        decision = self.service.push_voice_transcript("今天天气真好")
        self.assertIsNone(decision)
        self.assertEqual(SessionState.IDLE_OBSERVING, self.service.session_state)

    def test_runtime_stages_are_mirrored_as_structured_logs(self):
        self.service.boot()
        decision = self.service.push_voice_transcript("小灯")
        records = [json.loads(line) for line in self.logs]
        stages = [record["stage"] for record in records]
        self.assertIn("input_received", stages)
        self.assertIn("event_routed", stages)
        self.assertIn("awake_session_started", stages)
        self.assertTrue(all("trace_id" in record for record in records))
        self.assertTrue(any(record["trace_id"] == decision.trace_id for record in records))

    def test_sink_failure_does_not_block_estop(self):
        service = LampService(sink=lambda _: (_ for _ in ()).throw(OSError("disk full")))
        service.boot()
        decision = service.push_voice_transcript("急停")
        self.assertEqual(RouteKind.EMERGENCY, decision.routing.route)
        self.assertEqual(SafetyState.SAFE_DISABLED, service.runtime.safety.state)
        with self.assertRaises(SafetyError):
            service.runtime.safety.begin_homing()


if __name__ == "__main__":
    unittest.main()
