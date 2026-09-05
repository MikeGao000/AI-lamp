import unittest

from lamp_core.autonomous_contracts import (
    EventKind,
    EventPriority,
    EventSource,
    LampEvent,
    LampProfile,
    Language,
    PrivacyMode,
)
from lamp_core.autonomous_runtime import AutonomousRuntime
from lamp_core.behavior import BehaviorProposal, ScreenTarget
from lamp_core.event_router import EventRouter, RouteKind
from lamp_core.model_router import ModelEndpoint, ModelRouter, ModelTask, RouteStatus, RoutingRequest
from lamp_core.safety import SafetyController, SafetyState


def profile() -> LampProfile:
    return LampProfile(
        device_id="picture-lamp-01",
        board="raspberry_pi_3b",
        joint_names=("J1", "J2", "J3", "J4", "J5"),
        required_capabilities=frozenset({"motion", "audio", "vision", "system"}),
    )


def event(
    kind: EventKind,
    *,
    priority: EventPriority = EventPriority.INTERACTIVE,
    privacy: PrivacyMode = PrivacyMode.CLOUD_ALLOWED,
    language: Language | None = Language.DA_DK,
) -> LampEvent:
    return LampEvent(
        event_id=f"e-{kind.value}",
        trace_id="trace-1",
        timestamp_ms=0,
        source=EventSource.VOICE,
        kind=kind,
        priority=priority,
        privacy_mode=privacy,
        language_hint=language,
    )


class AutonomousRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.safety = SafetyController()
        self.safety.begin_homing()
        self.safety.complete_homing()
        self.safety.enable_drives()
        self.runtime = AutonomousRuntime(
            profile(),
            self.safety,
            EventRouter(),
            ModelRouter(
                [
                    ModelEndpoint(
                        provider="cloud",
                        model="danish-vision",
                        task=ModelTask.READING_VISION,
                        language=Language.DA_DK,
                        is_local=False,
                        supports_vision=True,
                        estimated_latency_ms=500,
                    )
                ]
            ),
        )

    def test_visual_event_reaches_only_a_matching_model_route(self):
        decision = self.runtime.process_event(
            event(EventKind.VISUAL_QUERY),
            now_ms=100,
            model_request=RoutingRequest(
                ModelTask.READING_VISION,
                Language.DA_DK,
                PrivacyMode.CLOUD_ALLOWED,
                requires_vision=True,
            ),
        )
        self.assertEqual(decision.routing.route, RouteKind.MODEL)
        self.assertEqual(decision.model_route.status, RouteStatus.MODEL)
        self.assertEqual(decision.model_route.endpoint.model, "danish-vision")

    def test_stop_latches_estop_and_never_routes_to_a_model(self):
        decision = self.runtime.process_event(event(EventKind.STOP), now_ms=100)
        self.assertEqual(decision.routing.route, RouteKind.EMERGENCY)
        self.assertIsNone(decision.model_route)
        self.assertEqual(self.safety.state, SafetyState.SAFE_DISABLED)
        self.assertFalse(self.safety.drives_enabled)

    def test_local_only_event_never_creates_a_model_route(self):
        decision = self.runtime.process_event(
            event(EventKind.VISUAL_QUERY, privacy=PrivacyMode.LOCAL_ONLY),
            now_ms=100,
        )
        self.assertEqual(decision.routing.route, RouteKind.LOCAL_ONLY)
        self.assertIsNone(decision.model_route)

    def test_behavior_reuses_existing_allow_list_after_safety_ready(self):
        approved = self.runtime.approve_model_behavior(
            BehaviorProposal(action_id="READING_POSTURE", speech="我们开始读吧。"),
            {},
        )
        self.assertEqual(approved.action_id, "READING_POSTURE")

    def test_runtime_approves_a_lelamp_catalogue_expression_without_motion_io(self):
        approved = self.runtime.approve_model_behavior(
            BehaviorProposal(action_id="LELAMP_GREET_SMALL", intensity=0.3),
            {},
            trace_id="lelamp-trace",
            now_ms=200,
        )
        self.assertEqual("LELAMP_GREET_SMALL", approved.action_id)
        self.assertEqual("lelamp_greet_small", approved.motion_key)
        self.assertEqual(
            ("behavior_approved",),
            tuple(entry.stage for entry in self.runtime.flow_log.for_trace("lelamp-trace")),
        )

    def test_behavior_is_rejected_after_stop(self):
        self.runtime.process_event(event(EventKind.STOP), now_ms=100)
        with self.assertRaises(ValueError):
            self.runtime.approve_model_behavior(
                BehaviorProposal(action_id="READING_POSTURE"),
                {},
            )

    def test_runtime_records_a_traceable_flow_log(self):
        decision = self.runtime.process_event(
            event(EventKind.VISUAL_QUERY),
            now_ms=100,
            model_request=RoutingRequest(
                ModelTask.READING_VISION,
                Language.DA_DK,
                PrivacyMode.CLOUD_ALLOWED,
                requires_vision=True,
            ),
        )
        self.assertEqual("trace-1", decision.trace_id)
        self.assertEqual(
            ("input_received", "event_routed", "model_selected"),
            tuple(entry.stage for entry in self.runtime.flow_log.for_trace("trace-1")),
        )
