"""L0 regression checks for the Autonomous v0.2 architecture baseline.

These tests intentionally encode requirements that were not covered by the
earlier module tests.  A failure is a concrete specification gap, not a
hardware test and not permission to access CAN/GPIO or a model provider.
"""

from dataclasses import fields
import unittest

from lamp_core.autonomous_contracts import (
    EventKind,
    EventPriority,
    EventSource,
    LampEvent,
    Language,
    PrivacyMode,
)
from lamp_core.action_catalog import DEFAULT_ACTION_CATALOG
from lamp_core.behavior import ai_behavior_tool_schema
from lamp_core.event_router import EventRouter, RouteKind
from lamp_core.model_router import RoutingRequest
from lamp_core.safety import SafetyState


class AutonomousSpecificationRegressionTests(unittest.TestCase):
    """Requirements taken from docs/台灯_Autonomous_完整架构规格.md."""

    def test_event_router_deduplicates_a_repeated_voice_event_id(self):
        event = LampEvent(
            event_id="voice-42",
            trace_id="trace-42",
            timestamp_ms=42,
            source=EventSource.VOICE,
            kind=EventKind.VOICE_FINAL,
            priority=EventPriority.INTERACTIVE,
            privacy_mode=PrivacyMode.CLOUD_ALLOWED,
            language_hint=Language.ZH_CN,
        )
        router = EventRouter()
        self.assertEqual(RouteKind.MODEL, router.route(event, now_ms=42).route)
        self.assertNotEqual(
            RouteKind.MODEL,
            router.route(event, now_ms=43).route,
            "§4 requires EventRouter de-duplication; a repeated event_id must not reach a model twice",
        )

    def test_model_router_contract_includes_budget_and_network_health(self):
        request_fields = {field.name for field in fields(RoutingRequest)}
        self.assertTrue(
            {"session_budget", "network_health"}.issubset(request_fields),
            "§7.3 requires routing decisions to consider session_budget and network_health",
        )

    def test_safety_state_names_cover_the_fail_closed_architecture_states(self):
        state_names = set(SafetyState.__members__)
        required_states = {
            "SAFE_DISABLED",
            "HOMING_REQUIRED",
            "READY_HOLD",
            "MOVING",
            "PAUSED",
            "FAULT_LATCHED",
        }
        self.assertTrue(
            required_states.issubset(state_names),
            "§10.2 requires the documented fail-closed state taxonomy",
        )

    def test_behavior_tool_schema_matches_the_enabled_audited_catalogue(self):
        actual = set(ai_behavior_tool_schema()["parameters"]["properties"]["action_id"]["enum"])
        self.assertEqual(set(DEFAULT_ACTION_CATALOG.selectable_action_ids()), actual)
        self.assertNotIn("LELAMP_TOUCH_TARGET", actual)
