import unittest

from lamp_core.autonomous_contracts import (
    EventKind,
    EventPriority,
    EventSource,
    LampEvent,
    PrivacyMode,
)
from lamp_core.event_router import EventRouter, RouteKind


def event(
    kind: EventKind,
    *,
    source: EventSource = EventSource.CAMERA,
    priority: EventPriority = EventPriority.INTERACTIVE,
    privacy_mode: PrivacyMode = PrivacyMode.CLOUD_ALLOWED,
    event_id: str | None = None,
) -> LampEvent:
    return LampEvent(
        event_id=event_id or f"e-{kind.value}",
        trace_id="trace-1",
        timestamp_ms=0,
        source=source,
        kind=kind,
        priority=priority,
        privacy_mode=privacy_mode,
    )


class EventRouterTests(unittest.TestCase):
    def test_stop_is_always_an_emergency_and_never_waits_for_cooldown(self):
        router = EventRouter()
        decision = router.route(event(EventKind.STOP, source=EventSource.VOICE), now_ms=1)
        self.assertEqual(decision.route, RouteKind.EMERGENCY)

    def test_ambient_object_events_are_cooled_down(self):
        router = EventRouter()
        first = router.route(event(EventKind.OBJECT_STABLE, priority=EventPriority.AMBIENT, event_id="object-1"), now_ms=100)
        second = router.route(event(EventKind.OBJECT_STABLE, priority=EventPriority.AMBIENT, event_id="object-2"), now_ms=101)
        after_cooldown = router.route(event(EventKind.OBJECT_STABLE, priority=EventPriority.AMBIENT, event_id="object-3"), now_ms=20_100)
        self.assertEqual(first.route, RouteKind.MODEL)
        self.assertEqual(second.route, RouteKind.DROPPED_COOLDOWN)
        self.assertEqual(second.cooldown_remaining_ms, 19_999)
        self.assertEqual(after_cooldown.route, RouteKind.MODEL)

    def test_duplicate_event_id_is_dropped_before_model_routing(self):
        router = EventRouter()
        repeated = event(EventKind.VOICE_FINAL, source=EventSource.VOICE, event_id="voice-1")
        self.assertEqual(RouteKind.MODEL, router.route(repeated, now_ms=1).route)
        self.assertEqual(RouteKind.DROPPED_DUPLICATE, router.route(repeated, now_ms=2).route)

    def test_local_privacy_keeps_visual_request_off_the_cloud_route(self):
        router = EventRouter()
        decision = router.route(
            event(EventKind.VISUAL_QUERY, privacy_mode=PrivacyMode.LOCAL_ONLY),
            now_ms=100,
        )
        self.assertEqual(decision.route, RouteKind.LOCAL_ONLY)

    def test_status_query_is_deterministic_local_intent(self):
        router = EventRouter()
        decision = router.route(
            event(EventKind.STATUS_QUERY, source=EventSource.VOICE, priority=EventPriority.COMMAND),
            now_ms=100,
        )
        self.assertEqual(decision.route, RouteKind.LOCAL_INTENT)
