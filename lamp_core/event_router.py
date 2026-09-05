"""Deterministic local event routing before any model or motor action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from lamp_core.autonomous_contracts import (
    EventKind,
    EventPriority,
    EventSource,
    LampEvent,
    PrivacyMode,
)


class RouteKind(str, Enum):
    EMERGENCY = "emergency"
    LOCAL_INTENT = "local_intent"
    MODEL = "model"
    LOCAL_ONLY = "local_only"
    DROPPED_COOLDOWN = "dropped_cooldown"
    DROPPED_DUPLICATE = "dropped_duplicate"


@dataclass(frozen=True)
class RoutingDecision:
    route: RouteKind
    reason: str
    cooldown_remaining_ms: int = 0


class EventRouter:
    """Routes events without executing hardware or calling a model.

    Emergency and explicit local commands are never delayed behind an ambient
    event. Cloud-sensitive events remain local when privacy disallows upload.
    """

    DEFAULT_COOLDOWNS_MS: Mapping[EventKind, int] = {
        EventKind.BOOK_STABLE: 10_000,
        EventKind.OBJECT_STABLE: 20_000,
    }
    EMERGENCY_KINDS = frozenset(
        {
            EventKind.STOP,
            EventKind.ESTOP,
            EventKind.LIMIT_FAULT,
            EventKind.MOTOR_FAULT,
            EventKind.CAN_FAULT,
        }
    )
    LOCAL_KINDS = frozenset(
        {
            EventKind.MUTE,
            EventKind.RETURN_IDLE,
            EventKind.STATUS_QUERY,
        }
    )
    MEDIA_TO_MODEL_KINDS = frozenset(
        {
            EventKind.VOICE_FINAL,
            EventKind.VISUAL_QUERY,
            EventKind.BOOK_STABLE,
            EventKind.OBJECT_STABLE,
        }
    )

    def __init__(
        self,
        cooldowns_ms: Mapping[EventKind, int] | None = None,
        *,
        dedup_window_ms: int = 60_000,
        dedup_capacity: int = 4_096,
    ) -> None:
        if dedup_window_ms <= 0 or dedup_capacity <= 0:
            raise ValueError("dedup window and capacity must be positive")
        self._cooldowns_ms = dict(self.DEFAULT_COOLDOWNS_MS)
        if cooldowns_ms:
            for kind, duration_ms in cooldowns_ms.items():
                if duration_ms < 0:
                    raise ValueError("cooldown duration must be non-negative")
                self._cooldowns_ms[kind] = duration_ms
        self._last_accepted_ms: dict[tuple[EventSource, EventKind], int] = {}
        self._dedup_window_ms = dedup_window_ms
        self._dedup_capacity = dedup_capacity
        self._seen_event_ids: dict[str, int] = {}

    def _remember_event_id(self, event_id: str, now_ms: int) -> bool:
        """Return true when the ID was recently seen; keep memory bounded."""

        expired = [
            seen_id
            for seen_id, accepted_ms in self._seen_event_ids.items()
            if now_ms - accepted_ms >= self._dedup_window_ms
        ]
        for seen_id in expired:
            del self._seen_event_ids[seen_id]
        if event_id in self._seen_event_ids:
            return True
        if len(self._seen_event_ids) >= self._dedup_capacity:
            oldest = min(self._seen_event_ids, key=self._seen_event_ids.__getitem__)
            del self._seen_event_ids[oldest]
        self._seen_event_ids[event_id] = now_ms
        return False

    def route(self, event: LampEvent, now_ms: int) -> RoutingDecision:
        event.validate()
        if now_ms < 0:
            raise ValueError("now_ms must be non-negative")

        if event.priority is EventPriority.EMERGENCY or event.kind in self.EMERGENCY_KINDS:
            return RoutingDecision(RouteKind.EMERGENCY, "emergency events bypass cooldown and models")

        if self._remember_event_id(event.event_id, now_ms):
            return RoutingDecision(RouteKind.DROPPED_DUPLICATE, "event_id was already handled recently")

        if event.kind in self.LOCAL_KINDS:
            return RoutingDecision(RouteKind.LOCAL_INTENT, "deterministic local command")

        key = (event.source, event.kind)
        cooldown_ms = self._cooldowns_ms.get(event.kind, 0)
        previous_ms = self._last_accepted_ms.get(key)
        if previous_ms is not None and now_ms - previous_ms < cooldown_ms:
            remaining_ms = cooldown_ms - (now_ms - previous_ms)
            return RoutingDecision(RouteKind.DROPPED_COOLDOWN, "same event is cooling down", remaining_ms)

        self._last_accepted_ms[key] = now_ms
        if event.privacy_mode is PrivacyMode.LOCAL_ONLY and event.kind in self.MEDIA_TO_MODEL_KINDS:
            return RoutingDecision(RouteKind.LOCAL_ONLY, "privacy mode forbids model upload")

        return RoutingDecision(RouteKind.MODEL, "eligible for the configured model route")
