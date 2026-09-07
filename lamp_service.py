"""Fail-closed composition root for the future always-on lamp service.

The interaction session and physical safety controller are deliberately
independent. Booting this service enables observation only; it never homes or
enables a motor. Hardware adapters can be added behind the same event boundary
after their independent safety validation is complete.
"""

from __future__ import annotations

import json
from enum import Enum
from itertools import count
from time import monotonic
from typing import Callable, Mapping

from lamp_core.autonomous_contracts import (
    EventKind,
    EventPriority,
    EventSource,
    LampEvent,
    LampProfile,
    Language,
    PrivacyMode,
)
from lamp_core.autonomous_runtime import AutonomousRuntime, RuntimeDecision
from lamp_core.event_router import EventRouter, RouteKind
from lamp_core.flow_log import FlowLog
from lamp_core.model_router import ModelRouter, ModelTask, RoutingRequest
from lamp_core.safety import SafetyController
from lamp_core.voice import VoiceIntent, parse_intent


class SessionState(str, Enum):
    BOOTING = "booting"
    IDLE_OBSERVING = "idle_observing"
    AWAKE_SESSION = "awake_session"


class JournalFlowLog(FlowLog):
    """Keep the deterministic memory log and mirror entries to a service sink."""

    def __init__(self, sink: Callable[[str], None], capacity: int = 4_096) -> None:
        super().__init__(capacity)
        self._sink = sink

    def append(self, trace_id: str, timestamp_ms: int, stage: str, **details: str) -> None:
        super().append(trace_id, timestamp_ms, stage, **details)
        record = {
            "trace_id": trace_id,
            "timestamp_ms": timestamp_ms,
            "stage": stage,
            "details": details,
        }
        try:
            self._sink(json.dumps(record, ensure_ascii=False, sort_keys=True))
        except Exception:
            # Logging must never prevent an emergency event from reaching the
            # safety controller. A production sink should report its own health.
            pass


class LampService:
    """Synchronous, I/O-free service core suitable for deterministic tests."""

    _EMERGENCY_KINDS = frozenset(
        {
            EventKind.STOP,
            EventKind.ESTOP,
            EventKind.LIMIT_FAULT,
            EventKind.MOTOR_FAULT,
            EventKind.CAN_FAULT,
        }
    )
    _VOICE_EVENTS = {
        VoiceIntent.WAKE: EventKind.WAKE,
        VoiceIntent.READ: EventKind.READ,
        VoiceIntent.STOP: EventKind.SOFT_STOP,
        VoiceIntent.ESTOP: EventKind.ESTOP,
        VoiceIntent.REST: EventKind.RETURN_IDLE,
    }

    def __init__(
        self,
        *,
        runtime: AutonomousRuntime | None = None,
        clock: Callable[[], float] = monotonic,
        sink: Callable[[str], None] = print,
        awake_session_seconds: float = 60.0,
        privacy_mode: PrivacyMode = PrivacyMode.LOCAL_ONLY,
        language: Language = Language.ZH_CN,
        cancel_soft_effects: Callable[[], None] | None = None,
    ) -> None:
        if awake_session_seconds <= 0:
            raise ValueError("awake_session_seconds must be positive")
        self._clock = clock
        self._awake_session_seconds = awake_session_seconds
        self._privacy_mode = privacy_mode
        self._language = language
        self._cancel_soft_effects = cancel_soft_effects or (lambda: None)
        self._ids = count(1)
        self.session_state = SessionState.BOOTING
        self._awake_deadline: float | None = None

        if runtime is None:
            flow_log = JournalFlowLog(sink)
            runtime = AutonomousRuntime(
                LampProfile(
                    device_id="smart-lamp",
                    board="raspberry_pi_3b",
                    joint_names=("J1", "J2", "J3", "J4", "J5"),
                    required_capabilities=frozenset({"motion", "audio", "vision", "system"}),
                ),
                SafetyController(),
                EventRouter(),
                ModelRouter(()),
                flow_log,
            )
        self.runtime = runtime

    def _now_ms(self) -> int:
        return int(self._clock() * 1_000)

    def _record(self, trace_id: str, stage: str, **details: str) -> None:
        self.runtime.flow_log.append(trace_id, self._now_ms(), stage, **details)

    def boot(self) -> None:
        """Enable observation without changing physical motor safety state."""

        if self.session_state is not SessionState.BOOTING:
            return
        self.session_state = SessionState.IDLE_OBSERVING
        self._record("service", "boot_complete", session_state=self.session_state.value)

    def _default_model_request(self, kind: EventKind) -> RoutingRequest | None:
        task = {
            EventKind.VOICE_FINAL: ModelTask.CHAT,
            EventKind.VISUAL_QUERY: ModelTask.READING_VISION,
            EventKind.BOOK_STABLE: ModelTask.READING_VISION,
            EventKind.OBJECT_STABLE: ModelTask.OBJECT_GROUNDING,
        }.get(kind)
        if task is None:
            return None
        return RoutingRequest(
            task,
            self._language,
            self._privacy_mode,
            requires_vision=kind
            in {EventKind.VISUAL_QUERY, EventKind.BOOK_STABLE, EventKind.OBJECT_STABLE},
        )

    def push_event(
        self,
        kind: EventKind,
        *,
        source: EventSource,
        priority: EventPriority | None = None,
        payload: Mapping[str, object] | None = None,
        model_request: RoutingRequest | None = None,
    ) -> RuntimeDecision:
        """Create, route and locally apply one event after service boot."""

        if self.session_state is SessionState.BOOTING:
            raise RuntimeError("service must be booted before accepting events")
        sequence = next(self._ids)
        now_ms = self._now_ms()
        trace_id = f"trace-{sequence}"
        event = LampEvent(
            event_id=f"event-{sequence}",
            trace_id=trace_id,
            timestamp_ms=now_ms,
            source=source,
            kind=kind,
            priority=priority
            or (
                EventPriority.EMERGENCY
                if kind in self._EMERGENCY_KINDS
                else EventPriority.COMMAND
                if kind in EventRouter.LOCAL_KINDS
                else EventPriority.INTERACTIVE
            ),
            privacy_mode=self._privacy_mode,
            language_hint=self._language,
            payload=payload or {},
        )
        decision = self.runtime.process_event(
            event,
            now_ms,
            model_request=model_request or self._default_model_request(kind),
        )
        self._apply_local_effects(kind, decision, trace_id)
        return decision

    def _apply_local_effects(
        self, kind: EventKind, decision: RuntimeDecision, trace_id: str
    ) -> None:
        if decision.routing.route is RouteKind.EMERGENCY:
            self._cancel_effects_without_masking_safety(trace_id)
            self._enter_idle(trace_id, "emergency_session_stopped")
            return
        if decision.routing.route is not RouteKind.LOCAL_INTENT:
            return
        if kind in {EventKind.SOFT_STOP, EventKind.RETURN_IDLE}:
            self._cancel_effects_without_masking_safety(trace_id)
            self._enter_idle(trace_id, "interaction_stopped")
        elif kind in {EventKind.WAKE, EventKind.READ}:
            self.session_state = SessionState.AWAKE_SESSION
            self._awake_deadline = self._clock() + self._awake_session_seconds
            self._record(
                trace_id,
                "awake_session_started",
                trigger=kind.value,
                session_state=self.session_state.value,
            )

    def _cancel_effects_without_masking_safety(self, trace_id: str) -> None:
        try:
            self._cancel_soft_effects()
        except Exception as error:
            self._record(trace_id, "soft_effect_cancel_failed", error=type(error).__name__)

    def _enter_idle(self, trace_id: str, stage: str) -> None:
        self.session_state = SessionState.IDLE_OBSERVING
        self._awake_deadline = None
        self._record(trace_id, stage, session_state=self.session_state.value)

    def push_voice_transcript(self, transcript: str) -> RuntimeDecision | None:
        """Translate only audited local voice intents into service events."""

        command = parse_intent(transcript)
        kind = self._VOICE_EVENTS.get(command.intent)
        if kind is None:
            if self.session_state is SessionState.BOOTING:
                raise RuntimeError("service must be booted before accepting voice")
            self._record("voice", "voice_intent_ignored", intent=command.intent.value)
            return None
        return self.push_event(kind, source=EventSource.VOICE)

    def tick(self) -> None:
        """Expire the awake session when called by the future service loop."""

        if (
            self.session_state is SessionState.AWAKE_SESSION
            and self._awake_deadline is not None
            and self._clock() >= self._awake_deadline
        ):
            self._enter_idle("service", "awake_session_expired")


if __name__ == "__main__":
    service = LampService()
    service.boot()
    print("Smart-lamp service demo: enter a voice transcript, or Ctrl+C to exit.")
    try:
        while line := input("> "):
            service.push_voice_transcript(line)
            service.tick()
    except (EOFError, KeyboardInterrupt):
        pass
