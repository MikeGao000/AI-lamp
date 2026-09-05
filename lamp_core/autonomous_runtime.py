"""Pure integration point for the new Autonomous architecture.

This runtime is deliberately not a daemon and it never performs I/O. It joins
the new event/model contracts to the existing behavior allow-list and safety
state machine, so the integration can be tested before hardware deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from lamp_core.autonomous_contracts import LampEvent, LampProfile
from lamp_core.behavior import ApprovedBehavior, BehaviorProposal, ScreenTarget, approve_behavior
from lamp_core.event_router import EventRouter, RouteKind, RoutingDecision
from lamp_core.flow_log import FlowLog
from lamp_core.model_router import ModelRoute, ModelRouter, RoutingRequest
from lamp_core.safety import SafetyController, SafetyState


@dataclass(frozen=True)
class RuntimeDecision:
    trace_id: str
    routing: RoutingDecision
    model_route: ModelRoute | None


class AutonomousRuntime:
    """Safe composition root for event routing, model selection, and behaviors."""

    def __init__(
        self,
        profile: LampProfile,
        safety: SafetyController,
        event_router: EventRouter,
        model_router: ModelRouter,
        flow_log: FlowLog | None = None,
    ) -> None:
        profile.validate()
        self.profile = profile
        self.safety = safety
        self.event_router = event_router
        self.model_router = model_router
        self.flow_log = flow_log or FlowLog()

    def process_event(
        self,
        event: LampEvent,
        now_ms: int,
        model_request: RoutingRequest | None = None,
    ) -> RuntimeDecision:
        """Route an event without making a network call or commanding a motor."""

        self.flow_log.append(event.trace_id, now_ms, "input_received", event_kind=event.kind.value)
        routing = self.event_router.route(event, now_ms)
        self.flow_log.append(event.trace_id, now_ms, "event_routed", route=routing.route.value)
        if routing.route is RouteKind.EMERGENCY:
            self.safety.emergency_stop(f"event: {event.kind.value}")
            self.flow_log.append(event.trace_id, now_ms, "emergency_stopped")
            return RuntimeDecision(event.trace_id, routing, None)

        if routing.route is not RouteKind.MODEL:
            if routing.route is RouteKind.LOCAL_INTENT:
                self.flow_log.append(event.trace_id, now_ms, "local_intent_matched")
            return RuntimeDecision(event.trace_id, routing, None)

        if model_request is None:
            raise ValueError("a model-routed event requires a routing request")
        if event.language_hint is not None and model_request.language is not event.language_hint:
            raise ValueError("event language_hint and model request language must match")
        model_route = self.model_router.route(model_request)
        stage = "model_selected" if model_route.endpoint is not None else "model_unavailable"
        self.flow_log.append(event.trace_id, now_ms, stage, status=model_route.status.value)
        return RuntimeDecision(event.trace_id, routing, model_route)

    def approve_model_behavior(
        self,
        proposal: BehaviorProposal,
        visible_objects: Mapping[str, ScreenTarget],
        *,
        trace_id: str = "standalone-behavior",
        now_ms: int = 0,
    ) -> ApprovedBehavior:
        """Apply the existing behavior allow-list only in a safe ready state."""

        safety_ready = self.safety.state is SafetyState.READY_HOLD and self.safety.drives_enabled
        approved = approve_behavior(proposal, visible_objects, safety_ready=safety_ready)
        self.flow_log.append(trace_id, now_ms, "behavior_approved", action_id=approved.action_id)
        if approved.action_id == "STOP":
            self.safety.emergency_stop("model behavior stop")
            self.flow_log.append(trace_id, now_ms, "emergency_stopped")
        return approved
