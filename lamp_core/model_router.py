"""Provider-neutral routing for language, vision, and audio requests.

The router selects a configured endpoint only. It does not hold keys, make
network calls, or grant a model hardware permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from lamp_core.autonomous_contracts import Language, PrivacyMode


class ModelTask(str, Enum):
    LOCAL_COMMAND = "local_command"
    CHAT = "chat"
    READING_VISION = "reading_vision"
    OBJECT_GROUNDING = "object_grounding"
    STT = "stt"
    TTS = "tts"
    REALTIME_VOICE = "realtime_voice"


class RouteStatus(str, Enum):
    MODEL = "model"
    LOCAL = "local"
    UNAVAILABLE = "unavailable"


class NetworkHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass(frozen=True)
class SessionBudget:
    """A provider-neutral per-session model budget; None means unrestricted."""

    requests_remaining: int | None = None
    tokens_remaining: int | None = None

    def validate(self) -> None:
        for name, value in (("requests_remaining", self.requests_remaining), ("tokens_remaining", self.tokens_remaining)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when supplied")

    @property
    def exhausted(self) -> bool:
        return self.requests_remaining == 0 or self.tokens_remaining == 0


@dataclass(frozen=True)
class ModelEndpoint:
    provider: str
    model: str
    task: ModelTask
    language: Language | None
    is_local: bool
    supports_audio: bool = False
    supports_vision: bool = False
    estimated_latency_ms: int = 0
    priority: int = 100

    def validate(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider and model must not be empty")
        if self.estimated_latency_ms < 0:
            raise ValueError("estimated_latency_ms must be non-negative")


@dataclass(frozen=True)
class RoutingRequest:
    task: ModelTask
    language: Language
    privacy_mode: PrivacyMode
    requires_audio: bool = False
    requires_vision: bool = False
    max_latency_ms: int | None = None
    session_budget: SessionBudget = SessionBudget()
    network_health: NetworkHealth = NetworkHealth.HEALTHY

    def validate(self) -> None:
        if self.max_latency_ms is not None and self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive when supplied")
        self.session_budget.validate()


@dataclass(frozen=True)
class ModelRoute:
    status: RouteStatus
    endpoint: ModelEndpoint | None
    reason: str
    fallbacks: tuple[ModelEndpoint, ...] = ()


class ModelRouter:
    """Selects an endpoint by task, language, modality, privacy, and latency."""

    def __init__(self, endpoints: Iterable[ModelEndpoint]) -> None:
        self._endpoints = tuple(endpoints)
        for endpoint in self._endpoints:
            endpoint.validate()

    def route(self, request: RoutingRequest) -> ModelRoute:
        request.validate()
        if request.task is ModelTask.LOCAL_COMMAND:
            return ModelRoute(RouteStatus.LOCAL, None, "local commands never use a model")

        candidates = [
            endpoint
            for endpoint in self._endpoints
            if endpoint.task is request.task
            and (endpoint.language is None or endpoint.language is request.language)
            and (not request.requires_audio or endpoint.supports_audio)
            and (not request.requires_vision or endpoint.supports_vision)
            and (request.privacy_mode is not PrivacyMode.LOCAL_ONLY or endpoint.is_local)
        ]

        if request.max_latency_ms is not None:
            candidates = [
                endpoint
                for endpoint in candidates
                if endpoint.estimated_latency_ms <= request.max_latency_ms
            ]

        if (
            request.privacy_mode is PrivacyMode.LOCAL_ONLY
            or request.network_health is NetworkHealth.OFFLINE
            or request.session_budget.exhausted
        ):
            candidates = [endpoint for endpoint in candidates if endpoint.is_local]

        if not candidates:
            return ModelRoute(
                RouteStatus.UNAVAILABLE,
                None,
                "no endpoint satisfies task, language, modality, privacy, and latency constraints",
            )

        def rank(endpoint: ModelEndpoint) -> tuple[int, int, int, int]:
            generic_language_penalty = 0 if endpoint.language is request.language else 1
            local_penalty = 0 if request.network_health is NetworkHealth.DEGRADED and endpoint.is_local else 1
            return local_penalty, generic_language_penalty, endpoint.priority, endpoint.estimated_latency_ms

        ranked = tuple(sorted(candidates, key=rank))
        selected = ranked[0]
        return ModelRoute(RouteStatus.MODEL, selected, "selected configured endpoint", ranked[1:])
