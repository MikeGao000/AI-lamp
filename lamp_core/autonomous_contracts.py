"""Executable contracts derived from the approved Autonomous lamp architecture.

These contracts are intentionally hardware-neutral. They make the architecture
testable before a real MKS CAN driver, cloud provider, or camera is connected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ContractError(ValueError):
    """Raised when an architecture contract is malformed or unsafe."""


class Language(str, Enum):
    ZH_CN = "zh-CN"
    EN = "en"
    DA_DK = "da-DK"

    @classmethod
    def coerce(cls, value: str) -> "Language":
        aliases = {
            "zh": cls.ZH_CN,
            "zh-cn": cls.ZH_CN,
            "cmn": cls.ZH_CN,
            "en": cls.EN,
            "en-us": cls.EN,
            "en-gb": cls.EN,
            "da": cls.DA_DK,
            "da-dk": cls.DA_DK,
        }
        try:
            return aliases[value.strip().lower()]
        except KeyError as error:
            raise ContractError(f"unsupported language: {value}") from error


class PrivacyMode(str, Enum):
    LOCAL_ONLY = "local_only"
    CLOUD_ALLOWED = "cloud_allowed"


class EventPriority(str, Enum):
    EMERGENCY = "emergency"
    COMMAND = "command"
    INTERACTIVE = "interactive"
    AMBIENT = "ambient"


class EventSource(str, Enum):
    VOICE = "voice"
    CAMERA = "camera"
    UI = "ui"
    MOTOR = "motor"
    LIMIT = "limit"
    SYSTEM = "system"


class EventKind(str, Enum):
    VOICE_FINAL = "voice_final"
    VISUAL_QUERY = "visual_query"
    BOOK_STABLE = "book_stable"
    OBJECT_STABLE = "object_stable"
    STOP = "stop"
    MUTE = "mute"
    RETURN_IDLE = "return_idle"
    STATUS_QUERY = "status_query"
    ESTOP = "estop"
    LIMIT_FAULT = "limit_fault"
    MOTOR_FAULT = "motor_fault"
    CAN_FAULT = "can_fault"


class Direction(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    UNKNOWN = "unknown"


class MotorMode(str, Enum):
    DISABLED = "disabled"
    HOLDING = "holding"
    MOVING = "moving"
    FAULT = "fault"


@dataclass(frozen=True)
class LampProfile:
    """The local device declaration used to fail closed during startup."""

    device_id: str
    board: str
    joint_names: tuple[str, ...]
    required_capabilities: frozenset[str]
    optional_capabilities: frozenset[str] = frozenset()
    safety_mode: str = "fail_closed"

    def validate(self) -> None:
        if not self.device_id.strip():
            raise ContractError("device_id must not be empty")
        if not self.board.strip():
            raise ContractError("board must not be empty")
        if len(self.joint_names) != 5:
            raise ContractError("this lamp profile must declare exactly five joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ContractError("joint_names must be unique")
        if any(not name.strip() for name in self.joint_names):
            raise ContractError("joint_names must not contain empty names")
        required = {"motion", "audio", "vision", "system"}
        missing = required.difference(self.required_capabilities)
        if missing:
            raise ContractError(f"missing required capabilities: {sorted(missing)}")
        overlap = self.required_capabilities.intersection(self.optional_capabilities)
        if overlap:
            raise ContractError(f"capabilities cannot be both required and optional: {sorted(overlap)}")
        if self.safety_mode != "fail_closed":
            raise ContractError("motion-capable lamp must use fail_closed safety mode")


@dataclass(frozen=True)
class LampEvent:
    """A sensor, user, or fault event. It is never a motor command."""

    event_id: str
    trace_id: str
    timestamp_ms: int
    source: EventSource
    kind: EventKind
    priority: EventPriority
    privacy_mode: PrivacyMode
    language_hint: Language | None = None
    payload: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.event_id.strip() or not self.trace_id.strip():
            raise ContractError("event_id and trace_id must not be empty")
        if self.timestamp_ms < 0:
            raise ContractError("timestamp_ms must be non-negative")
        if not isinstance(self.payload, Mapping):
            raise ContractError("event payload must be a mapping")


@dataclass(frozen=True)
class VisualObject:
    """A grounded, normalized observation. It cannot mark itself safe to move toward."""

    object_id: str
    label: str
    bbox_norm: tuple[float, float, float, float]
    confidence: float

    def validate(self) -> None:
        if not self.object_id.strip() or not self.label.strip():
            raise ContractError("visual object id and label must not be empty")
        if len(self.bbox_norm) != 4:
            raise ContractError("bbox_norm must contain x1, y1, x2, y2")
        x1, y1, x2, y2 = self.bbox_norm
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ContractError("bbox_norm must be normalized and have positive area")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("visual confidence must be normalized to [0, 1]")


@dataclass(frozen=True)
class VisualObservation:
    """A fact report from local vision, cloud vision, or a LAN GPU service."""

    source: str
    frame_age_ms: int
    direction: Direction
    objects: tuple[VisualObject, ...] = ()

    def validate(self) -> None:
        if not self.source.strip():
            raise ContractError("visual source must not be empty")
        if self.frame_age_ms < 0:
            raise ContractError("frame_age_ms must be non-negative")
        seen_ids: set[str] = set()
        for item in self.objects:
            item.validate()
            if item.object_id in seen_ids:
                raise ContractError("visual observation cannot contain duplicate object ids")
            seen_ids.add(item.object_id)


@dataclass(frozen=True)
class ConversationSession:
    """Language and service choices are locked per session, not per utterance."""

    session_id: str
    language: Language
    stt_provider: str
    chat_provider: str
    tts_provider: str
    voice_id: str
    privacy_mode: PrivacyMode

    def validate(self) -> None:
        values = {
            "session_id": self.session_id,
            "stt_provider": self.stt_provider,
            "chat_provider": self.chat_provider,
            "tts_provider": self.tts_provider,
            "voice_id": self.voice_id,
        }
        empty = [name for name, value in values.items() if not value.strip()]
        if empty:
            raise ContractError(f"session fields must not be empty: {empty}")


@dataclass(frozen=True)
class MotorTelemetry:
    """Normalized per-axis feedback without inventing unavailable MKS fields."""

    joint: str
    node_id: int
    mode: MotorMode
    target_position_rad: float | None
    actual_position_rad: float | None
    velocity_rad_s: float | None
    following_error_rad: float | None
    current_or_torque: float | None
    temperature_c: float | None
    bus_voltage_v: float | None
    fault_code: str | None
    last_heartbeat_age_ms: int

    def validate(self) -> None:
        if not self.joint.strip():
            raise ContractError("telemetry joint must not be empty")
        if self.node_id <= 0:
            raise ContractError("telemetry node_id must be positive")
        if self.last_heartbeat_age_ms < 0:
            raise ContractError("heartbeat age must be non-negative")
        if self.following_error_rad is not None and self.following_error_rad < 0:
            raise ContractError("following_error_rad must be absolute and non-negative")
