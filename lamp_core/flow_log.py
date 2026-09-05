"""Small in-memory FlowLog for deterministic Autonomous trace verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FlowLogEntry:
    trace_id: str
    timestamp_ms: int
    stage: str
    details: Mapping[str, str]


class FlowLog:
    """A bounded, I/O-free trace store; persistence belongs to a later service."""

    def __init__(self, capacity: int = 4_096) -> None:
        if capacity <= 0:
            raise ValueError("FlowLog capacity must be positive")
        self._capacity = capacity
        self._entries: list[FlowLogEntry] = []

    def append(self, trace_id: str, timestamp_ms: int, stage: str, **details: str) -> None:
        if not trace_id.strip() or timestamp_ms < 0 or not stage.strip():
            raise ValueError("FlowLog entry requires trace_id, non-negative time, and stage")
        if len(self._entries) >= self._capacity:
            del self._entries[0]
        self._entries.append(FlowLogEntry(trace_id, timestamp_ms, stage, dict(details)))

    def for_trace(self, trace_id: str) -> tuple[FlowLogEntry, ...]:
        return tuple(entry for entry in self._entries if entry.trace_id == trace_id)
