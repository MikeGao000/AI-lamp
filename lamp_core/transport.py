from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Transport(Protocol):
    def write(self, payload: bytes) -> None: ...

    def read(self, size: int) -> bytes: ...


@dataclass
class MemoryTransport:
    """Test-only transport. A production adapter must enforce RS-485 direction and timeout."""

    writes: list[bytes] = field(default_factory=list)
    incoming: bytearray = field(default_factory=bytearray)

    def write(self, payload: bytes) -> None:
        self.writes.append(bytes(payload))

    def read(self, size: int) -> bytes:
        result = bytes(self.incoming[:size])
        del self.incoming[:size]
        return result

