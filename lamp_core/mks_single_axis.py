"""Single-axis MKS CAN motion probe.

This module deliberately has no knowledge of the five-axis lamp geometry.  It
uses the already verified MKS CAN V1.0.9 codec to make one bounded encoder
increment on one node, then reads the result back.  Joint calibration and
five-axis motion remain a later layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep
from typing import Protocol

from lamp_core.mks_can_protocol import (
    CanFrame,
    ChecksumMode,
    absolute_coordinate_move,
    parse_accumulated_encoder,
    parse_motor_rpm,
    read_accumulated_encoder,
    read_motor_rpm,
    set_bus_enabled,
)


COUNTS_PER_REVOLUTION = 16_384
MAX_INITIAL_DELTA_COUNTS = 4_096
MAX_INITIAL_SPEED_RPM = 10
# The MKS encoder and RPM reports are integral-valued.  A closed-loop axis can
# legitimately settle a few counts either side of the requested coordinate and
# report ±1 RPM while mechanically at rest.
POSITION_SETTLE_TOLERANCE_COUNTS = 8
RPM_SETTLE_TOLERANCE = 1


class CanTransport(Protocol):
    """The narrow transport contract used by the single-axis probe."""

    def send(self, frame: CanFrame) -> None: ...

    def receive(self, timeout_s: float) -> CanFrame | None: ...


@dataclass(frozen=True)
class MotorSnapshot:
    encoder_counts: int
    rpm: int


@dataclass(frozen=True)
class MotionResult:
    before: MotorSnapshot
    target_counts: int
    after: MotorSnapshot
    reached_target: bool

    @property
    def position_error_counts(self) -> int:
        return self.after.encoder_counts - self.target_counts


class MksSingleAxisProbe:
    """Query and make one deliberately small, absolute-coordinate move."""

    def __init__(self, transport: CanTransport, node_id: int, checksum_mode: ChecksumMode) -> None:
        if not 1 <= node_id <= 0x7FF:
            raise ValueError("node_id must be a standard non-zero CAN ID")
        self._transport = transport
        self._node_id = node_id
        self._checksum_mode = checksum_mode

    def _request(self, request: CanFrame, expected_command: int, timeout_s: float = 1.0) -> CanFrame:
        self._transport.send(request)
        deadline = monotonic() + timeout_s
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MKS node {self._node_id} did not answer command 0x{expected_command:02X}")
            reply = self._transport.receive(remaining)
            if reply is not None and reply.arbitration_id == self._node_id and reply.data[0] == expected_command:
                return reply

    def snapshot(self) -> MotorSnapshot:
        encoder = parse_accumulated_encoder(
            self._request(read_accumulated_encoder(self._node_id, self._checksum_mode), 0x31),
            self._checksum_mode,
        )
        rpm = parse_motor_rpm(
            self._request(read_motor_rpm(self._node_id, self._checksum_mode), 0x32),
            self._checksum_mode,
        )
        return MotorSnapshot(encoder_counts=encoder, rpm=rpm)

    def move_relative_for_initial_test(
        self,
        *,
        delta_counts: int = MAX_INITIAL_DELTA_COUNTS,
        speed_rpm: int = MAX_INITIAL_SPEED_RPM,
        acceleration: int = 1,
        timeout_s: float = 15.0,
    ) -> MotionResult:
        """Move at most one quarter motor revolution from the live position.

        This leaves motor configuration unchanged.  It does enable position
        control for the move and does not disable it afterwards, avoiding an
        unexpected loss of holding torque on an installed axis.
        """
        if not 1 <= abs(delta_counts) <= MAX_INITIAL_DELTA_COUNTS:
            raise ValueError(f"initial delta must be within ±{MAX_INITIAL_DELTA_COUNTS} encoder counts")
        if not 1 <= speed_rpm <= MAX_INITIAL_SPEED_RPM:
            raise ValueError(f"initial speed must be within 1..{MAX_INITIAL_SPEED_RPM} RPM")
        if not 0 <= acceleration <= 255:
            raise ValueError("acceleration must be in 0..255")

        before = self.snapshot()
        if before.rpm != 0:
            raise RuntimeError(
                f"node {self._node_id} is already moving at {before.rpm} RPM; "
                "stop it and confirm a stationary encoder before this initial probe"
            )
        target = before.encoder_counts + delta_counts
        if not -(2**23) <= target <= 2**23 - 1:
            raise ValueError("target coordinate exceeds MKS absolute-coordinate range")

        self._transport.send(set_bus_enabled(self._node_id, True, self._checksum_mode))
        self._transport.send(
            absolute_coordinate_move(
                self._node_id,
                speed_rpm=speed_rpm,
                acceleration=acceleration,
                coordinate=target,
                mode=self._checksum_mode,
            )
        )

        deadline = monotonic() + timeout_s
        after = before
        while monotonic() < deadline:
            sleep(0.10)
            after = self.snapshot()
            if (
                abs(after.encoder_counts - target) <= POSITION_SETTLE_TOLERANCE_COUNTS
                and abs(after.rpm) <= RPM_SETTLE_TOLERANCE
            ):
                return MotionResult(before, target, after, reached_target=True)
        return MotionResult(before, target, after, reached_target=False)
