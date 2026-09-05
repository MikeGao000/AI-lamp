"""Pure MKS SERVO42D/57D CAN V1.0.9 frame codec; no CAN I/O.

Reference: ``MKS SERVO42&57D 闭环步进电机_CAN使用说明 v1.0.9.pdf``:
chapter 4, 5.1.2/5.1.3, 9.2, 11.4 and 12.3.  This module deliberately
does not choose the motor's configurable CRC mode or open a CAN device.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChecksumMode(str, Enum):
    ADDITIVE = "additive"
    FIXED_6B = "fixed_6b"


@dataclass(frozen=True)
class CanFrame:
    """A standard CAN arbitration ID and at most eight data bytes."""

    arbitration_id: int
    data: bytes

    def __post_init__(self) -> None:
        if not 0 <= self.arbitration_id <= 0x7FF:
            raise ValueError("MKS CAN address must be a standard 11-bit ID")
        if not 1 <= len(self.data) <= 8:
            raise ValueError("MKS CAN payload must contain 1..8 bytes")


def _checksum(address: int, body: bytes, mode: ChecksumMode) -> int:
    return 0x6B if mode is ChecksumMode.FIXED_6B else (address + sum(body)) & 0xFF


def _frame(address: int, body: bytes, mode: ChecksumMode) -> CanFrame:
    return CanFrame(address, body + bytes([_checksum(address, body, mode)]))


def _signed_from(data: bytes) -> int:
    return int.from_bytes(data, byteorder="big", signed=True)


def validate_checksum(frame: CanFrame, mode: ChecksumMode) -> bool:
    """Validate an MKS payload CRC without accepting a mismatched mode."""
    return len(frame.data) >= 2 and frame.data[-1] == _checksum(
        frame.arbitration_id, frame.data[:-1], mode
    )


def read_accumulated_encoder(address: int, mode: ChecksumMode) -> CanFrame:
    """Command 0x31: request signed int48 coordinate (16384 counts/revolution)."""
    return _frame(address, b"\x31", mode)


def read_motor_rpm(address: int, mode: ChecksumMode) -> CanFrame:
    """Command 0x32: request signed int16 motor RPM."""
    return _frame(address, b"\x32", mode)


def set_bus_enabled(address: int, enabled: bool, mode: ChecksumMode) -> CanFrame:
    """Command 0xF3, valid only after the motor is in a serial bus mode."""
    return _frame(address, bytes((0xF3, int(enabled))), mode)


def set_synchronised_start(address: int, enabled: bool, mode: ChecksumMode) -> CanFrame:
    """Command 0x4A: cache a move until a broadcast 0x4B is received."""
    return _frame(address, bytes((0x4A, int(enabled))), mode)


def trigger_synchronised_start(mode: ChecksumMode) -> CanFrame:
    """Broadcast command 0x4B. MKS documents no response for this frame."""
    return _frame(0, b"\x4B", mode)


def absolute_coordinate_move(
    address: int,
    speed_rpm: int,
    acceleration: int,
    coordinate: int,
    mode: ChecksumMode,
) -> CanFrame:
    """Command 0xF5: absolute signed-int24 encoder coordinate movement.

    The firmware supports updating this command while the prior 0xF5 motion is
    active. ``coordinate`` is an encoder coordinate, *not* a lamp joint angle.
    Joint conversion belongs to the calibrated MKS driver layer.
    """
    if not 0 <= speed_rpm <= 3000:
        raise ValueError("MKS speed must be in 0..3000 RPM")
    if not 0 <= acceleration <= 255:
        raise ValueError("MKS acceleration must be in 0..255")
    if not -(2**23) <= coordinate <= 2**23 - 1:
        raise ValueError("MKS absolute coordinate must fit signed int24")
    body = bytes((0xF5,)) + speed_rpm.to_bytes(2, "big") + bytes((acceleration,))
    body += coordinate.to_bytes(3, "big", signed=True)
    return _frame(address, body, mode)


def parse_accumulated_encoder(frame: CanFrame, mode: ChecksumMode) -> int:
    """Parse a response to 0x31 and return its signed int48 coordinate."""
    if len(frame.data) != 8 or frame.data[0] != 0x31 or not validate_checksum(frame, mode):
        raise ValueError("invalid MKS 0x31 accumulated-encoder response")
    return _signed_from(frame.data[1:7])


def parse_motor_rpm(frame: CanFrame, mode: ChecksumMode) -> int:
    """Parse a response to 0x32 and return signed RPM."""
    if len(frame.data) != 4 or frame.data[0] != 0x32 or not validate_checksum(frame, mode):
        raise ValueError("invalid MKS 0x32 motor-RPM response")
    return _signed_from(frame.data[1:3])
