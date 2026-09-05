from __future__ import annotations

import unittest

from lamp_core.mks_can_protocol import (
    CanFrame,
    ChecksumMode,
    absolute_coordinate_move,
    parse_accumulated_encoder,
    parse_motor_rpm,
    read_accumulated_encoder,
    read_motor_rpm,
    set_bus_enabled,
    set_synchronised_start,
    trigger_synchronised_start,
)


class MksCanProtocolTests(unittest.TestCase):
    """Vectors from MKS CAN V1.0.9, chapters 4, 9, 11 and 12."""

    def test_documented_additive_vectors(self) -> None:
        mode = ChecksumMode.ADDITIVE
        self.assertEqual(b"\x31\x32", read_accumulated_encoder(1, mode).data)
        self.assertEqual(b"\xF3\x01\xF5", set_bus_enabled(1, True, mode).data)
        self.assertEqual(b"\x4A\x01\x4B", set_synchronised_start(0, True, mode).data)
        self.assertEqual(b"\x4B\x4B", trigger_synchronised_start(mode).data)

    def test_documented_f5_absolute_coordinate_vector(self) -> None:
        frame = absolute_coordinate_move(1, 600, 2, 0x4000, ChecksumMode.ADDITIVE)
        self.assertEqual(1, frame.arbitration_id)
        self.assertEqual(bytes.fromhex("F5 02 58 02 00 40 00 92"), frame.data)

    def test_parses_signed_telemetry_and_rejects_bad_crc(self) -> None:
        mode = ChecksumMode.ADDITIVE
        self.assertEqual(-1, parse_accumulated_encoder(CanFrame(1, bytes.fromhex("31 FF FF FF FF FF FF 2C")), mode))
        self.assertEqual(-300, parse_motor_rpm(CanFrame(1, bytes.fromhex("32 FE D4 05")), mode))
        with self.assertRaises(ValueError):
            parse_motor_rpm(CanFrame(1, bytes.fromhex("32 FE D4 00")), mode)

    def test_requires_explicit_crc_mode_and_range(self) -> None:
        self.assertEqual(b"\x32\x6B", read_motor_rpm(7, ChecksumMode.FIXED_6B).data)
        with self.assertRaises(ValueError):
            absolute_coordinate_move(1, 3001, 2, 0, ChecksumMode.ADDITIVE)
        with self.assertRaises(ValueError):
            absolute_coordinate_move(1, 1, 2, 2**23, ChecksumMode.ADDITIVE)
