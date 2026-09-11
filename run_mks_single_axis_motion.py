"""Run one bounded physical MKS CAN motion test on Linux SocketCAN.

By default this is a dry run: it prints the exact MKS V1.0.9 frames and
never opens ``can0``.  Add ``--execute`` only on the Raspberry Pi after the
motor, power and CAN connection have been physically prepared.

Example (first J1 test):
    python3 run_mks_single_axis_motion.py --node-id 1 --execute
"""

from __future__ import annotations

import argparse
import select
import socket
import struct
from dataclasses import dataclass

from lamp_core.mks_can_protocol import CanFrame, ChecksumMode, set_bus_enabled
from lamp_core.mks_single_axis import (
    COUNTS_PER_REVOLUTION,
    MAX_INITIAL_DELTA_COUNTS,
    MAX_INITIAL_SPEED_RPM,
    CanTransport,
    MksSingleAxisProbe,
)


@dataclass
class SocketCanTransport(CanTransport):
    interface: str

    def __post_init__(self) -> None:
        if not hasattr(socket, "AF_CAN"):
            raise RuntimeError("SocketCAN is available only on the Raspberry Pi/Linux host")
        self._socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self._socket.bind((self.interface,))

    def close(self) -> None:
        self._socket.close()

    def send(self, frame: CanFrame) -> None:
        payload = frame.data.ljust(8, b"\x00")
        self._socket.send(struct.pack("=IB3x8s", frame.arbitration_id, len(frame.data), payload))

    def receive(self, timeout_s: float) -> CanFrame | None:
        readable, _, _ = select.select((self._socket,), (), (), timeout_s)
        if not readable:
            return None
        raw = self._socket.recv(16)
        can_id, dlc, payload = struct.unpack("=IB3x8s", raw)
        return CanFrame(can_id & 0x7FF, payload[:dlc])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--interface", default="can0")
    result.add_argument("--node-id", type=int, default=1)
    result.add_argument("--checksum", choices=[mode.value for mode in ChecksumMode], default=ChecksumMode.ADDITIVE.value)
    result.add_argument("--delta-counts", type=int, default=MAX_INITIAL_DELTA_COUNTS,
                        help="relative encoder step; initial test allows ±4096 (one quarter motor revolution)")
    result.add_argument("--speed-rpm", type=int, default=MAX_INITIAL_SPEED_RPM)
    result.add_argument("--acceleration", type=int, default=1)
    result.add_argument("--timeout-s", type=float, default=15.0)
    result.add_argument("--execute", action="store_true", help="open SocketCAN and send the two physical commands")
    return result


def main() -> int:
    args = parser().parse_args()
    mode = ChecksumMode(args.checksum)
    if not 1 <= abs(args.delta_counts) <= MAX_INITIAL_DELTA_COUNTS:
        parser().error("--delta-counts must be within ±4096 for the initial physical test")
    if not 1 <= args.speed_rpm <= MAX_INITIAL_SPEED_RPM:
        parser().error("--speed-rpm must be within 1..10 for the initial physical test")

    print("MKS single-axis initial motion: node", args.node_id)
    print(f"bounded relative step: {args.delta_counts} counts ({args.delta_counts / COUNTS_PER_REVOLUTION * 360:.1f}° motor-axis equivalent)")
    print(f"speed: {args.speed_rpm} RPM; acceleration: {args.acceleration}; checksum: {mode.value}")
    print("enable frame:", set_bus_enabled(args.node_id, True, mode))
    print("motion frame is calculated from the encoder position read immediately before execution.")
    if not args.execute:
        print("DRY RUN ONLY. Re-run with --execute on the Raspberry Pi to use can0.")
        return 0

    transport = SocketCanTransport(args.interface)
    try:
        probe = MksSingleAxisProbe(transport, args.node_id, mode)
        result = probe.move_relative_for_initial_test(
            delta_counts=args.delta_counts,
            speed_rpm=args.speed_rpm,
            acceleration=args.acceleration,
            timeout_s=args.timeout_s,
        )
    finally:
        transport.close()
    print("before:", result.before)
    print("target encoder:", result.target_counts)
    print("after:", result.after)
    print("position error:", result.position_error_counts, "counts")
    if not result.reached_target:
        print("FAIL: target was not reached before timeout.")
        return 2
    print("PASS: encoder settled within the MKS feedback tolerance of the commanded target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
