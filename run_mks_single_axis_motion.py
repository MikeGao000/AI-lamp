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
    GEARED_SPEED_CURVE,
    MAX_INITIAL_DELTA_COUNTS,
    MAX_INITIAL_OUTPUT_DEGREES,
    MAX_INITIAL_SPEED_RPM,
    MAX_GEARED_TEST_SPEED_RPM,
    CanTransport,
    GENTLE_SPEED_CURVE,
    MksSingleAxisProbe,
    alternating_cycle_deltas,
    motor_counts_for_joint_degrees,
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
                        help="relative motor-encoder step; raw diagnostic mode allows ±4096")
    result.add_argument(
        "--joint-degrees",
        type=float,
        help="requested output-joint displacement (±90°); converted through --gear-ratio and overrides --delta-counts",
    )
    result.add_argument("--speed-rpm", type=int, default=MAX_INITIAL_SPEED_RPM)
    result.add_argument("--acceleration", type=int, default=1)
    result.add_argument("--timeout-s", type=float, default=15.0)
    result.add_argument(
        "--gear-ratio",
        type=float,
        default=1.0,
        help="motor revolutions per joint-output revolution; 13.7 enables the geared 60 RPM curve cap",
    )
    result.add_argument(
        "--profile",
        choices=("constant", "curve"),
        default="constant",
        help="constant uses --speed-rpm/--acceleration; curve streams live F5 speed updates",
    )
    result.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="complete forward/reverse cycles; 10 means 20 motion segments and returns near the starting position",
    )
    result.add_argument("--execute", action="store_true", help="open SocketCAN and send physical commands")
    return result


def main() -> int:
    args = parser().parse_args()
    mode = ChecksumMode(args.checksum)
    if args.gear_ratio < 1:
        parser().error("--gear-ratio must be at least 1.0")
    if args.joint_degrees is None:
        if not 1 <= abs(args.delta_counts) <= MAX_INITIAL_DELTA_COUNTS:
            parser().error("--delta-counts must be within ±4096 for raw motor-axis diagnostics")
        delta_counts = args.delta_counts
        max_delta_counts = MAX_INITIAL_DELTA_COUNTS
    else:
        try:
            delta_counts = motor_counts_for_joint_degrees(args.joint_degrees, args.gear_ratio)
        except ValueError as error:
            parser().error(str(error))
        max_delta_counts = abs(delta_counts)
    if args.cycles < 0:
        parser().error("--cycles must be zero (one-way test) or a positive number of forward/reverse cycles")

    deltas = (delta_counts,) if args.cycles == 0 else alternating_cycle_deltas(delta_counts, args.cycles)
    geared = args.gear_ratio > 1.0
    max_speed_rpm = MAX_GEARED_TEST_SPEED_RPM if geared else MAX_INITIAL_SPEED_RPM
    curve = GEARED_SPEED_CURVE if geared else GENTLE_SPEED_CURVE
    if not 1 <= args.speed_rpm <= max_speed_rpm:
        parser().error(f"--speed-rpm must be within 1..{max_speed_rpm} for this drive configuration")

    print("MKS single-axis initial motion: node", args.node_id)
    print(f"bounded relative step: {delta_counts} motor counts ({delta_counts / COUNTS_PER_REVOLUTION * 360:.1f}° motor-axis equivalent)")
    print(f"gear ratio: {args.gear_ratio:g}:1; output equivalent: {delta_counts / COUNTS_PER_REVOLUTION * 360 / args.gear_ratio:.2f}°")
    if args.joint_degrees is not None:
        print(f"requested output angle: {args.joint_degrees:g}° (limit ±{MAX_INITIAL_OUTPUT_DEGREES:g}°)")
    if args.profile == "curve":
        print(
            "live speed curve:",
            " -> ".join(f"{stage.speed_rpm} RPM / acc {stage.acceleration}" for stage in curve),
        )
        print(f"peak joint-output speed: {max(stage.speed_rpm for stage in curve) / args.gear_ratio:.2f} RPM")
    else:
        print(f"speed: {args.speed_rpm} RPM; acceleration: {args.acceleration}")
    print(f"checksum: {mode.value}")
    if args.cycles:
        print(f"repeatability run: {args.cycles} forward/reverse cycles ({len(deltas)} motion segments)")
    print("enable frame:", set_bus_enabled(args.node_id, True, mode))
    print("motion frame is calculated from the encoder position read immediately before execution.")
    if not args.execute:
        print("DRY RUN ONLY. Re-run with --execute on the Raspberry Pi to use can0.")
        return 0

    transport = SocketCanTransport(args.interface)
    try:
        probe = MksSingleAxisProbe(transport, args.node_id, mode)
        for segment_index, delta_counts in enumerate(deltas, start=1):
            print(f"segment {segment_index}/{len(deltas)}: {delta_counts:+d} counts")
            if args.profile == "curve":
                result = probe.move_relative_with_speed_curve_for_initial_test(
                    delta_counts=delta_counts,
                    stages=curve,
                    timeout_s=args.timeout_s,
                    max_speed_rpm=max_speed_rpm,
                    max_delta_counts=max_delta_counts,
                )
            else:
                result = probe.move_relative_for_initial_test(
                    delta_counts=delta_counts,
                    speed_rpm=args.speed_rpm,
                    acceleration=args.acceleration,
                    timeout_s=args.timeout_s,
                    max_speed_rpm=max_speed_rpm,
                    max_delta_counts=max_delta_counts,
                )
            print("before:", result.before)
            print("target encoder:", result.target_counts)
            print("after:", result.after)
            print("position error:", result.position_error_counts, "counts")
            if not result.reached_target:
                print(f"FAIL: segment {segment_index} did not reach its target before timeout.")
                return 2
    finally:
        transport.close()
    print(f"PASS: {len(deltas)} segment(s) settled within the MKS feedback tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
