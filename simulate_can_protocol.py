"""Offline CAN telemetry demonstration; never opens hardware or sends frames."""

from lamp_core.mks_can_protocol import (
    CanFrame,
    ChecksumMode,
    parse_accumulated_encoder,
    parse_motor_rpm,
    read_accumulated_encoder,
    read_motor_rpm,
)


def show(direction: str, frame: CanFrame) -> None:
    print(f"{direction} ID={frame.arbitration_id:03X} DLC={len(frame.data)} DATA={frame.data.hex(' ').upper()}")


def fake_reply(address: int, command: int, value: int, width: int) -> CanFrame:
    body = bytes([command]) + value.to_bytes(width, "big", signed=True)
    return CanFrame(address, body + bytes([(address + sum(body)) & 0xFF]))


def main() -> None:
    mode = ChecksumMode.ADDITIVE
    print("OFFLINE SIMULATION ONLY - no CAN device opened, no frames transmitted")
    print("Assumptions: node 1, additive checksum, project MKS CAN V1.0.9 codec")
    for label, counts, rpm in [("stationary", 0, 0), ("synthetic telemetry", 4096, 30)]:
        print(f"\n[{label}; all RX data is fabricated]")
        show("SIM TX", read_accumulated_encoder(1, mode))
        reply = fake_reply(1, 0x31, counts, 6)
        show("SIM RX", reply)
        decoded = parse_accumulated_encoder(reply, mode)
        assert decoded == counts
        print(f"Parsed motor encoder: {decoded} counts = {decoded * 360 / 16384:.2f} degrees (not calibrated joint angle)")
        show("SIM TX", read_motor_rpm(1, mode))
        reply = fake_reply(1, 0x32, rpm, 2)
        show("SIM RX", reply)
        decoded_rpm = parse_motor_rpm(reply, mode)
        assert decoded_rpm == rpm
        print(f"Parsed motor speed: {decoded_rpm} RPM")

    print("\n[Corrupt response]")
    corrupt = CanFrame(1, bytes.fromhex("32 00 1E 00"))
    show("SIM RX", corrupt)
    try:
        parse_motor_rpm(corrupt, mode)
    except ValueError as error:
        print(f"Rejected as expected: {error}")
    else:
        raise AssertionError("Corrupt response was accepted")
    print("\nPASS: offline examples decoded; invalid checksum rejected.")
    print("Hardware wiring, bitrate, firmware compatibility and motor behavior remain untested.")


if __name__ == "__main__":
    main()
