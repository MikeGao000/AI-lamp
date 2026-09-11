import unittest

from lamp_core.mks_can_protocol import CanFrame, ChecksumMode
from lamp_core.mks_single_axis import (
    GENTLE_SPEED_CURVE,
    MAX_INITIAL_DELTA_COUNTS,
    MotionStage,
    MksSingleAxisProbe,
    alternating_cycle_deltas,
)


def reply(node_id: int, command: int, value: int, width: int) -> CanFrame:
    body = bytes((command,)) + value.to_bytes(width, "big", signed=True)
    return CanFrame(node_id, body + bytes(((node_id + sum(body)) & 0xFF,)))


class FakeCanTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def send(self, frame):
        self.sent.append(frame)

    def receive(self, timeout_s):
        return self.responses.pop(0) if self.responses else None


class MksSingleAxisProbeTests(unittest.TestCase):
    def test_ten_cycles_are_twenty_alternating_relative_segments(self):
        self.assertEqual((4096, -4096) * 10, alternating_cycle_deltas(4096, 10))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            alternating_cycle_deltas(4096, 0)

    def test_probe_refuses_to_join_a_motor_already_in_motion(self):
        transport = FakeCanTransport((
            reply(1, 0x31, 100, 6), reply(1, 0x32, 3, 2),
        ))
        probe = MksSingleAxisProbe(transport, node_id=1, checksum_mode="additive")

        with self.assertRaisesRegex(RuntimeError, "already moving at 3 RPM"):
            probe.move_relative_for_initial_test()

        self.assertEqual([frame.data[0] for frame in transport.sent], [0x31, 0x32])

    def test_relative_probe_reads_live_position_and_sends_bounded_absolute_target(self):
        # before snapshot: encoder 100 / rpm 0; after snapshot: target 4196 / rpm 0
        transport = FakeCanTransport((
            reply(1, 0x31, 100, 6), reply(1, 0x32, 0, 2),
            reply(1, 0x31, 4196, 6), reply(1, 0x32, 0, 2),
        ))
        probe = MksSingleAxisProbe(transport, 1, ChecksumMode.ADDITIVE)
        result = probe.move_relative_for_initial_test(delta_counts=4096, speed_rpm=10, acceleration=1)

        self.assertTrue(result.reached_target)
        self.assertEqual(100, result.before.encoder_counts)
        self.assertEqual(4196, result.target_counts)
        self.assertEqual(0xF3, transport.sent[2].data[0])
        self.assertEqual(0xF5, transport.sent[3].data[0])
        self.assertEqual(4196, int.from_bytes(transport.sent[3].data[4:7], "big", signed=True))

    def test_probe_accepts_the_documented_encoder_and_rpm_settle_tolerance(self):
        # A real MKS test settled four counts short and reported -1 RPM.
        transport = FakeCanTransport((
            reply(1, 0x31, 0, 6), reply(1, 0x32, 0, 2),
            reply(1, 0x31, 4092, 6), reply(1, 0x32, -1, 2),
        ))
        probe = MksSingleAxisProbe(transport, 1, ChecksumMode.ADDITIVE)
        result = probe.move_relative_for_initial_test()

        self.assertTrue(result.reached_target)
        self.assertEqual(-4, result.position_error_counts)

    def test_next_segment_accepts_a_previous_settle_reading_of_minus_one_rpm(self):
        transport = FakeCanTransport((
            # First +4096 step, which settles at 4092 / -1 RPM.
            reply(1, 0x31, 0, 6), reply(1, 0x32, 0, 2),
            reply(1, 0x31, 4092, 6), reply(1, 0x32, -1, 2),
            # Its reverse step must accept that -1 RPM snapshot as stationary.
            reply(1, 0x31, 4092, 6), reply(1, 0x32, -1, 2),
            reply(1, 0x31, -4, 6), reply(1, 0x32, 0, 2),
        ))
        probe = MksSingleAxisProbe(transport, 1, ChecksumMode.ADDITIVE)

        first = probe.move_relative_for_initial_test(delta_counts=4096)
        second = probe.move_relative_for_initial_test(delta_counts=-4096)

        self.assertTrue(first.reached_target)
        self.assertTrue(second.reached_target)
        self.assertEqual(-4, second.after.encoder_counts)

    def test_curve_streams_live_f5_speed_updates_to_one_target(self):
        transport = FakeCanTransport((
            reply(1, 0x31, 100, 6), reply(1, 0x32, 0, 2),
            reply(1, 0x31, 4194, 6), reply(1, 0x32, 0, 2),
        ))
        probe = MksSingleAxisProbe(transport, 1, ChecksumMode.ADDITIVE)
        stages = (
            MotionStage(speed_rpm=3, acceleration=8, hold_s=0),
            MotionStage(speed_rpm=6, acceleration=24, hold_s=0),
            MotionStage(speed_rpm=10, acceleration=40, hold_s=0),
            MotionStage(speed_rpm=4, acceleration=12, hold_s=0),
        )

        result = probe.move_relative_with_speed_curve_for_initial_test(
            delta_counts=4096,
            stages=stages,
        )

        self.assertTrue(result.reached_target)
        f5_frames = [frame for frame in transport.sent if frame.data[0] == 0xF5]
        self.assertEqual([3, 6, 10, 4], [int.from_bytes(frame.data[1:3], "big") for frame in f5_frames])
        self.assertEqual([4196] * 4, [int.from_bytes(frame.data[4:7], "big", signed=True) for frame in f5_frames])

    def test_default_curve_stays_within_the_initial_speed_cap(self):
        self.assertEqual([3, 6, 10, 4], [stage.speed_rpm for stage in GENTLE_SPEED_CURVE])
        self.assertTrue(all(stage.speed_rpm <= 10 for stage in GENTLE_SPEED_CURVE))

    def test_probe_rejects_a_larger_than_quarter_revolution_step(self):
        probe = MksSingleAxisProbe(FakeCanTransport(()), 1, ChecksumMode.ADDITIVE)
        with self.assertRaisesRegex(ValueError, "4096"):
            probe.move_relative_for_initial_test(delta_counts=MAX_INITIAL_DELTA_COUNTS + 1)


if __name__ == "__main__":
    unittest.main()
