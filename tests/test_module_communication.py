"""Offline, end-to-end message-flow simulations between the lamp modules."""

from __future__ import annotations

import unittest

from lamp_core.cloud import StaticStoryClient
from lamp_core.coordinator import AppEvent, ReadingCompanionCoordinator
from lamp_core.motion import JointLimit
from lamp_core.smart_motion import JointMove, SmartMotionDispatcher
from lamp_core.teach import TeachingRecorder, queue_safe_replay, smooth
from lamp_core.transport import MemoryTransport
from lamp_core.vision import StillnessGate
from lamp_core.virtual_hardware import SpeechStub, VirtualMotorBus


LIMITS = {
    "j1_base_yaw": JointLimit(-1.57, 1.57, 0.80),
    "j2_shoulder": JointLimit(-0.78, 0.78, 0.45),
    "j3_elbow": JointLimit(-0.95, 0.95, 0.55),
    "j4_neck_pitch": JointLimit(-0.70, 0.70, 0.75),
    "j5_head_yaw": JointLimit(-1.05, 1.05, 0.90),
}


class RecordingMotionBackend:
    """Verified-adapter stand-in; records dispatcher-to-driver calls only."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def prepare_move(self, move: JointMove) -> None:
        self.calls.append(("prepare", move.label))

    def start_prepared_move(self) -> None:
        self.calls.append(("start", ""))

    def cancel_motion(self) -> None:
        self.calls.append(("cancel", ""))


class ModuleCommunicationSimulationTests(unittest.TestCase):
    def test_page_to_story_to_safe_motion_to_speech(self) -> None:
        """Vision -> cloud -> coordinator -> trajectory/bus -> speech, with no I/O."""
        gate = StillnessGate(required_seconds=1.5, threshold=8.0)
        bus = VirtualMotorBus(LIMITS)
        speaker = SpeechStub()
        controller = ReadingCompanionCoordinator(LIMITS, bus, speaker)
        story = StaticStoryClient("Det er en glad ræv på siden.")
        controller.home()

        # A moving page resets the vision gate and inhibits a capture.
        self.assertFalse(gate.observe(9.0, now=0.0))
        controller.handle(AppEvent.BOOK_MOVED)
        self.assertFalse(gate.observe(0.5, now=0.1))

        # A continuously still page permits the cloud-story and motor flow.
        self.assertTrue(gate.observe(0.5, now=1.6))
        controller.set_pending_story(story.describe_page(b"simulated-jpeg"))
        controller.handle(AppEvent.BOOK_STILL)

        self.assertEqual("READY_HOLD", controller.safety.state.name)
        self.assertTrue(controller.safety.drives_enabled)
        self.assertGreater(len(bus.received_frames), 1)
        self.assertEqual(
            {
                "j1_base_yaw": 0.25,
                "j2_shoulder": -0.20,
                "j3_elbow": 0.30,
                "j4_neck_pitch": 0.12,
                "j5_head_yaw": -0.18,
            },
            bus.positions_rad,
        )
        self.assertEqual(["Det er en glad ræv på siden."], speaker.messages)

    def test_teach_replay_reaches_smart_motion_backend_in_order(self) -> None:
        """Teaching recorder -> smoothing -> replay queue -> motor adapter calls."""
        recorder = TeachingRecorder("page-turn", LIMITS)
        recorder.append(0.0, {name: 0.0 for name in LIMITS})
        recorder.append(0.4, {name: 0.1 for name in LIMITS})
        recorder.append(0.8, {name: 0.2 for name in LIMITS})
        recording = smooth(recorder.finish(), LIMITS)
        backend = RecordingMotionBackend()
        dispatcher = SmartMotionDispatcher(LIMITS, backend)

        self.assertEqual(2, queue_safe_replay(recording, LIMITS, dispatcher))
        dispatcher.tick(motion_finished=True)
        dispatcher.tick(motion_finished=True)

        self.assertEqual(
            [
                ("prepare", "teach:page-turn:0"),
                ("start", ""),
                ("prepare", "teach:page-turn:1"),
                ("start", ""),
            ],
            backend.calls,
        )

    def test_memory_transport_round_trip(self) -> None:
        """Host-to-driver command and driver acknowledgement stay byte-exact."""
        transport = MemoryTransport()
        command = b"SIM:MOVE:j1=0.25"
        acknowledgement = b"SIM:ACK"

        transport.write(command)
        transport.incoming.extend(acknowledgement)

        self.assertEqual([command], transport.writes)
        self.assertEqual(acknowledgement, transport.read(64))
        self.assertEqual(b"", transport.read(1))
