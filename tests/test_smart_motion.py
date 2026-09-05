import unittest

from lamp_core.motion import JointLimit
from lamp_core.smart_motion import JointMove, MotionMode, SmartMotionDispatcher


class BackendStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def prepare_move(self, move: JointMove) -> None:
        self.calls.append(("prepare", move.label))

    def start_prepared_move(self) -> None:
        self.calls.append(("start", ""))

    def cancel_motion(self) -> None:
        self.calls.append(("cancel", ""))


LIMITS = {
    "j1": JointLimit(-1.0, 1.0, 1.0),
    "j2": JointLimit(-1.0, 1.0, 1.0),
}


def move(label: str) -> JointMove:
    return JointMove({"j1": 0.1, "j2": -0.1}, 0.5, label)


class SmartMotionTests(unittest.TestCase):
    def test_sequence_starts_one_segment_per_completion(self) -> None:
        backend = BackendStub()
        dispatcher = SmartMotionDispatcher(LIMITS, backend)
        dispatcher.submit_sequence(move("nod-down"))
        dispatcher.submit_sequence(move("nod-up"))
        dispatcher.tick(motion_finished=True)
        dispatcher.tick(motion_finished=True)
        self.assertEqual(
            [("prepare", "nod-down"), ("start", ""), ("prepare", "nod-up"), ("start", "")],
            backend.calls,
        )
        self.assertEqual(MotionMode.SEQ, dispatcher.active_mode)

    def test_interrupt_cancels_sequence_and_trajectory(self) -> None:
        backend = BackendStub()
        dispatcher = SmartMotionDispatcher(LIMITS, backend)
        dispatcher.submit_sequence(move("queued"))
        dispatcher.submit_trajectory_segment(move("trajectory"))
        dispatcher.submit_interrupt(move("stop-looking"))
        self.assertEqual(0, dispatcher.queued_sequence_count)
        self.assertEqual(0, dispatcher.queued_trajectory_count)
        self.assertEqual(MotionMode.INT, dispatcher.active_mode)
        self.assertEqual(
            [("cancel", ""), ("prepare", "stop-looking"), ("start", "")], backend.calls
        )

    def test_trajectory_runs_only_after_sequence_is_empty(self) -> None:
        backend = BackendStub()
        dispatcher = SmartMotionDispatcher(LIMITS, backend)
        dispatcher.submit_sequence(move("expression"))
        dispatcher.submit_trajectory_segment(move("smooth"))
        dispatcher.tick(motion_finished=True)
        dispatcher.tick(motion_finished=True)
        self.assertEqual(MotionMode.TRJ, dispatcher.active_mode)
        self.assertEqual("smooth", backend.calls[-2][1])
