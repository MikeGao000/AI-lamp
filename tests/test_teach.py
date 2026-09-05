import tempfile
import unittest
from pathlib import Path

from lamp_core.motion import JointLimit
from lamp_core.smart_motion import SmartMotionDispatcher
from lamp_core.teach import TeachingRecorder, load, queue_safe_replay, save, smooth


LIMITS = {"j1": JointLimit(-1.0, 1.0, 1.0), "j2": JointLimit(-1.0, 1.0, 1.0)}


class BackendStub:
    def prepare_move(self, move):
        pass

    def start_prepared_move(self):
        pass

    def cancel_motion(self):
        pass


def recording():
    recorder = TeachingRecorder("hello", LIMITS)
    recorder.append(0.0, {"j1": 0.0, "j2": 0.0})
    recorder.append(0.2, {"j1": 0.6, "j2": -0.2})
    recorder.append(0.4, {"j1": 0.4, "j2": -0.4})
    return recorder.finish()


class TeachTests(unittest.TestCase):
    def test_recording_round_trip(self):
        original = recording()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            save(original, LIMITS, path)
            self.assertEqual(original, load(path, LIMITS))

    def test_smoothing_preserves_endpoints_and_limits(self):
        edited = smooth(recording(), LIMITS)
        self.assertEqual(0.0, edited.samples[0].positions_rad["j1"])
        self.assertEqual(0.4, edited.samples[-1].positions_rad["j1"])
        self.assertAlmostEqual(1.0 / 3.0, edited.samples[1].positions_rad["j1"])

    def test_safe_replay_slows_and_queues_each_segment(self):
        dispatcher = SmartMotionDispatcher(LIMITS, BackendStub())
        queued = queue_safe_replay(recording(), LIMITS, dispatcher, playback_rate=0.5)
        self.assertEqual(2, queued)
        self.assertEqual(2, dispatcher.queued_trajectory_count)
