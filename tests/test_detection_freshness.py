import queue
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from camera_preview import CameraStream
from lamp_core.salient_page import ProcessPageHypothesis
from lamp_core.text_recognition import CrnnTextRecognizer


class DetectionFreshnessTests(unittest.TestCase):
    def test_expired_preview_result_is_not_used(self):
        stream = CameraStream(640, 480, 10, 80, 0)
        stream._detection_lock = threading.Lock()
        stream._detection_latest = (1, 2, 3, 4)
        stream._detection_at = time.monotonic() - 2
        self.assertIsNone(stream._detected())

    def test_detector_owns_unannotated_frame_copy(self):
        stream = CameraStream(640, 480, 10, 80, 0)
        stream._detection_lock = threading.Lock()
        stream._detection_wake = threading.Event()
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        stream._submit_detection(image)
        image[:] = 255
        self.assertFalse(stream._detection_frame[0].any())

    def test_missing_hypothesis_clears_previous_box(self):
        results = queue.Queue()
        hypothesis = ProcessPageHypothesis(None, autostart=False, result_queue=results)
        hypothesis.box = (0.1, 0.1, 0.9, 0.9)
        results.put(None)
        hypothesis.drain()
        self.assertIsNone(hypothesis.box)

    def test_late_model_result_is_discarded(self):
        results = queue.Queue()
        hypothesis = ProcessPageHypothesis(None, autostart=False, result_queue=results)
        results.put({"box": (0.1, 0.1, 0.9, 0.9), "captured_at": time.monotonic() - 9})
        hypothesis.drain()
        self.assertIsNone(hypothesis.box)

    def test_transcript_crops_ink_instead_of_gaps(self):
        recognizer = CrnnTextRecognizer.__new__(CrnnTextRecognizer)
        image = np.full((40, 100, 3), 255, dtype=np.uint8)
        image[10:20] = 0
        with patch.object(CrnnTextRecognizer, "available", new_callable=unittest.mock.PropertyMock, return_value=True), patch.object(recognizer, "read_crop", return_value="hej") as read:
            recognizer.transcript(image, [(0, 0, 1, 1)])
        self.assertEqual(1, read.call_count)
        self.assertLess(read.call_args.args[0].mean(), 100)
