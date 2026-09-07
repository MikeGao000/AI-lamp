import unittest

from lamp_core.object_localization import (
    ObjectLocalizationError,
    SimulatedLocalTargetDetector,
    TargetDetection,
    locate_target,
)


class FakeVisionClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def describe_page(self, jpeg, prompt=None, system_instructions=None, on_output_text_delta=None):
        self.calls.append((jpeg, prompt, system_instructions))
        return self.response


class ObjectLocalizationTests(unittest.TestCase):
    def test_parses_and_validates_a_target_box_from_a_vision_response(self):
        client = FakeVisionClient(
            '{"found":true,"label":"blue package","bbox_norm":[0.20,0.30,0.60,0.80],"confidence":0.91}'
        )
        detection = locate_target(b"jpeg", client, "the blue package", object_id="box-1")
        self.assertEqual("box-1", detection.object_id)
        self.assertEqual("blue package", detection.label)
        self.assertAlmostEqual(0.4, detection.center_x)
        self.assertAlmostEqual(0.55, detection.center_y)
        self.assertIn("blue package", client.calls[0][1])

    def test_not_found_is_a_valid_vision_result(self):
        self.assertIsNone(locate_target(b"jpeg", FakeVisionClient('{"found":false}'), "a book"))

    def test_invalid_box_is_rejected_before_motion_can_consume_it(self):
        client = FakeVisionClient(
            '{"found":true,"label":"book","bbox_norm":[0.8,0.2,0.1,0.7],"confidence":0.8}'
        )
        with self.assertRaises(ObjectLocalizationError):
            locate_target(b"jpeg", client, "a book")

    def test_simulated_local_detector_uses_the_same_validated_detection_contract(self):
        detector = SimulatedLocalTargetDetector(
            {"sample.jpeg": TargetDetection("box-1", "box", (0.2, 0.3, 0.7, 0.8), 0.9)}
        )
        detection = detector.locate("sample.jpeg", b"jpeg")
        self.assertEqual("box-1", detection.object_id)
        self.assertIsNone(detector.locate("unknown.jpeg", b"jpeg"))
