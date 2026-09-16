import unittest

from lamp_core.object_localization import (
    ObjectLocalizationError,
    PageReading,
    SimulatedLocalTargetDetector,
    TargetDetection,
    choose_candidate_index,
    confirmation_prompt,
    confirm_target_present,
    locate_page,
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
        self.assertIn("json", client.calls[0][1])

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

    def test_confirmation_accepts_a_partial_book_without_a_box(self):
        self.assertTrue(
            confirm_target_present(b"jpeg", FakeVisionClient('{"found":true}'), "a book page")
        )

    def test_confirmation_rejects_when_nothing_is_found(self):
        self.assertFalse(
            confirm_target_present(b"jpeg", FakeVisionClient('{"found":false}'), "a book page")
        )

    def test_confirmation_requires_an_explicit_found_flag(self):
        with self.assertRaises(ObjectLocalizationError):
            confirm_target_present(b"jpeg", FakeVisionClient("{}"), "a book page")

    def test_cropped_confirmation_scopes_the_question_to_the_crop_only(self):
        prompt = confirmation_prompt("a children's picture book page", cropped_region=True)
        self.assertIn("crop", prompt)
        self.assertIn("children's picture book page", prompt)
        self.assertIn("not whatever else", prompt)

    def test_cropped_confirmation_asks_the_client_about_the_crop(self):
        client = FakeVisionClient('{"found":false}')
        self.assertFalse(
            confirm_target_present(
                b"jpeg", client, "a picture book page", cropped_region=True
            )
        )
        self.assertIn("crop", client.calls[0][1])

    def test_candidate_selection_returns_the_chosen_rectangle_number(self):
        client = FakeVisionClient('{"choice":2}')
        self.assertEqual(2, choose_candidate_index(b"jpeg", client, "a book page", 4))
        self.assertIn("4 numbered rectangles", client.calls[0][1])

    def test_candidate_selection_zero_means_none_of_them(self):
        self.assertEqual(
            0, choose_candidate_index(b"jpeg", FakeVisionClient('{"choice":0}'), "a book", 3)
        )

    def test_candidate_selection_rejects_a_choice_outside_the_offered_set(self):
        with self.assertRaises(ObjectLocalizationError):
            choose_candidate_index(b"jpeg", FakeVisionClient('{"choice":9}'), "a book", 4)
        with self.assertRaises(ObjectLocalizationError):
            choose_candidate_index(b"jpeg", FakeVisionClient('{"choice":"two"}'), "a book", 4)

    def test_locate_page_returns_the_box_and_the_text_in_one_call(self):
        client = FakeVisionClient(
            '{"found":true,"bbox_norm":[0.1,0.2,0.9,0.9],"visible_text":"Hej med dig","confidence":0.8}'
        )
        reading = locate_page(b"jpeg", client, "a children's picture book page")
        self.assertIsInstance(reading, PageReading)
        self.assertEqual((0.1, 0.2, 0.9, 0.9), reading.bbox_norm)
        self.assertEqual("Hej med dig", reading.visible_text)
        self.assertAlmostEqual(0.8, reading.confidence)
        self.assertEqual(1, len(client.calls))

    def test_locate_page_treats_found_false_as_no_page(self):
        self.assertIsNone(
            locate_page(b"jpeg", FakeVisionClient('{"found":false}'), "a book page")
        )

    def test_locate_page_requires_a_usable_box(self):
        client = FakeVisionClient('{"found":true,"bbox_norm":[0.9,0.2,0.1,0.9]}')
        with self.assertRaises(ObjectLocalizationError):
            locate_page(b"jpeg", client, "a book page")

    def test_locate_page_validates_the_reported_confidence(self):
        client = FakeVisionClient(
            '{"found":true,"bbox_norm":[0.1,0.2,0.9,0.9],"confidence":1.5}'
        )
        with self.assertRaises(ObjectLocalizationError):
            locate_page(b"jpeg", client, "a book page")
