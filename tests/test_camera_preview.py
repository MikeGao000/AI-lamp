import unittest
from unittest.mock import patch

import camera_preview


class CameraPreviewTests(unittest.TestCase):
    def test_defaults_are_suitable_for_pi_3_lan_preview(self):
        args = camera_preview.build_parser().parse_args([])
        self.assertEqual((args.width, args.height, args.fps), (960, 720, 10.0))
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.host, "127.0.0.1")

    def test_page_uses_live_stream_and_exposes_snapshot(self):
        page = camera_preview.INDEX_HTML.decode("utf-8")
        self.assertIn('src="/stream.mjpg"', page)
        self.assertIn('href="/snapshot.jpg"', page)
        self.assertIn("/alignment.json", page)
        self.assertIn("/shutdown", page)
        self.assertIn("method:'POST'", page)

    def test_semantic_target_excludes_background_text_and_packaging(self):
        description = camera_preview.SemanticBookTracker.TARGET_DESCRIPTION
        self.assertIn("picture-book page", description)
        self.assertIn("computer screens", description)
        self.assertIn("product packaging", description)

    def test_kcf_is_default_local_tracker(self):
        args = camera_preview.build_parser().parse_args([])
        self.assertEqual("kcf", args.opencv_tracker)

    def test_dense_text_has_priority_over_page_rectangle(self):
        text_box = (10, 20, 100, 30)
        page_box = (5, 5, 200, 250)
        with (
            patch("camera_preview.detect_text_bbox", return_value=text_box),
            patch("camera_preview.detect_document_bbox", return_value=page_box),
        ):
            self.assertEqual((text_box, "text_dense"), camera_preview.detect_priority_target(object(), object()))

    def test_page_rectangle_is_used_when_text_is_missing(self):
        page_box = (5, 5, 200, 250)
        with (
            patch("camera_preview.detect_text_bbox", return_value=None),
            patch("camera_preview.detect_document_bbox", return_value=page_box),
        ):
            self.assertEqual(
                (page_box, "page_rectangle"),
                camera_preview.detect_priority_target(object(), object()),
            )

    def test_three_api_misses_request_that_the_book_be_moved(self):
        tracker = camera_preview.SemanticBookTracker(object())
        with patch("lamp_core.object_localization.locate_target", return_value=None):
            tracker._locate(b"jpeg")
            tracker._locate(b"jpeg")
            self.assertFalse(tracker.help_requested)
            tracker._locate(b"jpeg")
            tracker._locate(b"jpeg")
        self.assertEqual(3, tracker.api_miss_count)
        self.assertTrue(tracker.help_requested)
        self.assertEqual("please_move_book", tracker.status)
        tracker.resume_after_stable_local_candidate()
        self.assertEqual(0, tracker.api_miss_count)
        self.assertFalse(tracker.help_requested)


if __name__ == "__main__":
    unittest.main()
