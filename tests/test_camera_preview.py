import time
import unittest
from unittest.mock import patch

import camera_preview
from lamp_core.alignment import HorizontalDirection, NarrowingScan, VisualServoPid
from lamp_core.mks_can_protocol import ChecksumMode
from lamp_core.object_localization import PageReading
from lamp_core.text_detection import TextBox


class _FakeTransport:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)


class _FakeTracker:
    def __init__(self):
        self.inited_bbox = None
        self._tracked = None

    def init(self, image, bbox):
        self.inited_bbox = bbox

    def update(self, image):
        return (True, self._tracked) if self._tracked is not None else (False, None)


class _FakeBuffer:
    def tobytes(self):
        return b"jpeg"


class _FakeCV:
    TrackerKCF_create = _FakeTracker
    TrackerCSRT_create = _FakeTracker
    IMWRITE_JPEG_QUALITY = 1
    MARKER_CROSS = 0
    FONT_HERSHEY_SIMPLEX = 0

    @staticmethod
    def imencode(extension, image, params=None):
        return True, _FakeBuffer()

    # Drawing calls are no-ops on the fake image, but they must exist so the
    # annotation path can be exercised without a real OpenCV build.
    @staticmethod
    def drawMarker(image, center, color, marker_type, size, thickness):
        return image

    @staticmethod
    def putText(image, text, org, font, scale, color, thickness):
        return image

    @staticmethod
    def rectangle(image, start, end, color, thickness):
        return image

    @staticmethod
    def circle(image, center, radius, color, thickness):
        return image

    @staticmethod
    def line(image, start, end, color, thickness):
        return image

    @staticmethod
    def arrowedLine(image, start, end, color, thickness, tipLength=0.1):
        return image


class _FakeImage:
    shape = (480, 640, 3)

    def __getitem__(self, key):
        return self

    def copy(self):
        return self


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
        self.assertIn("cropped by the image edge", description)
        self.assertIn("all visible pixels", description)
        self.assertIn("computer screens", description)
        self.assertIn("product packaging", description)

    def test_kcf_is_default_local_tracker(self):
        args = camera_preview.build_parser().parse_args([])
        self.assertEqual("kcf", args.opencv_tracker)

    def test_text_selects_the_containing_page_not_the_text_fragment(self):
        text_box = (10, 20, 100, 30)
        page_box = (5, 5, 200, 250)
        with (
            patch("camera_preview.detect_text_bbox", return_value=text_box),
            patch("camera_preview.detect_document_bbox", return_value=page_box),
        ):
            self.assertEqual(
                (page_box, "book_page_with_text"),
                camera_preview.detect_priority_target(object(), object()),
            )

    def test_text_is_only_a_fallback_when_no_page_is_visible(self):
        text_box = (10, 20, 100, 30)
        with (
            patch("camera_preview.detect_text_bbox", return_value=text_box),
            patch("camera_preview.detect_document_bbox", return_value=None),
        ):
            self.assertEqual(
                (text_box, "text_fallback"),
                camera_preview.detect_priority_target(object(), object()),
            )

    def test_semantic_book_outranks_local_text_fragment(self):
        text_box = (10, 20, 100, 30)
        book_box = (5, 5, 250, 300)
        self.assertEqual(
            (book_box, "semantic_book"),
            camera_preview.choose_tracking_target(text_box, "text_fallback", book_box),
        )

    def test_unconfirmed_local_shape_cannot_steer_the_motor(self):
        text_box = (10, 20, 100, 30)
        self.assertEqual(
            (None, "awaiting_semantic_text_fallback"),
            camera_preview.choose_tracking_target(text_box, "text_fallback", None),
        )

    def test_cloud_reacquisition_waits_for_a_stable_local_candidate(self):
        self.assertFalse(camera_preview.local_candidate_is_stable(10.0, 10.59))
        self.assertTrue(camera_preview.local_candidate_is_stable(10.0, 10.60))
        self.assertFalse(camera_preview.local_candidate_is_stable(0.0, 20.0))

    def test_bbox_stability_rejects_a_page_that_moved_far(self):
        self.assertAlmostEqual(
            1.0,
            camera_preview.bbox_intersection_over_union((10, 10, 100, 100), (10, 10, 100, 100)),
        )
        self.assertEqual(
            0.0,
            camera_preview.bbox_intersection_over_union((10, 10, 100, 100), (300, 10, 100, 100)),
        )
        self.assertLess(
            camera_preview.bbox_intersection_over_union((10, 10, 100, 100), (80, 10, 100, 100)),
            0.55,
        )

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
        with (
            patch("lamp_core.object_localization.locate_page", return_value=None),
            patch("camera_preview.detect_candidate_boxes", return_value=[]),
        ):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
            tracker._pick_candidate(_FakeImage(), _FakeCV())
            self.assertFalse(tracker.help_requested)
            tracker._pick_candidate(_FakeImage(), _FakeCV())
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual(3, tracker.api_miss_count)
        self.assertTrue(tracker.help_requested)
        self.assertEqual("please_move_book", tracker.status)
        tracker.resume_after_stable_local_candidate()
        self.assertEqual(0, tracker.api_miss_count)
        self.assertFalse(tracker.help_requested)

    def test_bbox_touches_image_edge_detects_partial_book(self):
        self.assertTrue(camera_preview.bbox_touches_image_edge((0, 10, 100, 100), 640, 480))
        self.assertTrue(camera_preview.bbox_touches_image_edge((590, 10, 60, 100), 640, 480))
        self.assertTrue(camera_preview.bbox_touches_image_edge((220, 0, 200, 100), 640, 480))
        self.assertFalse(camera_preview.bbox_touches_image_edge((220, 140, 200, 200), 640, 480))

    def test_edge_tracked_box_refreshes_sooner_than_the_timer(self):
        tracker = camera_preview.SemanticBookTracker(object(), reacquire_seconds=8.0)
        tracker._tracker = _FakeTracker()
        tracker._tracker._tracked = (0, 10, 100, 100)
        tracker._last_request_at = time.monotonic() - 3.0
        with patch.object(tracker, "_request_confirm") as request_confirm:
            box = tracker.update(_FakeImage(), object(), allow_reacquire=True)
        self.assertEqual((0, 10, 100, 100), box)
        request_confirm.assert_called_once()

    def test_interior_tracked_box_waits_for_the_timer(self):
        tracker = camera_preview.SemanticBookTracker(object(), reacquire_seconds=8.0)
        tracker._tracker = _FakeTracker()
        tracker._tracker._tracked = (220, 140, 200, 200)
        tracker._last_request_at = time.monotonic() - 3.0
        with patch.object(tracker, "_request_confirm") as request_confirm:
            box = tracker.update(_FakeImage(), object(), allow_reacquire=True)
        self.assertEqual((220, 140, 200, 200), box)
        request_confirm.assert_not_called()

    def test_validated_model_box_becomes_the_tracking_target(self):
        tracker = camera_preview.SemanticBookTracker(object())
        stale_tracker = _FakeTracker()
        tracker._tracker = stale_tracker
        tracker._pending_bbox = (64, 96, 512, 288)
        box = tracker.update(_FakeImage(), _FakeCV(), allow_reacquire=False)
        self.assertEqual((64, 96, 512, 288), box)
        self.assertIsNot(tracker._tracker, stale_tracker)
        self.assertEqual((64, 96, 512, 288), tracker._tracker.inited_bbox)
        self.assertIsNone(tracker._pending_bbox)

    def test_single_locate_call_supplies_the_box_and_the_page_text(self):
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "Once upon a time", 0.9)
        tracker = camera_preview.SemanticBookTracker(object())
        with patch("lamp_core.object_localization.locate_page", return_value=reading) as locate:
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual((64, 96, 512, 288), tracker._pending_bbox)
        self.assertEqual("Once upon a time", tracker.page_text)
        self.assertEqual("semantic_book_candidate_picked", tracker.status)
        locate.assert_called_once()

    def test_missing_page_falls_back_to_numbered_candidates(self):
        tracker = camera_preview.SemanticBookTracker(object())
        with (
            patch("lamp_core.object_localization.locate_page", return_value=None),
            patch(
                "camera_preview.detect_candidate_boxes", return_value=[(10, 20, 300, 200)]
            ),
            patch("camera_preview.annotate_candidate_boxes", return_value=_FakeImage()),
            patch("lamp_core.object_localization.choose_candidate_index", return_value=1),
        ):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual((10, 20, 300, 200), tracker._pending_bbox)

    def test_page_text_is_cleared_on_reset(self):
        tracker = camera_preview.SemanticBookTracker(object())
        tracker.page_text = "old page"
        tracker.reset()
        self.assertEqual("", tracker.page_text)

    def test_reset_forgets_tracker_and_pending_box(self):
        tracker = camera_preview.SemanticBookTracker(object())
        tracker._tracker = _FakeTracker()
        tracker._pending_bbox = (10, 10, 50, 50)
        tracker.api_miss_count = 2
        tracker.help_requested = True
        tracker.status = "opencv_kcf_tracking"
        tracker.reset()
        self.assertIsNone(tracker._tracker)
        self.assertIsNone(tracker._pending_bbox)
        self.assertEqual(0, tracker.api_miss_count)
        self.assertFalse(tracker.help_requested)
        self.assertEqual("waiting_for_semantic_book", tracker.status)

    def test_page_exposes_a_restart_button(self):
        page = camera_preview.INDEX_HTML.decode("utf-8")
        self.assertIn("/restart", page)
        self.assertIn('id="restart"', page)
        self.assertIn("当前位置为 0", page)

    def test_awaiting_confirmation_tracks_an_in_flight_crop(self):
        tracker = camera_preview.SemanticBookTracker(object())
        self.assertFalse(tracker.awaiting_confirmation)
        tracker._locating = True
        self.assertTrue(tracker.awaiting_confirmation)


class _FakeTextDetector:
    """Stands in for PpocrTextDetector in the tracker tests."""

    def __init__(self, boxes, available=True):
        self.boxes = boxes
        self._available = available
        self.calls = 0

    @property
    def available(self):
        return self._available

    def detect(self, image):
        self.calls += 1
        return self.boxes


class LocalTextDetectionTests(unittest.TestCase):
    def test_local_text_anchor_is_used_without_any_cloud_call(self):
        # Mirror of a measured frame: "Ma jeg pynte kagen," plus "mor?".
        detector = _FakeTextDetector(
            [
                TextBox((0.388, 0.105, 0.884, 0.223), 0.98),
                TextBox((0.547, 0.191, 0.688, 0.266), 0.98),
            ]
        )
        tracker = camera_preview.SemanticBookTracker(None, text_detector=detector)
        with patch("lamp_core.object_localization.locate_page") as locate:
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        locate.assert_not_called()
        self.assertEqual(1, detector.calls)
        self.assertEqual("local_text_book_candidate", tracker.status)
        self.assertIsNotNone(tracker._pending_bbox)
        self.assertGreater(tracker.page_score, 0.0)

    def test_detection_only_clears_stale_page_text(self):
        detector = _FakeTextDetector([TextBox((0.30, 0.10, 0.70, 0.30), 0.97)])
        tracker = camera_preview.SemanticBookTracker(None, text_detector=detector)
        tracker.page_text = "text read from an older frame"
        tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual("", tracker.page_text)

    def test_local_miss_falls_back_to_the_cloud(self):
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "Once upon a time", 0.9)
        detector = _FakeTextDetector([])
        tracker = camera_preview.SemanticBookTracker(object(), text_detector=detector)
        with patch("lamp_core.object_localization.locate_page", return_value=reading) as locate:
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        locate.assert_called_once()
        self.assertEqual((64, 96, 512, 288), tracker._pending_bbox)
        self.assertEqual("Once upon a time", tracker.page_text)
        self.assertEqual("semantic_book_candidate_picked", tracker.status)

    def test_local_text_cannot_bypass_semantic_confirmation_when_cloud_is_available(self):
        detector = _FakeTextDetector(
            [TextBox((0.388, 0.105, 0.884, 0.266), 0.98)]
        )
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "page", 0.9)
        tracker = camera_preview.SemanticBookTracker(object(), text_detector=detector)
        with patch("lamp_core.object_localization.locate_page", return_value=reading) as locate:
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        locate.assert_called_once()
        self.assertEqual("cloud", tracker._anchor_source)
        self.assertEqual("semantic_book_candidate_picked", tracker.status)

    def test_absent_model_never_runs_the_detector(self):
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "Once upon a time", 0.9)
        detector = _FakeTextDetector([], available=False)
        tracker = camera_preview.SemanticBookTracker(object(), text_detector=detector)
        with patch("lamp_core.object_localization.locate_page", return_value=reading):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual(0, detector.calls)
        self.assertEqual("semantic_book_candidate_picked", tracker.status)

    def test_local_only_mode_reports_a_miss_instead_of_failing(self):
        # With no cloud client the tracker must still work, and simply report
        # the same "no book" status the cloud path would.
        detector = _FakeTextDetector([])
        tracker = camera_preview.SemanticBookTracker(None, text_detector=detector)
        tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertIsNone(tracker._pending_bbox)
        self.assertEqual(0.0, tracker.page_score)
        self.assertEqual("semantic_book_not_found_1_of_3", tracker.status)

    def test_local_text_anchor_rejects_a_top_edge_only_detection(self):
        # The monitor taskbar row: nothing usable, so the cloud is consulted.
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "page", 0.9)
        detector = _FakeTextDetector([TextBox((0.59, 0.003, 0.61, 0.026), 0.96)])
        tracker = camera_preview.SemanticBookTracker(object(), text_detector=detector)
        with patch("lamp_core.object_localization.locate_page", return_value=reading) as locate:
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        locate.assert_called_once()
        self.assertEqual("semantic_book_candidate_picked", tracker.status)

    def test_local_text_detection_is_on_by_default_and_can_be_disabled(self):
        default = camera_preview.build_parser().parse_args([])
        self.assertTrue(default.local_text_detection)
        self.assertEqual(320, default.text_detect_long_side)
        disabled = camera_preview.build_parser().parse_args(["--no-local-text-detection"])
        self.assertFalse(disabled.local_text_detection)


class StickyAnchorSourceTests(unittest.TestCase):
    """A text-line box and a page box do not share a centre.

    Switching source on every refresh moved the aim point between the two and
    made the servo hunt, so whichever source owns the target keeps it until it
    has missed three refreshes in a row.
    """

    _TEXT_BOXES = [TextBox((0.40, 0.12, 0.88, 0.27), 0.98)]

    def _locked_on_local(self, client=None):
        detector = _FakeTextDetector(list(self._TEXT_BOXES))
        tracker = camera_preview.SemanticBookTracker(client, text_detector=detector)
        tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual("local_text", tracker._anchor_source)
        return tracker, detector

    def test_a_local_miss_does_not_hand_the_target_to_the_cloud(self):
        tracker, detector = self._locked_on_local()
        locked = tracker._pending_bbox
        self.assertIsNotNone(locked)
        detector.boxes = []  # e.g. motion blur destroys the text detection
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "page", 0.9)
        with patch("lamp_core.object_localization.locate_page", return_value=reading) as locate:
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        locate.assert_not_called()
        # The queued box is untouched, so the running tracker keeps its aim.
        self.assertEqual(locked, tracker._pending_bbox)
        self.assertEqual("local_text", tracker._anchor_source)
        self.assertEqual("local_text_soft_miss_1_of_3", tracker.status)
        # A soft miss is a tracking hiccup, not an empty scene.
        self.assertEqual(0, tracker.api_miss_count)

    def test_three_consecutive_misses_release_the_source(self):
        tracker, detector = self._locked_on_local()
        detector.boxes = []
        for _ in range(3):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertIsNone(tracker._anchor_source)
        self.assertEqual("semantic_book_not_found_1_of_3", tracker.status)

    def test_released_source_lets_the_cloud_take_over(self):
        tracker, detector = self._locked_on_local()
        tracker.client = object()
        detector.boxes = []
        for _ in range(3):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertIsNone(tracker._anchor_source)
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "page", 0.9)
        with patch("lamp_core.object_localization.locate_page", return_value=reading):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual("cloud", tracker._anchor_source)
        self.assertEqual("semantic_book_candidate_picked", tracker.status)

    def test_a_cloud_owned_target_is_not_stolen_by_a_good_local_frame(self):
        detector = _FakeTextDetector([])
        tracker = camera_preview.SemanticBookTracker(object(), text_detector=detector)
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "page", 0.9)
        with patch("lamp_core.object_localization.locate_page", return_value=reading):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual("cloud", tracker._anchor_source)
        detector.boxes = list(self._TEXT_BOXES)
        calls_before = detector.calls
        with patch("lamp_core.object_localization.locate_page", return_value=reading):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        # Cloud still owns the target, so the local detector is not even asked.
        self.assertEqual(calls_before, detector.calls)
        self.assertEqual("cloud", tracker._anchor_source)

    def test_reset_clears_the_anchor_source(self):
        tracker, _ = self._locked_on_local()
        tracker.reset()
        self.assertIsNone(tracker._anchor_source)
        self.assertEqual(0, tracker._source_miss_count)

    def test_cloud_sanity_check_confirms_in_place_without_relocalizing(self):
        tracker = camera_preview.SemanticBookTracker(object())
        running = _FakeTracker()
        tracker._tracker = running
        tracker._anchor_source = "cloud"
        tracker._anchor_bbox = (0.20, 0.20, 0.80, 0.80)
        tracker._tracked_bbox_norm = (0.25, 0.20, 0.75, 0.80)
        with (
            patch(
                "lamp_core.object_localization.confirm_target_present",
                return_value=True,
            ) as confirm,
            patch("lamp_core.object_localization.locate_page") as locate,
        ):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        confirm.assert_called_once()
        locate.assert_not_called()
        self.assertIs(running, tracker._tracker)
        self.assertIsNone(tracker._pending_bbox)
        self.assertEqual("semantic_book_confirmed_in_place", tracker.status)
        self.assertEqual((0.20, 0.20, 0.80, 0.80), tracker._anchor_bbox)

    def test_three_failed_crop_checks_drop_the_tracker_before_global_search(self):
        tracker = camera_preview.SemanticBookTracker(object())
        tracker._tracker = _FakeTracker()
        tracker._anchor_source = "cloud"
        tracker._anchor_bbox = (0.20, 0.20, 0.80, 0.80)
        tracker._tracked_bbox_norm = (0.25, 0.20, 0.75, 0.80)
        with (
            patch(
                "lamp_core.object_localization.confirm_target_present",
                return_value=False,
            ) as confirm,
            patch("lamp_core.object_localization.locate_page") as locate,
        ):
            for _ in range(camera_preview.SemanticBookTracker.SOURCE_MISS_LIMIT):
                tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertEqual(3, confirm.call_count)
        locate.assert_not_called()
        self.assertIsNone(tracker._tracker)
        self.assertIsNone(tracker._anchor_source)
        self.assertTrue(tracker.help_requested)


class EdgeRecentreTests(unittest.TestCase):
    """Regression tests for a clipped target freezing the J1 axis.

    Recording every sighting as "found" kept a clipped box inside the hold
    window for ever, so the axis sat still with the book half out of frame and
    the search below could never run.
    """

    @staticmethod
    def _follower(**overrides):
        follower = camera_preview.RealtimeJ1Follower.__new__(
            camera_preview.RealtimeJ1Follower
        )
        follower.gear_ratio = 1.0
        follower.positive_camera_direction = HorizontalDirection.LEFT
        follower.envelope_degrees = 10.0
        follower.search_degrees = 20.0
        follower.travel_degrees = 40.0
        follower.servo_kp = 120.0
        follower.servo_ki = 60.0
        follower.servo_kd = 10.0
        follower.servo_deadband = 0.03
        # These cases are about the velocity path, so keep direct aim out of them.
        follower.direct_aim = False
        follower.aim_settle_seconds = 0.8
        follower.aim_max_step_degrees = 8.0
        follower._aim_hold_until = 0.0
        follower._last_aim_delta = 0.0
        follower._aim_error_at_move = None
        follower._edge_best_error = None
        follower._edge_stall_count = 0
        follower._validate_hold_started = None
        follower._search_restarts = 0
        follower.degrees_per_error = 44.5
        follower.scan_step_degrees = 10.0
        follower.scan_dwell_seconds = 3.0
        follower.hold_seconds = 4.0
        follower.speed_rpm = 12
        follower.acceleration = 120
        follower.minimum_interval_s = 0.125
        # Mirrors __init__: speed_rpm / gear_ratio * 6.0 = 72 deg/s here.
        follower._servo = VisualServoPid(
            kp=120.0, ki=60.0, kd=10.0, max_velocity_degrees_s=72.0
        )
        follower._transport = _FakeTransport()
        follower._mode = ChecksumMode.ADDITIVE
        follower.node_id = 1
        follower.anchor_counts = 0
        follower._last_send_at = 0.0
        follower._last_step_at = time.monotonic()
        follower._last_target_counts = None
        follower._search_started_at = time.monotonic()
        follower._last_found_at = 0.0
        follower._edge_started_at = None
        follower._last_good_degrees = 0.0
        follower._offset_degrees = 0.0
        follower._scan = None
        follower._scan_target = None
        follower._scan_seen_pick = 0
        follower._scan_dwell_started = 0.0
        follower._scan_arrival_not_before = 0.0
        for name, value in overrides.items():
            setattr(follower, name, value)
        return follower

    @staticmethod
    def _alignment(bbox_norm, **extra):
        alignment = {
            "found": bbox_norm is not None,
            "bbox_norm": list(bbox_norm) if bbox_norm else [0.0, 0.0, 0.0, 0.0],
            "priority": "semantic_book",
            "message": "",
        }
        alignment.update(extra)
        return alignment

    def test_unclipped_target_is_followed(self):
        follower = self._follower()
        alignment = self._alignment((0.30, 0.30, 0.70, 0.70))
        follower.update(alignment)
        self.assertEqual("book_follow", alignment["tracking_mode"])

    def test_clipped_target_is_servoed_instead_of_frozen(self):
        follower = self._follower(_last_found_at=time.monotonic())
        alignment = self._alignment((0.0, 0.1417, 0.3198, 0.3097))
        follower.update(alignment)
        self.assertEqual("edge_recentre", alignment["tracking_mode"])
        # Clipped at the left, so the visible centre is left of frame centre and
        # the correction must drive the image to the right, i.e. a positive move.
        self.assertGreater(alignment["j1_velocity_degrees_s"], 0.0)
        self.assertGreater(follower._offset_degrees, 0.0)

    def test_clipped_on_the_right_moves_the_other_way(self):
        follower = self._follower(_last_found_at=time.monotonic())
        alignment = self._alignment((0.68, 0.20, 1.0, 0.60))
        follower.update(alignment)
        self.assertEqual("edge_recentre", alignment["tracking_mode"])
        self.assertLess(alignment["j1_velocity_degrees_s"], 0.0)
        self.assertLess(follower._offset_degrees, 0.0)

    def test_clipped_target_without_a_good_pose_holds_instead_of_blind_scanning(self):
        now = time.monotonic()
        follower = self._follower(
            _last_found_at=now - 30.0, _edge_started_at=now - 30.0
        )
        alignment = self._alignment((0.0, 0.1417, 0.3198, 0.3097), pick_count=0)
        follower.update(alignment)
        self.assertEqual("visible_hold", alignment["tracking_mode"])
        self.assertIsNone(follower._scan)

    def test_lost_target_holds_the_pose_that_worked(self):
        follower = self._follower(
            _last_found_at=time.monotonic(), _last_good_degrees=7.5
        )
        alignment = self._alignment(None)
        follower.update(alignment)
        self.assertEqual("centred_hold", alignment["tracking_mode"])
        self.assertAlmostEqual(7.5, follower._offset_degrees, places=6)

    def test_pending_detection_freezes_the_axis(self):
        follower = self._follower(_last_found_at=time.monotonic())
        alignment = self._alignment((0.0, 0.1417, 0.3198, 0.3097), awaiting_confirmation=True)
        follower.update(alignment)
        self.assertEqual("validate_hold", alignment["tracking_mode"])
        self.assertAlmostEqual(0.0, follower._offset_degrees, places=6)

    def test_pending_detection_does_not_starve_the_search(self):
        # Detections are issued back to back, so freezing on every one of them
        # held the axis still for as long as the book stayed out of view.
        now = time.monotonic()
        follower = self._follower(_last_found_at=now - 30.0)
        alignment = self._alignment(None, awaiting_confirmation=True, pick_count=0)
        follower.update(alignment)
        self.assertEqual("narrowing_scan", alignment["tracking_mode"])

    def test_edge_budget_is_bounded_by_the_hold_window(self):
        now = time.monotonic()
        follower = self._follower()
        self.assertTrue(follower._edge_recentre_active(now))
        self.assertTrue(follower._edge_recentre_active(now + 7.9))
        self.assertFalse(follower._edge_recentre_active(now + 8.1))

    def test_unclipped_sighting_clears_the_edge_budget(self):
        follower = self._follower(_edge_started_at=time.monotonic())
        follower.update(self._alignment((0.30, 0.30, 0.70, 0.70)))
        self.assertIsNone(follower._edge_started_at)


class SettledTargetRefreshTests(unittest.TestCase):
    """The end-game sluggishness: a settled loop kept being disturbed.

    Successive anchors move the box edge by about 0.17 of the frame on the real
    rig, so re-anchoring every couple of seconds while the aim was already good
    injected a fresh step for the servo to chase and it never came to rest.
    """

    def _tracker(self, age_seconds: float = 3.0):
        tracker = camera_preview.SemanticBookTracker(None, reacquire_seconds=2.0)
        tracker._last_request_at = time.monotonic() - age_seconds
        return tracker

    def _refresh(self, tracker, *, at_edge, settled):
        with patch.object(
            camera_preview.SemanticBookTracker, "_request_confirm"
        ) as request:
            tracker._maybe_refresh_target(
                _FakeImage(), _FakeCV(), at_edge=at_edge, settled=settled
            )
        return request

    def test_a_settled_target_is_left_alone(self):
        request = self._refresh(self._tracker(3.0), at_edge=False, settled=True)
        request.assert_not_called()

    def test_an_unsettled_target_is_still_refreshed(self):
        request = self._refresh(self._tracker(3.0), at_edge=False, settled=False)
        request.assert_called_once()

    def test_a_clipped_target_is_refreshed_even_when_settled(self):
        # Clipped means a new box is genuinely needed to re-frame the page.
        request = self._refresh(self._tracker(3.0), at_edge=True, settled=True)
        request.assert_called_once()

    def test_a_settled_target_is_re_checked_after_the_sanity_interval(self):
        tracker = self._tracker(
            camera_preview.SemanticBookTracker.SETTLED_SANITY_SECONDS + 1.0
        )
        request = self._refresh(tracker, at_edge=False, settled=True)
        request.assert_called_once()

    def test_a_centred_tracked_box_does_not_queue_a_new_anchor(self):
        tracker = camera_preview.SemanticBookTracker(None, reacquire_seconds=2.0)
        tracker._tracker = _FakeTracker()
        # 640-wide frame: centre of (256, 100, 128, 100) is exactly 0.5.
        tracker._tracker._tracked = (256.0, 100.0, 128.0, 100.0)
        tracker._last_request_at = time.monotonic() - 3.0
        with patch.object(
            camera_preview.SemanticBookTracker, "_request_confirm"
        ) as request:
            tracked = tracker.update(_FakeImage(), _FakeCV())
        self.assertEqual((256, 100, 128, 100), tracked)
        request.assert_not_called()

    def test_an_off_centre_tracked_box_does_queue_a_new_anchor(self):
        tracker = camera_preview.SemanticBookTracker(None, reacquire_seconds=2.0)
        tracker._tracker = _FakeTracker()
        tracker._tracker._tracked = (400.0, 100.0, 128.0, 100.0)
        tracker._last_request_at = time.monotonic() - 3.0
        with patch.object(
            camera_preview.SemanticBookTracker, "_request_confirm"
        ) as request:
            tracker.update(_FakeImage(), _FakeCV())
        request.assert_called_once()


class AnchorSmoothingTests(unittest.TestCase):
    """Successive anchors jitter by ~0.07-0.17 of the frame on the real rig.

    That jitter was reaching the servo as a fresh step every refresh, so the loop
    chased its own measurement instead of coming to rest.
    """

    def _tracker(self, tracking: bool = True):
        tracker = camera_preview.SemanticBookTracker(None)
        if tracking:
            tracker._tracker = _FakeTracker()
        return tracker

    def test_first_anchor_is_taken_as_it_is(self):
        tracker = self._tracker()
        box = (0.40, 0.20, 0.80, 0.40)
        self.assertEqual(box, tracker._accept_anchor(box))
        self.assertEqual(box, tracker._anchor_bbox)

    def test_a_nearby_anchor_is_blended_not_jumped_to(self):
        tracker = self._tracker()
        tracker._accept_anchor((0.40, 0.20, 0.80, 0.40))
        blended = tracker._accept_anchor((0.44, 0.20, 0.84, 0.40))
        # Offered centre was 0.64; blending halfway lands on 0.62, so the servo
        # sees half the step it otherwise would.
        self.assertAlmostEqual(0.62, (blended[0] + blended[2]) / 2.0, places=6)
        self.assertLess(blended[2], 0.84)
        self.assertGreater(blended[2], 0.80)

    def test_a_jumped_anchor_is_held_out_until_it_repeats(self):
        tracker = self._tracker()
        tracker._accept_anchor((0.10, 0.20, 0.30, 0.40))
        jumped = (0.60, 0.20, 0.90, 0.40)
        self.assertIsNone(tracker._accept_anchor(jumped))
        self.assertEqual(jumped, tracker._accept_anchor(jumped))

    def test_a_jump_that_does_not_repeat_never_wins(self):
        tracker = self._tracker()
        tracker._accept_anchor((0.10, 0.20, 0.30, 0.40))
        self.assertIsNone(tracker._accept_anchor((0.60, 0.20, 0.90, 0.40)))
        # A third, unrelated position is a new candidate, not a confirmation.
        self.assertIsNone(tracker._accept_anchor((0.20, 0.70, 0.40, 0.90)))
        self.assertAlmostEqual(
            0.20, (tracker._anchor_bbox[0] + tracker._anchor_bbox[2]) / 2.0, places=6
        )

    def test_no_tracker_means_no_screening(self):
        # With nothing tracking, refusing the candidate would never acquire.
        tracker = self._tracker(tracking=False)
        tracker._accept_anchor((0.10, 0.20, 0.30, 0.40))
        jumped = (0.60, 0.20, 0.90, 0.40)
        self.assertEqual(jumped, tracker._accept_anchor(jumped))

    def test_reset_clears_the_anchor_state(self):
        tracker = self._tracker()
        tracker._accept_anchor((0.10, 0.20, 0.30, 0.40))
        tracker.reset()
        self.assertIsNone(tracker._anchor_bbox)
        self.assertIsNone(tracker._anchor_candidate)
        self.assertEqual(0, tracker._anchor_candidate_count)

    def test_a_jumped_anchor_does_not_replace_a_running_tracker(self):
        detector = _FakeTextDetector([TextBox((0.40, 0.12, 0.88, 0.27), 0.98)])
        tracker = camera_preview.SemanticBookTracker(None, text_detector=detector)
        tracker._tracker = _FakeTracker()
        tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertIsNotNone(tracker._pending_bbox)
        tracker._pending_bbox = None
        detector.boxes = [TextBox((0.02, 0.70, 0.30, 0.90), 0.90)]
        tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertIsNone(tracker._pending_bbox)
        self.assertEqual("anchor_outlier_held", tracker.status)

    def test_releasing_the_source_drops_the_stale_anchor(self):
        detector = _FakeTextDetector([TextBox((0.40, 0.12, 0.88, 0.27), 0.98)])
        tracker = camera_preview.SemanticBookTracker(None, text_detector=detector)
        tracker._tracker = _FakeTracker()
        tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertIsNotNone(tracker._anchor_bbox)
        tracker.client = object()
        detector.boxes = []
        for _ in range(camera_preview.SemanticBookTracker.SOURCE_MISS_LIMIT):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        self.assertIsNone(tracker._anchor_bbox)
        reading = PageReading((0.1, 0.2, 0.9, 0.8), "page", 0.9)
        with patch("lamp_core.object_localization.locate_page", return_value=reading):
            tracker._pick_candidate(_FakeImage(), _FakeCV())
        # The cloud's box is accepted even though it sits far from the old anchor.
        self.assertEqual("cloud", tracker._anchor_source)


class DirectAimTests(unittest.TestCase):
    """Direct aim uses the calibrated field of view instead of integrating.

    One unit of normalized image error is worth ``degrees_per_error`` of joint
    rotation, so the correction can be computed outright rather than crawled
    toward. Moves are spaced by a settle gap because the image lags the command.
    """

    @staticmethod
    def _follower(**overrides):
        overrides.setdefault("direct_aim", True)
        return EdgeRecentreTests._follower(**overrides)

    def test_one_computed_move_nulls_the_error(self):
        follower = self._follower()
        # Centre 0.65, so the error is +0.15 and LEFT commands negative degrees.
        alignment = self._follower_alignment((0.55, 0.30, 0.75, 0.70))
        follower.update(alignment)
        self.assertAlmostEqual(-0.15 * 44.5, follower._offset_degrees, places=6)
        # The reported delta is rounded for the diagnostics payload, so allow the
        # rounding step rather than pinning the last bit of a float.
        self.assertAlmostEqual(
            -0.15 * 44.5, alignment["aim_delta_degrees"], delta=0.011
        )

    def test_the_other_direction_moves_the_other_way(self):
        follower = self._follower()
        follower.update(self._follower_alignment((0.25, 0.30, 0.45, 0.70)))
        self.assertAlmostEqual(+0.15 * 44.5, follower._offset_degrees, places=6)

    def test_one_move_is_capped_so_a_lagging_reading_cannot_run_away(self):
        # Measured on the rig: after a large fast move the tracker lags, so the
        # error stays stale for seconds. An uncapped move was repeated on that
        # stale reading until the axis hit its travel stop.
        follower = self._follower()
        alignment = self._follower_alignment((0.70, 0.30, 0.90, 0.70))
        follower.update(alignment)
        self.assertAlmostEqual(-8.0, follower._offset_degrees, places=6)
        self.assertAlmostEqual(-8.0, alignment["aim_delta_degrees"], places=6)

    def test_a_stale_reading_is_not_corrected_twice(self):
        follower = self._follower()
        alignment = self._follower_alignment((0.70, 0.30, 0.90, 0.70))
        follower.update(alignment)
        after_first = follower._offset_degrees
        # Same reading again, well past the settle window: the tracker has not
        # caught up, so this must not move the axis a second time.
        follower._aim_hold_until = 0.0
        follower.update(self._follower_alignment((0.70, 0.30, 0.90, 0.70)))
        self.assertAlmostEqual(after_first, follower._offset_degrees, places=6)

    def test_a_changed_reading_is_corrected_again(self):
        follower = self._follower()
        follower.update(self._follower_alignment((0.70, 0.30, 0.90, 0.70)))
        after_first = follower._offset_degrees
        self.assertAlmostEqual(-8.0, after_first, places=6)
        # The image caught up: the error is now smaller and still outside the
        # deadband, so direct aim takes the next step -- also capped at 8 deg.
        follower._aim_hold_until = 0.0
        follower.update(self._follower_alignment((0.60, 0.30, 0.80, 0.70)))
        self.assertLess(follower._offset_degrees, after_first)
        self.assertAlmostEqual(-16.0, follower._offset_degrees, places=6)

    def test_a_move_is_not_repeated_before_the_image_catches_up(self):
        follower = self._follower()
        alignment = self._follower_alignment((0.60, 0.30, 0.80, 0.70))
        follower.update(alignment)
        after_first = follower._offset_degrees
        # Same stale error on the very next frame: the axis has not moved yet, so
        # correcting again would overshoot.
        follower.update(self._follower_alignment((0.60, 0.30, 0.80, 0.70)))
        self.assertAlmostEqual(after_first, follower._offset_degrees, places=6)

    def test_an_error_inside_the_deadband_commands_nothing(self):
        follower = self._follower()
        # Centre 0.52, inside the 0.03 tolerance.
        follower.update(self._follower_alignment((0.42, 0.30, 0.62, 0.70)))
        self.assertAlmostEqual(0.0, follower._offset_degrees, places=6)
        self.assertAlmostEqual(0.0, follower._last_aim_delta, places=6)

    def test_a_move_is_capped_by_the_travel_limit(self):
        follower = self._follower(travel_degrees=5.0)
        # Centre 0.91 asks for 18.2 deg, past both caps. Kept clear of the frame
        # edge so this exercises aiming rather than edge recentring.
        follower.update(self._follower_alignment((0.86, 0.30, 0.96, 0.70)))
        self.assertAlmostEqual(-5.0, follower._offset_degrees, places=6)

    @staticmethod
    def _follower_alignment(bbox_norm):
        return {
            "found": True,
            "bbox_norm": list(bbox_norm),
            "priority": "semantic_book",
            "message": "",
        }

    def test_reanchor_forgets_a_pending_aim_hold(self):
        follower = self._follower()
        follower._aim_hold_until = time.monotonic() + 100.0
        follower.reanchor()
        self.assertEqual(0.0, follower._aim_hold_until)


class CaptureLoopContractTests(unittest.TestCase):
    """The capture loop reaches into the follower, and a typo there kills it.

    This happened for real: the loop asked for ``self.follower`` while the
    attribute is ``j1_follower``, so the very first frame raised and the whole
    preview served nothing but the error.
    """

    def test_the_follower_is_reachable_under_the_name_the_loop_uses(self):
        follower = EdgeRecentreTests._follower()
        stream = camera_preview.CameraStream(960, 720, 10.0, 82, 0, follower)
        self.assertIs(follower, stream.j1_follower)
        self.assertEqual(0.03, stream.j1_follower.servo_deadband)

    def test_a_stream_without_a_follower_still_has_the_attribute(self):
        stream = camera_preview.CameraStream(960, 720, 10.0, 82, 0)
        self.assertIsNone(stream.j1_follower)

    def test_stop_marks_the_stream_finished_so_mjpeg_clients_can_exit(self):
        stream = camera_preview.CameraStream(960, 720, 10.0, 82, 0)
        stream.stop()
        self.assertTrue(stream._stopping.is_set())
        self.assertEqual("preview stopped", stream.error)

    def test_annotate_bbox_accepts_the_servo_deadband(self):
        image, alignment = camera_preview.annotate_bbox(
            _FakeImage(), _FakeCV(), (100, 100, 200, 200), "semantic_book", 0.03
        )
        self.assertIsNotNone(image)
        self.assertTrue(alignment["found"])


class EdgeRecentreBoundTests(unittest.TestCase):
    """The edge path must be bounded exactly like normal following.

    Measured on the rig: an unbounded velocity here drove 40 deg to the travel
    stop in three seconds while the target stayed clipped the whole way, so the
    push achieved nothing and left the axis on the wrong side of the book.
    """

    @staticmethod
    def _follower(**overrides):
        overrides.setdefault("direct_aim", True)
        return EdgeRecentreTests._follower(**overrides)

    @staticmethod
    def _alignment(bbox_norm):
        return {
            "found": True,
            "bbox_norm": list(bbox_norm),
            "priority": "semantic_book",
            "message": "",
        }

    def test_an_edge_move_is_capped_like_a_normal_one(self):
        follower = self._follower()
        # Clipped at the right with a 0.40 error, which would ask for 17.8 deg.
        follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        self.assertAlmostEqual(-8.0, follower._offset_degrees, places=6)

    def test_a_stale_edge_reading_is_not_corrected_twice(self):
        follower = self._follower()
        follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        after_first = follower._offset_degrees
        follower._aim_hold_until = 0.0
        follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        self.assertAlmostEqual(after_first, follower._offset_degrees, places=6)

    def test_repeated_frames_do_not_count_as_repeated_motor_moves(self):
        follower = self._follower()
        follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        after_first = follower._offset_degrees
        # The reading never improves, but these are still frames from the same
        # physical move and must not consume three separate attempts.
        for _ in range(camera_preview.RealtimeJ1Follower.EDGE_STALL_LIMIT):
            follower._aim_hold_until = 0.0
            follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        self.assertAlmostEqual(after_first, follower._offset_degrees, places=6)
        self.assertEqual(0, follower._edge_stall_count)
        # The time budget, rather than frame count, eventually hands control to
        # search when the single verified correction did not help.
        follower._edge_started_at = time.monotonic() - follower.hold_seconds * 2.0 - 1.0
        self.assertFalse(follower._edge_recentre_active(time.monotonic(), 0.40))
        alignment = self._alignment((0.81, 0.14, 1.00, 0.28))
        follower.update(alignment)
        self.assertEqual("visible_hold", alignment["tracking_mode"])

    def test_a_vertically_cropped_book_is_still_followed_horizontally(self):
        follower = self._follower()
        alignment = self._alignment((0.60, 0.00, 0.90, 0.80))
        follower.update(alignment)
        self.assertEqual("book_follow", alignment["tracking_mode"])
        self.assertLess(follower._offset_degrees, 0.0)

    def test_a_visible_book_never_falls_into_a_blind_scan(self):
        follower = self._follower()
        follower._edge_started_at = time.monotonic() - follower.hold_seconds * 2.0 - 1.0
        alignment = self._alignment((0.81, 0.14, 1.00, 0.78))
        follower.update(alignment)
        self.assertEqual("visible_hold", alignment["tracking_mode"])
        self.assertIsNone(follower._scan)

    def test_progress_keeps_the_edge_attempt_alive(self):
        follower = self._follower()
        follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        # The image caught up and the target is less clipped, so keep going.
        follower._aim_hold_until = 0.0
        alignment = self._alignment((0.70, 0.14, 0.98, 0.30))
        follower.update(alignment)
        self.assertEqual("edge_recentre", alignment["tracking_mode"])
        self.assertEqual(0, follower._edge_stall_count)

    def test_a_full_sighting_resets_the_edge_episode(self):
        follower = self._follower()
        follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        self.assertIsNotNone(follower._edge_best_error)
        follower.update(self._alignment((0.30, 0.30, 0.70, 0.70)))
        self.assertIsNone(follower._edge_started_at)
        self.assertIsNone(follower._edge_best_error)
        self.assertEqual(0, follower._edge_stall_count)

    def test_reanchor_clears_the_edge_episode(self):
        follower = self._follower()
        follower.update(self._alignment((0.81, 0.14, 1.00, 0.28)))
        follower.reanchor()
        self.assertIsNone(follower._edge_started_at)
        self.assertIsNone(follower._edge_best_error)
        self.assertEqual(0, follower._edge_stall_count)


class ValidateHoldBoundTests(unittest.TestCase):
    """Waiting for a detection must not freeze the axis indefinitely.

    Cloud calls take over a second and are re-issued every couple of seconds, so
    the wait was almost always active: the axis once sat frozen for 36 seconds
    while the book stayed clipped at the frame edge.
    """

    @staticmethod
    def _follower(**overrides):
        return EdgeRecentreTests._follower(**overrides)

    @staticmethod
    def _alignment(bbox_norm, **extra):
        alignment = {
            "found": True,
            "bbox_norm": list(bbox_norm),
            "priority": "semantic_book",
            "message": "",
        }
        alignment.update(extra)
        return alignment

    _CLIPPED = (0.82, 0.14, 1.00, 0.79)

    def test_a_pending_detection_holds_the_axis_at_first(self):
        follower = self._follower()
        alignment = self._alignment(self._CLIPPED, awaiting_confirmation=True)
        follower.update(alignment)
        self.assertEqual("validate_hold", alignment["tracking_mode"])

    def test_the_wait_expires_so_the_edge_logic_can_run(self):
        follower = self._follower()
        follower._validate_hold_started = time.monotonic() - (
            camera_preview.RealtimeJ1Follower.VALIDATE_HOLD_SECONDS + 1.0
        )
        alignment = self._alignment(self._CLIPPED, awaiting_confirmation=True)
        follower.update(alignment)
        self.assertNotEqual("validate_hold", alignment["tracking_mode"])
        self.assertEqual("edge_recentre", alignment["tracking_mode"])

    def test_the_budget_restarts_once_no_detection_is_in_flight(self):
        follower = self._follower()
        follower._validate_hold_started = time.monotonic() - 100.0
        follower.update(self._alignment(self._CLIPPED, awaiting_confirmation=False))
        self.assertIsNone(follower._validate_hold_started)
        alignment = self._alignment(self._CLIPPED, awaiting_confirmation=True)
        follower.update(alignment)
        self.assertEqual("validate_hold", alignment["tracking_mode"])

    def test_reanchor_clears_the_wait(self):
        follower = self._follower()
        follower._validate_hold_started = time.monotonic()
        follower.reanchor()
        self.assertIsNone(follower._validate_hold_started)


class SearchPolicyTests(unittest.TestCase):
    """The sweep must reach a book that is a long way off.

    Measured on the rig: with the lamp nudged, the book sat 40 degrees away while
    the opening sweep only ever sampled 0, +-10 and +-20.
    """

    @staticmethod
    def _follower(**overrides):
        return EdgeRecentreTests._follower(**overrides)

    @staticmethod
    def _finished_scan(score):
        scan = NarrowingScan(
            envelope_degrees=20.0, step_degrees=10.0, minimum_step_degrees=10.0
        )
        while scan.next_target() is not None:
            scan.record(score)
        return scan

    @staticmethod
    def _alignment(**extra):
        alignment = {"pick_count": 0, "page_score": 0.0}
        alignment.update(extra)
        return alignment

    def test_the_search_defaults_to_the_whole_travel(self):
        # search_degrees defaulting to None is what makes it follow the travel.
        self.assertIsNone(camera_preview.build_parser().parse_args([]).j1_search_degrees)

    def test_a_salient_third_leads_the_sweep(self):
        follower = self._follower()
        follower.search_degrees = 40.0
        follower.scan_step_degrees = 10.0
        # A third's centre sits at 1/6, 1/2 or 5/6, so aiming it at the middle is
        # an ordinary aim correction worth about a third of the field of view.
        left = follower._new_scan(self._alignment(salient_third=0))
        self.assertAlmostEqual(44.5 / 3.0, left.next_target(), places=2)
        right = follower._new_scan(self._alignment(salient_third=2))
        self.assertAlmostEqual(-44.5 / 3.0, right.next_target(), places=2)
        centre = follower._new_scan(self._alignment())
        self.assertAlmostEqual(0.0, centre.next_target(), places=6)

    def test_a_finished_sweep_is_never_scored_again(self):
        follower = self._follower()
        scan = self._finished_scan(1.0)
        follower._scan = scan
        follower._scan_target = scan.best_degrees
        follower._scan_seen_pick = 0
        follower._scan_dwell_started = 0.0
        alignment = self._alignment(pick_count=1)
        # A new pick used to be recorded against the finished round, which raises.
        follower._advance_narrowing_scan(alignment, time.monotonic())
        self.assertEqual("narrowing_scan", alignment["tracking_mode"])

    def test_a_sweep_that_found_nothing_sweeps_again(self):
        follower = self._follower()
        scan = self._finished_scan(0.0)
        follower._scan = scan
        follower._scan_target = 0.0
        follower._scan_seen_pick = 0
        follower._scan_dwell_started = 0.0
        alignment = self._alignment(pick_count=1)
        follower._advance_narrowing_scan(alignment, time.monotonic())
        self.assertEqual(1, follower._search_restarts)
        self.assertEqual(1, alignment["scan_restarts"])
        self.assertFalse(follower._scan.finished)
        self.assertIsNotNone(follower._scan_target)

    def test_a_restart_shifts_the_samples_by_half_a_step(self):
        follower = self._follower()
        follower._search_restarts = 1
        follower.scan_step_degrees = 10.0
        follower.search_degrees = 20.0
        scan = follower._new_scan(self._alignment(salient_third=0))
        self.assertAlmostEqual(5.0, scan.next_target(), places=6)

    def test_restarts_are_capped_so_a_hopeless_sweep_settles(self):
        follower = self._follower()
        follower._search_restarts = camera_preview.RealtimeJ1Follower.SEARCH_RESTART_LIMIT
        scan = self._finished_scan(0.0)
        follower._scan = scan
        follower._scan_target = 0.0
        follower._scan_seen_pick = 0
        follower._scan_dwell_started = 0.0
        alignment = self._alignment(pick_count=1)
        follower._advance_narrowing_scan(alignment, time.monotonic())
        self.assertEqual(
            camera_preview.RealtimeJ1Follower.SEARCH_RESTART_LIMIT,
            follower._search_restarts,
        )
        self.assertAlmostEqual(0.0, follower._offset_degrees, places=6)
        self.assertTrue(follower._scan.finished)

    def test_a_real_sighting_clears_the_restart_count(self):
        follower = self._follower()
        follower._search_restarts = 2
        follower.update(
            {
                "found": True,
                "bbox_norm": [0.30, 0.30, 0.70, 0.70],
                "priority": "semantic_book",
                "message": "",
            }
        )
        self.assertEqual(0, follower._search_restarts)


class ConfirmationTrustTests(unittest.TestCase):
    """An unconfirmed tracking box must not be followed for ever.

    Measured on the rig: no detector confirmed anything for 60 seconds while the
    local tracker drifted onto a laptop, and the loop went on reporting
    "centred" with the book nowhere in frame.
    """

    @staticmethod
    def _tracker(**overrides):
        tracker = camera_preview.SemanticBookTracker(None)
        tracker._tracker = _FakeTracker()
        tracker._tracker._tracked = (200.0, 150.0, 200.0, 150.0)
        # Keep the re-acquire timer quiet so no picker thread runs and the status
        # stays the one under test.
        tracker._last_request_at = time.monotonic()
        for name, value in overrides.items():
            setattr(tracker, name, value)
        return tracker

    def test_a_recently_confirmed_box_is_still_followed(self):
        tracker = self._tracker()
        self.assertIsNotNone(tracker.update(_FakeImage(), _FakeCV(), allow_reacquire=False))

    def test_an_unconfirmed_box_is_dropped(self):
        tracker = self._tracker(
            _last_confirmed_at=time.monotonic()
            - camera_preview.SemanticBookTracker.CONFIRMATION_TIMEOUT_SECONDS
            - 1.0
        )
        self.assertIsNone(tracker.update(_FakeImage(), _FakeCV(), allow_reacquire=False))
        self.assertIsNone(tracker._tracker)
        self.assertEqual("target_unconfirmed_searching", tracker.status)

    def test_a_late_but_bounded_refresh_does_not_expire_the_tracker(self):
        tracker = self._tracker(
            _last_confirmed_at=time.monotonic()
            - camera_preview.SemanticBookTracker.CONFIRMATION_TIMEOUT_SECONDS
            - 1.0,
            _locating=True,
            _last_request_at=time.monotonic() - 2.0,
        )
        self.assertIsNotNone(
            tracker.update(_FakeImage(), _FakeCV(), allow_reacquire=False)
        )

    def test_dropping_it_also_releases_the_stale_anchor(self):
        tracker = self._tracker(
            _last_confirmed_at=time.monotonic()
            - camera_preview.SemanticBookTracker.CONFIRMATION_TIMEOUT_SECONDS
            - 1.0
        )
        tracker._anchor_bbox = (0.2, 0.2, 0.8, 0.5)
        tracker._anchor_source = "local_text"
        tracker.update(_FakeImage(), _FakeCV(), allow_reacquire=False)
        self.assertIsNone(tracker._anchor_bbox)
        self.assertIsNone(tracker._anchor_source)

    def test_a_fresh_confirmation_restarts_the_clock(self):
        tracker = self._tracker(
            _last_confirmed_at=time.monotonic()
            - camera_preview.SemanticBookTracker.CONFIRMATION_TIMEOUT_SECONDS
            - 1.0
        )
        # A pick that found the page is a confirmation, so the box survives.
        tracker._pending_bbox = (100, 100, 300, 300)
        self.assertIsNotNone(tracker.update(_FakeImage(), _FakeCV(), allow_reacquire=False))
        self.assertFalse(
            tracker._confirmation_expired(time.monotonic())
        )

    def test_reset_leaves_a_usable_confirmation_clock(self):
        tracker = self._tracker(
            _last_confirmed_at=time.monotonic() - 1000.0
        )
        tracker.reset()
        self.assertFalse(tracker._confirmation_expired(time.monotonic()))

    def test_a_detector_result_from_before_reset_is_discarded(self):
        detector = _FakeTextDetector(
            [TextBox((0.30, 0.10, 0.70, 0.30), 0.97)]
        )
        tracker = camera_preview.SemanticBookTracker(None, text_detector=detector)
        stale_generation = tracker._generation
        tracker.reset()
        tracker._pick_candidate(_FakeImage(), _FakeCV(), stale_generation)
        self.assertIsNone(tracker._pending_bbox)
        self.assertIsNone(tracker._anchor_bbox)


if __name__ == "__main__":
    unittest.main()
