"""Tests for the on-device PP-OCR text detector."""

import os
import tempfile
import unittest

from lamp_core.text_detection import (
    INPUT_MEAN,
    INPUT_SCALE,
    PpocrTextDetector,
    TextBox,
    candidate_model_paths,
    find_model_path,
    text_anchor_box,
    text_anchor_candidates,
)


class TextAnchorAssociationTests(unittest.TestCase):
    def test_previous_block_wins_over_a_new_larger_distant_block(self):
        boxes = [
            TextBox((0.08, 0.20, 0.42, 0.30), 0.96),
            TextBox((0.62, 0.20, 0.92, 0.28), 0.98),
        ]
        candidates = text_anchor_candidates(boxes, block_span=0.5)
        self.assertEqual(2, len(candidates))
        previous = (0.58, 0.17, 0.96, 0.31)
        selected = text_anchor_box(boxes, reference_bbox=previous, block_span=0.5)
        self.assertGreater((selected[0] + selected[2]) / 2.0, 0.5)


class _FakeFrame:
    def __init__(self, height: int, width: int) -> None:
        self.shape = (height, width, 3)


class _FakeDnnModel:
    """Records the configuration the detector applies and replays canned boxes."""

    def __init__(self, rectangles, confidences) -> None:
        self.rectangles = rectangles
        self.confidences = confidences
        self.mean = None
        self.scale = None
        self.input_size = None
        self.binary_threshold = None
        self.polygon_threshold = None
        self.max_candidates = None
        self.unclip_ratio = None

    def setBinaryThreshold(self, value) -> None:  # noqa: N802 - OpenCV API
        self.binary_threshold = value

    def setPolygonThreshold(self, value) -> None:  # noqa: N802 - OpenCV API
        self.polygon_threshold = value

    def setMaxCandidates(self, value) -> None:  # noqa: N802 - OpenCV API
        self.max_candidates = value

    def setUnclipRatio(self, value) -> None:  # noqa: N802 - OpenCV API
        self.unclip_ratio = value

    def setInputMean(self, value) -> None:  # noqa: N802 - OpenCV API
        self.mean = tuple(value)

    def setInputScale(self, value) -> None:  # noqa: N802 - OpenCV API
        self.scale = tuple(value)

    def setInputSize(self, width, height) -> None:  # noqa: N802 - OpenCV API
        self.input_size = (width, height)

    def setPreferableBackend(self, value) -> None:  # noqa: N802 - OpenCV API
        pass

    def setPreferableTarget(self, value) -> None:  # noqa: N802 - OpenCV API
        pass

    def detect(self, frame):
        return self.rectangles, self.confidences


class _FakeDnn:
    DNN_BACKEND_OPENCV = 3
    DNN_TARGET_CPU = 0

    def __init__(self, model: _FakeDnnModel) -> None:
        self.model = model

    def readNet(self, path):  # noqa: N802 - OpenCV API
        return object()

    def TextDetectionModel_DB(self, net):  # noqa: N802 - OpenCV API
        return self.model


class _FakeCv2:
    def __init__(self, model: _FakeDnnModel) -> None:
        self.dnn = _FakeDnn(model)
        self.resized_to = None

    def resize(self, image, size):
        self.resized_to = size
        return _FakeFrame(size[1], size[0])


def _detector_with(rectangles, confidences, **kwargs):
    handle = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    handle.write(b"stub")
    handle.close()
    model = _FakeDnnModel(rectangles, confidences)
    cv2 = _FakeCv2(model)
    detector = PpocrTextDetector(
        handle.name, cv2_module=cv2, long_side=kwargs.pop("long_side", 320), **kwargs
    )
    return detector, model, cv2, handle.name


class ModelPathTests(unittest.TestCase):
    def test_missing_model_reports_unavailable(self) -> None:
        detector = PpocrTextDetector("/definitely/not/here.onnx")
        self.assertFalse(detector.available)
        self.assertEqual(detector.detect(_FakeFrame(720, 960)), [])

    def test_detect_returns_nothing_without_model(self) -> None:
        detector = PpocrTextDetector("/definitely/not/here.onnx", cv2_module=_FakeCv2(_FakeDnnModel([], [])))
        self.assertEqual(detector.last_box_count, 0)
        self.assertEqual(detector.detect(_FakeFrame(720, 960)), [])

    def test_explicit_path_wins_and_expands_user(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
            name = handle.name
        try:
            self.assertEqual(find_model_path(explicit=name), name)
        finally:
            os.unlink(name)

    def test_search_paths_prefer_a_models_directory(self) -> None:
        paths = candidate_model_paths("x.onnx")
        self.assertTrue(any(path.endswith(os.path.join("models", "x.onnx")) for path in paths))


class FittedSizeTests(unittest.TestCase):
    def test_both_sides_are_multiples_of_32(self) -> None:
        for long_side in (320, 480, 640, 960):
            width, height = PpocrTextDetector.fitted_size(960, 720, long_side)
            self.assertEqual(width % 32, 0)
            self.assertEqual(height % 32, 0)
            self.assertEqual(max(width, height), min(long_side, 960))

    def test_matches_measured_pi_geometry(self) -> None:
        self.assertEqual(PpocrTextDetector.fitted_size(960, 720, 320), (320, 256))
        self.assertEqual(PpocrTextDetector.fitted_size(960, 720, 480), (480, 352))
        self.assertEqual(PpocrTextDetector.fitted_size(960, 720, 960), (960, 704))


class DetectTests(unittest.TestCase):
    def test_boxes_are_normalised_and_sorted_by_confidence(self) -> None:
        # (32, 32)-(288, 64) inside a 320x256 network input.
        rectangles = [[[32, 32], [288, 32], [288, 64], [32, 64]]]
        detector, _, cv2, name = _detector_with(rectangles, [0.93])
        try:
            boxes = detector.detect(_FakeFrame(720, 960))
        finally:
            os.unlink(name)
        self.assertEqual(cv2.resized_to, (320, 256))
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].bbox[0], 0.1, places=6)
        self.assertAlmostEqual(boxes[0].bbox[1], 32 / 256, places=6)
        self.assertAlmostEqual(boxes[0].bbox[2], 0.9, places=6)
        self.assertAlmostEqual(boxes[0].bbox[3], 64 / 256, places=6)
        self.assertAlmostEqual(boxes[0].confidence, 0.93, places=6)

    def test_paddleocr_normalisation_is_applied(self) -> None:
        # The regression that mattered: without mean/scale the DB head returns
        # near-noise, so the detector must configure exactly the zoo values.
        rectangles = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
        detector, model, _, name = _detector_with(rectangles, [0.9])
        try:
            detector.detect(_FakeFrame(720, 960))
        finally:
            os.unlink(name)
        self.assertEqual(model.mean, INPUT_MEAN)
        self.assertEqual(model.scale, INPUT_SCALE)
        self.assertEqual(model.input_size, (320, 256))

    def test_low_confidence_boxes_are_dropped(self) -> None:
        rectangles = [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[20, 20], [30, 20], [30, 30], [20, 30]],
        ]
        detector, _, _, name = _detector_with(rectangles, [0.2, 0.91])
        try:
            boxes = detector.detect(_FakeFrame(720, 960))
        finally:
            os.unlink(name)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].confidence, 0.91, places=6)

    def test_malformed_polygon_is_skipped(self) -> None:
        rectangles = ["not-a-polygon"]
        detector, _, _, name = _detector_with(rectangles, [0.9])
        try:
            boxes = detector.detect(_FakeFrame(720, 960))
        finally:
            os.unlink(name)
        self.assertEqual(boxes, [])

    def test_detect_records_timing_and_count(self) -> None:
        rectangles = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
        detector, _, _, name = _detector_with(rectangles, [0.9])
        try:
            detector.detect(_FakeFrame(720, 960))
        finally:
            os.unlink(name)
        self.assertEqual(detector.last_box_count, 1)
        self.assertGreaterEqual(detector.last_detect_seconds, 0.0)


class AnchorBoxTests(unittest.TestCase):
    def test_empty_input_has_no_anchor(self) -> None:
        self.assertIsNone(text_anchor_box([]))

    def test_top_edge_boxes_are_treated_as_background(self) -> None:
        # The monitor taskbar row measured at y 0.003-0.026 on a real frame. The
        # story text below it is full size, so it is the one that anchors.
        boxes = [
            TextBox((0.59, 0.003, 0.61, 0.026), 0.96),
            TextBox((0.40, 0.197, 0.88, 0.300), 0.98),
        ]
        anchor = text_anchor_box(boxes, margin=0.0)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor[0], 0.40, places=6)
        self.assertAlmostEqual(anchor[1], 0.197, places=6)

    def test_anchor_takes_the_dominant_block_and_its_neighbours(self) -> None:
        # Two stacked lines of one paragraph: the larger sets the block and the
        # smaller line sits under it, so both belong to the anchor.
        boxes = [
            TextBox((0.40, 0.10, 0.88, 0.22), 0.98),
            TextBox((0.55, 0.19, 0.69, 0.27), 0.98),
        ]
        anchor = text_anchor_box(boxes, margin=0.0)
        self.assertAlmostEqual(anchor[0], 0.40, places=6)
        self.assertAlmostEqual(anchor[1], 0.10, places=6)
        self.assertAlmostEqual(anchor[2], 0.88, places=6)
        self.assertAlmostEqual(anchor[3], 0.27, places=6)

    def test_a_stray_box_elsewhere_does_not_move_the_anchor(self) -> None:
        # Measured on a real frame: a false positive far to the right used to
        # drag the union centre a third of the frame away from the story text.
        boxes = [
            TextBox((0.388, 0.105, 0.884, 0.223), 0.98),
            TextBox((0.547, 0.191, 0.688, 0.266), 0.98),
            TextBox((0.891, 0.211, 0.978, 0.250), 0.97),
        ]
        anchor = text_anchor_box(boxes, margin=0.0)
        self.assertAlmostEqual(anchor[2], 0.884, places=6)
        self.assertLess((anchor[0] + anchor[2]) / 2.0, 0.70)

    def test_anchor_grows_by_margin(self) -> None:
        boxes = [TextBox((0.30, 0.20, 0.70, 0.40), 0.99)]
        tight = text_anchor_box(boxes, margin=0.0)
        grown = text_anchor_box(boxes, margin=0.10)
        self.assertLess(grown[0], tight[0])
        self.assertLess(grown[1], tight[1])
        self.assertGreater(grown[2], tight[2])
        self.assertGreater(grown[3], tight[3])

    def test_a_tiny_fragment_is_not_a_page_anchor(self) -> None:
        # Measured on the rig: a 0.07x0.08 fragment scoring 0.004 captured the
        # anchor and parked the axis against its travel stop, while real story
        # text measures 0.60-0.68 of the frame wide and scores 0.15-0.31.
        boxes = [TextBox((0.34, 0.07, 0.41, 0.15), 0.99)]
        self.assertIsNone(text_anchor_box(boxes))

    def test_a_tiny_fragment_does_not_drag_a_real_block(self) -> None:
        boxes = [
            TextBox((0.20, 0.11, 0.85, 0.35), 0.98),
            TextBox((0.34, 0.07, 0.41, 0.15), 0.99),
        ]
        anchor = text_anchor_box(boxes, margin=0.0)
        self.assertAlmostEqual(anchor[0], 0.20, places=6)
        self.assertAlmostEqual(anchor[2], 0.85, places=6)

    def test_a_small_but_real_line_still_anchors(self) -> None:
        # A speech bubble alone on a spread: small, but far above a fragment.
        boxes = [TextBox((0.71, 0.09, 1.00, 0.30), 0.95)]
        anchor = text_anchor_box(boxes, margin=0.0)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor[0], 0.71, places=6)

    def test_anchor_is_clamped_into_the_frame(self) -> None:
        boxes = [TextBox((0.02, 0.10, 0.98, 0.90), 0.99)]
        anchor = text_anchor_box(boxes, margin=0.5)
        self.assertGreaterEqual(anchor[0], 0.0)
        self.assertGreaterEqual(anchor[1], 0.0)
        self.assertLessEqual(anchor[2], 1.0)
        self.assertLessEqual(anchor[3], 1.0)

    def test_anchor_can_prefer_the_highest_confidence_subset(self) -> None:
        # Only the top-edge box is present, so there is nothing usable.
        boxes = [TextBox((0.10, 0.001, 0.90, 0.030), 0.99)]
        self.assertIsNone(text_anchor_box(boxes))


if __name__ == "__main__":
    unittest.main()
