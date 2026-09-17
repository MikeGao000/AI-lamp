"""Tests for the salient page hypothesis.

The box decision is pure numpy on purpose so this runs on the dev box, which has no
OpenCV: the numbers behind these cases come from the measured behaviour of the
model on 39 real frames.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lamp_core import salient_page  # noqa: E402
from lamp_core.salient_page import (  # noqa: E402
    SalientPageDetector,
    box_contains,
    find_model_path,
    region_box,
)


def heat_with_block(
    size: int = 32,
    *,
    x1: int = 8,
    y1: int = 6,
    x2: int = 24,
    y2: int = 26,
    peak: float = 1.0,
    noise: float = 0.0,
) -> list[list[float]]:
    heat = [[noise for _ in range(size)] for _ in range(size)]
    for y in range(y1, y2):
        for x in range(x1, x2):
            heat[y][x] = peak
    return heat


class RegionBoxTests(unittest.TestCase):
    def test_a_solid_block_is_found_where_it_is(self):
        box = region_box(heat_with_block(size=32, x1=8, y1=6, x2=24, y2=26))
        self.assertIsNotNone(box)
        self.assertAlmostEqual(8 / 32, box[0], places=3)
        self.assertAlmostEqual(6 / 32, box[1], places=3)
        self.assertAlmostEqual(24 / 32, box[2], places=3)
        self.assertAlmostEqual(26 / 32, box[3], places=3)

    def test_the_decision_does_not_depend_on_absolute_brightness(self):
        dim = region_box(heat_with_block(peak=0.05, noise=0.0))
        bright = region_box(heat_with_block(peak=200.0, noise=0.0))
        self.assertEqual(dim, bright)

    def test_a_flat_map_has_no_region(self):
        self.assertIsNone(region_box([[0.5] * 32 for _ in range(32)]))

    def test_a_blank_map_has_no_region(self):
        self.assertIsNone(region_box([[0.0] * 32 for _ in range(32)]))

    def test_a_speck_is_rejected(self):
        box = region_box(heat_with_block(size=64, x1=30, y1=30, x2=31, y2=31))
        self.assertIsNone(box)

    def test_the_widest_band_wins_over_a_narrow_one(self):
        heat = heat_with_block(size=64, x1=40, y1=4, x2=48, y2=60)
        for y in range(8, 56):
            for x in range(2, 30):
                heat[y][x] = 1.0
        box = region_box(heat)
        self.assertIsNotNone(box)
        # The wide band on the left, not the tall narrow one on the right.
        self.assertLess(box[0], 0.2)

    def test_a_one_dimensional_map_is_refused(self):
        self.assertIsNone(region_box([[1.0, 1.0]]))
        self.assertIsNone(region_box([]))


class BoxContainsTests(unittest.TestCase):
    def test_a_contained_box_passes(self):
        self.assertTrue(
            box_contains((0.1, 0.1, 0.9, 0.9), (0.3, 0.3, 0.7, 0.7))
        )

    def test_a_box_that_overhangs_is_rejected(self):
        self.assertFalse(
            box_contains((0.3, 0.3, 0.7, 0.7), (0.1, 0.3, 0.7, 0.7))
        )

    def test_slack_absorbs_a_padding_mismatch(self):
        # A padded text anchor against a smoothed saliency box: this exact mismatch
        # produced a false 21%-containment figure earlier, which is why slack exists.
        self.assertTrue(
            box_contains((0.30, 0.30, 0.70, 0.70), (0.29, 0.29, 0.70, 0.70))
        )


class DetectorTests(unittest.TestCase):
    def test_a_missing_model_reports_unavailable(self):
        detector = SalientPageDetector("definitely/not/here.onnx")
        self.assertFalse(detector.available)
        self.assertIsNone(detector.page_box(object()))

    def test_the_model_is_looked_for_in_the_project_and_tmp(self):
        paths = find_model_path("u2netp.onnx")
        self.assertTrue(paths is None or paths.endswith("u2netp.onnx"))


class TextOwnershipTests(unittest.TestCase):
    """Which text belongs to the book -- the question that blocked everything else.

    The measured anchors on this rig include the monitor's own text, so a padded
    anchor spanning x 0.40..1.00 at y 0.07..0.41 must not be treated as the book's.
    """

    class _Box:
        def __init__(self, bbox):
            self.bbox = bbox

    def test_text_on_the_page_is_kept(self):
        page = (0.1, 0.1, 0.8, 0.9)
        boxes = [self._Box((0.2, 0.2, 0.6, 0.3)), self._Box((0.5, 0.5, 0.7, 0.6))]
        self.assertEqual(2, len(salient_page.text_boxes_inside(page, boxes)))

    def test_text_on_the_monitor_is_dropped(self):
        page = (0.1, 0.3, 0.8, 0.9)
        boxes = [self._Box((0.40, 0.07, 1.00, 0.20))]
        self.assertEqual([], salient_page.text_boxes_inside(page, boxes))

    def test_a_line_mostly_on_the_page_is_kept(self):
        page = (0.2, 0.2, 0.8, 0.8)
        # 80% of this line is inside the page.
        self.assertEqual(
            1, len(salient_page.text_boxes_inside(page, [self._Box((0.15, 0.3, 0.65, 0.4))]))
        )

    def test_the_anchor_is_rebuilt_from_the_kept_boxes_only(self):
        page = (0.1, 0.3, 0.8, 0.9)
        boxes = [
            self._Box((0.40, 0.07, 1.00, 0.20)),   # monitor
            self._Box((0.20, 0.35, 0.60, 0.45)),   # page
        ]
        anchor = salient_page.saliency_text_anchor(page, boxes)
        self.assertAlmostEqual(0.20, anchor[0], places=3)
        self.assertAlmostEqual(0.60, anchor[2], places=3)
        self.assertGreaterEqual(anchor[1], 0.3)

    def test_no_text_on_the_page_gives_no_anchor(self):
        page = (0.1, 0.5, 0.4, 0.9)
        boxes = [self._Box((0.6, 0.05, 0.95, 0.15))]
        self.assertIsNone(salient_page.saliency_text_anchor(page, boxes))

    def test_plain_tuples_work_as_well_as_box_objects(self):
        page = (0.1, 0.1, 0.9, 0.9)
        self.assertEqual(1, len(salient_page.text_boxes_inside(page, [(0.2, 0.2, 0.4, 0.3)])))


class _CountingDetector:
    def __init__(self, box=(0.1, 0.2, 0.7, 0.8), available=True):
        self.box = box
        self._available = available
        self.calls = 0

    @property
    def available(self):
        return self._available

    def page_box(self, image):
        self.calls += 1
        return self.box


class PageHypothesisTests(unittest.TestCase):
    def test_the_model_runs_once_per_interval(self):
        detector = _CountingDetector()
        hypothesis = salient_page.PageHypothesis(detector, interval_seconds=6.0)
        self.assertEqual((0.1, 0.2, 0.7, 0.8), hypothesis.update(object(), 0.0))
        hypothesis.update(object(), 1.0)
        hypothesis.update(object(), 5.9)
        self.assertEqual(1, detector.calls)
        hypothesis.update(object(), 6.0)
        self.assertEqual(2, detector.calls)

    def test_a_failed_refresh_keeps_the_previous_box(self):
        detector = _CountingDetector()
        hypothesis = salient_page.PageHypothesis(detector, interval_seconds=1.0)
        hypothesis.update(object(), 0.0)
        detector.box = None
        self.assertEqual((0.1, 0.2, 0.7, 0.8), hypothesis.update(object(), 2.0))
        self.assertEqual(1, hypothesis.refreshes)

    def test_an_unavailable_detector_yields_nothing(self):
        hypothesis = salient_page.PageHypothesis(_CountingDetector(available=False))
        self.assertFalse(hypothesis.available)
        self.assertIsNone(hypothesis.update(object(), 0.0))

    def test_no_detector_at_all_is_handled(self):
        self.assertIsNone(salient_page.PageHypothesis(None).update(object(), 0.0))


class AsyncPageHypothesisTests(unittest.TestCase):
    """The model must never run on the capture loop's thread.

    Called inline it cost 5.5 s per iteration and dropped the preview to 0.28 fps.
    """

    def test_submit_does_not_call_the_model(self):
        detector = _CountingDetector()
        hypothesis = salient_page.AsyncPageHypothesis(
            detector, interval_seconds=60.0, autostart=False
        )
        for _ in range(5):
            self.assertIsNone(hypothesis.submit(object()))
        self.assertEqual(0, detector.calls)

    def test_a_refresh_stores_the_box_and_submit_returns_it(self):
        detector = _CountingDetector()
        hypothesis = salient_page.AsyncPageHypothesis(
            detector, interval_seconds=60.0, autostart=False
        )
        hypothesis.submit(object())
        self.assertEqual((0.1, 0.2, 0.7, 0.8), hypothesis.refresh_once())
        self.assertEqual(1, detector.calls)
        self.assertEqual((0.1, 0.2, 0.7, 0.8), hypothesis.submit(object()))
        self.assertEqual(1, detector.calls)
        self.assertEqual(1, hypothesis.refreshes)

    def test_a_failed_refresh_keeps_the_last_box_and_records_the_error(self):
        class Exploding(_CountingDetector):
            def page_box(self, image):
                self.calls += 1
                raise RuntimeError("model blew up")

        hypothesis = salient_page.AsyncPageHypothesis(
            Exploding(), interval_seconds=60.0, autostart=False
        )
        hypothesis.box = (0.2, 0.3, 0.6, 0.8)
        hypothesis.submit(object())
        self.assertEqual((0.2, 0.3, 0.6, 0.8), hypothesis.refresh_once())
        self.assertIn("RuntimeError", hypothesis.last_error)

    def test_no_detector_means_no_box_and_no_thread(self):
        hypothesis = salient_page.AsyncPageHypothesis(None, autostart=False)
        self.assertFalse(hypothesis.available)
        self.assertIsNone(hypothesis.submit(object()))
        self.assertIsNone(hypothesis.refresh_once())


class _FakeQueue:
    """Minimal stand-in for a multiprocessing queue."""

    def __init__(self, full=False, results=()):
        self.full = full
        self.results = list(results)
        self.items = []

    def put_nowait(self, item):
        if self.full:
            raise Exception("queue full")
        self.items.append(item)

    def get_nowait(self):
        if not self.results:
            raise Exception("empty")
        return self.results.pop(0)


class ProcessPageHypothesisTests(unittest.TestCase):
    """The model must not run in the capture loop's process at all.

    Measured, the thread version stopped the loop's iteration counter dead for seconds
    at a time (published 3.00/s, received 2.89/s), so the parent must never run the
    model and must never block.
    """

    def build(self, detector=None, frames=None, results=()):
        frames = frames if frames is not None else _FakeQueue()
        return (
            salient_page.ProcessPageHypothesis(
                detector or _CountingDetector(),
                autostart=False,
                frame_queue=frames,
                result_queue=_FakeQueue(results=results),
            ),
            frames,
        )

    def test_submit_never_runs_the_model(self):
        detector = _CountingDetector()
        hypothesis, _ = self.build(detector)
        for _ in range(5):
            self.assertIsNone(hypothesis.submit(object()))
        self.assertEqual(0, detector.calls)

    def test_submit_hands_the_frame_over(self):
        hypothesis, frames = self.build()
        hypothesis.submit("frame", now=100.0)
        self.assertEqual(["frame"], [item[0] for item in frames.items])
        self.assertEqual(1, hypothesis.submitted)

    def test_only_one_frame_is_enqueued_per_interval(self):
        # Enqueueing every frame cost a 2 MB pickle per loop iteration and held the
        # loop at 3.5 iterations a second against a 56 ms stage total.
        hypothesis, frames = self.build()
        hypothesis.interval_seconds = 8.0
        hypothesis.submit("first", now=100.0)
        for offset in range(1, 8):
            hypothesis.submit("again", now=100.0 + offset)
        self.assertEqual(["first"], [item[0] for item in frames.items])
        self.assertEqual(1, hypothesis.submitted)
        hypothesis.submit("later", now=108.0)
        self.assertEqual(["first", "later"], [item[0] for item in frames.items])
        self.assertEqual(2, hypothesis.submitted)

    def test_a_full_queue_drops_the_frame_instead_of_blocking(self):
        hypothesis, _ = self.build(frames=_FakeQueue(full=True))
        hypothesis.submit("frame")
        self.assertEqual(1, hypothesis.dropped)
        self.assertEqual(0, hypothesis.submitted)

    def test_results_are_drained_and_the_newest_box_kept(self):
        hypothesis, _ = self.build(
            results=[(0.1, 0.2, 0.3, 0.4), (0.2, 0.3, 0.4, 0.5)]
        )
        self.assertEqual((0.2, 0.3, 0.4, 0.5), hypothesis.submit(object()))
        self.assertEqual(2, hypothesis.refreshes)

    def test_a_reported_error_is_recorded_not_raised(self):
        hypothesis, _ = self.build(results=["RuntimeError: boom", None])
        hypothesis.submit(object())
        self.assertEqual("RuntimeError: boom", hypothesis.last_error)

    def test_no_detector_means_nothing_is_submitted(self):
        hypothesis = salient_page.ProcessPageHypothesis(None, autostart=False)
        self.assertFalse(hypothesis.available)
        self.assertIsNone(hypothesis.submit(object()))

    def test_stop_is_safe_when_no_process_was_started(self):
        hypothesis, _ = self.build()
        hypothesis.stop()

    def test_the_interval_is_read_on_every_call(self):
        # Adaptive refresh depends on this: the caller shortens the interval while
        # acquiring and lengthens it once a box is held.
        detector = _CountingDetector()
        hypothesis, frames = self.build(detector)
        hypothesis.interval_seconds = 30.0
        hypothesis.submit("first", now=100.0)
        hypothesis.interval_seconds = 3.0
        hypothesis.submit("second", now=104.0)
        self.assertEqual(["first", "second"], [item[0] for item in frames.items])

    def test_the_two_adaptive_intervals_are_ordered(self):
        self.assertLess(
            salient_page.PAGE_REFRESH_SECONDS_ACQUIRING,
            salient_page.PAGE_REFRESH_SECONDS_TRACKED,
        )


class WorkerWatchdogTests(unittest.TestCase):
    """A dead worker must not wedge the parent.

    Measured: the u2netp worker died after about an hour and the preview froze -- no
    frames for ten seconds, health "ok", the machine idle and every parent thread parked
    on a futex, because a multiprocessing queue is not safe against a process dying
    mid-operation.
    """

    class _FakeProcess:
        def __init__(self, alive):
            self.alive = alive

        def is_alive(self):
            return self.alive

        def start(self):
            # Deliberately does not revive the process: this models a worker that is
            # already gone, which is the case the watchdog exists for.
            pass

        def terminate(self):
            self.alive = False

    def build(self, alive):
        created = []

        def factory(**kwargs):
            process = self._FakeProcess(alive)
            created.append(process)
            return process

        hypothesis = salient_page.ProcessPageHypothesis(
            _CountingDetector(),
            autostart=False,
            frame_queue=_FakeQueue(),
            result_queue=_FakeQueue(),
            process_factory=factory,
        )
        hypothesis.interval_seconds = 0.0
        hypothesis.start()
        return hypothesis, created

    def test_a_live_worker_is_left_alone(self):
        hypothesis, created = self.build(alive=True)
        self.assertEqual(1, len(created))
        hypothesis.submit("frame", now=0.0)
        self.assertEqual(1, len(created))
        self.assertEqual(0, hypothesis.restarts)

    def test_a_dead_worker_is_replaced_with_fresh_queues(self):
        hypothesis, created = self.build(alive=False)
        self.assertEqual(1, len(created))
        hypothesis.submit("frame", now=0.0)
        self.assertEqual(2, len(created), "the dead worker should have been replaced")
        self.assertEqual(1, hypothesis.restarts)

    def test_an_untouched_object_never_starts_a_process(self):
        # autostart=False must stay inert for tests that inject their own queues.
        hypothesis = salient_page.ProcessPageHypothesis(
            _CountingDetector(), autostart=False, frame_queue=_FakeQueue()
        )
        hypothesis.submit("frame", now=0.0)
        self.assertEqual(0, hypothesis.restarts)
        self.assertIsNone(hypothesis._process)


if __name__ == "__main__":
    unittest.main()
