"""Salient-region page hypothesis, and the text that belongs to it.

Why this exists, with the measurements that forced it:

* The salient model returns a page-sized box on 39 of 39 frames of the current
  scene, cross-frame spread about 0.01.
* The text-driven page contour fires on only 13 of 39 (33%) frames, because the
  "dominant text block" it is seeded with includes the monitor's own text (measured
  anchors such as x 0.40..1.00, y 0.07..0.41), and no page contour can cover a band
  that runs from the screen onto the page.
* So the salient region is the page hypothesis, and whichever text lies inside it
  is the book's text. That answers both "the box is not the page" and "which text
  belongs to the book" with one signal.

The box decision is pure numpy so it can be tested without OpenCV; the model call
is the only part that needs cv2.
"""

from __future__ import annotations

import os
import time

DEFAULT_MODEL_FILENAME = "u2netp.onnx"
INPUT_SIZE = 320
#: u2net output is a saliency map; the page is the region above this share of the
#: map's own peak, so no absolute brightness constant is involved.
REGION_FLOOR = 0.25
#: A column or row must be at least this covered to count as part of the region.
LINE_FLOOR = 0.25
#: Regions smaller than this share of the frame are specks, not pages.
MINIMUM_AREA = 0.004
#: A region covering essentially everything means the map had no localised peak, so
#: there is no distinct object to point at. Measured salient page boxes peaked at
#: 0.725 of the frame, so this only rejects the uninformative answer.
MAXIMUM_AREA = 0.98
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def candidate_model_paths(filename: str = DEFAULT_MODEL_FILENAME) -> tuple[str, ...]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return (
        os.path.join(root, "models", filename),
        os.path.join(root, filename),
        os.path.join("/tmp", filename),
    )


def find_model_path(filename: str = DEFAULT_MODEL_FILENAME) -> str | None:
    for path in candidate_model_paths(filename):
        if os.path.isfile(path):
            return path
    return None


def _longest_run(flags: list[bool]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index - 1)
            start = None
    return best


def region_box(
    heat: list,
    *,
    region_floor: float = REGION_FLOOR,
    line_floor: float = LINE_FLOOR,
    minimum_area: float = MINIMUM_AREA,
    maximum_area: float = MAXIMUM_AREA,
) -> tuple[float, float, float, float] | None:
    """Bounding box of the widest salient band, normalised, or None.

    ``heat`` is any 2-D array of saliency values; it is normalised by its own peak
    first, so the decision never depends on how bright the scene was.
    """

    if not heat:
        return None
    height = len(heat)
    width = len(heat[0]) if height else 0
    if height < 2 or width < 2:
        return None
    peak = max(max(row) for row in heat)
    if peak <= 0:
        return None

    mask = [[value / peak > region_floor for value in row] for row in heat]
    column_hits = [sum(1 for row in mask if row[x]) / height for x in range(width)]
    row_hits = [sum(1 for x in range(width) if mask[y][x]) / width for y in range(height)]
    columns = _longest_run([value >= line_floor for value in column_hits])
    rows = _longest_run([value >= line_floor for value in row_hits])
    if columns is None or rows is None:
        return None
    x1, x2 = columns
    y1, y2 = rows
    box = (x1 / width, y1 / height, (x2 + 1) / width, (y2 + 1) / height)
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area < minimum_area or area > maximum_area:
        return None
    return box


def box_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    *,
    slack: float = 0.02,
) -> bool:
    """Whether ``inner`` lies inside ``outer`` within ``slack``.

    Slack is needed because these boxes come from different measurements -- a
    padded text anchor against a smoothed saliency map -- so exact containment
    would reject correct answers by a few pixels, which is what produced a false
    21%-containment figure earlier in this work.
    """

    return (
        outer[0] <= inner[0] + slack
        and outer[1] <= inner[1] + slack
        and outer[2] >= inner[2] - slack
        and outer[3] >= inner[3] - slack
    )


def box_fraction_inside(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> float:
    """Share of ``inner`` that lies inside ``outer``."""

    width = inner[2] - inner[0]
    height = inner[3] - inner[1]
    if width <= 0 or height <= 0:
        return 0.0
    overlap_x = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    overlap_y = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    return (overlap_x * overlap_y) / (width * height)


def _is_adjacent(
    page_box: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
    margin: float,
) -> bool:
    """Whether a box sits just beside the page box, level with it.

    A spread is two pages side by side, and the salient model routinely boxes only
    one of them: measured at -5 to +5 degrees it boxed the left page while every
    detected line sat on the right one, so strict containment kept nothing. Requiring
    overlap in y and a small horizontal gap accepts the facing page without letting
    a box elsewhere on the desk through.
    """

    if margin <= 0:
        return False
    vertical = min(page_box[3], box[3]) - max(page_box[1], box[1])
    if vertical <= 0:
        return False
    gap = max(0.0, max(page_box[0] - box[2], box[0] - page_box[2]))
    return gap <= margin


def text_boxes_inside(
    page_box: tuple[float, float, float, float],
    boxes,
    *,
    minimum_fraction: float = 0.6,
    adjacent_margin: float = 0.15,
):
    """The detected text boxes that belong to the page hypothesis.

    This is the step the measurements kept pointing at. The dominant text block on
    this rig routinely includes the monitor's own text (measured anchors such as
    x 0.40..1.00, y 0.07..0.41), so any rule of the form "the page must cover the
    text" fails, and the page contour fired on only 13 of 39 frames because of it.
    Asking the other way round -- which text lies inside the page -- uses the one
    signal that is reliable over the whole sweep, and it yields the book's text as
    a by-product.

    A box counts as inside when most of its area is, so a line straddling the page
    edge is judged by where it mostly is rather than by either extreme.
    """

    kept = []
    for box in boxes:
        bbox = getattr(box, "bbox", box)
        values = tuple(float(v) for v in bbox)
        if box_fraction_inside(page_box, values) >= minimum_fraction:
            kept.append(box)
        elif _is_adjacent(page_box, values, adjacent_margin):
            kept.append(box)
    return kept


def saliency_text_anchor(
    page_box, boxes, *, minimum_fraction: float = 0.6, adjacent_margin: float = 0.15
):
    """Anchor of the book's own text, or None when none of it is on the page.

    The anchor is rebuilt from the kept boxes so that a padded anchor including
    off-page text can never be what steers the axis.
    """

    kept = text_boxes_inside(
        page_box,
        boxes,
        minimum_fraction=minimum_fraction,
        adjacent_margin=adjacent_margin,
    )
    if not kept:
        return None
    xs = [float(getattr(b, "bbox", b)[0]) for b in kept]
    ys = [float(getattr(b, "bbox", b)[1]) for b in kept]
    x2s = [float(getattr(b, "bbox", b)[2]) for b in kept]
    y2s = [float(getattr(b, "bbox", b)[3]) for b in kept]
    return (min(xs), min(ys), max(x2s), max(y2s))


class PageHypothesis:
    """Rate-limited page hypothesis.

    The salient model costs about 5.5 s on this Pi, so running it every capture
    iteration would wreck a loop whose other stages total about 0.6 s. The page does
    not move that fast, so the box is refreshed on an interval and reused in between.
    A failed refresh keeps the previous box rather than dropping the target on the
    strength of one uninformative frame.
    """

    def __init__(self, detector, *, interval_seconds: float = 6.0) -> None:
        self.detector = detector
        self.interval_seconds = interval_seconds
        self.box: tuple[float, float, float, float] | None = None
        self.updated_at = 0.0
        self.attempts = 0
        self.refreshes = 0

    @property
    def available(self) -> bool:
        return self.detector is not None and getattr(self.detector, "available", False)

    def update(self, image, now: float):
        """Return the current page box, refreshing it only when it is stale."""

        if not self.available:
            return None
        stale = self.box is None or (now - self.updated_at) >= self.interval_seconds
        if not stale:
            return self.box
        self.attempts += 1
        self.updated_at = now
        found = self.detector.page_box(image)
        if found is not None:
            self.box = found
            self.refreshes += 1
        return self.box


class AsyncPageHypothesis:
    """Refresh the page hypothesis off the capture loop's thread.

    The model costs about 5.5 s on this Pi. Calling it from the capture loop dropped
    the MJPEG preview to 0.28 fps (measured: three frames in 10.6 s), which reads to
    a user as "the picture never refreshes". A worker thread keeps the box fresh while
    the loop stays at frame rate; the loop only hands over the newest frame and reads
    whatever box is current.

    ``autostart=False`` builds the object without a thread, so the behaviour can be
    tested deterministically.

    The interval is deliberately long. Measured, inference holds the GIL: with the
    worker running every 8 s the capture loop stalled for 0.9-1.5 s at a time and the
    preview averaged 3 fps, while hiding the model gave a rock-steady 10 fps (nine
    consecutive 100 ms iterations). Capping OpenCV's threads did not help, which is
    what points at the GIL rather than at CPU share -- a separate *process* is the
    real fix. Until then the refresh is rare enough that the page box stays current
    (the tracker and the text detector carry the frames in between) without the
    preview stuttering every few seconds.
    """

    def __init__(
        self,
        detector,
        *,
        interval_seconds: float = 30.0,
        autostart: bool = True,
    ) -> None:
        self.detector = detector
        self.interval_seconds = interval_seconds
        self.box: tuple[float, float, float, float] | None = None
        self.refreshes = 0
        self.attempts = 0
        self.last_error: str | None = None
        self._latest_image = None
        self._wake = None
        self._stop = False
        self._thread = None
        if autostart:
            self.start()

    @property
    def available(self) -> bool:
        return self.detector is not None and getattr(self.detector, "available", False)

    def start(self) -> None:
        if self._thread is not None or not self.available:
            return
        import threading

        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._wake is not None:
            self._wake.set()

    def submit(self, image):
        """Hand over the newest frame and return the current box. Never blocks."""

        if not self.available:
            return None
        self._latest_image = image
        if self._wake is not None:
            self._wake.set()
        return self.box

    def refresh_once(self) -> tuple[float, float, float, float] | None:
        """Run the model on the newest frame, once. Kept separate for tests."""

        image = self._latest_image
        if image is None or not self.available:
            return self.box
        self.attempts += 1
        try:
            found = self.detector.page_box(image)
        except Exception as error:  # noqa: BLE001 - a worker must not die on one frame
            self.last_error = f"{type(error).__name__}: {error}"
            return self.box
        self.last_error = None
        if found is not None:
            self.box = found
            self.refreshes += 1
        return self.box

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=self.interval_seconds)
            self._wake.clear()
            if self._stop:
                return
            self.refresh_once()
            time.sleep(self.interval_seconds)


class ProcessPageHypothesis:
    """Refresh the page hypothesis in a separate process.

    Measured, the thread version stalls the capture loop: while the worker is inside
    the model the loop's iteration counter stops advancing entirely for seconds at a
    time, and the preview settled at 3 fps (published 3.00/s, received 2.89/s -- the
    client was never the limit). Capping OpenCV's threads made it worse, because the
    inference then took longer while still holding the GIL, so the only fix that
    removes the stall is to run the model where there is no shared interpreter.

    The parent never blocks: it drops the newest frame into a one-slot queue and reads
    whatever box the worker last produced. Losing frames is correct here -- only the
    most recent view matters and the refresh interval is seconds long.
    """

    def __init__(
        self,
        detector,
        *,
        interval_seconds: float = 8.0,
        autostart: bool = True,
        frame_queue=None,
        result_queue=None,
        process_factory=None,
    ) -> None:
        self.detector = detector
        self.interval_seconds = interval_seconds
        self.box: tuple[float, float, float, float] | None = None
        self.refreshes = 0
        self.submitted = 0
        self.dropped = 0
        self.restarts = 0
        self.last_submitted_at = 0.0
        self.box_received_at = 0.0
        self.last_error: str | None = None
        self._frame_queue = frame_queue
        self._result_queue = result_queue
        self._process_factory = process_factory
        self._process = None
        self._started = False
        self._stop = False
        if autostart:
            self.start()

    @property
    def available(self) -> bool:
        return self.detector is not None and getattr(self.detector, "available", False)

    def start(self) -> None:
        if self._stop or self._process is not None or not self.available:
            return
        import multiprocessing

        if self._frame_queue is None:
            self._frame_queue = multiprocessing.Queue(maxsize=1)
        if self._result_queue is None:
            self._result_queue = multiprocessing.Queue()
        factory = self._process_factory or multiprocessing.Process
        self._process = factory(
            target=_hypothesis_worker,
            args=(
                self.detector,
                self._frame_queue,
                self._result_queue,
                self.interval_seconds,
            ),
            daemon=True,
        )
        self._process.start()
        self._started = True

    def _ensure_worker(self) -> None:
        """Replace a worker that died, with fresh queues.

        Measured: the worker died after about an hour and the preview froze completely --
        no frames published for ten seconds, health still "ok", the machine idle and the
        parent's threads all parked on a futex. A multiprocessing queue's internals are
        not safe against a process dying mid-operation, so the queues are rebuilt too
        rather than reused.
        """

        process = self._process
        if process is None:
            return
        if process.is_alive():
            return
        self.restarts += 1
        self.box = None
        self._process = None
        self._frame_queue = None
        self._result_queue = None
        self.start()

    def stop(self) -> None:
        self._stop = True
        process = self._process
        self._process = None
        if process is not None:
            try:
                process.terminate()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass

    def submit(self, image, now: float | None = None):
        """Hand the newest frame over and return the current box. Never blocks.

        Only one frame per interval is enqueued. Enqueueing every frame cost a 2 MB
        pickle per loop iteration -- the worker drains the slot within milliseconds, so
        the slot was almost always empty and the parent paid that copy ~10 times a
        second. Measured with the per-frame enqueue, iterations ran at 3.5/s against a
        56 ms stage total, i.e. roughly 150 ms an iteration was being spent here.

        ``interval_seconds`` is read on every call, so the caller can make the refresh
        adaptive: quick while there is no page to hold on to, slow once there is. The
        model costs about 5.5 s of one core, and the page box barely moves once it is
        being tracked, so a fixed short interval was paying that cost for nothing.
        """

        if self._stop or not self.available:
            return None
        if self._started:
            self._ensure_worker()
        if self._frame_queue is None:
            return None
        self.drain()
        stamp = time.monotonic() if now is None else now
        if time.monotonic() - self.box_received_at > 8.0:
            self.box = None
        if stamp - self.last_submitted_at < self.interval_seconds:
            return self.box
        self.last_submitted_at = stamp
        try:
            self._frame_queue.put_nowait((image, time.monotonic()))
            self.submitted += 1
        except Exception:  # noqa: BLE001 - a full queue just means a frame is pending
            self.dropped += 1
        return self.box

    def drain(self) -> None:
        """Take every result the worker has produced; keep the newest box."""

        if self._result_queue is None:
            return
        while True:
            try:
                result = self._result_queue.get_nowait()
            except Exception:  # noqa: BLE001 - Empty is the normal case
                return
            if result is None:
                self.box = None
                continue
            if isinstance(result, str):
                self.last_error = result
                self.box = None
                continue
            if isinstance(result, dict):
                captured_at = result["captured_at"]
                if time.monotonic() - captured_at > 8.0:
                    self.box = None
                    continue
                result = result["box"]
                if result is None:
                    self.box = None
                    continue
            self.box = tuple(result)
            self.box_received_at = time.monotonic()
            self.refreshes += 1


def _hypothesis_worker(detector, frame_queue, result_queue, interval_seconds):
    """Child process body: refresh the box from the newest frame, for ever."""

    while True:
        try:
            image, captured_at = frame_queue.get(timeout=interval_seconds)
        except Exception:  # noqa: BLE001 - nothing to do; keep waiting
            continue
        try:
            box = detector.page_box(image)
            result_queue.put_nowait({"box": box, "captured_at": captured_at})
        except Exception as error:  # noqa: BLE001 - report, never die
            try:
                result_queue.put_nowait(f"{type(error).__name__}: {error}")
            except Exception:  # noqa: BLE001
                pass


#: How long the page hypothesis may go unrefreshed once a box is being held. The model
#: costs about 5.5 s of one core and the box barely moves while it is tracked, so the
#: steady-state refresh is deliberately slow; acquisition uses the fast value.
PAGE_REFRESH_SECONDS_TRACKED = 30.0
#: The refresh asked for while there is no usable box -- i.e. while acquiring.
PAGE_REFRESH_SECONDS_ACQUIRING = 3.0


class SalientPageDetector:
    """Runs the salient model and returns the page hypothesis."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        input_size: int = INPUT_SIZE,
        region_floor: float = REGION_FLOOR,
        line_floor: float = LINE_FLOOR,
        minimum_area: float = MINIMUM_AREA,
    ) -> None:
        self.model_path = model_path or find_model_path()
        self.input_size = input_size
        self.region_floor = region_floor
        self.line_floor = line_floor
        self.minimum_area = minimum_area
        self._net = None
        self.last_milliseconds = 0.0

    @property
    def available(self) -> bool:
        return bool(self.model_path) and os.path.isfile(self.model_path)

    def _load(self):
        if self._net is None:
            import cv2

            net = cv2.dnn.readNetFromONNX(self.model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._net = net
        return self._net

    def saliency_map(self, image) -> list:
        """Return the model's saliency map as nested lists."""

        import cv2
        import numpy as np

        size = self.input_size
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        scaled = resized.astype(np.float32) / 255.0
        normalised = (scaled - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(
            IMAGE_STD, dtype=np.float32
        )
        blob = np.transpose(normalised, (2, 0, 1))[None].astype(np.float32)
        net = self._load()
        net.setInput(blob)
        # Cap the worker's thread use. Measured, the model busy for 5.5 s out of every
        # 7.5 s left the capture loop at 1.72 fps even though its own per-frame work
        # (CAN read 4.6 ms, draw 2.6 ms, JPEG 19 ms) is trivial -- it was CPU
        # contention, not the loop's own cost. Two threads leave the other two cores
        # for the camera loop.
        previous = cv2.getNumThreads()
        try:
            cv2.setNumThreads(min(2, max(1, previous)))
            heat = np.asarray(net.forward()).reshape(size, size)
        finally:
            cv2.setNumThreads(previous)
        return heat.tolist()

    def page_box(self, image) -> tuple[float, float, float, float] | None:
        """Page hypothesis as a normalised box, or None when nothing stands out."""

        if not self.available:
            return None
        import time

        started = time.perf_counter()
        try:
            heat = self.saliency_map(image)
        finally:
            self.last_milliseconds = (time.perf_counter() - started) * 1000.0
        return region_box(
            heat,
            region_floor=self.region_floor,
            line_floor=self.line_floor,
            minimum_area=self.minimum_area,
        )
