"""Local text recognition (CRNN).

**Status: not wired in.** Written for an anti-hallucination check that was replaced by
cloud self-consistency (see ``reading_check``). It is kept because the measurement that
ruled it out is worth preserving: the integrated detector returns *region* boxes, not
lines -- one box of 63% x 35% of the page at 0.99 confidence, holding the story text and
the illustration together -- so a recogniser fed from it produced nothing usable, and
line splitting inside such a box finds no gaps because the illustration inks every row.

The rest of this docstring describes the model contract, for whoever revives it.

Contract taken from the OpenCV Zoo CRNN model this is built for:
input 100x32, mean 127.5 with scale 1/127.5, greedy CTC decode, and the charset chosen
by model name (``_EN_`` 36 classes, ``_CH_`` 94, ``_CN_`` 3944).
"""

from __future__ import annotations

import os
import re

#: The language token in a model name, delimited by non-alphanumerics or the ends.
_LANGUAGE_TOKEN = re.compile(r"(?:^|[^A-Z0-9])(EN|CH|CN)(?=[^A-Z0-9]|$)")

DEFAULT_MODEL_FILENAME = "text_recognition_crnn_ch.onnx"
INPUT_SIZE = (100, 32)
MEAN = 127.5
SCALE = 1.0 / 127.5

CHARSET_EN_36 = list("0123456789abcdefghijklmnopqrstuvwxyz")
CHARSET_CH_94 = list(
    "0123456789abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


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


def charset_for_model(model_path: str) -> list[str]:
    """Pick the charset the model was trained with.

    The zoo's own implementation does this by name, and getting it wrong silently
    garbles every reading, so an unrecognised name is an error rather than a guess.

    The language is matched as a *token*, not as a substring: the zoo writes
    ``..._CRNN_CH_2021sep.onnx`` but a local copy named ``..._crnn_ch.onnx`` used to be
    rejected, and a name that merely contains "en" somewhere must not count.
    """

    stem = os.path.splitext(os.path.basename(model_path or ""))[0].upper()
    match = _LANGUAGE_TOKEN.search(stem)
    if match is None:
        raise ValueError(f"cannot tell which charset {model_path!r} needs")
    language = match.group(1)
    if language == "EN":
        return CHARSET_EN_36
    if language == "CH":
        return CHARSET_CH_94
    raise ValueError(
        "the Chinese charset is 3944 characters and not bundled here; "
        "use an _EN_ or _CH_ model"
    )


def decode_ctc(output, charset: list[str]) -> str:
    """Greedy CTC decode of one sequence.

    ``output`` is a 2-D array of per-timestep class scores; index 0 is the CTC blank.
    Adjacent repeats collapse, which is what makes the blank class necessary.
    """

    if output is None or len(charset) == 0:
        return ""
    text = []
    for step in output:
        if step is None or len(step) == 0:
            continue
        best = int(max(range(len(step)), key=lambda index: step[index]))
        if best == 0:
            text.append("-")
        elif best - 1 < len(charset):
            text.append(charset[best - 1])
        else:
            text.append("-")
    collapsed = []
    for index, char in enumerate(text):
        if char != "-" and not (index > 0 and char == text[index - 1]):
            collapsed.append(char)
    return "".join(collapsed)


def line_ranges(ink_rows, *, gap_fraction: float = 0.18) -> list[tuple[int, int]]:
    """Split a box's row-wise ink profile into text lines.

    CRNN reads one line at a time, and the detector returns paragraph blocks: measured,
    feeding a two-line block straight in produced the garbage "l N". This finds the gaps
    between lines from the profile, which needs no threshold beyond each box's own peak.
    """

    if ink_rows is None or len(ink_rows) < 3:
        return []
    peak = max(float(value) for value in ink_rows)
    if peak <= 0.0:
        return []
    floor = peak * gap_fraction
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(ink_rows):
        if float(value) > floor:
            if start is None:
                start = index
        elif start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(ink_rows)))
    # Merge runs separated by a single row: a broken glyph stroke is not a new line.
    merged: list[tuple[int, int]] = []
    for begin, end in ranges:
        if merged and begin - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))
    return [span for span in merged if span[1] - span[0] >= 3]


def sequence_from_output(output):
    """Pull the per-timestep class scores out of a CRNN output tensor.

    The zoo model emits ``(T, 1, C)`` and its own implementation indexes
    ``outputBlob[i][0]``. Indexing ``output[0]`` instead -- as this did -- decodes only
    the first timestep, which is why every line came back as a single letter ('t', 'c',
    'a') no matter what was in the crop. Both layouts are accepted because the batch axis
    is not guaranteed to sit on one side.
    """

    if output is None:
        return None
    if output.ndim == 2:
        return output
    if output.ndim != 3:
        return None
    if output.shape[1] == 1:
        return output[:, 0, :]
    if output.shape[0] == 1:
        return output[0]
    return None


class CrnnTextRecognizer:
    """Reads the text inside given boxes."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        input_size: tuple[int, int] = INPUT_SIZE,
    ) -> None:
        self.model_path = model_path or find_model_path()
        self.input_size = input_size
        self._net = None
        self._charset: list[str] | None = None
        self.last_milliseconds = 0.0

    @property
    def available(self) -> bool:
        return bool(self.model_path) and os.path.isfile(self.model_path)

    def _load(self):
        if self._net is None:
            import cv2

            self._charset = charset_for_model(self.model_path)
            net = cv2.dnn.readNet(self.model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._net = net
        return self._net

    def _prepare(self, picture, cv2):
        height, width = picture.shape[:2]
        if height < 2 or width < 2:
            return None
        vertices = [
            [0.0, float(height - 1)],
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
        ]
        import numpy as np

        target = np.array(
            [
                [0.0, float(self.input_size[1] - 1)],
                [0.0, 0.0],
                [float(self.input_size[0] - 1), 0.0],
                [float(self.input_size[0] - 1), float(self.input_size[1] - 1)],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(
            np.array(vertices, dtype="float32"), target
        )
        warped = cv2.warpPerspective(picture, matrix, self.input_size)
        return cv2.dnn.blobFromImage(
            warped, size=self.input_size, mean=MEAN, scalefactor=SCALE
        )

    def read_crop(self, picture) -> str:
        """Read one already-cropped line."""

        if not self.available:
            return ""
        import time

        import cv2
        import numpy as np

        blob = self._prepare(picture, cv2)
        if blob is None:
            return ""
        net = self._load()
        started = time.perf_counter()
        try:
            net.setInput(blob)
            output = np.asarray(net.forward())
        finally:
            self.last_milliseconds = (time.perf_counter() - started) * 1000.0
        if output.ndim != 3:
            return ""
        sequence = sequence_from_output(output)
        if sequence is None:
            return ""
        return decode_ctc(sequence, self._charset or [])

    def transcript(self, image, text_boxes) -> str:
        """Read every box, line by line, and join the words.

        Boxes are normalised (as the text detector returns them) and are padded before
        cropping, because the detector's boxes touch the glyphs and CRNN does better with
        a little margin around them. A box taller than one line is split first: the
        detector returns paragraph blocks and CRNN reads a single line.
        """

        if not self.available or image is None or not text_boxes:
            return ""
        import numpy as np

        height, width = image.shape[:2]
        words: list[str] = []
        for box in text_boxes:
            bbox = getattr(box, "bbox", box)
            try:
                x1, y1, x2, y2 = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            pad_x = (x2 - x1) * 0.03
            pad_y = (y2 - y1) * 0.10
            left = max(0, int((x1 - pad_x) * width))
            top = max(0, int((y1 - pad_y) * height))
            right = min(width, int((x2 + pad_x) * width))
            bottom = min(height, int((y2 + pad_y) * height))
            if right - left < 4 or bottom - top < 4:
                continue
            block = image[top:bottom, left:right]
            grey = (
                block
                if block.ndim == 2
                else block.mean(axis=2)
            )
            ink = 255.0 - np.asarray(grey, dtype="float32")
            spans = line_ranges(ink.mean(axis=1).tolist())
            if not spans:
                continue
            # Split at the gaps, keeping a little of the neighbouring rows.
            for begin, end in spans:
                strip = block[max(0, begin - 1):min(end + 1, block.shape[0]), :]
                if strip.shape[0] < 4 or strip.shape[1] < 4:
                    continue
                text = self.read_crop(strip)
                if text:
                    words.append(text)
        return " ".join(words)
