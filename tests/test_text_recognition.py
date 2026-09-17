"""Tests for local text recognition.

The CTC decode and the charset choice are pure logic and are tested directly; the model
call needs OpenCV and is not exercised here.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lamp_core.text_recognition import (  # noqa: E402
    CHARSET_CH_94,
    CHARSET_EN_36,
    CrnnTextRecognizer,
    charset_for_model,
    decode_ctc,
    find_model_path,
    line_ranges,
    sequence_from_output,
)


def scores(*classes: int, classes_count: int = 95) -> np.ndarray:
    """One row per step, argmax at the given class index."""

    rows = []
    for value in classes:
        row = np.zeros(classes_count, dtype="float32")
        row[value] = 1.0
        rows.append(row)
    return np.array(rows, dtype="float32")


class DecodeTests(unittest.TestCase):
    """CTC semantics: class 0 is blank, adjacent repeats collapse, a blank separates."""

    A = CHARSET_EN_36.index("a") + 1
    B = CHARSET_EN_36.index("b") + 1

    def test_adjacent_repeats_collapse(self):
        self.assertEqual("ab", decode_ctc(scores(self.A, self.A, 0, self.B), CHARSET_EN_36))

    def test_a_blank_between_repeats_keeps_both(self):
        self.assertEqual("aa", decode_ctc(scores(self.A, 0, self.A), CHARSET_EN_36))

    def test_an_all_blank_sequence_reads_nothing(self):
        self.assertEqual("", decode_ctc(scores(0, 0, 0), CHARSET_EN_36))

    def test_out_of_range_classes_are_treated_as_blank(self):
        self.assertEqual("", decode_ctc(scores(90, 90), CHARSET_EN_36))

    def test_empty_input_is_safe(self):
        self.assertEqual("", decode_ctc(None, CHARSET_EN_36))
        self.assertEqual("", decode_ctc(scores(), CHARSET_EN_36))
        self.assertEqual("", decode_ctc(scores(1), []))


class CharsetTests(unittest.TestCase):
    def test_the_english_model_gets_36_classes(self):
        self.assertEqual(CHARSET_EN_36, charset_for_model("text_recognition_CRNN_EN_x.onnx"))

    def test_the_shared_model_gets_94_classes(self):
        self.assertEqual(CHARSET_CH_94, charset_for_model("text_recognition_CRNN_CH_x.onnx"))

    def test_matching_ignores_case(self):
        # The name must not be a silent switch: a lower-case file used to be rejected.
        self.assertEqual(CHARSET_CH_94, charset_for_model("text_recognition_crnn_ch.onnx"))
        self.assertEqual(CHARSET_EN_36, charset_for_model("text_recognition_crnn_en.onnx"))

    def test_a_chinese_model_is_refused_not_guessed(self):
        # Decoding with the wrong charset garbles every reading, so this must raise.
        with self.assertRaises(ValueError):
            charset_for_model("text_recognition_CRNN_CN_x.onnx")

    def test_an_unknown_name_is_refused(self):
        with self.assertRaises(ValueError):
            charset_for_model("mystery.onnx")

    def test_a_name_that_merely_contains_en_is_refused(self):
        # "garden" contains "en"; matching substrings instead of tokens would accept it.
        with self.assertRaises(ValueError):
            charset_for_model("garden.onnx")


class RecognizerTests(unittest.TestCase):
    def test_a_missing_model_is_unavailable_and_reads_nothing(self):
        recognizer = CrnnTextRecognizer("definitely/not/here.onnx")
        self.assertFalse(recognizer.available)
        self.assertEqual("", recognizer.read_crop(object()))
        self.assertEqual("", recognizer.transcript(object(), []))

    def test_no_boxes_means_no_transcript(self):
        recognizer = CrnnTextRecognizer("definitely/not/here.onnx")
        self.assertEqual("", recognizer.transcript(object(), []))

    def test_the_model_is_sought_in_the_project_and_tmp(self):
        path = find_model_path("text_recognition_crnn_ch.onnx")
        self.assertTrue(path is None or path.endswith("text_recognition_crnn_ch.onnx"))


class LineSplitTests(unittest.TestCase):
    """A paragraph block has to be split into lines before CRNN can read it."""

    def profile(self, *band_heights: int, gap: int = 4, value: float = 100.0) -> list[float]:
        rows: list[float] = []
        for index, height in enumerate(band_heights):
            rows.extend([value] * height)
            if index < len(band_heights) - 1:
                rows.extend([0.0] * gap)
        return rows

    def test_two_bands_become_two_lines(self):
        spans = line_ranges(self.profile(12, 12))
        self.assertEqual(2, len(spans))
        self.assertEqual((0, 12), spans[0])

    def test_one_band_is_one_line(self):
        self.assertEqual(1, len(line_ranges(self.profile(14))))

    def test_a_blank_profile_has_no_lines(self):
        self.assertEqual([], line_ranges([0.0] * 20))
        self.assertEqual([], line_ranges([]))
        self.assertEqual([], line_ranges(None))

    def test_specks_too_short_to_be_text_are_dropped(self):
        self.assertEqual([], line_ranges(self.profile(2)))

    def test_a_one_row_break_does_not_split_a_line(self):
        # A broken glyph stroke must not look like two lines: exactly one blank row.
        rows = [100.0] * 8 + [0.0] + [100.0] * 8
        self.assertEqual(1, len(line_ranges(rows)))


class SequenceLayoutTests(unittest.TestCase):
    """The CRNN output axis bug: every line decoded to a single letter."""

    def build(self, shape):
        return np.zeros(shape, dtype="float32")

    def test_the_zoo_layout_t_1_c_gives_one_row_per_timestep(self):
        output = self.build((7, 1, 95))
        sequence = sequence_from_output(output)
        self.assertEqual((7, 95), sequence.shape)

    def test_a_batch_first_layout_is_also_accepted(self):
        output = self.build((1, 7, 95))
        sequence = sequence_from_output(output)
        self.assertEqual((7, 95), sequence.shape)

    def test_a_two_dimensional_input_passes_through(self):
        output = self.build((7, 95))
        self.assertIs(sequence_from_output(output), output)

    def test_an_ambiguous_or_empty_tensor_is_refused(self):
        self.assertIsNone(sequence_from_output(None))
        self.assertIsNone(sequence_from_output(self.build((3, 4, 95))))
        self.assertIsNone(sequence_from_output(self.build((7,))))

    def test_the_zoo_layout_decodes_a_whole_line(self):
        # Two words' worth of timesteps, blank-separated, in the zoo's (T, 1, C) shape.
        a = CHARSET_EN_36.index("a") + 1
        b = CHARSET_EN_36.index("b") + 1
        steps = [a, a, 0, b]
        output = np.zeros((len(steps), 1, 95), dtype="float32")
        for index, value in enumerate(steps):
            output[index, 0, value] = 1.0
        sequence = sequence_from_output(output)
        self.assertEqual("ab", decode_ctc(sequence, CHARSET_EN_36))


if __name__ == "__main__":
    unittest.main()
