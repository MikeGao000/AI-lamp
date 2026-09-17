"""Tests for the cloud self-consistency check."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lamp_core.reading_check import (  # noqa: E402
    readings_agree,
    readings_are_trustworthy,
    transcript_overlap,
    words,
)

PAGE = "Nu kan vi begynde med at lave dejen"
BUBBLE = "Vi er allerede i gang"
HALLUCINATION = "De begynte med det lille elefanten. Hva skal vi gjøre?"


class WordsTests(unittest.TestCase):
    def test_punctuation_and_case_are_ignored(self):
        self.assertEqual({"nu", "kan", "vi"}, words("Nu, kan vi!"))

    def test_line_breaks_do_not_matter(self):
        self.assertEqual(words("nu kan vi"), words("nu\nkan   vi"))

    def test_accents_are_kept(self):
        self.assertIn("déjens", words("lave déjens"))

    def test_single_letters_are_dropped(self):
        self.assertEqual(set(), words("a i o"))

    def test_empty_and_non_text_are_safe(self):
        self.assertEqual(set(), words(""))
        self.assertEqual(set(), words("   "))
        self.assertEqual(set(), words(None))


class OverlapTests(unittest.TestCase):
    def test_an_identical_reading_scores_one(self):
        self.assertEqual(1.0, transcript_overlap(PAGE, PAGE))

    def test_the_measured_hallucination_scores_far_below_the_threshold(self):
        # It shares the Danish function words "med" and "vi", so the score is 0.25 rather
        # than zero -- which is exactly why the threshold must sit well above that.
        score = transcript_overlap(HALLUCINATION, PAGE)
        self.assertLess(score, 0.4)
        self.assertFalse(readings_agree(HALLUCINATION, PAGE))

    def test_the_shorter_side_is_the_reference(self):
        # One call catching an extra line is not disagreement.
        self.assertEqual(1.0, transcript_overlap(PAGE, f"{PAGE} {BUBBLE}"))

    def test_an_empty_side_is_zero_not_one(self):
        self.assertEqual(0.0, transcript_overlap(PAGE, ""))
        self.assertEqual(0.0, transcript_overlap("", PAGE))


class AgreementTests(unittest.TestCase):
    def test_two_matching_readings_agree(self):
        self.assertTrue(readings_agree(PAGE, PAGE))

    def test_a_reading_with_an_extra_line_agrees(self):
        self.assertTrue(readings_agree(PAGE, f"{PAGE}\n{BUBBLE}"))

    def test_the_hallucination_is_refused(self):
        self.assertFalse(readings_agree(HALLUCINATION, PAGE))

    def test_a_half_wrong_reading_is_refused_at_the_default_threshold(self):
        self.assertFalse(readings_agree("Nu kan vi helt andre ord her", PAGE))

    def test_the_threshold_is_configurable(self):
        # Three shared words out of six: 0.5, so it passes a 0.4 gate but not a 0.6 one.
        mixed = "nu kan vi noget helt andet"
        self.assertTrue(readings_agree(mixed, PAGE, minimum_overlap=0.4))
        self.assertFalse(readings_agree(mixed, PAGE, minimum_overlap=0.6))

    def test_an_empty_reading_never_agrees(self):
        self.assertFalse(readings_agree("", PAGE))
        self.assertFalse(readings_agree(PAGE, ""))


class TrustworthyTests(unittest.TestCase):
    def test_two_agreeing_readings_are_trustworthy(self):
        self.assertTrue(readings_are_trustworthy([PAGE, PAGE]))

    def test_a_single_reading_is_never_enough(self):
        # Nothing distinguishes one reading from the invented sentence.
        self.assertFalse(readings_are_trustworthy([PAGE]))
        self.assertFalse(readings_are_trustworthy([]))
        self.assertFalse(readings_are_trustworthy(None))

    def test_disagreeing_readings_are_refused(self):
        self.assertFalse(readings_are_trustworthy([PAGE, HALLUCINATION]))

    def test_empty_readings_are_ignored_rather_than_counted(self):
        self.assertFalse(readings_are_trustworthy([PAGE, "", None]))


if __name__ == "__main__":
    unittest.main()
