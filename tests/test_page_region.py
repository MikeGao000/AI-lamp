"""Tests for the fused page-region anchor.

The synthetic background here is deliberately *not* clean. An earlier version of
this module leaned on saturation and was only ever tested against a perfectly
flat synthetic desk, so it passed while returning a box covering 0.995 of the
frame on every one of 39 real frames -- the real desk is speckled, and cables,
a laptop and a monitor all carry structure. Two of the cases below exist
specifically to fail that version: a speckled desk, and a textureless blob.
"""

import unittest

import numpy as np

from lamp_core.page_region import (
    MINIMUM_HEIGHT,
    MINIMUM_WIDTH,
    PageRegion,
    page_region_box,
)


class _NumpyCV2:
    """Just enough OpenCV for page_region_box, backed by numpy."""

    COLOR_BGR2GRAY = 1
    CV_32F = 5
    BORDER_REFLECT = 4

    @staticmethod
    def resize(image, size):
        height, width = image.shape[:2]
        rows = np.clip(np.arange(size[1]) * height // size[1], 0, height - 1)
        columns = np.clip(np.arange(size[0]) * width // size[0], 0, width - 1)
        return image[rows][:, columns]

    @staticmethod
    def cvtColor(image, code):
        return image[..., :3].astype(np.float32).mean(axis=2)

    @staticmethod
    def GaussianBlur(image, size, sigma):
        padded = np.pad(image.astype(np.float32), 1, mode="edge")
        return (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

    @staticmethod
    def Laplacian(image, dtype):
        return (
            np.roll(image, 1, axis=0)
            + np.roll(image, -1, axis=0)
            + np.roll(image, 1, axis=1)
            + np.roll(image, -1, axis=1)
            - 4.0 * image
        )


class _Box:
    """Stands in for a TextBox, which is what the detector actually hands over."""

    def __init__(self, bbox):
        self.bbox = bbox


def _flat_desk():
    return np.full((240, 320, 3), 150, dtype=np.uint8)


def _textured_block(image, y0, y1, x0, x1, amplitude=90):
    """Line work inside a region: what a printed illustration actually looks like."""
    block = np.full((y1 - y0, x1 - x0, 3), 150, dtype=np.uint8)
    block[::4, :] = 150 - amplitude
    block[:, ::4] = 150 - amplitude
    image[y0:y1, x0:x1] = block


def _speckle(image, rate=0.04, seed=7):
    """Low-level noise everywhere: the real desk, cables and laptop edges."""
    generator = np.random.default_rng(seed)
    noise = generator.random(image.shape[:2]) < rate
    image[noise] = 230


class PageRegionTests(unittest.TestCase):
    def test_flat_desk_has_no_page(self):
        self.assertIsNone(page_region_box(_flat_desk(), _NumpyCV2()))

    def test_a_speckled_desk_does_not_balloon_the_box(self):
        # The regression that mattered: on real frames the box covered 0.995 of
        # the frame because background structure was treated as content.
        image = _flat_desk()
        _speckle(image)
        _textured_block(image, 70, 180, 90, 240)
        region = page_region_box(image, _NumpyCV2(), text_boxes=[_Box((0.30, 0.35, 0.70, 0.55))])
        self.assertIsNotNone(region)
        area = (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
        self.assertLess(area, 0.55)

    def test_a_textured_illustration_locates_a_page(self):
        image = _flat_desk()
        _textured_block(image, 70, 180, 90, 240)
        region = page_region_box(image, _NumpyCV2())
        self.assertIsNotNone(region)
        self.assertEqual("content", region.source)
        self.assertLess(region.bbox[0], 0.35)
        self.assertGreater(region.bbox[2], 0.65)

    def test_a_textureless_blob_is_not_a_page(self):
        # A solid colour patch has edges only at its border, so it carries no
        # content structure of its own.
        image = _flat_desk()
        image[70:180, 90:240] = (40, 40, 200)
        self.assertIsNone(page_region_box(image, _NumpyCV2()))

    def test_text_marks_the_source_and_raises_confidence(self):
        image = _flat_desk()
        _textured_block(image, 70, 180, 90, 240)
        without = page_region_box(image, _NumpyCV2())
        with_text = page_region_box(
            image, _NumpyCV2(), text_boxes=[_Box((0.30, 0.35, 0.70, 0.50))]
        )
        self.assertEqual("content+text", with_text.source)
        self.assertGreater(with_text.confidence, without.confidence)

    def test_monitor_chrome_above_the_frame_is_ignored(self):
        image = _flat_desk()
        _textured_block(image, 70, 180, 90, 240)
        _textured_block(image, 0, 8, 0, 320)
        region = page_region_box(image, _NumpyCV2())
        self.assertIsNotNone(region)
        self.assertGreater(region.bbox[1], 0.10)

    def test_a_box_is_always_inside_the_frame(self):
        image = _flat_desk()
        _textured_block(image, 0, 240, 0, 320)
        region = page_region_box(image, _NumpyCV2())
        if region is not None:
            self.assertGreaterEqual(region.bbox[0], 0.0)
            self.assertGreaterEqual(region.bbox[1], 0.0)
            self.assertLessEqual(region.bbox[2], 1.0)
            self.assertLessEqual(region.bbox[3], 1.0)

    def test_a_too_small_blob_is_not_a_page(self):
        image = _flat_desk()
        _textured_block(image, 118, 126, 158, 166)
        self.assertIsNone(page_region_box(image, _NumpyCV2()))

    def test_confidence_is_density_not_area(self):
        # A box the size of the frame explains the content poorly, so it must not
        # look more confident than a page-sized one.
        image = _flat_desk()
        _speckle(image, rate=0.02)
        _textured_block(image, 70, 180, 90, 240)
        page = page_region_box(image, _NumpyCV2(), text_boxes=[_Box((0.30, 0.35, 0.70, 0.55))])
        self.assertIsNotNone(page)
        self.assertLess(page.density, 0.75)
        self.assertGreater(page.density, 0.0)

    def test_the_bounds_are_the_documented_minimums(self):
        self.assertEqual(0.15, MINIMUM_WIDTH)
        self.assertEqual(0.10, MINIMUM_HEIGHT)

    def test_the_box_is_stable_when_only_the_detected_text_changes(self):
        # The whole point: the page box must not move because a different text
        # line was found this frame.
        image = _flat_desk()
        _textured_block(image, 70, 180, 90, 240)
        first = page_region_box(image, _NumpyCV2(), text_boxes=[_Box((0.30, 0.32, 0.70, 0.38))])
        second = page_region_box(image, _NumpyCV2(), text_boxes=[_Box((0.42, 0.50, 0.58, 0.56))])
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertLess(abs(first.bbox[0] - second.bbox[0]), 0.02)
        self.assertLess(abs(first.bbox[2] - second.bbox[2]), 0.02)

    def test_the_result_is_a_page_region(self):
        image = _flat_desk()
        _textured_block(image, 70, 180, 90, 240)
        self.assertIsInstance(page_region_box(image, _NumpyCV2()), PageRegion)


if __name__ == "__main__":
    unittest.main()
