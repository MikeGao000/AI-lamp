"""Find the page, not just a line of its text.

Measured on this rig, every wobble in the aim traced back to one thing: the
anchor was a *text block*, so it changed whenever a different line was detected
(a large story line scores 0.18-0.20, a speech bubble 0.09) and the aim point
moved about 6 degrees each time. An illustration-only page produced no anchor at
all. Nothing in the pipeline was looking for the page.

**What actually separates a page from this desk.** Dumping the intermediate maps
on a real frame settled it:

* *Saturation is useless here.* The lamp colours the whole scene, so 99.2% of
  pixels measure above 0.15 saturation -- the page, the desk, the cables and the
  monitor alike. An earlier version of this module leaned on saturation and
  produced a box covering 0.995 of the frame on 39 out of 39 real frames.
* *Edge energy works.* Only 13% of pixels exceed 0.15, because the page's margins
  and the desk are both smooth: structure is concentrated on printed content.

So the page is found as **the dense, contiguous region of structure**, grown
outward from the text that was detected, stopping where the structure thins out.
Growing a box ring by ring (rather than taking a global bounding box, or doing
connected components) is what keeps the monitor and the cables out: a smooth gap
stops the growth before it can reach them.

Without any text the densest patch seeds the growth instead, which is what lets
an illustration-only spread still be found.

Confidence is the structure density inside the box, which is a meaningful number
-- a page's content region measures far denser than the whole frame does.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Structure above this counts as content. 0.15 keeps ~13% of pixels on this rig,
#: which is the printed content and little else.
CONTENT_THRESHOLD = 0.15

#: Content above this line is monitor or window chrome, not the working area,
#: mirroring the top-edge rule the text anchor already uses.
CONTENT_TOP_MARGIN = 0.06

#: How far the box grows per accepted step, as a fraction of the frame.
GROW_STEP = 0.04

#: A ring joins the box only while it is at least this dense relative to the box.
#: Halving is enough to cross the gaps between letters and lines while still
#: stopping at the page margin.
GROW_FRACTION = 0.5
MAX_GROWTH_STEPS = 25

#: Seed size when there is no text to start from.
SEED_FRACTION = 0.08

#: Content sits inside the page margins, so the box is grown a little at the end.
PAGE_MARGIN = 0.06

#: A box outside these bounds is not a page and must not become the target.
MINIMUM_AREA = 0.02
MINIMUM_WIDTH = 0.15
MINIMUM_HEIGHT = 0.10


@dataclass(frozen=True)
class PageRegion:
    """A page-sized target with an honest confidence."""

    bbox: tuple[float, float, float, float]
    confidence: float
    source: str
    density: float


def _structure_map(image, cv2_module, sample_width: int, top_margin: float):
    """Normalised edge energy, with monitor chrome above the working area removed."""

    import numpy as np

    height, width = image.shape[:2]
    sample_height = max(1, int(round(sample_width * height / float(width))))
    small = cv2_module.resize(image, (sample_width, sample_height))
    gray = cv2_module.cvtColor(small, cv2_module.COLOR_BGR2GRAY)
    # Smooth before differencing: a Laplacian on raw pixels is dominated by
    # single-pixel sensor noise, which would make speckle look like content.
    gray = cv2_module.GaussianBlur(gray, (3, 3), 0)
    energy = np.abs(np.asarray(cv2_module.Laplacian(gray, cv2_module.CV_32F)))
    # Scale by a high quantile rather than the maximum, so one hot pixel cannot
    # flatten everything else.
    reference = float(np.quantile(energy, 0.995))
    if reference <= 0.0:
        return None, 0, 0
    energy = np.clip(energy / reference, 0.0, 1.0)
    top_rows = int(round(sample_height * top_margin))
    if top_rows:
        energy[:top_rows, :] = 0.0
    return energy, sample_width, sample_height


def _density(mask, box) -> float:
    height, width = mask.shape
    left = max(0, int(box[0] * width))
    right = min(width, max(left + 1, int(box[2] * width)))
    top = max(0, int(box[1] * height))
    bottom = min(height, max(top + 1, int(box[3] * height)))
    patch = mask[top:bottom, left:right]
    return float(patch.mean()) if patch.size else 0.0


def _grow_box(mask, seed, *, step: float, fraction: float, max_steps: int):
    """Expand a box ring by ring while the new ring stays dense enough."""

    x1, y1, x2, y2 = seed
    current = _density(mask, (x1, y1, x2, y2))
    if current <= 0.0:
        return seed
    for _ in range(max_steps):
        grew = False
        for candidate in (
            (max(0.0, x1 - step), y1, x2, y2),
            (x1, y1, min(1.0, x2 + step), y2),
            (x1, max(0.0, y1 - step), x2, y2),
            (x1, y1, x2, min(1.0, y2 + step)),
        ):
            if candidate == (x1, y1, x2, y2):
                continue
            if _density(mask, candidate) >= fraction * current:
                x1, y1, x2, y2 = candidate
                current = _density(mask, candidate)
                grew = True
        if not grew:
            break
    return (x1, y1, x2, y2)


def _densest_seed(mask, fraction: float, *, columns: int = 8, rows: int = 6):
    """Seed box at the densest coarse cell, for frames with no text at all."""

    height, width = mask.shape
    best = None
    best_mass = -1.0
    for row in range(rows):
        for column in range(columns):
            left = column / columns
            right = (column + 1) / columns
            top = row / rows
            bottom = (row + 1) / rows
            mass = _density(mask, (left, top, right, bottom))
            if mass > best_mass:
                best_mass = mass
                best = (left, top, right, bottom)
    if best is None or best_mass <= 0.0:
        return None
    # Shrink the winning cell to a seed-sized box around its middle so the first
    # growth step is measured against content, not against a whole cell.
    centre_x = (best[0] + best[2]) / 2.0
    centre_y = (best[1] + best[3]) / 2.0
    half = fraction / 2.0
    return (
        max(0.0, centre_x - half),
        max(0.0, centre_y - half),
        min(1.0, centre_x + half),
        min(1.0, centre_y + half),
    )


def page_region_box(
    image: object,
    cv2_module: object,
    *,
    text_boxes: object = None,
    sample_width: int = 160,
    content_threshold: float = CONTENT_THRESHOLD,
    margin: float = PAGE_MARGIN,
    top_margin: float = CONTENT_TOP_MARGIN,
    grow_step: float = GROW_STEP,
    grow_fraction: float = GROW_FRACTION,
) -> PageRegion | None:
    """Bounding box of the page implied by the structure around the detected text.

    ``text_boxes`` may be an empty list or None; without text the densest patch
    seeds the growth, which is what lets an illustration-only spread be found.
    Returns None when nothing page-shaped is there, so the caller can fall back
    rather than aim at a guess.
    """

    import numpy as np

    structure, sample_width_used, sample_height = _structure_map(
        image, cv2_module, sample_width, top_margin
    )
    if structure is None:
        return None
    mask = structure > content_threshold
    if not mask.any():
        return None

    source = "content"
    seed = None
    if text_boxes:
        xs = []
        ys = []
        for box in text_boxes:
            bbox = getattr(box, "bbox", box)
            xs += [float(bbox[0]), float(bbox[2])]
            ys += [float(bbox[1]), float(bbox[3])]
        if xs and ys:
            seed = (
                max(0.0, min(xs) - grow_step),
                max(0.0, min(ys) - grow_step),
                min(1.0, max(xs) + grow_step),
                min(1.0, max(ys) + grow_step),
            )
            source = "content+text"
    if seed is None:
        seed = _densest_seed(mask, SEED_FRACTION)
    if seed is None:
        return None

    x1, y1, x2, y2 = _grow_box(
        mask, seed, step=grow_step, fraction=grow_fraction, max_steps=MAX_GROWTH_STEPS
    )

    span_x = x2 - x1
    span_y = y2 - y1
    box = (
        max(0.0, x1 - span_x * margin),
        max(0.0, y1 - span_y * margin),
        min(1.0, x2 + span_x * margin),
        min(1.0, y2 + span_y * margin),
    )
    if (box[2] - box[0]) < MINIMUM_WIDTH or (box[3] - box[1]) < MINIMUM_HEIGHT:
        return None
    if (box[2] - box[0]) * (box[3] - box[1]) < MINIMUM_AREA:
        return None

    density = _density(mask, box)
    if density <= 0.0:
        return None
    # Density is the honest confidence: a page's content region measures far
    # denser than the whole frame, so a frame-sized box scores poorly by itself.
    confidence = density if text_boxes else density * 0.6
    return PageRegion(box, round(min(1.0, confidence), 4), source, round(density, 4))
