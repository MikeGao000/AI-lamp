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

**Status: evaluated on real frames and NOT adopted. Nothing imports this module.**

The offline evaluation on 39 real frames (`evaluate_page_region.py`) plus the
overlays it writes say the same thing as the picture: this does not find the page.
Measured after four rounds of fixes:

* On the synthetic block the box overshoots horizontally by up to 0.09 and moves
  by 0.078 depending on which text line seeded it, so the one property it was
  built for -- a target that does not move when a different text line is found --
  does not hold.
* On real frames the resulting box measures 0.18-0.26 of the frame while the text
  anchor measures 0.09-0.15: about twice the text block, which is not obviously
  the page.
* With no text it returned a confident box around a **laptop keyboard** on a
  frame where the book was out of view. That path is off by default for that
  reason.

The measurements behind this are worth keeping; the code is not. It is left in
place only as the record of a tested-and-rejected approach. The cloud model
already returns a page-shaped box on this rig (measured: x[0.13,1.00] y[0.35,1.00]
and x[0.37,0.63] y[0.12,0.84]), which is the better source for a page-level
target, and a *reliable local* page detector needs the fine-tuning route rather
than more hand-tuned thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Structure above this counts as content. 0.15 kept ~13% of pixels on this rig,
#: but a real frame's laptop bezel, keyboard, cables and speckled desk all carry
#: structure too, and at 0.15 the box grew straight over them onto a keyboard.
#: 0.25 keeps ~6% and is what actually stops at the page's own content.
CONTENT_THRESHOLD = 0.25

#: Content above this line is monitor or window chrome, not the working area,
#: mirroring the top-edge rule the text anchor already uses.
CONTENT_TOP_MARGIN = 0.06

#: How far the box grows per accepted step, as a fraction of the frame.
GROW_STEP = 0.04

#: A ring joins the box only while it is at least this dense relative to the box.
#: Halving is enough to cross the gaps between letters and lines while still
#: stopping at the page margin.
GROW_FRACTION = 0.5

#: Absolute floor for a ring as well, because this rig's clutter -- a laptop
#: keyboard notably -- is structured enough to pass a purely relative test.
GROW_MIN_DENSITY = 0.25
MAX_GROWTH_STEPS = 25

#: Seed size when there is no text to start from.
SEED_FRACTION = 0.08

#: Content sits inside the page margins, so the box is grown a little at the end.
PAGE_MARGIN = 0.02

#: A box outside these bounds is not a page and must not become the target.
MINIMUM_AREA = 0.02
MINIMUM_WIDTH = 0.15
MINIMUM_HEIGHT = 0.10

#: A box larger than this is the whole scene rather than a page, which the first
#: version of this module returned on 39 out of 39 real frames.
MAXIMUM_AREA = 0.90


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
    # Drop the monitor chrome *before* scaling, not after: a bright band across
    # the top of the frame otherwise sets the reference that everything else is
    # measured against, which shrank the page's own content below the threshold.
    top_rows = int(round(sample_height * top_margin))
    if top_rows:
        energy[:top_rows, :] = 0.0
    # Scale by a high quantile rather than the maximum, so one hot pixel cannot
    # flatten everything else.
    reference = float(np.quantile(energy, 0.995))
    if reference <= 0.0:
        return None, 0, 0
    energy = np.clip(energy / reference, 0.0, 1.0)
    return energy, sample_width, sample_height


def _region_mass(mask, box):
    """Structure mass and pixel count inside a normalized box."""

    height, width = mask.shape
    left = max(0, int(box[0] * width))
    right = min(width, max(left + 1, int(box[2] * width)))
    top = max(0, int(box[1] * height))
    bottom = min(height, max(top + 1, int(box[3] * height)))
    patch = mask[top:bottom, left:right]
    return float(patch.sum()), float(patch.size)


def _density(mask, box) -> float:
    mass, area = _region_mass(mask, box)
    return mass / area if area else 0.0


def _extend(x1: float, y1: float, x2: float, y2: float, axis: str, delta: float):
    """One step outward on a single side, built from the *current* box."""

    if axis == "x1":
        return (max(0.0, x1 + delta), y1, x2, y2)
    if axis == "x2":
        return (x1, y1, min(1.0, x2 + delta), y2)
    if axis == "y1":
        return (x1, max(0.0, y1 + delta), x2, y2)
    return (x1, y1, x2, min(1.0, y2 + delta))


def _grow_box(
    mask,
    seed,
    *,
    step: float,
    fraction: float,
    min_density: float,
    max_steps: int,
):
    """Expand a box while the material it newly covers still looks like content.

    Two details here were both wrong in the first version, and both only showed
    up when the intermediates were printed:

    * The candidate boxes were built once per pass from the coordinates at the
      start of that pass, so accepting a later side silently overwrote an earlier
      one. Since "down" was evaluated last, the net effect was a box that could
      only ever grow downward -- measured, it walked from y 0.46 to the bottom
      edge of the frame while x never moved.
    * The test used the box's *average* density, which lets a box that already
      covers content keep absorbing empty space (1.00 to 0.61 without ever
      failing). The test is now on the ring being added, so growth stops where
      the content stops.

    The ring must also pass an absolute floor. A purely relative test let a real
    frame's laptop keyboard and bezel keep the box growing until it covered the
    whole scene.
    """

    x1, y1, x2, y2 = seed
    reference = max(_density(mask, seed) * fraction, min_density)
    if reference <= 0.0:
        return seed
    mass, area = _region_mass(mask, seed)
    for _ in range(max_steps):
        grew = False
        for axis, delta in (("x1", -step), ("x2", step), ("y1", -step), ("y2", step)):
            candidate = _extend(x1, y1, x2, y2, axis, delta)
            if candidate == (x1, y1, x2, y2):
                continue
            candidate_mass, candidate_area = _region_mass(mask, candidate)
            added = candidate_area - area
            if added <= 0.0:
                continue
            if (candidate_mass - mass) / added >= reference:
                x1, y1, x2, y2 = candidate
                mass, area = candidate_mass, candidate_area
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
    allow_textless: bool = False,
) -> PageRegion | None:
    """Bounding box of the page implied by the structure around the detected text.

    Without text there is nothing reliable to anchor to. That path was measured
    on a real frame where the book was out of view: it returned a confident box
    around a **laptop keyboard**. It is therefore off by default, so a page-sized
    target is only ever produced when detected text pins it to the page, and the
    caller otherwise keeps its existing behaviour.
    """

    import numpy as np

    if not text_boxes and not allow_textless:
        return None
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
    anchor = None
    if text_boxes:
        # Seed from the *vetted* dominant block, not from a union of the raw
        # boxes. Measured on a real frame, the raw union included the monitor
        # taskbar strip at y 0.004 and a stray box at y 0.55, so the seed was
        # already 0.84 x 0.62 of the frame before any growth and there was
        # nothing left to grow into.
        from lamp_core.text_detection import text_anchor_box

        anchor = text_anchor_box(text_boxes)
        if anchor is None:
            return None
        seed = (
            max(0.0, anchor[0] - grow_step),
            max(0.0, anchor[1] - grow_step),
            min(1.0, anchor[2] + grow_step),
            min(1.0, anchor[3] + grow_step),
        )
        source = "content+text"
    if seed is None:
        seed = _densest_seed(mask, SEED_FRACTION)
    if seed is None:
        return None

    x1, y1, x2, y2 = _grow_box(
        mask,
        seed,
        step=grow_step,
        fraction=grow_fraction,
        min_density=GROW_MIN_DENSITY,
        max_steps=MAX_GROWTH_STEPS,
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
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area < MINIMUM_AREA or area > MAXIMUM_AREA:
        return None

    density = _density(mask, box)
    if density <= 0.0:
        return None
    # Density is the honest confidence: a page's content region measures far
    # denser than the whole frame, so a frame-sized box scores poorly by itself.
    confidence = density if text_boxes else density * 0.6
    return PageRegion(box, round(min(1.0, confidence), 4), source, round(density, 4))
