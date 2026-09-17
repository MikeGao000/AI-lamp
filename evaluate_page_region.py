"""Compare the fused page anchor against the text-block anchor on real frames.

Reference-free by design: it reports whether the page box contains the text the
detector found, how much bigger it is, and how often it is available at all, then
writes overlays so the boxes can be judged by eye rather than by a number.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output-dir", default="/tmp/pagereg")
    parser.add_argument("--text-model-path", default=None)
    parser.add_argument("--long-side", type=int, default=320)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def _contains(outer, inner) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _text_union(text_boxes):
    """Union of the detected text boxes, with no padding applied.

    Containment has to be judged against this rather than against the anchor:
    the anchor is deliberately grown by 15% while the page box's seed is grown by
    4%, so comparing the two reported 18 of 38 real frames as "the box misses the
    text" when the only excess was the anchor's own padding.
    """

    if not text_boxes:
        return None
    xs = [v for box in text_boxes for v in (box.bbox[0], box.bbox[2])]
    ys = [v for box in text_boxes for v in (box.bbox[1], box.bbox[3])]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def main() -> None:
    args = build_parser().parse_args()
    import cv2

    from lamp_core.page_region import page_region_box
    from lamp_core.text_detection import PpocrTextDetector, text_anchor_box

    detector = PpocrTextDetector(
        args.text_model_path,
        long_side=args.long_side,
        score_threshold=args.score_threshold,
    )
    if not detector.available:
        print("no text detection model found", file=sys.stderr)
        return

    paths = sorted(glob.glob(os.path.join(args.directory, "*.jpg")))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no frames in {args.directory}", file=sys.stderr)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    page_areas: list[float] = []
    text_areas: list[float] = []
    contains = 0
    both = 0
    page_missing = 0
    text_missing = 0

    print(f"{'frame':<30} {'text area':>9} {'page area':>9} {'holds':>6} {'conf':>6} {'source':<13}")
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        text_boxes = detector.detect(image)
        text_anchor = text_anchor_box(text_boxes)
        text_union = _text_union(text_boxes)
        region = page_region_box(image, cv2, text_boxes=text_boxes)
        name = os.path.basename(path)

        if text_anchor is None:
            text_missing += 1
            text_area_text = "-"
        else:
            text_area = (text_anchor[2] - text_anchor[0]) * (text_anchor[3] - text_anchor[1])
            text_areas.append(text_area)
            text_area_text = f"{text_area:.3f}"
        if region is None:
            page_missing += 1
            print(f"{name:<30} {text_area_text:>9} {'-':>9} {'-':>6} {'-':>6} {'-':<13}")
            region_box = None
        else:
            page_area = (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
            page_areas.append(page_area)
            holds = "-"
            if text_union is not None:
                both += 1
                ok = _contains(region.bbox, text_union)
                contains += int(ok)
                holds = "yes" if ok else "no"
            print(
                f"{name:<30} {text_area_text:>9} {page_area:.3f}   {holds:>6} "
                f"{region.confidence:>6.2f} {region.source:<13}"
            )
            region_box = region.bbox

        vis = image.copy()
        height, width = image.shape[:2]
        if text_anchor is not None:
            x1, y1, x2, y2 = (
                int(text_anchor[0] * width), int(text_anchor[1] * height),
                int(text_anchor[2] * width), int(text_anchor[3] * height),
            )
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        if region_box is not None:
            x1, y1, x2, y2 = (
                int(region_box[0] * width), int(region_box[1] * height),
                int(region_box[2] * width), int(region_box[3] * height),
            )
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(args.output_dir, f"pagereg_{name}"), vis)

    print()
    if page_areas:
        print(f"frames with a page box      : {len(page_areas)} / {len(paths)}"
              f"   (mean area {sum(page_areas) / len(page_areas):.3f})")
    else:
        print("no frame produced a page box")
    if text_areas:
        print(f"frames with a text anchor   : {len(text_areas)}"
              f"   (mean area {sum(text_areas) / len(text_areas):.3f})")
    if page_areas and text_areas:
        print(f"page box holds the text     : {contains} / {both}"
              f"   ({100.0 * contains / max(1, both):.0f}%)")
        print(f"page box is larger by       : "
              f"{(sum(page_areas) / len(page_areas)) / (sum(text_areas) / len(text_areas)):.1f}x")
    print(f"page box unavailable        : {page_missing} frames")
    print(f"text anchor unavailable     : {text_missing} frames")
    print(f"overlays -> {args.output_dir}")


if __name__ == "__main__":
    main()
