"""Evaluate the *existing* page-contour upgrade on real frames.

camera_preview.detect_document_bbox() turns a detected text box into a page box
when a contour surrounds it, and the tracker follows that page centre instead of
the paragraph centre. That is the local half of the plan-B design, so this
measures whether it actually produces a page on this rig -- the same way an
earlier hand-tuned page finder was measured and rejected.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, "/home/lamp/AI-lamp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output-dir", default="/tmp/docbbox")
    parser.add_argument("--long-side", type=int, default=320)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import cv2

    import camera_preview
    from lamp_core.text_detection import PpocrTextDetector, text_anchor_box

    detector = PpocrTextDetector(None, long_side=args.long_side)
    if not detector.available:
        print("no text model", file=sys.stderr)
        return
    paths = sorted(glob.glob(os.path.join(args.directory, "*.jpg")))
    if not paths:
        print("no frames", file=sys.stderr)
        return
    os.makedirs(args.output_dir, exist_ok=True)

    found = 0
    contains = 0
    ratios: list[float] = []
    print(f"{'frame':<28} {'text area':>9} {'page area':>9} {'ratio':>6} {'holds':>6}")
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = detector.detect(image)
        anchor = text_anchor_box(boxes)
        if anchor is None:
            print(f"{os.path.basename(path):<28} {'-':>9} {'-':>9} {'-':>6} {'-':>6}")
            continue
        tx1 = int(anchor[0] * width)
        ty1 = int(anchor[1] * height)
        tx2 = int(anchor[2] * width)
        ty2 = int(anchor[3] * height)
        text_box = (tx1, ty1, tx2 - tx1, ty2 - ty1)
        text_area = (tx2 - tx1) * (ty2 - ty1) / float(width * height)

        page_box = camera_preview.detect_document_bbox(
            image, cv2, text_evidence=text_box
        )
        vis = image.copy()
        cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (0, 165, 255), 2)
        if page_box is None:
            print(f"{os.path.basename(path):<28} {text_area:.3f} {'-':>9} "
                  f"{'-':>6} {'-':>6}")
        else:
            px, py, pw, ph = page_box
            page_area = pw * ph / float(width * height)
            holds = (px <= tx1 and py <= ty1
                     and px + pw >= tx2 and py + ph >= ty2)
            ratio = page_area / max(1e-6, text_area)
            found += 1
            contains += int(holds)
            ratios.append(ratio)
            print(f"{os.path.basename(path):<28} {text_area:.3f} {page_area:.3f} "
                  f"{ratio:6.1f} {'yes' if holds else 'no':>6}")
            cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(args.output_dir, "doc_" + os.path.basename(path)), vis)

    print()
    print(f"page box found      : {found} / {len(paths)}")
    if ratios:
        ratios.sort()
        print(f"page/text area ratio: min {ratios[0]:.1f}  "
              f"median {ratios[len(ratios) // 2]:.1f}  max {ratios[-1]:.1f}")
        print(f"page box holds text : {contains} / {found}")
    print(f"overlays -> {args.output_dir}")


if __name__ == "__main__":
    main()
