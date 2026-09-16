"""Report what the text detector sees in each frame of a dataset directory.

Built to answer one question after a sweep: at which J1 angle is the book
actually in view? It also doubles as a way to triage captured frames before
labelling, and to spot the small stray text fragments that must not be allowed
to capture the anchor.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--text-model-path", default=None)
    parser.add_argument("--long-side", type=int, default=320)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import cv2

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
    if not paths:
        print(f"no frames in {args.directory}", file=sys.stderr)
        return

    print(f"{'frame':<28} {'boxes':>5} {'anchor':<26} {'area':>7}")
    best: tuple[float, str, tuple[float, float, float, float]] | None = None
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"{os.path.basename(path):<28} unreadable")
            continue
        boxes = detector.detect(image)
        anchor = text_anchor_box(boxes)
        if anchor is None:
            print(f"{os.path.basename(path):<28} {len(boxes):>5} {'-':<26} {'-':>7}")
            continue
        area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
        centre = (anchor[0] + anchor[2]) / 2.0
        shown = f"x[{anchor[0]:.2f},{anchor[2]:.2f}] c={centre:.2f}"
        print(f"{os.path.basename(path):<28} {len(boxes):>5} {shown:<26} {area:>7.3f}")
        if best is None or area > best[0]:
            best = (area, os.path.basename(path), anchor)

    print()
    if best is None:
        print("no frame yielded a page anchor -- the book is not in view anywhere in this sweep")
        return
    print(f"largest anchor: {best[1]} area={best[0]:.3f} "
          f"x[{best[2][0]:.2f},{best[2][2]:.2f}]")


if __name__ == "__main__":
    main()
