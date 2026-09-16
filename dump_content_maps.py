"""Dump the intermediate maps behind page_region_box so they can be looked at.

Designing a content threshold from synthetic tests went wrong once already: the
synthetic background was perfectly flat, the real desk is covered in speckle,
cables and a laptop, and the resulting box was the entire frame. This prints the
distribution and writes the maps as images, which is the only honest way to pick
a threshold.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--output-prefix", default="/tmp/maps")
    parser.add_argument("--sample-width", type=int, default=160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import cv2
    import numpy as np

    image = cv2.imread(args.image)
    if image is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return
    height, width = image.shape[:2]
    sample_height = max(1, int(round(args.sample_width * height / float(width))))
    small = cv2.resize(image, (args.sample_width, sample_height))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    peak = float(energy.max())
    energy = energy / peak if peak > 0 else energy

    print(f"image {width}x{height} -> working {args.sample_width}x{sample_height}")
    for name, plane in (("saturation", saturation), ("edge energy", energy)):
        quantiles = np.quantile(plane, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        print(f"  {name:<12} mean={plane.mean():.3f}  "
              + "  ".join(f"p{int(q * 100)}={v:.3f}" for q, v in zip([10, 25, 50, 75, 90, 99], quantiles)))
    for threshold in (0.15, 0.25, 0.35, 0.5):
        sat_fraction = float((saturation > threshold).mean())
        energy_fraction = float((energy > threshold).mean())
        print(f"  above {threshold:.2f}: saturation {sat_fraction * 100:5.1f}% of pixels, "
              f"edge energy {energy_fraction * 100:5.1f}%")

    for label, plane in (
        ("sat", saturation),
        ("energy", energy),
        ("content", np.clip(saturation + energy, 0.0, 2.0) / 2.0),
    ):
        scaled = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{args.output_prefix}_{label}.jpg", scaled)
        print(f"  wrote {args.output_prefix}_{label}.jpg")


if __name__ == "__main__":
    main()
