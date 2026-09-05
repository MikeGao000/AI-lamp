"""Offline edit helper for recorded manual lamp motions."""

from __future__ import annotations

import argparse

from lamp_core.teach import load, save, smooth
from simulate_system import LIMITS


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--radius", type=int, default=1)
args = parser.parse_args()
save(smooth(load(args.input, LIMITS), LIMITS, args.radius), LIMITS, args.output)
print(f"saved edited motion: {args.output}")
