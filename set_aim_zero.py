"""Move the axis to an offset, then adopt that position as zero.

Needed because the service re-zeros at start-up: after a sweep ends at +30 deg the
book (which sat at -15 deg of the sweep's frame) is 45 deg away and outside the
service's +-40 search range, so it can never be found. Shifting the zero puts the
book back near the middle of the search envelope.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/lamp/AI-lamp")

from lamp_core.mks_can_protocol import (  # noqa: E402
    ChecksumMode,
    absolute_coordinate_move,
    set_bus_enabled,
    set_working_mode,
    set_zero_point,
)
from lamp_core.mks_single_axis import (  # noqa: E402
    COUNTS_PER_REVOLUTION,
    MksSingleAxisProbe,
)
from run_mks_single_axis_motion import SocketCanTransport  # noqa: E402

OFFSET = float(sys.argv[1]) if len(sys.argv) > 1 else -35.0
NODE = 1
MODE = ChecksumMode("additive")
COUNTS_PER_DEGREE = COUNTS_PER_REVOLUTION / 360.0

transport = SocketCanTransport("can0")
transport.send(set_working_mode(NODE, 0x05, MODE))
transport.send(set_zero_point(NODE, MODE))
probe = MksSingleAxisProbe(transport, NODE, MODE)
time.sleep(0.2)
anchor = probe.snapshot().encoder_counts
transport.send(set_bus_enabled(NODE, True, MODE))
time.sleep(0.2)

# Fast settings: measured 0.61 s for 20 deg with 0.18 deg accuracy.
target = anchor + round(OFFSET * COUNTS_PER_DEGREE)
transport.send(absolute_coordinate_move(NODE, 120, 255, target, MODE))
deadline = time.time() + 15.0
while time.time() < deadline:
    snapshot = probe.snapshot()
    if snapshot.rpm == 0 and abs(snapshot.encoder_counts - target) <= 24:
        break
    time.sleep(0.05)
time.sleep(0.8)
moved = (probe.snapshot().encoder_counts - anchor) / COUNTS_PER_DEGREE
print(f"moved {moved:+.2f} deg")

# Re-zero here so the book sits near the middle of the +-40 search envelope.
transport.send(set_zero_point(NODE, MODE))
time.sleep(0.3)
print(f"new zero set; position now {probe.snapshot().encoder_counts} counts")
