#!/bin/sh
# Bring the MKS servo bus up at every boot.
#
# The Pi has an mcp251x SPI CAN HAT (`can0`), and nothing configured it persistently:
# after the reboot on 2026-09-16 the interface came up DOWN, so camera_preview.py died
# in its first encoder read with "OSError: [Errno 100] Network is down" and the preview
# looked like a code failure when it was a missing boot step.
#
# The bitrate was never recorded in this project -- the docs still say
# "<fill in per the MKS CAN manual>". It is now measured on this rig: at
# `bitrate 500000` (sample-point 0.833) the encoder answers, so that is the default.
# Pass another value if a different motor is ever fitted:
#
#     sudo sh scripts/enable_can0.sh 1000000
#
# Verify with:
#     python3 -c "import sys; sys.path.insert(0,'.'); \
#       from lamp_core.mks_can_protocol import ChecksumMode; \
#       from lamp_core.mks_single_axis import MksSingleAxisProbe; \
#       from run_mks_single_axis_motion import SocketCanTransport; \
#       print(MksSingleAxisProbe(SocketCanTransport('can0'),1,ChecksumMode('additive')).snapshot())"
set -e

BITRATE="${1:-500000}"
INTERFACE="${2:-can0}"
UNIT="/etc/systemd/system/${INTERFACE}-up.service"

echo "installing ${UNIT} at ${BITRATE} bit/s"

cat > "${UNIT}" <<EOF
[Unit]
Description=Bring up ${INTERFACE} for the MKS servo bus
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
# The "down" first is what makes this idempotent. Without it, running while the
# interface is already up fails with "RTNETLINK answers: Device or resource busy" and
# exit status 2 -- which is exactly how the first install of this unit failed, even
# though the bus itself was fine.
ExecStartPre=-/sbin/ip link set ${INTERFACE} down
ExecStart=/sbin/ip link set ${INTERFACE} up type can bitrate ${BITRATE}
ExecStop=/sbin/ip link set ${INTERFACE} down

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${INTERFACE}-up.service"
sleep 1
ip -details link show "${INTERFACE}" | head -3
echo "done: ${INTERFACE} should now survive a reboot"
