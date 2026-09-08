"""Local five-axis pose library used by the ideal action compiler.

The public LeLamp design is used as an *expression reference*, not as a
source of directly copyable angles.  Its joints are ``yaw, pitch, pitch,
roll, pitch`` while this lamp is ``yaw, pitch, pitch, pitch, yaw``.  The
reference therefore contributes coordinated pose motifs (look, nod, shake,
and curious lean); the values below are calibrated local radians for the
photo-based five-axis MuJoCo geometry.
"""

from __future__ import annotations


IDLE_POSE = {
    "j1_base_yaw": 0.0,
    "j2_shoulder": 0.0,
    "j3_elbow": 0.0,
    "j4_neck_pitch": 0.0,
    "j5_head_yaw": 0.0,
}

POSE_LIBRARY = {
    "look_left": {
        "j1_base_yaw": -0.55,
        "j2_shoulder": -0.08,
        "j3_elbow": 0.12,
        "j4_neck_pitch": 0.10,
        "j5_head_yaw": 0.30,
    },
    "look_right": {
        "j1_base_yaw": 0.55,
        "j2_shoulder": -0.08,
        "j3_elbow": 0.12,
        "j4_neck_pitch": 0.10,
        "j5_head_yaw": -0.30,
    },
    "reading_pose": {
        "j1_base_yaw": 0.25,
        "j2_shoulder": -0.20,
        "j3_elbow": 0.30,
        "j4_neck_pitch": 0.12,
        "j5_head_yaw": -0.18,
    },
    "nod_down": {
        "j1_base_yaw": 0.0,
        "j2_shoulder": -0.06,
        "j3_elbow": 0.08,
        "j4_neck_pitch": -0.18,
        "j5_head_yaw": 0.0,
    },
    # LeLamp's two pitch joints are expressed here as a coordinated J2--J4
    # move.  This is the upward half of the nod and is deliberately smaller
    # than nod_down so the greeting feels like a lamp acknowledgement.
    "nod_up": {
        "j1_base_yaw": 0.0,
        "j2_shoulder": 0.05,
        "j3_elbow": -0.07,
        "j4_neck_pitch": 0.14,
        "j5_head_yaw": 0.0,
    },
    # In the reference lamp, a listening posture is a slight forward lean,
    # not a yaw gesture.  J2--J4 retain the upright S silhouette.
    "listening_pose": {
        "j1_base_yaw": 0.0,
        "j2_shoulder": -0.12,
        "j3_elbow": 0.18,
        "j4_neck_pitch": -0.05,
        "j5_head_yaw": 0.0,
    },
    # This mechanism has no wrist-roll joint.  A small body yaw plus a head
    # yaw in the same direction is its local equivalent of a curious tilt.
    "curious_left": {
        "j1_base_yaw": -0.18,
        "j2_shoulder": -0.10,
        "j3_elbow": 0.18,
        "j4_neck_pitch": 0.22,
        "j5_head_yaw": -0.12,
    },
    "curious_right": {
        "j1_base_yaw": 0.18,
        "j2_shoulder": -0.10,
        "j3_elbow": 0.18,
        "j4_neck_pitch": 0.22,
        "j5_head_yaw": 0.12,
    },
    # Unlike look_left/right, these hold the base nearly still and use the
    # terminal yaw motor, so they visibly read as a head shake rather than a
    # whole-body scan in this J1--J5 layout.
    "head_shake_left": {
        "j1_base_yaw": -0.06,
        "j2_shoulder": 0.0,
        "j3_elbow": 0.0,
        "j4_neck_pitch": 0.0,
        "j5_head_yaw": -0.50,
    },
    "head_shake_right": {
        "j1_base_yaw": 0.06,
        "j2_shoulder": 0.0,
        "j3_elbow": 0.0,
        "j4_neck_pitch": 0.0,
        "j5_head_yaw": 0.50,
    },
    "near_configured_envelope": {
        "j1_base_yaw": 1.35,
        "j2_shoulder": -0.68,
        "j3_elbow": 0.82,
        "j4_neck_pitch": 0.60,
        "j5_head_yaw": -0.90,
    },
}
