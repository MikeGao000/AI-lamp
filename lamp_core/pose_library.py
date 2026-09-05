"""Local five-axis pose library used by the ideal action compiler.

The numeric poses are the existing project profiles, moved here unchanged so
the simulator and the LeLamp expression layer share one source of truth.
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
    "near_configured_envelope": {
        "j1_base_yaw": 1.35,
        "j2_shoulder": -0.68,
        "j3_elbow": 0.82,
        "j4_neck_pitch": 0.60,
        "j5_head_yaw": -0.90,
    },
}
