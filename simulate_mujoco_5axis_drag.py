"""Open an interactive, offline pose-teaching window for the five-axis lamp.

Run from the repository root:
    .\\.venv-mujoco\\Scripts\\python.exe simulate_mujoco_5axis_drag.py

Use the right-side ``Actuator`` sliders (j1_position through j5_position) to
drag each simulated motor target.  The model is gravity-neutral in this mode
so a manually chosen pose remains visible.  This program never opens CAN/GPIO
and does not command real motors.
"""

from __future__ import annotations

import mujoco
import mujoco.viewer

from lamp_core.pose_library import IDLE_POSE


MODEL_PATH = "simulations/mujoco/lamp_5axis.xml"
ACTUATOR_TO_JOINT = {
    "j1_position": "j1_base_yaw",
    "j2_position": "j2_shoulder",
    "j3_position": "j3_elbow",
    "j4_position": "j4_neck_pitch",
    "j5_position": "j5_head_yaw",
}


def initialise_drag_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Start at idle, with each slider synchronized to its joint angle."""
    model.opt.gravity[:] = (0.0, 0.0, 0.0)
    for actuator_name, joint_name in ACTUATOR_TO_JOINT.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_index = model.jnt_qposadr[joint_id]
        target = IDLE_POSE[joint_name]
        data.qpos[qpos_index] = target
        data.ctrl[actuator_id] = target
    mujoco.mj_forward(model, data)


def formatted_pose(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    values: list[str] = []
    for joint_name in ACTUATOR_TO_JOINT.values():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        values.append(f"{joint_name}={data.qpos[model.jnt_qposadr[joint_id]]:+.3f}")
    return ", ".join(values)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    initialise_drag_pose(model, data)
    print("Offline five-axis drag/teach mode")
    print("Drag the five Actuator sliders in the right panel to set J1–J5.")
    print("For a temporary force test: double-click a body, then hold Ctrl and drag.")
    print("Press F1 in the window for all MuJoCo controls; close the window to print the final pose.")
    mujoco.viewer.launch(model, data, show_left_ui=True, show_right_ui=True)
    print(f"Final drag pose: {formatted_pose(model, data)}")


if __name__ == "__main__":
    main()
