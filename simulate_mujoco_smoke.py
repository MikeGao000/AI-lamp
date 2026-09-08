"""Minimal headless MuJoCo verification for the lamp training environment.

Run with:
    .\\.venv-mujoco\\Scripts\\python.exe simulate_mujoco_smoke.py
"""

from __future__ import annotations

import mujoco


MODEL_XML = """
<mujoco model="lamp_smoke">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <joint name="j1_base_yaw" type="hinge" axis="0 0 1"
             range="-1 1" limited="true" damping="2" armature="0.01"/>
      <geom type="box" size="0.08 0.08 0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="base_yaw_position" joint="j1_base_yaw" kp="30"/>
  </actuator>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    data.ctrl[0] = 0.4
    for _ in range(1_000):
        mujoco.mj_step(model, data)

    if abs(data.qpos[0] - 0.4) > 0.01:
        raise RuntimeError(f"position actuator did not converge: {data.qpos[0]:.3f} rad")
    print(
        f"PASS MuJoCo {mujoco.__version__}: "
        f"nq={model.nq}, actuator={model.nu}, j1={data.qpos[0]:.3f} rad"
    )


if __name__ == "__main__":
    main()
