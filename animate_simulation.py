"""Open a Tkinter window that animates the offline five-axis lamp trajectory."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

from lamp_core.motion import plan_synchronised_minimum_jerk
from simulate import JOINT_LIMITS


START = {name: 0.0 for name in JOINT_LIMITS}
TARGET = {
    "j1_base_yaw": 0.60,
    "j2_shoulder": -0.35,
    "j3_elbow": 0.45,
    "j4_neck_pitch": 0.22,
    "j5_head_yaw": -0.50,
}


class LampSimulationApp:
    WIDTH, HEIGHT = 920, 700
    ORIGIN = (455, 535)
    LINK_LENGTHS = (165, 135, 70)

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("五轴阅读伴侣灯 — 离线运动仿真")
        self.frames = plan_synchronised_minimum_jerk(
            START, TARGET, JOINT_LIMITS, sample_period_s=0.02
        )
        self.index = 0
        self.running = False

        self.canvas = tk.Canvas(root, width=self.WIDTH, height=self.HEIGHT, bg="#faf8f2")
        self.canvas.pack(padx=12, pady=(12, 6))
        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle_play)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="重置", command=self.reset).pack(side="left", padx=8)
        self.status = ttk.Label(controls, text="状态：READY（离线仿真，不控制真实电机）")
        self.status.pack(side="left", padx=18)
        self.draw()

    @staticmethod
    def point(origin: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
        return origin[0] + length * math.cos(angle), origin[1] + length * math.sin(angle)

    def draw_link(self, left: tuple[float, float], right: tuple[float, float], width: int = 28) -> None:
        self.canvas.create_line(*left, *right, width=width, fill="#d7bf94", capstyle="round")
        self.canvas.create_line(*left, *right, width=2, fill="#68563f", capstyle="round")

    def draw_joint(self, center: tuple[float, float], name: str, angle: float) -> None:
        x, y = center
        radius = 28 if name in ("J2", "J3") else 21
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#4b5557", outline="#1c2426", width=2)
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#d7bf94", outline="")
        self.canvas.create_text(x, y - radius - 16, text=f"{name}  {math.degrees(angle):+.1f}°", fill="#273238", font=("Segoe UI", 10, "bold"))

    def draw(self) -> None:
        point = self.frames[self.index]
        p = point.positions_rad
        self.canvas.delete("all")
        self.canvas.create_text(self.WIDTH / 2, 26, text="五轴步进灯：逐帧离线轨迹", fill="#273238", font=("Segoe UI", 18, "bold"))
        self.canvas.create_text(self.WIDTH / 2, 53, text=f"t = {point.time_s:.2f} s    帧 {self.index + 1} / {len(self.frames)}", fill="#52636a", font=("Segoe UI", 11))

        ox, oy = self.ORIGIN
        self.canvas.create_oval(ox - 150, oy - 56, ox + 150, oy + 56, fill="#d9d0bd", outline="#68563f", width=2)
        self.canvas.create_oval(ox - 100, oy - 33, ox + 100, oy + 33, fill="#ebe4d6", outline="#68563f", width=2)
        self.canvas.create_text(ox, oy + 82, text="低重心底座", fill="#52636a", font=("Segoe UI", 11))
        # J1 is yaw: in this side-view it is shown as the rotating marker in the base.
        marker_end = self.point((ox, oy), -math.pi / 2 + p["j1_base_yaw"], 26)
        self.canvas.create_line(ox, oy, *marker_end, fill="#cc4b45", width=4, arrow="last")

        a2 = -math.pi / 2 + p["j2_shoulder"]
        p2 = self.point((ox, oy - 10), a2, self.LINK_LENGTHS[0])
        a3 = a2 + p["j3_elbow"]
        p3 = self.point(p2, a3, self.LINK_LENGTHS[1])
        a4 = a3 + p["j4_neck_pitch"]
        p4 = self.point(p3, a4, self.LINK_LENGTHS[2])
        head_angle = a4 + p["j5_head_yaw"]
        head_center = self.point(p4, head_angle, 30)

        self.draw_link((ox, oy - 10), p2, 34)
        self.draw_link(p2, p3, 30)
        self.draw_link(p3, p4, 24)
        self.draw_joint((ox, oy - 10), "J1", p["j1_base_yaw"])
        self.draw_joint((ox, oy - 10), "J2", p["j2_shoulder"])
        self.draw_joint(p2, "J3", p["j3_elbow"])
        self.draw_joint(p3, "J4", p["j4_neck_pitch"])
        self.draw_joint(p4, "J5", p["j5_head_yaw"])

        hx, hy = head_center
        self.canvas.create_oval(hx - 46, hy - 30, hx + 46, hy + 30, fill="#e6dcc8", outline="#68563f", width=2)
        beam_end = self.point(head_center, head_angle, 44)
        self.canvas.create_line(hx, hy, *beam_end, fill="#f0b84c", width=8, capstyle="round")
        self.canvas.create_text(hx, hy + 48, text="灯头", fill="#52636a", font=("Segoe UI", 10))

        self.canvas.create_text(18, 670, anchor="w", text="红色箭头：J1 底座偏航在侧视图中的方向指示；其余关节为侧视平面投影。", fill="#52636a", font=("Segoe UI", 10))

    def toggle_play(self) -> None:
        self.running = not self.running
        self.play_button.configure(text="暂停" if self.running else "播放")
        self.status.configure(text="状态：MOVING（离线仿真）" if self.running else "状态：READY（离线仿真）")
        if self.running:
            self.advance()

    def advance(self) -> None:
        if not self.running:
            return
        self.index += 1
        if self.index >= len(self.frames):
            self.index = len(self.frames) - 1
            self.running = False
            self.play_button.configure(text="播放")
            self.status.configure(text="状态：READY（目标姿态已到达；未控制硬件）")
        self.draw()
        if self.running:
            self.root.after(20, self.advance)

    def reset(self) -> None:
        self.running = False
        self.index = 0
        self.play_button.configure(text="播放")
        self.status.configure(text="状态：READY（离线仿真，不控制真实电机）")
        self.draw()


if __name__ == "__main__":
    app_root = tk.Tk()
    LampSimulationApp(app_root)
    app_root.mainloop()
