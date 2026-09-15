# 外部姿态参考映射

更新日期：2026-09-08

## 采用顺序

1. **Watti** 是当前的首要参照：它与本项目同为 `base → shoulder → elbow → neck → head` 五轴桌面灯。我们采用它的“每轴姿态关键帧 + 转场时间 + 虚拟预览”的场景组织方式；本项目的 MuJoCo 五轴模型、`POSE_LIBRARY` 与 minimum-jerk 转场就是该方式的本地实现。Watti 目前尚未发布源代码或示例动作文件，所以没有可下载的数值角度库。
2. **LeLamp** 是表达动作和手动示教/回放流程的参照，但其第 4、5 轴与本项目不等价。
3. **通用六轴机械臂与 LeRobot** 提供标定、示教、数据集和模仿训练流程；不导入它们的绝对关节角或连杆比例。

## LeLamp 的转译

LeLamp 可以作为本项目的表达动作参考，但不能把它的关节角直接抄到五轴步进台灯上。

| 表达含义 | LeLamp 关节协同 | 本项目 J1--J5 转译 |
| --- | --- | --- |
| 看向 | base yaw 为主，头端配合 | J1 与 J5 反向部分抵消，保持灯头面向目标 |
| 点头 / 问候 | base/elbow/wrist pitch 联动 | J2、J3、J4 联动：`nod_up` → `nod_down` |
| 倾头好奇 | pitch + wrist roll 的小幅组合 | 本机构无 roll，改为 J1、J2--J4、J5 同向的 `curious_left/right` |
| 摇头 | 头端关节为主 | J5 主导，J1 只作 0.06 rad 的伴随，使用 `head_shake_left/right` |
| 倾听 | 轻微向前姿态 | J2--J4 的 `listening_pose`，不用 yaw 伪装 |

LeLamp 的轴序为 `base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch`；本项目为 `J1 base_yaw, J2 shoulder_pitch, J3 elbow_pitch, J4 neck_pitch, J5 head_yaw`。所以 LeLamp 的 wrist roll 没有一对一等价轴，而 J5 在本结构中是灯头水平 yaw。

数值姿态不从 LeLamp 直接复制：其公开运行时将姿态录为每台设备本地校准后的 CSV，再回放，零位、舵机方向和连杆安装角均属于具体实体设备。当前本地弧度值是对 Watti 风格几何和 MuJoCo 静态安装角的重新定向结果，保存在 `lamp_core/pose_library.py`，并由全有向转场训练与 MuJoCo 回放验证。

参考：

- Watti：<https://github.com/Nikolay-Tyulkin/Watti/blob/main/docs/ARCHITECTURE.md>、<https://github.com/Nikolay-Tyulkin/Watti/blob/main/docs/HARDWARE.md>
- LeLamp：<https://github.com/humancomputerlab/LeLamp/blob/master/docs/2.%20Servos%20Setup.md>、<https://github.com/humancomputerlab/lelamp_runtime>
- LeRobot：<https://github.com/huggingface/lerobot/blob/main/docs/source/il_robots.mdx>
