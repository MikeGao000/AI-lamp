# INNFOS NE30-36 接入预研

更新：2026-09-15。用户提出改用 INNFOS NE 30-36，当前按 QDD Lite-NE30-36 关节模组理解；具体铭牌、替换轴数和通信配件待确认。本次为资料预研，未安装 SDK、未修改运行配置、未连接硬件。

## 已核实的项目入口

### 已找回的用户硬件记录

用户确认没有 ECB，使用微雪 CAN HAT。已从历史对话「接入RS485 CAN HAT」找回用户提供的教程：https://www.waveshare.net/wiki/RS485_CAN_HAT 。device/DEVICE.md 同样记录为 Waveshare RS485 CAN HAT（基础版）。历史对话记录显示，该板与 SERVO42D 的只读通信已验证成功；这不等于已验证 INNFOS 通信。后续按已有板卡与 CAN 环境研究直接 CAN 适配，无需再次向用户索取型号或原 Wiki。

- https://github.com/mintasca/innfos-gluon-controller ：README 明确对应 QDD Lite-NE30-36 六轴机械臂，包含接线、示教和再现流程。其 500g 末端负载、42V 供电等是整臂参数，不作为单关节额定参数。历史接线流程为电脑 → 以太网 ECB → HUB → 执行器，提到终端电阻和回馈制动电容。
- https://github.com/mintasca/innfos-cpp-sdk ：旧 innfos/ActuatorController_SDK 地址重定向到此。包含 example、sdk、tools；README 指向以太网 C++ SDK 文档，列出旧版 Windows/Linux 构建环境。不能据此认定兼容当前树莓派 ARM64。
- https://github.com/mintasca/INNFOS_CAN_SDK_STM32 ：CAN 控制参考工程，README 标注 STM32F429VET6、V1.5.3；参考底层固定为 1Mbps。SCA_Protocol.c/h 负责封包解包，SCA_API.c/h 提供参数读写接口，SCA_APP.c/h 是演示，bsp_can.c/h 是硬件 CAN 层。该速率仍需与实物配置核对。README 标注 All rights reserved，公开可读不等于授权任意复制分发。
- https://github.com/mintasca/innfos-gluon-cpp-sdk-raspi ：存在树莓派版本，包含 example 和 sdk；尚未验证二进制架构、系统兼容性及单关节接口。
- https://github.com/mintasca/ros_gluon ：可作为后续机械臂集成参考，当前台灯接入不必先引入 ROS。

## 对当前项目的影响与建议

本地已有 MKS CAN 单轴路径，docs/实机单轴动作测试.md 使用 MKS counts、RPM、acc 和可选 13.7:1 减速比。以上数值不能套用 NE30-36。lamp_core/motor_interface.py 的通用实机入口目前仍为未配置即拒绝执行。

建议继续让上层轨迹使用输出关节弧度，在新增 INNFOS 适配层实现单位换算、方向和零点映射、反馈及故障处理。更换关节后还需重新核对质量、惯量、机械限位与运动速度；原有姿态须重新标定。

两条候选路线：

1. 有匹配的 ECB/HUB：先用匹配的上位机验证单轴，再核对 C++ SDK 与树莓派兼容性。
2. 直接 CAN：参考 STM32 协议层，移植到现有树莓派 CAN 通道；硬件适配器是否复用取决于接口、电气和驱动支持，不能仅凭同为 CAN 判断。

## 下一步必须核实的细节

- 铭牌完整型号、固件版本、替换关节数量及轴位。
- 电源规格、插头引脚、CAN 终端、电源回馈处理、配套 ECB/HUB 或 USB-CAN 型号。
- 协议位置单位究竟对应电机轴还是输出轴，减速比、字节序、有符号及定点数格式。
- 扫描 ID、读取位置/速度/电流/温度/故障、使能、模式切换、目标写入和超时处理的实际 API。
- 抱闸与失能行为。历史六轴 README 要求失能前托住机械臂，因此不能假定失能后仍能承重。

验证顺序：读取身份和状态 → 确认单位与当前角度 → 低速小角度单轴动作 → 验证停止及失联行为 → 多轴轨迹。

## 本次证据边界

已阅读仓库页面、README 与本地接口；尝试读取 CAN 源文件的 raw 地址失败，尚未逐字节核验协议，也没有完成编译或实机测试。单关节额定扭矩、峰值扭矩、额定电流和实际减速比暂不填写，避免把整臂或其他型号的数据混入。
