---
device_id: mks_can_embodied_lamp
schema: lamp.device.v1
motion_axes: 5
---

# 设备声明（实际安装后必须更新）

## 主控与总线

- 主控：Raspberry Pi 3B（第一版）；升级目标 Raspberry Pi 5。
- 运动总线：Waveshare RS485 CAN HAT（基础版）→ CANH/CANL 主干。
- 电机：计划 MKS SERVO42D CAN × 5。仅当卖家提供的固件与 CAN 手册匹配时才启用真实驱动。
- 供电：24 V 电机电源独立并联分支、每轴熔断；Pi 使用独立降压 5 V 电源。CAN 不是电源线。

## 已声明能力

| 能力 | 初版状态 | 备注 |
|---|---|---|
| motion | `planned` | 必须单轴校准后才可变为 `ready` |
| camera | `ready` | Pi Camera Module 3 |
| voice_input | `planned` | 固定底座 USB UAC 单麦克风 |
| voice_output | `planned` | 扬声器/TTS；运行时半双工 |
| light | `planned` | 状态/阅读 LED 待选型 |
| distance | `planned` | VL53L1X 后续接入 I²C |
| display | `absent` | 第一版不安装眼睛屏幕 |

## 运动关节

`J1` 底座、`J2` 肩、`J3` 肘、`J4` 颈俯仰、`J5` 灯头。真实零点、齿比（当前预期 1:1）和每轴软限位必须在装配/回零后填写到 `SAFETY.md`；未填写视为不可动。

