# Autonomous OS 逻辑拆解与本项目落地

> 本文研究的是公开架构与 Markdown 契约，**不复制其 GPL/Apache 源码，也不把其舵机协议移植到 MKS**。本项目保留既定硬件：Pi 3B + CAN HAT + MKS SERVO42D CAN × 5；实际 CAN 帧必须以购买电机所附、版本匹配的手册为准。

## 结论：应该借什么，不该照搬什么

Autonomous OS 的核心价值不是某一个动作，而是把“会思考”与“能动硬件”彻底分层：

```text
云端/本地 AI：理解、对话、提出有限行为建议
        ↓  结构化意图（绝不含 CAN 帧或关节角）
本地行为编排：事件去重、冷却、情绪连续性、SEQ / INT / TRJ
        ↓  已批准的动作 ID / 目标物 ID
本地安全门：急停、限位、温度、CAN 心跳、软限位、速度/加速度
        ↓  安全后的轨迹
MKS CAN：各轴编码器位置闭环、FOC、驱动本轴运动
```

可直接借鉴：设备声明、人格声明、技能白名单、安全声明、事件驱动感知、短促表情动作、历史记录与有限习惯推断。

不可照搬：其串口舵机驱动、其重型 Go + Python + React 全栈、实时云语音依赖、用户身份/人脸存储、以及将每次对话都绑定一条动作的策略。Pi 3B 1 GB 和普通单 USB 麦克风不适合完整复刻这些组件。

## 运行决策（2026-08-28）

- 主控固定为 **Raspberry Pi 3B**。可以验证并采用 Autonomous OS，但以轻量配置运行：低频、事件触发的视觉；本地固定指令/VAD；复杂视觉与开放对话按需交给云端。持续运行本地大视觉模型、人脸身份库或重型实时语音不在第一版范围。
- 本项目的 `J4`（颈俯仰）和 `J5`（灯头）并不等同于参考灯的“头部”坐标。OS 的 `look_at`、`expression` 等语义必须由本地动作适配层映射为本机已校准的五轴姿态/动作 ID；绝不照搬参考系统的关节角、限位或舵机动作。
- 电机只接受本地已批准的 CAN 位置/速度/加速度命令。到达位置后是否保持、保持电流和通信丢失后的动作，均以实物的 MKS 固件手册和单轴带载测试为准；未验证前不得把“位置到达”视为“可长期承重”。

## 1. 四份声明文件：采用为本项目的固定契约

| Autonomous OS 概念 | 本项目文件 | 本地消费者 | 作用 |
|---|---|---|---|
| `DEVICE.md` | [`../device/DEVICE.md`](../device/DEVICE.md) | 启动检查、UI、驱动装配 | 声明真正安装的板子、轴、传感器与能力 |
| `SOUL.md` | [`../personality/SOUL.md`](../personality/SOUL.md) | AI 提示词构造器 | 定义稳定角色，不定义电机角度 |
| `SKILL.md` | 本文第 5 节的本地技能表 | `behavior.py` / 未来行为服务 | 限制 AI 可以提议的能力 |
| `SAFETY.md` | [`../device/SAFETY.md`](../device/SAFETY.md) | `safety.py`、真实 CAN 驱动 | 可机读的物理边界；在 AI 之下强制执行 |

这四个文件不是给人“看一下就算了”。启动时读取 `DEVICE.md` 和 `SAFETY.md`；每次 AI 请求前读取或缓存 `SOUL.md` 与技能目录；每一个会产生运动的请求都再次经过安全门。

## 2. 感知逻辑：事件，不是视频流直接控制电机

参考系统每约 2 秒读取一帧，并将运动、人物存在、环境亮度和声音变成事件，再给不同事件设置冷却时间。这个方向适合本项目，但应做更轻量的版本：

| 本地事件 | 初版来源 | 建议冷却 | 默认处理 | 是否询问 AI |
|---|---|---:|---|---|
| `person_enter` | 人体/人脸框（不做身份） | 15 s | `GREET` 或温和注视 | 否 |
| `hand_enter` | 手部检测 | 5 s | 短暂 `LOOK_AT` | 否 |
| `book_stable` | 静止页 + 书本框 | 10 s | `READ_POSE` | 仅需要讲解时 |
| `object_stable` | 支持类别的目标框 | 20 s | `LOOK_AT` / `CURIOUS_TILT` | 需要语义时 |
| `voice_wake` | 本地 VAD + 唤醒词 | 0 s | `LISTENING` | 指令/对话时 |
| `direction_correction` | 本地允许语音意图 | 0 s | `INT` → 左/右/中注视姿态 | 否 |
| `limit_or_can_fault` | GPIO/CAN 心跳 | 0 s | 停机、拒绝一切运动 | 永不询问 |

必须实现的三个保护：

1. **稳定门**：目标连续出现 N 帧才触发，丢失 2–3 次才离开，避免灯来回摇摆。
2. **冷却门**：同一来源在冷却期内仅更新内部注意力，不重复说话或播放大动作。
3. **动作静止门**：要拍照或请求云端视觉时，先停止追踪，等待关节停止后的 0.3–0.5 秒再抓帧；完成后才允许继续追踪。

### 人物方位的 Pi 3B 适配

先实现三段式 `LEFT / CENTER / RIGHT` 注意力，不实现逐帧追脸。默认未知时可以低速 `RIGHT → LEFT` 各扫描一次；视觉一旦稳定确认某扇区便锁定。用户说“我在这里 / 另一边 / 你看错了 / 看左边”时，本地意图解析器直接触发 `INT`，切换至指定或相反扇区并重新确认。这样保留了“会找人”的互动感，却不会把检测框抖动放大成五轴抖动。

## 3. 语音逻辑：快速本地路径与慢速 AI 路径分开

参考系统把闲聊和复杂任务分开。我们不用其云端实时语音堆栈，但采用相同原则：

```text
USB Mic → 本地 VAD → 唤醒词「小灯小灯」
     ├─ STOP / 急停 / 回待机 / 开始或停止示教 → 本地立即执行
     ├─ 简单状态查询 → 本地模板回答
     └─ 阅读、解释、开放式聊天、物体语义 → 发送给 AI
```

- 只有在 TTS 静音后才重开单麦克风识别（半双工），防止台灯听见自己。
- `STOP` 必须绕开云端、动作队列和冷却逻辑，直接走安全中断。
- AI 语音回复中附带的动作也只是提议；语音即使成功，动作被拒绝也不应重试猛动。

## 4. 情绪与性格：短暂表达 + 连续状态，而非“每句话演一次”

参考系统将 `emotion` 映射为伺服动作、LED 和显示表情的组合，并将环境灯光场景与人格表达分开。这是非常好的产品逻辑。我们的适配：

```text
场景 Scene（长期）：阅读 / 夜间 / 静音 / 待机
       与
表达 Expression（短暂）：好奇 / 倾听 / 认可 / 欢迎 / 思考 / 关怀
       与
状态 State（连续）：energy、attention、current_target、last_action
```

不要一比一照搬其二十余种情绪。第一版只开放以下 9 个表达：

`idle`、`curious`、`listening`、`thinking`、`acknowledge`、`greeting`、`caring`、`nod`、`headshake`。

每一个表达在本地映射为：动作片段 ID、最大幅度、最大速度、LED 覆盖时长。AI 只能选择枚举名和 `0.0–1.0` 强度；本地将强度限制为 0.25–0.65（`greeting` 等安全动画例外），并添加 3–10 秒冷却。`idle` 自动回归，AI 不应不断发送它。

## 5. 本项目技能白名单（替换自由文本“控制机器人”）

| 技能 | AI 可提供的参数 | 本地必须验证 | 允许结果 |
|---|---|---|---|
| `attention.look_at` | `target_id` | 目标当前可见、置信度、软限位 | 小幅注视轨迹 |
| `expression.show` | 枚举情绪、强度 | 枚举、冷却、运动状态 | 动作片段 + LED 临时层 |
| `reading.start` | 可选 `book_id` | 书存在且稳定 | 阅读姿态、语音流程 |
| `motion.replay` | 录制轨迹名 | 文件签名、轨迹校验、低速 | `TRJ` 回放 |
| `interaction.touch_target` | `target_id` | ToF、可达性、软胶头、障碍/限位、低速 | 极低速接近、立即回退 |
| `voice.speak` | 短文本 | 长度、静音时段 | TTS；不影响安全 |
| `scene.set` | `reading/night/idle` | 时段与亮度上限 | 灯光/状态层 |

**禁止技能**：`can.raw_frame`、`motor.set_current`、`motor.set_foc`、`joint.set_angle`、任意 shell、任意 Python。AI 永远不拥有它们。

## 6. 习惯与“自主学习”：只学习偏好，不学习危险动作

参考系统的 `habit` 技能从多天的存在、活动、姿态等 JSONL 记录中，至少收集 3 天再推断时间相关习惯，并要求数据不足时不编造。这个克制原则应保留。

本项目第一版只存以下非敏感事件，并默认可在网页清除：

```json
{"ts":"2026-08-24T19:10:00+02:00","type":"book_stable","duration_s":420}
{"ts":"2026-08-24T19:12:00+02:00","type":"reading_started","source":"voice"}
{"ts":"2026-08-24T19:25:00+02:00","type":"reading_stopped","source":"object_lost"}
```

推断要求：至少 3 天、至少 2 次同类事件才形成“弱偏好”；至少 5 天、3 次以上才可做一次**非打扰式建议**。例如“通常这个时段会看书，需要我调到阅读姿态吗？”不得自动开始说故事、移动到人脸附近或上传旧图像。

动作学习分两类：

- **允许：**你手动示教 → 保存编码器轨迹 → 离线平滑/限速/碰撞检查 → 标注动作名 → `TRJ` 回放。
- **暂不允许：**让模型通过试错自行探索碰撞、快速触碰、关节极限。未来即便接入 ACT/SmolVLA 等策略，所有目标仍必须逐点经过 `SafetyGate`，先在仿真和受限台架验证。

## 7. 安全引擎：采用它最重要的原则，但收紧默认策略

参考项目强调“安全在大脑之下”，并将限制写成可测试的纯函数，在真正执行前的单一入口强制检查。我们将它落实为：

```text
行为请求 / UI 请求 / 回放请求 / 视觉跟踪请求
             ↓
SafetyGate.check_and_clamp()
  急停、限位、CAN 心跳、校准状态、温度、目标、速度/加速度、软限位
             ↓
SmartMotionDispatcher → MKS CAN adapter
```

与参考项目不同，本项目对**运动**采用更保守的默认：安全文件缺失、轴未校准、MKS 协议版本未知、CAN 心跳丢失，均为**拒绝运动**，而不是放行。因为这是有 5 个大扭矩步进轴的自建原型。

验收必须包含三层：纯函数单元测试、接真实 MKS 的单轴低速测试、以及绕过审计（AI/API/UI/回放/追踪都无法绕开同一个安全入口）。

## 8. 代码落点与下一批实现

现有核心已有对应基础：

| 目标 | 已有文件 | 本轮后应增加/修改 |
|---|---|---|
| SEQ/INT/TRJ | `lamp_core/smart_motion.py` | 统一让 UI、AI、追踪都走它 |
| 安全拒绝 | `lamp_core/safety.py` | 读取 `device/SAFETY.md`，加 CAN/校准状态 |
| 行为提议 | `lamp_core/behavior.py` | 增加技能白名单、表达强度、冷却与稳定门 |
| 手动示教 | `lamp_core/teach.py` | 接入 MKS 位置读回后完成真实记录 |
| 语音意图 | `lamp_core/voice.py` | VAD/唤醒词和半双工音频流 |
| 云端理解 | `lamp_core/cloud.py` | 返回受限 `BehaviorProposal`，不返回电机指令 |

推荐的下一次编码顺序：先做 `EventRouter + CooldownRegistry` 的纯 Python 测试；再让 `behavior.py` 输出“表达动作 ID”；最后才在单台 MKS 上接真实 CAN 适配器。这样每一层都可离线验证。

## 公开来源与采用边界

- [Autonomous OS 总览](https://github.com/autonomous-ai/autonomous-os) — 设备声明、技能、HAL、安全在 AI 下方的分层思想。
- [Architecture Overview](https://github.com/autonomous-ai/autonomous-os/blob/main/docs/overview.md) — 感知事件、冷却与三层服务组织。
- [Safety Engine](https://github.com/autonomous-ai/autonomous-os/blob/main/docs/safety.md) — 单一执行闸门、可测试边界、旁路审计。
- [Realtime Voice Agent](https://github.com/autonomous-ai/autonomous-os/blob/main/docs/realtime-voice.md) — 快慢路径分流、拍照前稳定、图像调用冷却。
- [Autonomous Lamp / LeLamp Runtime](https://github.com/autonomous-ai/autonomous-lamp/blob/main/lelamp/README.md) — 舵机校准、示教与回放的产品能力参考。
