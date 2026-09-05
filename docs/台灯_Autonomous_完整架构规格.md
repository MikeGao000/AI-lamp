# 绘本伴侣灯：Autonomous 完整架构规格（待确认）

版本：0.2  
日期：2026-09-01  
状态：架构基线；已完成不依赖协议的控制接口与仿真验证，未接入真实 CAN

## 0. 文档目的与备份

本文件定义“完整系统如何适配到五轴绘本伴侣灯”，范围包含设备声明、视觉与物体识别、语音识别与合成、实时对话、模型调用、记忆、技能、动作安全、遥测、部署和验证。

它吸收 Autonomous OS 的结构思想，但不直接移植其完整运行时或硬件代码。现有策略仍然有效，并已在本次写入前保留为只读快照：

- [整体方案与决策基线备份](备份_2026-08-31_整体方案与决策基线.md)
- [Autonomous OS 逻辑拆解备份](备份_2026-08-31_Autonomous_OS_逻辑拆解与本项目落地.md)

本文件与上述两份文档发生冲突时，以“已验证的安全边界”为优先；涉及 MKS 协议、轴负载、温升和校准的数据，始终以版本匹配的电机手册与实测为准。

## 1. 一句话决策

Raspberry Pi 3B 是身体、安全和 CAN 主机；云端或局域网计算负责高质量语音、视觉和对话；任何模型都只能提出受限的语义行为，不能直接控制电机、GPIO、CAN 或 Shell。

Autonomous OS 适合被当作架构参考，不适合原样安装。上游 Lamp 面向 Pi 4/5、Feetech 串口舵机和重型多进程依赖；本项目需要轻量、独立实现的 Pi 3B 版本，并为 MKS CAN 建立新的驱动和硬件在环验证。

## 2. 目标、非目标与硬件边界

### 2.1 产品目标

- 五轴台灯能安全地看、听、说、用姿态表达。
- 第一目标是绘本陪伴：阅读姿态、看向书页、回答问题、自然对话。
- 支持中文、英文、丹麦语，并允许每种语言采用不同的服务商或模型。
- 断网、模型出错或视觉不确定时，设备仍然能急停、停止发声、保持安全状态并说明有限的本地状态。
- 后续可以扩展物体定位、局域网 GPU 视觉、全双工实时语音和远程只读监控。

### 2.2 明确非目标

- 不在 Pi 3B 本地运行大语言模型、10B 级视觉模型或持续高帧率视觉推理。
- 不让模型自行探索电机动作、调 PID、电流、FOC 或机械极限。
- 第一版不进行人脸身份、声纹身份、儿童情绪画像或长期保存音视频。
- 不为了“看起来像完整 Autonomous OS”而引入 OpenClaw、通用在线技能商店、任意 Shell 工具或自动 OTA。
- 不把物体框、语音转写或远程模型输出直接接到关节坐标。

### 2.3 当前硬件边界

| 部件 | 固定职责 |
|---|---|
| Raspberry Pi 3B | Linux、CAN 主站、安全状态、动作调度、音视频接入、事件路由、网络客户端 |
| Waveshare CAN HAT | Pi 与 MKS CAN 总线的物理接口 |
| MKS SERVO42D CAN × 5 | 本轴编码器闭环、FOC、位置保持和底层故障报告 |
| Pi Camera 3 | 按需或低频视觉输入 |
| USB UAC 麦克风 / 扬声器 | 语音采集与播放 |
| 急停、NC 限位、24 V 电机电源支路 | 不依赖 Pi 或网络的最终物理保护 |
| 可选 ToF | 近距离障碍、接近动作的冗余保护 |
| 可选 ESP P4+C6 | 仅当实测证明 Pi 音频前端不足时承担媒体前端；不拥有 CAN 写权限 |
| 可选局域网 GPU | 开放词汇物体定位、重视觉模型或未来 VLX-Seek 类服务 |

## 3. 采用与拒绝：从 Autonomous OS 取什么

| 上游能力 | 本项目决定 | 原因 |
|---|---|---|
| 设备能力声明、人格、安全声明 | 采用并收紧 | 启动时能验证硬件、能力和安全配置是否一致 |
| 感知事件、冷却、优先级和运行追踪 | 采用 | 很适合避免重复说话、重复动作和模型费用失控 |
| 实时语音处理闲聊、主智能体处理复杂任务 | 采用其分层 | 语音响应更自然，复杂任务仍可有完整上下文 |
| 图像先描述、再给文本智能体 | 采用 | 让文字模型不必直接处理原始图像，也可防止视觉失败时猜测 |
| 模型 / 运行时可替换 | 采用并增强 | 本项目必须按任务和语言动态选择，而非单个全局模型 |
| 技能目录与能力门控 | 部分采用 | 保留受审计的功能声明；拒绝任意下载、任意执行的技能 |
| 直接硬件标记和通用 HTTP 调用 | 拒绝 | 模型不得构造硬件路径、关节角或 CAN 帧 |
| Feetech 舵机驱动、逐帧伺服追踪 | 拒绝 | MKS CAN 与本机五轴几何完全不同，必须重新实现 |
| 人脸库、声纹、情绪画像 | 默认不采用 | 儿童/家庭场景的隐私与资源成本不匹配 |
| 完整 Go + Python + React + OpenClaw 部署 | 拒绝 | Pi 3B 不适合承担这一整套运行时，且安全面过大 |
| JSONL 事件日志、SSE 监控 | 采用 | 便于现场调试、远程只读观察和复现一次动作 |
| 物理模拟器 | 只借接口模拟思想 | 仍需我们自己的 L1 至 L4 验证体系与真实机构模型 |

## 4. 系统总架构

    Pi Camera / USB Mic / 按钮 / ToF / 限位 / MKS CAN 反馈
                                 ↓
                  本地快速层：急停、VAD、唤醒、状态采样
                                 ↓
      EventRouter：去重、优先级、冷却、会话、trace_id、隐私过滤
                 ┌───────────────┼────────────────┐
                 ↓               ↓                ↓
          LocalIntent      VoiceSession      VisionAdapter
          固定本地命令      STT / TTS         帧筛选 / 描述 / 定位
                 └───────────────┼────────────────┘
                                 ↓
          ModelRouter：语言、任务、模态、延迟、预算、隐私模式
                                 ↓
          BehaviorPlanner：文本回复 + 受限 BehaviorProposal
                                 ↓
       SafetyGate：状态、限位、轨迹、温度、心跳、目标有效性、预算
                                 ↓
     MotionCompiler：语义动作 → 本机姿态 / 已验证轨迹 / MKS CAN 指令
                                 ↓
      MKSCanDriver：唯一 CAN 写入者 → 5 台 MKS SERVO42D
                                 ↓
        TelemetryStore + FlowLog + 本地只读监控 API / SSE

所有向下的箭头都是权限收缩。任何上层都可以建议或请求，只有安全层和电机进程可以真正写 CAN。

## 5. 进程、权限与启动顺序

Pi 上先实现三个独立服务，避免一个语音或模型问题影响电机安全。

| 服务 | 权限 | 负责 | 绝不负责 |
|---|---|---|---|
| lamp-motor-safety | CAN、限位、急停状态、只写电机遥测 | 安全状态机、校准、轨迹执行、CAN 心跳、故障停机 | 云端调用、图像上传、对话 |
| lamp-interaction | 相机、麦克风、扬声器、网络、只读电机状态 | VAD、STT/TTS、视觉、会话、模型路由、行为建议 | 原始 CAN、关节越界执行 |
| lamp-monitor | 只读日志和状态 | 本地页面、SSE、导出、诊断 | 修改电机参数、读取密钥、开放硬件接口 |

启动顺序如下：

1. 系统启动时，电机服务首先读取急停、限位、CAN 总线和设备配置。
2. 未验证校准、未发现全部必需节点、急停按下、故障码未清除或配置不完整时，进入 SAFE_DISABLED。
3. 只有通过单轴和整机条件后，才进入 READY_HOLD；此状态仍不自动移动。
4. interaction 服务等待 motor-safety 报告 READY 后才开启带动作的行为功能。
5. monitor 服务任何时候只能显示状态；网络失败不应影响 motor-safety。
6. 任何服务重启或心跳消失，都让 motor-safety 拒绝新的自由运动。

## 6. 统一数据契约

### 6.1 设备能力声明

以现有 DEVICE.md 和 SAFETY.md 为基础，未来增加机器可读的 LampProfile。它至少声明：

    device_id: picture-lamp-01
    board: raspberry_pi_3b
    motion: mks_servo42d_can
    joint_count: 5
    sensors: camera, microphone, speaker, limits
    required_capabilities: motion, audio, vision, system
    optional_capabilities: tof, esp_media, lan_gpu
    privacy_defaults: no_face_id, no_voice_id, snapshot_ttl
    safety_mode: fail_closed

启动时，配置与真实发现到的节点不一致就失败，不以“尽量启动”代替安全。

### 6.2 事件 Event

每个输入都转换为统一事件。事件不是动作命令。

    {
      "event_id": "uuid",
      "trace_id": "uuid",
      "ts_monotonic_ms": 0,
      "source": "voice | camera | ui | motor | limit | system",
      "type": "voice_final | visual_query | object_stable | motor_fault | stop",
      "priority": "emergency | command | interactive | ambient",
      "language_hint": "zh-CN | en | da-DK | unknown",
      "privacy_mode": "local_only | cloud_allowed",
      "payload": {},
      "image_ref": "ephemeral reference or null"
    }

急停、限位、CAN 故障和本地 STOP 使用 emergency 优先级，不排队、不等待模型。

### 6.3 视觉观察 VisualObservation

视觉服务返回事实与置信度，不返回电机角度。

    {
      "request_id": "uuid",
      "objects": [
        {
          "label": "book",
          "bbox_norm": [0.20, 0.18, 0.75, 0.91],
          "confidence": 0.89,
          "tracking_state": "stable"
        }
      ],
      "direction": "left | center | right | unknown",
      "frame_age_ms": 120,
      "source": "local | cloud_vlm | lan_gpu",
      "safe_to_act": false
    }

safe_to_act 只能由本地稳定门和安全层决定，云端模型无权把它设为 true。

### 6.4 会话 ConversationSession

每次对话明确保存语言和服务选择，避免中途换声线、换上下文或混合隐私策略。

    {
      "session_id": "uuid",
      "language": "zh-CN | en | da-DK",
      "stt_provider": "selected adapter",
      "chat_provider": "selected adapter",
      "tts_provider": "selected adapter",
      "voice_id": "selected voice",
      "mode": "half_duplex | realtime",
      "budget_remaining": {},
      "privacy_mode": "local_only | cloud_allowed"
    }

语言首先由 STT 自动识别；用户在本地 UI 或语音中明确指定时覆盖。会话内锁定，除非用户主动切换。

### 6.5 受限行为提案 BehaviorProposal

模型只能返回文字和 `ActionCatalog` 中**已启用、可由模型选择**的动作 ID。动作目录可逐步吸收 LeLamp 风格的问候、好奇、确认和拒绝等表达；每个条目必须声明来源、验证层级、强度范围、冷却、目标条件和本地 `motion_key`。模型不能发明目录外的动作。

    {
      "speech": "可以，我来看看。",
      "action_id": "LOOK_LEFT | ... | LELAMP_GREET_SMALL",
      "intensity": 0.0,
      "target_ref": "optional visual object id",
      "reason": "optional short text"
    }

首批基础动作可包括 `LOOK_LEFT`、`LOOK_CENTER`、`LOOK_RIGHT`、`READING_POSTURE`、`GENTLE_NOD`、`LISTENING_POSE`、`SET_LIGHT_SCENE`、`STOP` 和 `NONE`。后续动作仅在经过 L0 至适用验证层后进入目录；未登记或未验证的动作拒绝。接触/伸手类条目可以保存在目录中，但在 L4 目标可达性与实机验证前必须 `enabled=false` 且不可由模型选择。

当前 `LELAMP_GREET_SMALL`、`LELAMP_CURIOUS_TILT`、`LELAMP_ACKNOWLEDGE` 与 `LELAMP_HEAD_SHAKE` 采用“LeLamp 表达语义 + 本项目已验证姿态组合”的适配方式：通过本地 pose library 组合点头、看左、看右和回中，并由本项目的 minimum-jerk 多轴规划器生成 L1 理想轨迹；不复制 LeLamp 的硬件参数或驱动协议。

禁止字段包括：joint_angle、relative_degree、speed、torque、pid、can_frame、shell、url、script。`motion_key` 仅供本地 MotionCompiler 查找经过验证的动作库，模型无权提交轨迹或硬件参数。

### 6.6 电机遥测 MotorTelemetry

每个轴都应当有统一的可记录状态：

    {
      "joint": "J1",
      "node_id": 1,
      "mode": "disabled | holding | moving | fault",
      "target_position": 0.0,
      "actual_position": 0.0,
      "velocity": 0.0,
      "following_error": 0.0,
      "current_or_torque": 0.0,
      "temperature_c": 0.0,
      "bus_voltage_v": 0.0,
      "fault_code": "none",
      "last_heartbeat_ms": 0
    }

字段单位、MKS 可读性和轮询频率必须在拿到实物手册后冻结。无法读取的字段应明确标为 unavailable，而不是伪造为零。

## 7. 语音与三语言策略

### 7.1 快路径和慢路径

    麦克风
      ↓
    本地 VAD / 唤醒词 / 物理按钮
      ├─ STOP、静音、回待机、确认状态 → LocalIntent → SafetyGate
      └─ 正常语句
             ↓
          STT → 语言识别 → ConversationSession
             ├─ 简短闲聊 → 可选实时语音服务
             └─ 阅读、视觉、知识、记忆、动作建议 → 主模型

本地 STOP 必须覆盖中文、英文、丹麦语的确认短语，并且在 STT 不确定时允许物理按钮/网页停止。STOP 永远优先于 TTS、动作队列与正在进行的云端请求。

### 7.2 Pi 3B 的实际策略

- 第一版采用半双工：台灯说话时暂停普通单麦克风识别，避免听见自己。
- VAD、唤醒、短命令和音频采集在 Pi 本地运行。
- 高质量 STT、TTS 和聊天可以是云端适配器；网络失败时退回本地固定提示与安全动作。
- 全双工、打断、回声消除和连续视频聊天是独立验收项目。只有 Pi 实测 CPU、RSS、掉音率和延迟合格才启用。
- 若不合格，再引入 ESP P4+C6 作为媒体事件生产者。Pi 仍验证每个事件，且仍是唯一 CAN 主机。

### 7.3 模型服务商原则

不预先承诺某一个品牌。每个候选服务须在中文、英文、丹麦语上实测：

- STT：转写准确率、混说表现、首个结果延迟、断网降级。
- TTS：发音自然度、丹麦语可懂度、流式首音延迟、价格、可中断性。
- 实时语音：打断、回声环境、工具委托是否可靠。
- 对话模型：儿童内容约束、三语言质量、视觉能力、费用和可用性。
- 视觉模型：绘本、书页、常见物体、拒识、方向描述的准确度。

ModelRouter 的输入为 language、task、requires_audio、requires_vision、max_latency_ms、session_budget、privacy_mode、network_health。它始终有首选、备用和本地降级三条路径。

## 8. 视觉、物体识别与注视

### 8.1 三层视觉

| 层 | 在哪里运行 | 用途 | 不做什么 |
|---|---|---|---|
| FrameGate | Pi 3B | 稳定检测、运动检测、画面质量、抓取快照 | 连续大模型推理 |
| VisionAdapter | 云端 VLM | “这是什么”“读这一页”“画面里有什么” | 直接生成电机动作 |
| GroundingAdapter | 可选 LAN GPU / 云端检测 | “书在哪里”“红色积木在哪一边” | 直接控制关节或长期存人脸 |

每次发送云端图像前，先确保机械臂暂停或进入可预测姿态，再等待稳定时间。原始图像默认只在当前请求生命周期内保存；调试保存必须显式开启并带 TTL。

### 8.2 第一期能力

第一期仅支持：

- 检测“书/页面在不在”或用户主动要求看一眼。
- 使用 LEFT / CENTER / RIGHT 三个注视扇区。
- 用户明确说“看左边”“看我这里”“看书”时，按语义动作转向。
- 识别失败或目标丢失时停止追踪，回到监听姿态；不盲扫、不持续扭头。

### 8.3 后续定位与追踪

只有在以下全部条件满足后，才能启用连续视觉注视：

1. 目标连续多帧稳定，置信度与框大小合理。
2. 相机坐标到本机姿态的映射已通过标定。
3. 当前动作状态允许视觉修正，且没有阅读、碰撞、限位或高负载动作冲突。
4. 视觉结果未过期，失锁或重检失败立即 HOLD。
5. 电机真实反馈、速度、加速度和温度仍在允许范围。

视觉只提出目标方向或已校准注视姿态。不得把检测误差放大为逐帧五轴 PID 控制。

## 9. 行为、技能、人格与记忆

### 9.1 行为状态

长期场景与短期表达必须分开：

- Scene：reading、night、quiet、idle。
- Expression：listening、thinking、curious、acknowledge、greeting、caring、nod、headshake。
- State：attention_target、last_action、energy、safety_state、voice_session。

模型只能选择表达枚举和有限强度；本地根据当前场景、动作冷却、目标是否有效和安全状态决定是否执行。

### 9.2 技能安全模型

技能是经代码审计的功能声明，不是让模型下载并执行说明文本。

| 技能 | 可接受输入 | 本地验证 | 输出 |
|---|---|---|---|
| attention.look | target_ref 或方向 | 目标新鲜度、稳定度、状态 | 已验证注视动作 |
| reading.start | 可选书本引用 | 书页稳定、场景可用 | 阅读姿态与语音流程 |
| expression.show | 枚举、强度 | 冷却、姿态冲突、限速 | 短表达动作 |
| scene.set | 枚举场景 | 时段、亮度、安全状态 | 灯光与行为状态 |
| voice.speak | 短文本 | 长度、静音时段、队列 | TTS 请求 |
| system.stop | 无 | 永远允许 | 取消动作与发声 |

绝不提供 can.raw_frame、motor.set_current、joint.set_angle、shell、任意 URL 执行或在线技能安装。

### 9.3 记忆与隐私

记忆分开保存，不能用遥测或原始媒体替代用户记忆：

| 类别 | 默认 | 内容 | 删除方式 |
|---|---|---|---|
| 会话记忆 | 临时 | 当前对话、语言、页面、短期上下文 | 会话结束或超时 |
| 用户偏好 | 需同意 | 喜欢的语言、阅读偏好、音量、场景 | 本地 UI 一键删除 |
| 行为事件 | 最小化 | reading_started、book_stable、stop 等 | 设置保留期 |
| 安全遥测 | 必需 | 故障、温度、跟随误差、动作结果 | 滚动保留、导出 |
| 原始音视频 | 默认关闭 | 仅调试快照 | 显式开启、短 TTL |

不做脸库、声纹库或“情绪画像”。任何未来习惯建议至少基于多天、足够次数、明确同意的数据，且只能提出建议，不可自行移动或主动上传历史媒体。

## 10. SafetyGate 与运动控制

### 10.1 唯一执行入口

    AI / 语音 / UI / 视觉 / 示教回放 / 自动行为
                             ↓
                 SafetyGate.check_and_clamp()
       急停、限位、校准、CAN、温度、供电、软限位、
       速度、加速度、jerk、目标有效性、动作预算、状态冲突
                             ↓
          SmartMotionDispatcher：SEQ / INT / TRJ 仲裁
                             ↓
                    MotionCompiler
                             ↓
                    MKSCanDriver

### 10.2 状态机

- SAFE_DISABLED：急停、未校准、总线缺失、严重故障；不允许运动。
- HOMING_REQUIRED：需要人工确认或受限回零；不允许自由动作。
- READY_HOLD：已校准，可维持姿态，但仅接受批准动作。
- MOVING：正在执行轨迹；所有新请求仲裁。
- PAUSED：动作暂挂，等待恢复或取消。
- FAULT_LATCHED：故障锁存，必须人工确认和诊断后恢复。

默认拒绝原则：安全配置缺失、MKS 协议未知、任何轴心跳丢失、真实位置不可读、温度不可读且策略要求它、限位异常、模型超时或目标过期时，拒绝新的可选运动。

### 10.3 位置保持的边界

电机内部闭环可以维持已给位置，但系统不能因此假设机构安全。长期保持还依赖：

- 实际负载与质心力矩；
- 结构刚度、轴承游隙、联轴器与材料弹性；
- 连续电流、温升、供电能力；
- 动作后跟随误差；
- 断电、通信丢失和急停时的机械后果。

J2/J3 等受重力轴优先考虑弹簧、配重或几何减载。速度低并不会降低静态重力矩。

## 11. 监控、日志与远程调试

### 11.1 FlowLog

每个 trace_id 记录：

    input_received
    local_intent_matched
    vision_requested
    model_selected
    model_response
    behavior_proposed
    behavior_approved_or_rejected
    motion_started
    motor_feedback
    motion_completed_or_faulted
    tts_started_or_cancelled

日志采用每日 JSONL，附带内存环形缓冲和本地 SSE。日志中绝不写 API 密钥，不默认写原始音频或完整图像。

### 11.2 只读控制台

控制台显示：

- SafetyState、急停与限位状态；
- 每轴 target / actual / error / 温度 / 电流 / 故障 / 心跳；
- 当前动作 ID、来源和批准原因；
- 摄像头、麦克风、网络、STT/TTS 队列状态；
- 每个模型调用的服务名、任务、耗时、失败与费用估算；
- 最近 trace 的完整时间线。

首次实机阶段，控制台只读。任何电机参数修改都必须走单独的本地维护模式、物理确认和审计日志，不能从 AI 对话或公网页面写入。

### 11.3 Codex 的后续接入边界

当 Pi 通过 SSH、受限本地 API 或日志同步接入后，我可以读取上述只读数据、协助分析故障、生成变更补丁和指导测试。没有你的明确授权，我不会远程写入电机参数、发送运动命令或绕开安全流程。

## 12. 网络与安全

- CAN 与电机控制 API 仅绑定本机 Unix socket 或 127.0.0.1，不对局域网暴露。
- monitor 页面默认只在本机或经 SSH 隧道访问；若日后开放 LAN，必须鉴权、TLS 和明确的只读角色。
- interaction 服务只允许访问配置的模型服务域名；密钥由权限最小化的配置文件或秘密管理保存。
- 原始图片上传前必须通过隐私模式、稳定门和请求目的检查。
- 云端不可达时不积压危险动作；只保留有限本地语音、停止和安全状态。
- 更新采用手动审核、版本锁定、健康检查与可回滚部署；不启用未验证的自动在线更新。

## 13. 验证体系

已有 L0 与 L1 基础必须继续保留，并扩展为：

| 层级 | 目标 | 核心验收 |
|---|---|---|
| L0 代码契约 | schema、状态机、行为白名单、拒绝路径 | 单元测试与属性测试 |
| L1 理想轨迹 | 限位、速度、加速度、jerk、动作仲裁 | 理想 plant 与全转场测试 |
| L2 非理想数字孪生 | 重力、摩擦、柔性、延迟、跟随误差、碰撞 | 参数扫描与最差情形 |
| L3 单轴硬件在环 | 真实 MKS 帧、方向、回零、温升、故障 | 无负载后带已知负载 |
| L4 五轴受限实机 | 五轴姿态、限位、急停、视觉静止门 | 低速、小包络、有人看守 |
| L5 互动验收 | 三语、视觉、网络、打断、隐私 | 明确场景与失败降级 |

任何新技能、动作库、视觉跟踪或模型服务都要经过 L0 至适用层级后才进入默认功能。

## 14. 分阶段实施路线

### 阶段 A：规格冻结，不改运动代码

- 审核本文件、DEVICE.md、SAFETY.md 和姿态命名。
- 收集 MKS 手册、固件版本、节点号、通信与故障字段。
- 完成每轴质量、质心、结构、减载和持续保持需求表。
- 定义 LampProfile、Event、BehaviorProposal、MotorTelemetry 的测试样例。

### 阶段 B：电机与安全优先

- 实现只读 MKS 状态采集，不使能运动。
- 单轴低速回零、限位和急停验证。
- 增加实际速度、加速度、jerk、跟随误差和温度策略。
- 让所有 UI、示教、测试和未来 AI 路径进入同一个 SafetyGate。

### 阶段 C：事件、语音和本地控制

- EventRouter、trace_id、CooldownRegistry、FlowLog。
- 本地 STOP、静音、回待机、状态查询。
- 半双工 VAD、STT/TTS 适配器、三语会话和网络降级。
- 本地只读监控页。

### 阶段 D：按需视觉与模型路由

- FrameGate、稳定快照、隐私开关。
- ModelRouter 与统一的文本 / 视觉 / TTS 适配器。
- 主动问答型视觉：物体说明、书页说明、方向识别。
- 仅启用 LEFT / CENTER / RIGHT 的经过验证注视。

### 阶段 E：可选高级能力

- 局域网 GPU GroundingAdapter。
- 连续但受限的目标注视。
- 实时全双工语音；若 Pi 3B 不达标再加入 ESP 媒体前端。
- 经审计的长期偏好与建议，不引入自动危险动作学习。

## 15. 当前仓库的对应关系

| 已有基础 | 下一步角色 |
|---|---|
| lamp_core/behavior.py + action_catalog.py | BehaviorProposal 与可审计动作目录；模型只选择已启用 action_id |
| lamp_core/safety.py | 扩展为 fail-closed 状态、校准、CAN、温度与故障策略 |
| lamp_core/motion.py | 增加实际加速度 / jerk 限制与参数化验证 |
| lamp_core/smart_motion.py | 保留 SEQ / INT / TRJ，成为唯一动作仲裁器 |
| lamp_core/motor_interface.py | 替换未配置驱动为经 HIL 验证的 MKS CAN 适配器 |
| lamp_core/voice.py | 演进为 VAD、唤醒、三语 STT 会话适配层 |
| lamp_core/vision.py | 演进为 FrameGate 和按需视觉请求入口 |
| lamp_core/cloud.py | 演进为 provider-neutral ModelRouter 的一个适配器 |
| ideal_plant.py 与测试 | 作为 L1，后续接入 L2 参数化模型 |

## 16. 进入编码前仍需确认的事项

1. MKS SERVO42D 的完整、版本匹配 CAN 手册与实物固件信息。
2. 每轴机械图、限位、零位、负载、重心、轴承与减载设计。
3. 急停的实际切断对象：使能、24 V 动力，或两者。
4. Pi Camera、麦克风、扬声器和供电布线后的真实噪声与回声。
5. 三语言 STT、TTS、对话和视觉服务的对比测试结果。
6. 是否购买局域网 GPU，以及其隐私/网络位置。
7. 哪些原始媒体可以保存、保存多久、谁可以访问。
8. 第一版可用动作库及每个动作的最大幅度、速度、频率和冷却。

在这些事项未确认前，不开始真实五轴运动和连续视觉追踪。

## 17. 与现有运动代码的融合（2026-09-01）

本节将公开项目的控制思路适配进本项目；**不是复制其电机协议、关节参数或机械尺寸**。备份文件继续保留，本文成为新功能的架构入口。

### 17.1 已融合的可复用部分

`lamp_core/mechanicalarm_port.py` 是对 MechanicalArm_Code_V2 的 Python 化、接口化改写，保留其有价值的外环结构：

    100 Hz Cartesian reference
      -> 标定后的 IK
      -> 前馈速度 + 关节位置误差 P 项
      -> 闭环电机网络的速度命令

其中轨迹参考采用其“正弦加速—匀速—正弦减速”的结构；`MechanicalArmControlLoop` 每次循环先读真实关节反馈，再生成五轴速度命令。SmallRobotArm 仅借鉴“几何、脉冲/减速比换算须显式配置”的工程习惯；不采用它的 Arduino 阻塞脉冲输出，因为 MKS SERVO42D CAN 已拥有本轴闭环。

这条路径在本项目中位于 `MotionCompiler` 之后，不能被 AI、视觉框或语音文本直接调用：

    BehaviorProposal -> SafetyGate -> MotionCompiler
       -> MKS 位置段/同步模式（待协议确认）
       或 -> MechanicalArmControlLoop（100 Hz 外环，待标定）
       -> MKSCanDriver -> CAN

两种执行方式并存是有意设计，而不是二选一的猜测。MKS 手册确认后，按它是否提供可靠的同步位置段、状态反馈与速度命令，选择每个动作的执行方式。

### 17.2 明确留空、不得猜测的适配项

| 标记 | 待补信息 | 将填入的位置 | 在信息缺失时的行为 |
|---|---|---|---|
| TBD-MKS-PROTOCOL | 已收到 CAN V1.0.9 手册；仍需实物读取的固件、每轴节点号、当前 CRC 模式、工作模式与应答设置 | `mks_can_protocol.py` / `MKSCanDriver` | 协议编解码可离线测试；真实驱动保持 fail-closed，直到单轴读取核对完成 |
| TBD-KINEMATICS | 五轴 CAD 尺寸、轴方向、零位、连杆长度、关节正负方向 | FK/IK 标定模块 | 不启用 Cartesian 100 Hz 外环 |
| TBD-TRANSMISSION | 每轴实际传动比（若直驱则为 1）、编码器单位、机械零偏 | joint↔motor 单位换算 | 不产生真实角度或 RPM |
| TBD-CONTROL-GAINS | 空载及带负载实测后的外环 P 增益、采样抖动、跟随误差容许值 | `MechanicalArmControlLoop` 配置 | 不提供臆造默认增益 |
| TBD-POSE-LIBRARY | 阅读、点头、看左/中/右等姿态的实测关节坐标、动作包络与回退姿态 | `MotionCompiler` 姿态库 | 仅仿真或拒绝未登记动作 |
| TBD-HOMING | 每轴限位电气逻辑、回零方向、零位偏移、恢复流程 | 安全状态机 / 校准 | 停留在 `SAFE_DISABLED` |

`MechanicalArmControlLoop` 已刻意要求调用方注入 IK、反馈网络和 P 增益；它不藏入任何来自参考项目的机械常数。`lamp_core/motion.py` 的同步轨迹仍可用于已经标定的关节姿态段，二者共享 SafetyGate 和遥测，不互相绕过。

### 17.3 当前可验证范围

- `tests/test_mechanicalarm_port.py`：参考轨迹端点、零位移、反馈/前馈/P 外环输出。
- `tests/test_feedback.py`：反馈陈旧、故障与跟随误差的协议无关判定。
- `lamp_core/mks_can_protocol.py` / `tests/test_mks_can_protocol.py`：按 MKS CAN V1.0.9 的标准 CAN 帧、CRC、`0x31` 编码器读取、`0x32` 转速读取、`0xF3` 使能、`0xF5` 坐标绝对运动，以及 `0x4A/0x4B` 五轴同步缓存/启动进行字节级离线验证；不打开 CAN 设备。
- 全部离线测试：`python -m unittest discover -s tests -v`。

这证明软件接口和算法结构能在 Python 中运行；不等同于已证明 MKS CAN 帧、真实五轴 IK 或机构负载可运行。上述 TBD 按表中证据逐项关闭，才进入 L3/L4。

## 18. 参考实现优先的研发规则

本项目后续编程遵循“**先研读、再适配、后验证**”，不以通用生成代码替代已经落地的工程经验。

| 领域 | 首要参考 | 可迁移内容 | 必须重新实现或确认的内容 |
|---|---|---|---|
| 闭环机械臂外环与协同轨迹 | MechanicalArm_Code_V2 | 反馈读取→FK/IK→轨迹参考→前馈 + 误差校正→多轴命令的循环结构 | MKS CAN 帧、五轴几何、单位、增益、回零 |
| 步进机械臂几何和示教思路 | SmallRobotArm | 关节/末端几何、动作录制与插补的工程拆分 | Arduino 脉冲输出、无反馈位置估计、其传动比 |
| 台灯互动、性格、事件和语音分层 | Autonomous OS / Autonomous Lamp / LeLamp | 事件路由、能力声明、性格约束、语音与复杂智能体分层、只读遥测 | Feetech 舵机驱动、Pi 4/5 依赖、上游运行时与硬件配置 |

每个后续模块或提交必须在模块文档、代码注释或测试名称中写清楚：

1. **参考来源**：仓库、具体文件/函数及其版本或提交号；
2. **保留内容**：从该实现验证过的控制结构或数据流；
3. **适配差异**：为何因 MKS CAN、五轴机构、Pi 3B 或本项目交互目标而调整；
4. **未证实项**：以 `TBD-*` 标记，并附上关闭它所需的手册、CAD、标定或硬件测试；
5. **验证层级**：L0--L5 中至少对应哪一层测试通过。

禁止将参考项目的寄存器地址、CAN/串口帧、PID 数值、减速比、关节范围或机械参数在没有来源和实测的情况下改名后直接用于本机。相反，确认无关的抽象接口和离线仿真可以先完成，真实驱动必须等版本匹配的 MKS 文档与单轴硬件在环结果。
