# 自主常驻运行与 AI 行为方案

## 目标

将台灯从“手动运行一次脚本，完成一次演示”的模式，改为 Raspberry Pi 开机后长期常驻的设备服务：

```text
Pi 开机
  -> 自动启动台灯服务
  -> 相机 / 语音 / 日志 / 状态机就绪
  -> 待机观察
  -> 唤醒词、书页、人物、时间等事件触发互动
  -> AI 决定高层意图
  -> 本地安全层决定是否允许并具体执行
```

这里的“自主”是指台灯不必每次等待用户从终端运行命令；它可以根据环境事件主动发起已批准的互动。自主不等于让模型直接控制关节、电流、CAN 帧或安全状态。

## 现状

- `app_main.py --mode pi` 已经是常驻相机循环：启动相机、检测页面移动/静止、触发识图与语音。
- 目前 Pi 主入口仍使用 `VirtualMotorBus`，因此可实现相机、云端识图、TTS 的自动流程，但不等于真实电机 ready。
- `AutonomousRuntime`、`EventRouter`、`FlowLog` 与 `SafetyController` 已组成可测试的 `LampService` 核心，但尚未接入真实 I/O 适配器形成设备守护进程。
- `voice.py` 已接入 `LampService` 的 WAKE / READ / SOFT_STOP / ESTOP 本地路径，但还没有持续麦克风采集和唤醒音频会话。

### 第一阶段实施状态（2026-09-07）

第一阶段常驻服务核心已经落地并通过完整离线回归：

- 新增根目录 `lamp_service.py`，负责会话状态、事件构造、本地路由与结构化日志；
- `SessionState` 与 `SafetyState` 保持正交，服务启动不会自动回零或使能驱动；
- 新增 `WAKE`、`READ`、`SOFT_STOP` 事件，三者均走确定性本地路径；
- 普通“停止”映射为 `SOFT_STOP`，明确“急停”映射为 `ESTOP`；原有 `EventKind.STOP` 保留紧急语义以兼容现有调用；
- 修复“停止录制”被“停止”吞掉及 `Hello Lamp` 空格归一化不一致的问题；
- 未显式分类的事件默认拒绝，不会自动进入模型路径；
- `BOOK_STABLE`、`OBJECT_STABLE` 可按页面指纹或对象 ID 分别冷却；
- `FlowLog` 在保留内存记录的同时，可镜像为单行 JSON，供未来 journald 收集；
- 当前完整测试为 122 项，全部通过。

本阶段仍不包含真实麦克风循环、相机事件接线、可中断 TTS、systemd unit 或真实 CAN 驱动。

## 产品行为原则

### 1. 开机自动运行，不开机自动乱动

服务启动后默认应进入：

```text
SAFE_DISABLED
```

此时相机、麦克风、TTS、日志可以工作，但真实电机不使能、不运动。只有以下条件都成立，才可进入 `READY_HOLD`：

1. 物理急停已释放；
2. CAN 总线、五轴心跳、反馈和故障码正常；
3. 回零和限位检查已完成；
4. 人工确认允许进入运行状态。

当前没有真实 CAN 驱动时，服务只能运行“虚拟电机模式”；不能把仿真状态写成实体电机已就绪。

### 2. AI 自主决定意图，本地系统掌握执行权

```text
AI / 模型
  -> 场景、情绪、说话内容、注意力目标、动作计划
  -> 例如："看书 -> 好奇歪头 -> 阅读姿态"

本地执行系统
  -> 校验安全状态、目标有效性、冷却时间、动作冲突、动作预算
  -> 将动作 ID 映射到已验证姿态/轨迹
  -> 通过 SafetyGate 和 SmartMotionDispatcher 执行
```

模型可以自主决定：

- 是否主动问候、保持安静或回待机；
- 看书、看人、看玩具或进入倾听姿态；
- 对话内容、故事内容、语气和情绪；
- 已验证动作的组合、顺序与强度；
- 灯光场景和互动节奏。

模型不可直接决定：

- 任意 CAN 帧、关节角度、速度、电流或驱动模式；
- 绕过急停、限位、温度、反馈、总线故障或状态机；
- 未验证的实体动作；
- 任意 shell、网络地址或在线安装行为。

## 运行状态与事件

设备互动状态与物理安全状态必须是两套正交状态机，而不是一条混合状态链。

```text
SafetyState（物理安全）：
  SAFE_DISABLED -> HOMING_REQUIRED -> READY_HOLD <-> MOVING
  任意异常 -> FAULT_LATCHED / SAFE_DISABLED（ESTOP 锁存）

SessionState（互动会话）：
  BOOTING -> IDLE_OBSERVING
  -> AWAKE_SESSION
  -> READING / LISTENING（后续阶段）
  -> IDLE_OBSERVING

两者可以组合，例如：SafetyState=SAFE_DISABLED、SessionState=AWAKE_SESSION。
```

### 推荐触发规则

| 事件 | 当前状态 | 行为 |
|---|---|---|
| 唤醒词（如“小灯”） | `IDLE_OBSERVING` | 进入 `AWAKE_SESSION`，例如维持 60 秒 |
| 稳定书页 | `AWAKE_SESSION` | 看向书、识别、朗读、进入 `READING` |
| “开始读” | `IDLE_OBSERVING` / `AWAKE_SESSION` | 创建或刷新会话，允许下一张稳定书页触发阅读；是否运动另由 `SafetyState` 决定 |
| 人/手稳定出现 | `IDLE_OBSERVING` | 可选一次轻微欢迎或注视；须冷却 |
| 会话超时 | `AWAKE_SESSION` | 回到待机或倾听姿态 |
| “停止” / `SOFT_STOP` | 任意互动状态 | 取消语音、模型请求和软动作，退出会话；不清除回零状态 |
| “急停” / `ESTOP` | 任意状态 | 立即禁用驱动并锁存；要求人工复位和重新安全检查 |
| 限位、CAN、反馈故障 | 任意状态 | 立即锁存故障，禁止新动作 |

`BOOK_STABLE`、`OBJECT_STABLE` 必须有去重和冷却。当前 `EventRouter` 已有基础冷却与事件 ID 去重机制；新增服务必须通过它而不是绕过它。

## 主服务结构

建议新增 `lamp_service.py`（或 `autonomous_service.py`）作为唯一常驻入口。它不是第二套业务逻辑，而是组装现有模块。

```text
CameraAdapter ----------> EventQueue ---->
Microphone/STTAdapter --> EventQueue ----> EventRouter
Timer/PresenceAdapter --> EventQueue ----> AutonomousRuntime
Motor/CAN telemetry ---> EventQueue ----> SafetyGate
Physical E-stop/fault -> independent safety path -> disable drives immediately
                                      |             -> EventQueue（记录/同步）
                                              |
                                              v
                                      AI behavior planner
                                              |
                                              v
                                      Behavior allow-list
                                              |
                                              v
                             ActionCatalog / MotionCompiler / TTS / LED
                                              |
                                              v
                               SmartMotionDispatcher -> MKSCanDriver
```

### 每个循环必须做的事

1. 读取相机、麦克风、CAN 反馈和定时器事件；
2. 为每个事件生成唯一 `event_id` 与 `trace_id`；
3. 普通事件交给 `EventRouter` 做分类、去重、冷却与隐私判定；硬件急停和故障必须先走独立安全路径，再进入事件流记录；
4. 只有被允许的事件才交给 `AutonomousRuntime` / 模型路由；
5. 模型输出必须是结构化的 `BehaviorProposal`，不是自由文本指令；
6. `approve_behavior()` 通过后再从 `ActionCatalog` 选择动作；
7. 所有运动再经过统一 `SafetyGate`、`SmartMotionDispatcher` 和电机适配器；
8. 每一步写 `FlowLog`，包括拒绝原因、模型路径、动作开始/完成与故障。

## AI 行为提案格式

AI 的输出可以比“固定应答”自由，但必须结构化。第一版一次只审批一个动作，例如：

```json
{
  "speech": "这只小熊好像发现了什么，我们一起看看？",
  "target_ref": "book-1",
  "action_id": "LELAMP_CURIOUS_TILT",
  "intensity": 0.3,
  "reason": "用户刚唤醒且书页稳定"
}
```

多动作 `plan[]` 属于后续 C 级能力。实现前必须新增独立的计划契约，对整组动作做原子验证、最大动作数、总时长、冷却、预算和中途重新校验；不能把多个单动作审批简单串联后立即执行。

本地校验项：

- `action_id` 必须存在于启用的 `ActionCatalog`；
- 目标必须仍在画面中且未过期；
- 强度、频率、动作持续时间与动作预算必须在限制内；
- 同一类动作必须遵守冷却；
- 阅读、说话、大幅姿态动作之间不得冲突；
- 当前安全状态不是 `READY_HOLD` 时，拒绝一切可选运动；
- `SOFT_STOP` / `ESTOP` 永远不调用模型、没有冷却、优先取消动作与语音。

## 自主程度分级

| 级别 | 能力 | 是否适合第一版 |
|---|---|---|
| A | 自动待机、唤醒、朗读、固定动作 | 是 |
| B | AI 选择说话内容、情绪、已验证动作 | 是 |
| C | AI 编排多个已验证动作、依环境调整顺序 | 是，需冷却/预算 |
| D | AI 提议新动作，先在仿真中审查后人工批准入库 | 后续 |
| E | AI 直接生成关节/CAN/电流命令并实体执行 | 不允许 |

第一版先完成 B：台灯能主动观察并从经过验证的动作库选择单个表达。C 级多动作编排在取消、预算和计划级验证完成后再启用。

## 唤醒与阅读会话

最小可用设计：

```text
默认 IDLE_OBSERVING：只监听唤醒词、SOFT_STOP、ESTOP、故障和低成本视觉
  -> WAKE：创建 AWAKE_SESSION，保存超时时间
  -> BOOK_STABLE 且会话仍有效：运行阅读流程
  -> READ：可直接允许当前稳定页开始阅读
  -> SOFT_STOP / 会话超时：取消语音与软动作，回 IDLE_OBSERVING
  -> ESTOP：立即走独立安全路径并锁存，同时结束互动会话
```

不要把唤醒词识别和 TTS 放在同一阻塞线程。第一版可采用半双工：TTS 播放时暂停普通识别，物理急停必须始终可用。若要求播放期间语音 STOP 仍可用，需要保留独立的本地关键词检测通道，或实现经过验证的回声抑制；不能在暂停识别时同时承诺语音 STOP 可用。

## Raspberry Pi 开机服务

部署时由 systemd 启动主服务，而不是人工 SSH 后运行命令：

```ini
[Unit]
Description=Smart Lamp Runtime
Wants=sound.target
After=local-fs.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-lamp
ExecStart=/usr/bin/python3 -u /home/pi/smart-lamp/lamp_service.py
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
# Optional deployment-specific configuration; keep secrets out of the unit.
EnvironmentFile=-/etc/smart-lamp.env

[Install]
WantedBy=multi-user.target
```

部署前将 `User`、项目路径和 Python 路径替换为 Pi 实际值。启用命令：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now smart-lamp.service
systemctl status smart-lamp.service
journalctl -u smart-lamp.service -f
```

`Restart=on-failure` 只能重启应用，不能把故障状态自动变为电机可动状态；每次异常后都应重新做安全检查。

## 推荐实施顺序

1. ~~新建 `lamp_service.py`，先实现不接触电机的服务核心；~~ 已完成；
2. ~~接入 `EventRouter`、`AutonomousRuntime`、`FlowLog`，实现 `IDLE_OBSERVING` / `AWAKE_SESSION`；~~ 已完成；
3. 接 USB 麦克风采集和 Vosk，验证 WAKE / STOP / READ；
4. 将现有 `accept_page_jpeg()` 作为稳定页被批准后的阅读执行函数；
5. 接入可取消的 TTS/云任务与统一 cancellation token，再让单动作 `ActionCatalog` 提案进入执行链；
6. 添加 systemd 服务和启动/重启/日志验证；
7. 通过完整离线回归和 Pi 无电机模拟验收；
8. 最后才将真实 MKS CAN 驱动接到同一执行入口，并逐轴完成硬件在环验证。

## 验收标准

在没有真实电机时：

- Pi 重启后服务自动启动；
- 相机、TTS、日志进入 ready；
- 未唤醒时不朗读、不发动作；
- 唤醒后，稳定书页可自动进入阅读；
- 同一页不会因稳定帧重复触发；
- STOP 可立即取消阅读会话和待执行软动作；
- 服务异常重启后回到安全待机，而不是恢复未完成动作；
- 所有路径保留 `trace_id` 和可读日志。

接入真实 CAN 后，额外要求：

- 开机默认 `SAFE_DISABLED`；
- 只有人工确认回零、限位、心跳和故障检查后进入 `READY_HOLD`；
- 任意总线、反馈、限位、急停异常都禁止自动运动；
- 不存在模型直达 CAN 或关节命令的路径。
