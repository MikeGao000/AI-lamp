# AI-lamp（给 Claude 读的项目规则）

## 项目
树莓派陪读台灯：相机 + 麦克风 + TTS + 云端识图 + 5 轴 MKS SERVO42D/57D CAN 闭环步进机械臂。
纯 Python（无 ROS2 / CMake / 交叉编译）。日常开发在 `develop` 分支，`main` 已落后。
Pi 主入口 `app_main.py --mode pi` 的电机仍是 `VirtualMotorBus`；真实 MKS 只通过单轴脚本在台架上验证。

## 常用命令
- 全部单元测试（改完任何代码必跑，新增行为要补回归测试）：`python3 -m unittest discover -s tests`
- 完整回归：`python3 run_full_test_suite.py`
- 仿真场景：`python3 simulate_system.py <normal|timeout|limit|camera|estop>`
- 摄像头预览：`python3 camera_preview.py`（默认只监听 Pi 本机 127.0.0.1:8000；从电脑看需 `ssh -L 8765:127.0.0.1:8000`，再开 http://127.0.0.1:8765；与 `app_main.py` 不能同时占用摄像头）
- Pi 主入口：`python3 app_main.py --mode pi`

## 真实硬件红线
- 会打开真实 CAN 总线或发电机指令的脚本：`bench_motor.py`、`run_mks_single_axis_motion.py`、
  `measure_axis_direction.py`、`set_aim_zero.py`、`bench_loop_cost.py`、`collect_book_dataset.py`，
  以及带 `--execute-j1` 的 `camera_preview.py`。
- 除非用户在对话里明确说“现在上电测试”，不得运行它们，也不得用 `cansend` 等任何方式向 `can0` 发帧。
  `run_mks_single_axis_motion.py` 默认是 dry-run（只打印帧）；其余脚本未逐个核对，一律按真机处理。
- 真机前提：先仿真 / dry-run，人在旁边，急停随手可达。
- 上面这些真机脚本只能在树莓派（Linux SocketCAN）上跑；Windows 上只能跑单元测试和 dry-run。
- 代码里目前没有“停止电机”的 CAN 帧（`mks_can_protocol.py` 只有使能 0xF3 和绝对位置 0xF5）；
  Ctrl+C / 超时 / 异常退出时代码不会发任何停止指令；按协议推断电机会继续走完最后一条 F5（未用硬件验证）。真机时急停靠断电。
- `--gear-ratio`、`--j1-gear-ratio` 只要略大于 1 就会放开 120/360 RPM 上限；传参前人工核对减速比。

## 架构硬规矩
1. LLM 只能提议高层意图，并且只能从 `ActionCatalog` 的已验证动作里选；任何路径都不得让模型输出直达 CAN 帧、关节角度、速度或电流。
2. STOP / ESTOP 必须本地、确定性、不调用模型、没有冷却。
3. 云功能默认关闭、显式 opt-in，离线优先；不得新增默认开启的云依赖。
4. 没用硬件验证过的假设，必须在代码注释和回复里明确标注“未验证”，不得当成事实。

## 工作方式
- 在 `develop`（或它的分支）上工作。大改动前确认 git 工作区干净，小步提交。
  不要用 `git checkout .` / `git reset --hard` 丢弃未提交改动，除非用户明确要求。
- 不改 `.env`，不把密钥写进仓库。
- 改语音短语（`lamp_core/voice.py` 的 `INTENT_PHRASES`）后跑 `tests/test_voice.py`；
  复合短语（如“停止录制”）不得被短语“停止”吞掉。

## 已知未完成（2026-11-02，核对对象 develop 6daa3d8；之后若有提交请重新核对）
- `app_main.py` 的麦克风回调 `on_transcript` 没有本地 STOP 判断：孩子说“停止”会被当成问题发给云端。
  `lamp_service.py` 里有 `SOFT_STOP` 接线，但没有接到这条麦克风路径上。
- `run_pi()` 只有 Ctrl+C（`KeyboardInterrupt`）会触发 ESTOP，没有 SIGTERM 处理；用 systemd 部署前必须补。
- `intensity` 仍未参与动作的运动编译（只在行为校验里检查范围）。
