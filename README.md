# 绘本伴侣灯：可落地基线

本仓库不是原附件的“直接上电版”。它把系统拆成可独立验收的安全层、运动规划层和设备适配层；默认只运行仿真，**不会向电机发送任何指令**。

先阅读：[方案审阅与实施路线](docs/方案审阅与实施路线.md)。
当前唯一的完整架构入口（硬件、音视频、多模型、行为和安全决策）见：[Autonomous 完整架构规格](docs/台灯_Autonomous_完整架构规格.md)；两份 2026-08-31 决策文档已在其中作为只读备份保留。
分层的运动验证计划、理想仿真命令和实机验收边界见：[验证体系与仿真边界](docs/验证体系与仿真边界.md)。
今天的完整代码索引、硬件决定与验证流程见：[今日最终代码与流程](docs/今日最终代码与流程.md)。
完整的互动机器人台灯功能范围与行为规范见：[产品功能与互动行为规格](docs/产品功能与互动行为规格.md)。
Autonomous OS / LeLamp 的公开设计逻辑已整理为本项目可用的分层、事件、性格、技能与安全契约，见：[Autonomous OS 逻辑拆解与本项目落地](docs/Autonomous_OS_逻辑拆解与本项目落地.md)。设备、安全与初版人格文件在：[DEVICE.md](device/DEVICE.md)、[SAFETY.md](device/SAFETY.md)、[SOUL.md](personality/SOUL.md)。
第一次在 Raspberry Pi 上部署且不接电机的流程见：[树莓派安全模拟部署](docs/树莓派安全模拟部署.md)。
两套开源步进机械臂的控制方式、适配结论与已融合代码见：[MechanicalArm 与 SmallRobotArm 电机控制审阅](docs/MechanicalArm与SmallRobotArm电机控制审阅.md)。
MKS CAN 手册与厂商 STM32 例程的逐项研读、协议确认和 Pi 适配计划见：[MKS CAN 厂商例程研读与 Pi 适配计划](docs/MKS_CAN_厂商例程研读与Pi适配计划.md)。

## 当前包含什么

- `lamp_core/safety.py`：必须经过回零、限位确认与使能确认才允许运动的状态机。
- `lamp_core/motion.py`：带关节限位、速度限制和最小加加速度（minimum-jerk）多轴同步轨迹生成器。
- `lamp_core/transport.py`：RS-485 帧收发的抽象与内存仿真器；没有伪造 MKS 寄存器地址。
- `tests/`：离线单元测试，可在电脑或树莓派上运行。

## 本地验证

```powershell
python3 -m unittest discover -s tests -v
```

## 全项目安全回归测试

日常开发按本次改动的模块运行测试，而不是全扫描。例如，改绘本识别/提示词时：

```powershell
python run_full_test_suite.py --component reading
```

可用模块是 `reading`、`safety`、`motion`、`can`、`autonomous`、`system`。所有模块
测试仍只调用本项目的真实代码入口，不连接 GPIO、串口、CAN 或真实电机。

若需要一轮较快的全代码回归，可不指定模块：

```powershell
python run_full_test_suite.py
```

仅在改动安全/运动逻辑、准备发布或人工要求时，才运行完整 `full` 扫描；它额外
验证 992 条关节极限转场、动作库和所有系统故障场景：

```powershell
python run_full_test_suite.py --profile full
```

在 Pi 上还可额外把一张真实 JPEG 走过正式的云端识图、绘本提示词和 WAV 输出路径：

```bash
python3 run_full_test_suite.py --with-cloud-image test-page.jpg
```

测试报告写入 `full-test-report.json`。准备好私有 Git 远程仓库后，Pi 可运行例如
`bash scripts/update_and_test.sh reading`：它总是拉取完整同版本代码，但只运行
`reading` 模块测试；不会自动开启真实电机。

## 逐帧运动仿真

以下命令不访问串口或 GPIO，只会输出一段五轴动作的状态机转换和每一帧关节角度：

```powershell
python simulate.py --sample-period 0.10
```

## 二维动画窗口

运行下列命令会打开一个 Tkinter 窗口；点击“播放”查看五轴关节逐帧运动。该窗口只做离线可视化。

```powershell
python animate_simulation.py
```

## 主程序事件仿真

以下场景分别验证正常阅读、总线超时、硬限位、相机断流和急停。它们只使用虚拟设备。

```powershell
python simulate_system.py normal
python simulate_system.py timeout
python simulate_system.py limit
python simulate_system.py camera
python simulate_system.py estop
```

## 自动找书与阅读动作演示

以下命令使用虚拟摄像头位置、离线识别文本和五轴虚拟电机，完整运行“找到书 → 逐帧定位至画面中央 → 快照 → 识别 → 阅读姿态 → 朗读”流程。`--book-x` 是书本在初始画面中的归一化横向中心；例如 `0.18` 为左侧、`0.50` 为中间、`0.82` 为右侧。演示会根据每帧书本中心与 `0.50` 的误差，协同调整相机承载的五个关节、重新取图，直至误差不超过 `±0.04`。

```powershell
python simulate_book_reading_flow.py --book-x 0.18
```

当前阅读姿态使用已验证的五轴配置：J1=0.25、J2=-0.20、J3=0.30、J4=0.12、J5=-0.18 rad。该演示只驱动 `VirtualMotorBus`，用于验证互动流程和轨迹组织。

对真实照片做“目标识别 → 二维居中规划”时，先在 `.env` 配置可用的视觉模型，然后运行：

```powershell
python simulate_detected_target_centering.py C:\Users\Bruger\Desktop\book_x015.jpeg C:\Users\Bruger\Desktop\book_x090.jpeg C:\Users\Bruger\Desktop\book_x0901.jpeg
```

该入口会请求模型返回目标框 `bbox_norm`，根据相机标定矩阵把二维误差分配给全部五个相机承载关节；后续位置变化仍由虚拟相机与 `VirtualMotorBus` 仿真，不调用真实电机。

没有相机和本地模型时，可用已标注的真实样本运行 Pi 本地检测器的离线仿真：

```powershell
python simulate_local_target_centering.py C:\Users\Bruger\Desktop\book_x015.jpeg C:\Users\Bruger\Desktop\book_x020.jpeg C:\Users\Bruger\Desktop\book_x035.jpeg C:\Users\Bruger\Desktop\book_x050.jpeg C:\Users\Bruger\Desktop\book_x070.jpeg C:\Users\Bruger\Desktop\book_x090.jpeg C:\Users\Bruger\Desktop\book_x0901.jpeg
```

该测试使用人工标注的目标框模拟未来 Pi/OpenNI/TFLite 检测器输出，验证的是本地接口、二维位置误差与轨迹规划；它不宣称已经测得真实模型的识别率或帧率。

## 完整主程序

电脑上的完整离线流程（模拟相机页、云端回复与五轴总线）：

```powershell
python app_main.py --mode simulate
```

## 手动示教、编辑与回放

`lamp_core/teach.py` 定义了手动示教的安全文件格式：每条记录包含时间戳及 J1–J5 的关节角度。它会验证关节名称、严格递增的时间戳和软限位；回放默认降速至原始动作的 35%，并进入 `TRJ` 有限缓冲区。编辑已有记录可用：

```powershell
python teach_edit.py 输入动作.json 输出动作.json --radius 1
```

真实示教必须在电机失能或经实测安全的低保持力矩模式下进行。实际的 MKS 42D 编码器轮询和串口回放适配器会在确认随货手册、固件和实物通讯后接入；当前模块只做离线验证，不会访问串口。

树莓派验证相机、静止检测、云端视觉与本地语音（电机仍是虚拟的）：

```bash
sudo apt install -y python3-picamera2 python3-opencv espeak-ng
cp .env.example .env
# 编辑 .env：将 ENABLE_CLOUD_VISION 改为 true，并填写 OPENAI_API_KEY。
# 程序会读取 .env；不要把它提交到 Git。
python3 app_main.py --mode pi
```

云端接口遵循 [OpenAI Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)，以 JPEG data URL 发送稳定页面，并设置 `store: false`。该客户端在未启用云端或未配置密钥时不会发出网络请求。

## 不可跳过的硬件前置条件

1. 每一轴有独立的硬限位或可靠的回零方案，并在低速、低电流下逐轴验证。
2. 急停必须是硬件动作：切断驱动器使能/动力电源；软件 `ESTOP` 只作为第二道保护。
3. 先只装一轴、无负载、低速测试通讯和方向，再逐轴增加。不得用真实人脸/书本跟踪闭环驱动未验证的机构。
4. 购买前确认完整的驱动器型号、固件版本和厂商手册。SERVO42C 与带 RS-485/Modbus 的 SERVO42D/57D 不能混用协议。
