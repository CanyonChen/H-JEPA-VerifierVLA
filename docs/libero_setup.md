# LIBERO 环境、EGL 与真实观测检查

本步骤衔接已通过的官方 π₀.₅-LIBERO 合成观测推理：先渲染真实 LIBERO 场景，再通过 WebSocket 调用模型，最后执行最多 5 个策略动作。检查默认使用 `libero_10` 的 task 0、初始状态 0 和 seed 7；没有运行完整 episode，不能据此报告任务成功率。

## 两个环境的职责

| 环境 | 用途 | 主要版本 |
| --- | --- | --- |
| 根目录 `.venv` | OpenPI、H-JEPA、VeriVLA 共享模型环境；π₀.₅ 使用 JAX GPU 推理 | Python 3.11.13、PyTorch 2.7.1、JAX 0.5.3 |
| 根目录 `.venv-libero` | LIBERO 仿真、相机渲染和 WebSocket 客户端 | Python 3.8.20、PyTorch 1.11.0+cpu、NumPy 1.22.4 |

三个模型仍然共用原来的模型环境。独立的仿真环境由 [envs/libero/pyproject.toml](../envs/libero/pyproject.toml) 和对应 `uv.lock` 管理，目标平台为 Linux x86_64。安装脚本显式选择 `.venv-libero`，不会同步或修改根模型环境。

仿真端保留官方示例的 robosuite 1.4.1、MuJoCo 3.2.3、Numba 0.53.1、NumPy 1.22.4 等版本。针对本次运行路径，主要适配是：

- 官方示例使用 `torch==1.11.0+cu113`，这里改为 `torch==1.11.0+cpu`。这条运行路径通过 PyTorch 读取仓库自带的初始状态，不在仿真进程中执行神经网络推理，因此不需要给客户端再安装一套旧 CUDA 运行库。
- MuJoCo 物理步进与 PyTorch 无关；相机通过 OpenGL/EGL 使用 NVIDIA 驱动渲染。CPU 版 PyTorch 不会把 EGL 渲染改成 CPU 渲染。
- 安装仿真运行依赖与 `openpi-client`，不安装 LIBERO 训练入口所需的完整训练栈。客户端锁定 `websockets==13.1`，保持 Python 3.8 支持；模型服务沿用根锁文件中的 15.0.1。

这些是本项目的环境适配，不是未经修改的官方完整环境。版本依据见仓库中的[官方示例依赖](../openpi/examples/libero/requirements.txt)、[初始状态加载代码](../openpi/third_party/libero/libero/libero/benchmark/__init__.py)及[官方评测入口](../openpi/examples/libero/main.py)。

## 1. 安装并检查 EGL

在服务器项目根目录运行：

```bash
cd ~/canyon/H-JEPA-VerifierVLA
git pull --ff-only origin main &&
MUJOCO_EGL_DEVICE_ID=1 bash scripts/setup_libero_env.sh
```

安装脚本自动获取 Python 3.8.20，使用提交的锁文件执行 `uv sync --frozen`，随后执行 `uv pip check` 和仅渲染检查。下载设置沿用模型环境的 `UV_PYTHON_DOWNLOADS=automatic`、`UV_HTTP_TIMEOUT=600`、`UV_CONCURRENT_DOWNLOADS=4`。下载中断后可以重跑同一命令，复用 uv 缓存。

检查入口在导入 LIBERO 前写入 `.venv-libero/libero-config/config.yaml`，指向仓库自带的 BDDL、初始状态与模型资源，避免交互式询问路径。`data/libero/datasets/` 仅作为预留路径创建；这一步不下载训练数据集。

仅渲染检查会加载初始状态，执行 10 个等待步，获取外部相机和手腕相机图像。图像按官方入口旋转 180 度并缩放至 224×224，检查非恒定 RGB 图像和有限的 8 维机器人状态，保存 PNG 与原始输入。

通过时末尾 JSON 包含：

```json
{
  "status": "PASS",
  "mode": "render_only",
  "egl_render": "PASS",
  "policy_roundtrip": "NOT_RUN",
  "action_steps": 0
}
```

该步骤需要服务器的 NVIDIA EGL 驱动可用。`MUJOCO_EGL_DEVICE_ID=1` 选择 EGL 枚举中的设备 1；它与模型进程的 `CUDA_VISIBLE_DEVICES` 分别生效，不应因为 JAX 显示 `cuda:0` 而直接改成 0。在当前服务器先沿用 GPU 1，设备枚举及渲染是否可用以本次实际检查为准。报告同时记录实际 OpenGL vendor、renderer 和版本，用来核对渲染后端。

## 2. 启动模型服务

在同一终端、EGL 检查通过后运行并保持该终端开启：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_pi05_libero.sh
```

脚本使用根模型环境和上游 `serve_policy.py`，明确加载 `pi05_libero`，默认端口为 8000。权重沿用 `OPENPI_DATA_HOME`，未设置时使用 `~/.cache/openpi`，复用此前已下载的检查点及 norm stats。它关闭 JAX 显存预分配并要求 CUDA 后端。

等待服务出现监听日志，再执行下一步。加载检查点和首次真实观测的 JIT 编译仍需时间；同一服务进程的后续调用复用编译结果。

需要更换端口时设置 `PI05_PORT`；已有检查点在其他位置时设置 `PI05_CHECKPOINT=/absolute/path/to/pi05_libero`。服务脚本按上游行为监听 `0.0.0.0`，本步骤的客户端通过本机回环地址连接。

## 3. 发送真实观测并执行动作

打开第二个服务器终端，运行：

```bash
cd ~/canyon/H-JEPA-VerifierVLA
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_libero_check.sh
```

该入口重复场景初始化与相机检查，然后把真实观测发送到 `127.0.0.1:8000`。预处理与上游一致：两路 RGB 图像、末端位置、由四元数转换的轴角、夹爪状态和任务文字。接收到有限的 `(10, 7)` 动作块后执行前 5 步；若环境提前返回成功则结束。

通过时报告包含 `status: PASS`、`mode: real_observation_and_actions`、`egl_render: PASS`、`policy_roundtrip: PASS`、`action_shape: [10, 7]`，以及实际执行的 `action_steps`（1–5）。`task_success_observed` 只记录这几步内的环境信号，不是完整 episode 的评测结果。

可通过参数指定端口或不同任务，例如：

```bash
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_libero_check.sh --port 8001 --task-id 1
```

无需激活环境；各入口都显式调用所属环境的 Python，并清除 `PYTHONPATH`，避免已有 ROS 等路径混入。

## 产物与验证范围

每次检查保存到独立的 `outputs/libero_check/run_*/`：

| 文件 | 内容 |
| --- | --- |
| `report.json` | 成功或异常、代码提交、版本、任务、设备选择、数据形状和通信耗时 |
| `run.log` | Python 日志与异常堆栈 |
| `agentview.png`、`wrist.png` | 送入模型的两路 RGB 图像 |
| `observation.npz` | 图像、机器人状态和任务文字 |
| `actions.npy` | 模型返回的 `(10, 7)` 动作块，仅通信模式生成 |
| `agentview_after.png` | 执行动作后的相机图像，仅通信模式生成 |

`roundtrip_seconds` 包含通信与整个推理调用，首次调用还包含 JIT，不是稳定吞吐率。通过接口检查后，再进行 [B2：10 个任务各 1 个完整 episode](i_baseline.md)；本入口没有替代 B2 或后续成功率评测。

2026-11-10 本地 Windows CPU 验证结果：

- Linux x86_64 / glibc 2.39 的冻结安装预演通过；锁文件包含 75 个包条目，计划实际安装 74 个包。预演不等于 Linux 安装成功。
- Python 3.8.20 / PyTorch 1.11.0+cpu 成功读取 `libero_10` 全部 10 个任务、每任务 50 个初始状态，数值全部有限；NumPy 1.22.4 与 Numba 0.53.1 的 CPU JIT 小计算通过。
- 两项观测测试在 Python 3.8 和 3.11 中均通过，覆盖图像方向、状态排列、四元数转换、缩放与消息序列化。复跑命令为 `python -m unittest discover -s tests -p 'test_libero_observation.py' -v`。
- Python 3.8 / websockets 13.1 客户端与 Python 3.11 / websockets 15.0.1 服务端通过本机回环通信，完整传输两张图像、状态和文字，并收到预期的 `(10, 7)` 动作。该检查使用探针策略，没有加载模型权重。
- Python 3.8 语法及检查入口帮助、三个 Bash 脚本语法、服务参数与上游 Tyro 数据类定义的一致性检查通过。LIBERO 和 `openpi-client` 的本地 editable 包构建通过。

本地验证没有运行 NVIDIA GPU、EGL 或真实模型服务推理；服务器安装及上述两个运行模式的结果仍待回传。根模型环境的 `pyproject.toml` 与 `uv.lock` 未修改。
