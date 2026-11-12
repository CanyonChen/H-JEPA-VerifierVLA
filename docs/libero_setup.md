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
- 显式安装 LIBERO 原版依赖表中的 `future==0.18.2`；`bddl==1.0.1` 的运行代码使用它，但没有将其声明为安装依赖。

这些是本项目的环境适配，不是未经修改的官方完整环境。版本依据见仓库中的[官方示例依赖](../openpi/examples/libero/requirements.txt)、[初始状态加载代码](../openpi/third_party/libero/libero/libero/benchmark/__init__.py)及[官方评测入口](../openpi/examples/libero/main.py)。

## 1. 安装并检查 EGL

当前服务器项目路径为 `~/canyon43079/H-JEPA-VerifierVLA`。若从旧父目录整体改名而来，先按[父目录改名后的恢复](environment.md#父目录改名后的恢复)备份并重新创建两个虚拟环境。

在服务器项目根目录运行：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main &&
MUJOCO_EGL_DEVICE_ID=1 bash scripts/setup_libero_env.sh
```

安装脚本自动获取 Python 3.8.20，使用提交的锁文件执行 `uv sync --frozen --no-active --reinstall-package libero`，重建 LIBERO 的 editable 安装，随后执行 `uv pip check`、隔离进程中的 LIBERO / BDDL 导入检查和仅渲染检查。`--no-active` 明确使用 `.venv-libero`，不采用当前终端中激活的模型环境。定向重装用于修复此前生成的 LIBERO 空壳包；其他已安装依赖继续复用。下载设置沿用模型环境的 `UV_PYTHON_DOWNLOADS=automatic`、`UV_HTTP_TIMEOUT=600`、`UV_CONCURRENT_DOWNLOADS=4`。下载中断后可以重跑同一命令，复用 uv 缓存。

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

脚本使用根模型环境，明确加载 `pi05_libero`，默认端口为 8000。B3 起由项目的 `serve_pi05_libero.py` 调用上游策略加载器与 WebSocket 服务，另记录检查点指纹和服务请求编号。权重沿用 `OPENPI_DATA_HOME`，未设置时使用 `~/.cache/openpi`，复用此前已下载的检查点及 norm stats；计算指纹会读取缓存文件。入口关闭 JAX 显存预分配并要求 CUDA 后端。

等待服务出现监听日志，再执行下一步。加载检查点和首次真实观测的 JIT 编译仍需时间；同一服务进程的后续调用复用编译结果。

需要更换端口时设置 `PI05_PORT`；已有检查点在其他位置时设置 `PI05_CHECKPOINT=/absolute/path/to/pi05_libero`。服务脚本按上游行为监听 `0.0.0.0`，本步骤的客户端通过本机回环地址连接。

## 3. 发送真实观测并执行动作

打开第二个服务器终端，运行：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
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

- Linux x86_64 / glibc 2.39 的冻结安装预演通过；补齐 `future` 后，锁文件包含 76 个包条目，计划实际安装 75 个包。预演不等于 Linux 安装成功。
- Python 3.8.20 / PyTorch 1.11.0+cpu 成功读取 `libero_10` 全部 10 个任务、每任务 50 个初始状态，数值全部有限；NumPy 1.22.4 与 Numba 0.53.1 的 CPU JIT 小计算通过。
- 两项观测测试在 Python 3.8 和 3.11 中均通过，覆盖图像方向、状态排列、四元数转换、缩放与消息序列化。复跑命令为 `python -m unittest discover -s tests -p 'test_libero_observation.py' -v`。
- Python 3.8 / websockets 13.1 客户端与 Python 3.11 / websockets 15.0.1 服务端通过本机回环通信，完整传输两张图像、状态和文字，并收到预期的 `(10, 7)` 动作。该检查使用探针策略，没有加载模型权重。
- Python 3.8 语法及检查入口帮助、三个 Bash 脚本语法、服务参数与上游 Tyro 数据类定义的一致性检查通过。LIBERO 和 `openpi-client` 的本地 editable 包构建通过。

本地验证没有运行 NVIDIA GPU、EGL 或真实模型服务推理。服务器此前遇到的 LIBERO 包发现及 BDDL 缺少 `future` 问题已修复，过程见下文。2026-11-12，用户确认新路径中的环境重建完成，并回传下述 EGL、真实观测通信和 5 步动作执行结果。根模型环境的 `pyproject.toml` 与 `uv.lock` 未修改。

## 服务器结果：真实观测与 5 步动作执行

2026-11-12，用户回传提交 `4e8cdae86a11125abe844464a1f1fdd20e5f917d` 的运行结果，记录见 [pi05_libero_real_observation_seed7.json](results/pi05_libero_real_observation_seed7.json)。该文件是终端 JSON 的副本；服务器的原始 PNG、NPZ、动作和日志尚未拉取到本地。

| 项目 | 本次结果 |
| --- | --- |
| 任务 | `libero_10` / task 0 / init state 0 / seed 7 |
| 任务描述 | put both the alphabet soup and the tomato sauce in the basket |
| EGL | `PASS`；NVIDIA H100 80GB HBM3，OpenGL 4.6.0 / NVIDIA 580.159.03 |
| 观测 | 两路 `224×224×3` RGB 图像和 8 维机器人状态 |
| 策略通信 | `PASS`；返回有限的 `(10, 7)` 动作块 |
| 动作执行 | 5 步；`task_success_observed: false` |
| 首次往返 | 34.589 秒，包含首次真实观测调用的 JIT；不是稳定延迟 |
| 服务器输出 | `outputs/libero_check/run_20261112T091006_006571Z/` |

模型服务日志确认权重从 `~/.cache/openpi` 恢复，并从检查点 assets 加载 norm stats。客户端的 `libOpenGL.so.0` 加载提示、缺少可选 `OpenGL_accelerate`、robosuite 私有宏提示及旧 Gym 提示没有阻止本次运行；实际渲染后端和报告证明这次 EGL 检查通过。`task_success_observed: false` 表示这 5 步内没有完成任务，不是接口检查失败，也不是完整回合的失败结果。

## 4. 十个任务各一个完整 episode

B0/B1 与 [B2 smoke](libero_smoke.md)均已在服务器通过：LIBERO-Long 十个任务各一个完整 episode，初始状态 0、环境 seed 7，10/10 成功。复跑 B2 仍使用 `bash scripts/run_libero_smoke.sh`；当前下一步是 [B3：每任务 50 个初始状态](libero_baseline.md)，正式 500 回合结果待服务器执行。

## LIBERO 安装导入失败与修复

2026-11-10，用户回传的服务器日志显示 Python 3.8.20、74 个包的依赖检查通过，但运行 `outputs/libero_check/run_20261110T075210_873786Z` 在 `from libero.libero import benchmark` 处报 `ModuleNotFoundError: No module named 'libero'`。该失败发生在创建仿真环境之前，不能据此判断 EGL 是否可用。

本地复现确认：外层 `libero/` 缺少 `__init__.py`，上游 `setup.py` 的 `find_packages()` 返回空列表，安装后有发行包元数据，却没有可导入模块。这符合 [Setuptools 的包发现规则](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)。先前本地状态读取检查主动加入了源码路径，只证明源码可以执行，没有验证 editable 安装的导入行为，因而漏掉了此问题。

修复补入外层包标记，并在安装脚本中定向重建 LIBERO、执行 `python -I` 导入检查。新增 [安装回归检查](../tests/test_libero_installation.py)先在修复前复现同一异常，修复后从临时工作目录启动隔离 Python 进程，不注入源码搜索路径，实际导入安装包并读取全部 500 个初始状态；该检查和两项观测测试均通过。锁文件一致性、Linux 冻结安装预演与 Bash 语法检查也通过。

在安装好的 LIBERO 环境中复跑回归检查：

```bash
.venv-libero/bin/python -m unittest discover -s tests -p 'test_libero_*.py' -v
```

## BDDL 缺少 future 的修复

2026-11-10，服务器运行 `outputs/libero_check/run_20261110T084632_850202Z` 已打印 `LIBERO package import: PASS`，随后在导入 `OffScreenRenderEnv` 的依赖链中，由 `bddl.backend_abc` 报 `ModuleNotFoundError: No module named 'future'`。此时仍未创建仿真环境。

`future==0.18.2` 原本列在 [LIBERO 官方依赖表](../openpi/third_party/libero/requirements.txt)，初版仿真运行依赖子集漏掉了它。BDDL 1.0.1 的安装元数据只声明 pytest、numpy、networkx、jupytext，没有声明代码实际使用的 `future`，所以 `uv pip check` 无法发现该遗漏。修复将原版版本补回仿真环境，并把安装前置检查扩展到 `bddl.parsing`；安装回归检查同时解析 LIBERO-10 的十个真实任务文件。

本地验证先在修复前复现相同的 `ModuleNotFoundError`，补齐 `future` 后三项回归全部通过。额外的 CPU 检查实际导入 `OffScreenRenderEnv` 及其依赖链，使用 LIBERO 解析器读取十个任务，确认任务类型均已注册；未创建仿真环境。锁文件只新增 `future`，其余依赖版本保持原样，离线锁文件校验与 Linux 冻结安装预演通过。

完整导入检查的本地条件是 Windows / Python 3.8.20、robosuite 1.4.1、MuJoCo 3.2.3。仅在忽略的 `.venv-libero-check` 内，配置 `MUJOCO_GPU_RENDERING=False`、`MUJOCO_GL=disable`，并将 MuJoCo 自带 DLL 放到 robosuite 旧版 Windows 加载路径，避免触发图形设备初始化。此配置用于本地导入验证；服务器入口继续使用 EGL，实际 GPU 渲染后来已由 2026-11-12 的服务器报告确认通过。
