# B2：π₀.₅-LIBERO 十个完整 episode

2026-11-12，服务器已通过本步骤：`libero_10` 十个任务各运行初始状态 0、环境 seed 7，10/10 成功，没有超时或基础设施异常。该结果确认完整回合运行链路，不能将十次 smoke 检查当作每任务 50 次的正式 baseline。下一步见 [B3：500 个 episode](libero_baseline.md)。

## 服务器结果：10/10 成功

用户回传的汇总保存在 [pi05_libero_smoke_seed7.json](results/pi05_libero_smoke_seed7.json)。以下逐任务数字抄自同次终端日志，策略步数合计 2486、推理请求合计 500；每行请求数均等于 `ceil(policy_steps / 5)`，符合固定执行前缀。这里的 500 是请求数，episode 数为 10。

| Task ID | 结果 | 策略步数 | 推理请求数 |
| --- | --- | --- | --- |
| 0 | success | 267 | 54 |
| 1 | success | 245 | 49 |
| 2 | success | 210 | 42 |
| 3 | success | 220 | 44 |
| 4 | success | 256 | 52 |
| 5 | success | 170 | 34 |
| 6 | success | 198 | 40 |
| 7 | success | 233 | 47 |
| 8 | success | 434 | 87 |
| 9 | success | 253 | 51 |

产物目录为 `outputs/pi05_libero/pi05_libero_10_smoke_seed7_20261112T093146_955970Z/`。原始 `run_config.json`、`episodes.jsonl`、视频和 NPZ 尚未拉取到本地，不能将终端摘录写成这些文件的副本，也尚未逐帧检查视频。粘贴的终端文本中 task 7 的开始行重复出现，但完成行只有一条，汇总也只有 10 个有效回合；仅凭这行重复不能认定任务执行了两次。

客户端先等待服务，随后在 17:32:45 开始 task 0，最后在 17:36:25 完成 task 9。首次回合包含 JIT 和环境初始化，不能用它估算稳定单次推理延迟。任务 8 使用 434 个策略步，仍在 Long 的 520 步预算内。

## 运行

在模型服务所在的终端按 `Ctrl+C` 停止上一轮服务，然后执行：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA &&
git pull --ff-only origin main &&
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_pi05_libero.sh
```

重启的目的是从相同的模型随机数序列起点开始。上游 JAX 策略使用默认 key 0，每次推理继续 split；客户端 seed 7 控制仿真侧，`client.reset()` 不会重置模型 RNG。此次运行只使用一个客户端，不先发送额外预热或检查请求；第一个 episode 的第一次请求包含 JIT。

服务监听 8000 端口后，在第二个终端执行：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA &&
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_libero_smoke.sh
```

无需重新安装依赖或手动激活环境。脚本显式使用 `.venv-libero`，模型服务使用 `.venv`。每个任务开始和结束都会输出日志；最终输出 `summary.json` 的内容。

单任务诊断可加 `--task-id 0`，指定初始化可加 `--init-state-id 0`；默认参数才是十个任务各一次。诊断是单独的运行，不自动替换已经完成的任务失败。需要改端口时，在模型终端设置 `PI05_PORT=8001`，客户端传入 `--port 8001`。

## 执行与判定

- 保留官方观测预处理、任务语言和初始状态加载方式；仿真分辨率 256，两路图像旋转 180° 后缩放到 224。
- 每个 episode 先执行 10 个 `[0,0,0,0,0,0,-1]` 等待步，这些不计入策略步数。
- 每次请求一个有限的 `(10, 7)` 动作块，只执行前 5 步，再观测与推理。
- Long 最多执行 520 个策略步；其他可选 suite 保留官方预算：Spatial 220、Object 280、Goal 300。
- 固定版本 LIBERO 的 `BDDLBaseDomain.step()` 将 `done` 设置为 `_check_success()`。收到该信号即记录 `success`；耗尽策略预算仍未成功则记录 `timeout`，这是有效的未成功回合。
- 模型通信、仿真、视频保存等异常记为 `infrastructure_error`，保留已完成的动作数和产物，停止本批次；未运行的任务计入 `unattempted_episodes`。`Ctrl+C` 记为中断。没有自动重试或挑选成功回合的逻辑。

`summary.json` 的 `status: PASS` 表示十个回合全部有效记录，允许其中有任务超时；`successes` 和 `timeouts` 表示任务结果。存在异常、中断或未运行回合时为 `INCOMPLETE`，进程返回非零退出码，`success_rate` 为 `null`，避免悄悄排除异常后报告成功率。全部完成时，`success_rate = successes / planned_episodes`，仅代表这十个固定初始化的结果。

## 产物

每次运行写入独立目录：

```text
outputs/pi05_libero/pi05_libero_10_smoke_seed7_<UTC>/
  run_config.json
  run.log
  episodes.jsonl
  summary.json
  episodes/
    libero_10_task00_init00/
      result.json
      video.mp4
      decisions.jsonl
      observation_000001.npz
      ...
    ...
```

`video.mp4` 左侧为外部相机、右侧为手腕相机；保存每个策略动作前的画面及最终动作后的画面。视频沿用官方输出的 10 fps，仿真控制频率为 20 Hz，因此视频播放速度不是墙钟实时速度。等待步不录入视频。

每次策略请求的 NPZ 保存两路模型输入、8 维状态和任务文字；`decisions.jsonl` 保存请求序号、策略步位置、完整动作块、实际执行步数和客户端往返耗时。请求失败时保留本次请求的观测与记录。请求序号是本客户端的尝试顺序；在通信无异常且遵守服务重启、单客户端、无额外请求协议的运行中，它与服务端采样顺序对应。首次往返包含 JIT，往返耗时不含本地视频/NPZ 保存。

`run_config.json` 保存代码提交、工作区状态、依赖版本、任务与执行协议、连接端点和服务返回的 metadata。本次 B2 使用的旧服务 metadata 为空，因此 `expected_policy_config`、`expected_checkpoint` 只是启动约定。B3 已增加[带检查点指纹的服务入口](libero_baseline.md)，后续运行会记录服务实际加载的目录、内容指纹和请求编号；这不追溯补造本次 B2 的服务身份记录。

`episodes.jsonl` 每完成一个回合就追加一行。最终汇总重新读取这些行生成；每回合 `result.json` 同时保留结果、错误阶段、策略步数、耗时和渲染设备，视频不会被其他回合或其他运行覆盖。

## 本地验证范围

回归入口使用本地 Python 3.8，不加载模型或创建 MuJoCo 环境：

```bash
.venv-libero/bin/python -m unittest discover -s tests -p 'test_libero_*.py' -v
```

2026-11-12，本地 Windows / Python 3.8.20 的 12 项检查全部通过，包含原有的包安装、BDDL 任务解析及观测预处理回归。新增检查覆盖 5 步重规划与丢弃动作块后缀、反馈新观测、首步成功、520 步预算与 10 步等待分别计数、非法动作不执行、异常保留实际执行前缀、存在异常时不报告成功率，以及实际 FFmpeg 双视角 MP4 写入和解码。完整入口的探针运行使用真实 LIBERO 任务文件，验证两个成功回合及第三个回合发生异常时的 JSON/NPZ/视频记录与停止行为。Bash 语法及 Python 3.8 CLI 帮助检查也通过。测试中的策略、仿真和 OpenGL 为探针对象，不代表完整 episode 在服务器通过。
