# B3：π₀.₅-LIBERO-Long 的 500 个 episode

前置的 [B2 smoke](libero_smoke.md)已由服务器回传确认 10/10 成功。现在将每任务的初始化范围扩展到官方 `initial_states[0:50]`：10 个任务 × 50 个初始状态，共 500 个 episode，环境 seed 7。正式 500 回合尚未执行，不沿用 smoke 的 100% 作为正式结果。

## 1 号 H100 并行运行

当前选择在 1 号 H100 上运行多组模型服务与 LIBERO 仿真进程，显存目标为 **70 GiB，即 71680 MiB**。每组负责一个任务的 50 个初始状态，10 个任务合计 500 回合。

先在终端 1 按 `Ctrl+C` 停止当前手动启动的模型服务，再在一个终端执行下面的命令。这个入口负责启动和回收全部模型、仿真子进程。

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA &&
git pull --ff-only origin main &&
bash scripts/run_libero_parallel.sh --gpu 1 --egl-device 1 --target-vram-gib 70
```

调度过程如下。

1. 记录 GPU 原有占用，启动任务 0 的模型，预热后启动它的仿真进程。
2. 首个回合结束后，用这一阶段采样到的显存峰值增量估计一组模型和仿真的开销。这个回合计入 500 回合。
3. 按 `floor((71680 - 原有占用 MiB) / 单组开销 MiB)` 计算并发数，上限为剩余任务数和 `--max-workers`，后者默认 10。终端输出 `Measured ... GiB per warmed model/simulator, selected N parallel tasks`。
4. 逐组加载、预热，再启动各自的仿真。已有任务同时继续评测，新增组还要满足当时的剩余显存预算。一个任务完成后释放自己的服务，并调度下一个任务。

每个模型子进程设置 `CUDA_VISIBLE_DEVICES=1`，每个仿真子进程设置 `MUJOCO_EGL_DEVICE_ID=1`。JAX 使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 按需分配，各模型服务从 8100 端口起占用独立端口。显存目标通过采样和整组数量控制，实际占用需要观察服务器上的运行记录。按需分配选项的行为见 [JAX 官方说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。

这次并行实现选择了**每个任务独立的模型 RNG 起点 key 0**。预热完成后恢复这个起点，同一任务的 50 回合连续使用该服务的采样序列。环境 seed 为 7，任务内部按初始状态 0–49 顺序运行。每项任务的服务指纹、请求序号和客户端记录均单独保存，后续比较实验应使用这份采样协议。

结果位于 `outputs/pi05_libero_parallel/pi05_libero_10_gpu1_seed7_<UTC>/`。

- `run_config.json` 保存目标显存、测得的单组开销、选定并发数和采样协议。
- `gpu.csv` 每约两秒记录显存、GPU 利用率和活动任务数。终端每约 30 秒显示进度。
- `summary.json` 保存合并进度，结束时加入总耗时、每小时有效回合数和采样显存峰值。
- `episodes.jsonl` 在结束时汇总各任务记录并校验 task/init ID 的唯一覆盖。
- `tasks/taskXX/` 保存该任务的模型与仿真日志、检查点指纹、回合记录、视频和观测。

按 `Ctrl+C` 时，调度器先通知仿真保存当前记录，再结束模型服务。某项任务异常退出后，已运行任务继续完成各自的评测，新任务等待修复后另行安排。完整结果应记录 500 个有效回合，成功与超时一起计数。

### 已有速度与显存观测

2026-11-12 用户回传的客户端启动前快照显示 GPU 1 占用 **8753 MiB，约 8.55 GiB**，利用率为 0%。B2 日志中后九个回合合计约 162.2 秒，平均约 18 秒，500 回合串行粗估约 2.5–3 小时。并行运行的显存峰值、吞吐量和总耗时由这次服务器运行测量，用 `gpu.csv` 和最终 `summary.json` 一起核对。

## 串行启动

终端 1 先按 `Ctrl+C` 停止旧模型服务，再更新并启动：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA &&
git pull --ff-only origin main &&
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_pi05_libero.sh
```

服务会复用现有检查点，顺序读取缓存文件计算 SHA-256，再加载模型。哈希读取不会重新下载权重，但会增加启动时间；等待 `server listening ...:8000` 后再启动客户端。

终端 2：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA &&
MUJOCO_EGL_DEVICE_ID=1 bash scripts/run_libero_baseline.sh
```

无需安装新依赖。保持服务与客户端运行，不向此服务发送额外预热、诊断或其他客户端请求。每个 episode 结束后，终端输出任务/初始状态 ID、结果、策略步数和请求数，磁盘上的 `summary.json` 也会更新。

## 固定协议与记录

- 任务按 0–9 顺序；每个任务的初始状态按 0–49 顺序，完整顺序写入 `run_config.json`。
- 与官方 runner 一致，每个任务创建一个仿真环境并设置 seed 7，同任务的 50 个 episode 复用这个环境，每回合先 reset，再加载对应官方初始状态。
- 保留 10 步稳定等待、最多 520 个策略步、单候选 `(10, 7)` 动作和执行前 5 步后重规划；观测和成功条件沿用 B2。
- 新服务仍调用上游 `create_trained_policy` 与 `WebsocketPolicyServer`。记录包装器只附加服务 ID 和请求编号，不额外调用模型、不重置 RNG、不修改动作。
- JAX 初始 RNG 仍为上游默认 key 0，按推理请求顺序 split，跨 episode 连续使用。环境 seed 不是模型 seed。

多初始化评测启动时核对服务 metadata：必须为新的 `pi05_libero` JAX 服务，包含检查点指纹且尚未接收推理请求。每次响应的服务 ID 和请求编号也会被记录、核对；若发生交错请求，在执行该动作前报错并停止，避免把采样顺序改变的结果当成规定的串行评测。

服务记录位于 `outputs/pi05_servers/pi05_libero_<service_id>/`，包含 `service.json` 与 `run.log`。`service.json` 记录实际检查点路径、逐文件字节数/SHA-256、整体 manifest 指纹、norm stats 指纹、代码提交、运行库版本、CUDA 选择、JAX 设备和加载耗时。指纹标识本次使用的本地内容，并不等于与官方另行公布的哈希做过比对。客户端握手保存这份 metadata，每回合记录服务 ID 与请求区间。

客户端输出目录：

```text
outputs/pi05_libero/pi05_libero_10_baseline_seed7_<UTC>/
  run_config.json
  run.log
  episodes.jsonl
  summary.json
  episodes/libero_10_task00_init00/...
  episodes/libero_10_task00_init01/...
  ...
  episodes/libero_10_task09_init49/...
```

每回合的双视角视频、每次请求的输入 NPZ、完整动作及实际执行前缀继续保存，目录按 task/init ID 唯一命名。保存这些文件所需时间不包含在 `roundtrip_seconds` 中，首次模型调用仍含 JIT。

## 结果与中断

正常结束时应有 `planned_episodes: 500`、`valid_episodes: 500`、`infrastructure_errors: 0`、`unattempted_episodes: 0`。`status: PASS` 表示完整记录了 500 个有效回合；任务成功由 `successes` 计数，达到预算仍未完成的回合计入 `timeouts`，两者相加应为 500。

汇总从 `episodes.jsonl` 重算，包含总成功率、逐任务的成功数/有效数/成功率和 Wilson 95% 区间。区间描述这组固定任务与初始状态的计数，不代表多个训练种子的变异。完整 Long 结果与 [baseline 计划](i_baseline.md)中已固定的 Long 参考值比较；不把 Long 结果与四套平均混合比较。

通信、仿真、记录错误或手动中断会保留已经写下的结果并停止，未完成的总成功率为 `null`。此入口没有自动续跑或自动替换失败回合的功能。发生中断时先保留当前目录并回传汇总；修复后仅安排未运行或基础设施异常的 task/init ID，使用独立目录记录修复批次及新的服务/采样起点。合并前核对 ID 覆盖和采样差异，不能将修复批次描述为未中断的原始 RNG 序列。任务超时保留原结果，不以重跑结果替换。

`--task-id`、`--init-state-id`（起点）和 `--num-trials-per-task` 可指定单独的诊断范围；小于完整 500 组时命名为 smoke/subset，不标为正式整套 baseline。当前 Long 的运行选择为本页上方的单 GPU 任务并行入口。

## 本地验证范围

2026-11-12，本地 Python 3.8 的 20 项 CPU 检查通过（17 项 LIBERO 检查、3 项服务记录检查），覆盖 500 个 task/init 组合唯一覆盖、范围边界、同任务环境复用与逐回合 reset、服务器请求顺序、异常不混入成功率、双视角视频以及新旧 smoke 入口兼容。检查点指纹测试验证相同内容移动缓存路径后身份不变、同长度文件内容变化后身份改变；记录包装器测试验证动作内容与调用次数保持不变。三个 Bash 入口的语法、Python 3.8 评测 CLI 和 Python 3.11 服务 CLI 检查通过。服务端仍沿用已验证的上游加载与推理实现；带指纹的新入口实际加载权重、GPU 推理和 500 回合仍由服务器运行验证。

同日增加单 GPU 并行入口后，Python 3.8 的 30 项 CPU 检查通过。其中 13 项并行调度及服务检查也在 Python 3.11 通过，四个 Bash 入口语法和新入口 CLI 检查通过。新增检查覆盖显存预算、设备选择、模拟子进程调度与端口复用、退出时最后一批回合记录、任务异常后的汇总、预热恢复 RNG 和中断时的子进程清理。本机还用上游 WebSocket 服务与 CPU 测试策略验证了 `/healthz`、握手 metadata 和首个请求编号。本地使用 CPU 与模拟子进程验证这些控制逻辑，H100 上的实际并发性能等待服务器测量。
