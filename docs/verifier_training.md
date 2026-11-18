# B5 LIBERO-Long verifier 训练

更新日期为 2026-11-17。按用户本轮确认，**当前 B5 使用 LIBERO-Long（`libero_10`）的 10 个任务、500 个回合，推进数据检查、划分、打标、verifier 训练及闭环对照。** π₀.₅ 与视觉编码器冻结，verifier 基于官方结构和论文方法进行本项目复现。

2026-11-17，服务器已完成 Long 500 回合的数据审计及固定划分，训练／验证／测试分别为 400／50／50 回合。训练集包含 368 次成功、32 次超时和 22,213 个动作窗口。当前步骤为四条训练轨迹的 Robometer 进度打标试运行，随后检查标签质量、迁移预训练参数并训练 verifier。[服务器审计记录](results/pi05_libero_long_verifier_data_audit.json)、[划分回传](results/pi05_libero_long_verifier_split_seed7.json)

## 已核对的依据与实施边界

| 项目 | 已有依据 | 本项目处理 |
| --- | --- | --- |
| 官方发布 | [VeriVLA @ f95ab14](https://github.com/shirley430316/VeriVLA/blob/f95ab142d6b5b247b47b36137352eb6289dd86a7/README.md) 只勾选模型结构，训练与仿真集成仍在 TODO；本轮读取 [Releases](https://github.com/shirley430316/VeriVLA/releases) 返回空列表 | 复用模型代码，自行补齐训练、标注和闭环集成；不把随机进度头用于正式对照 |
| 采集规模 | [论文](../VeriVLA/paper/VeriVLA.md) Implementation Details：每任务 50 示范用于策略微调，另采 50 策略 rollout 用于 verifier；LIBERO 每种策略各训一套，覆盖 40 任务 | 已有适配 LIBERO 的 π₀.₅，先检查现有策略轨迹。Long 的 500 条只是按十任务折算的规模参考，不意味着这批评测数据可同时全量训练和全量测试 |
| 预训练 | 论文描述冻结 V-JEPA 2 编码器、DROID 23k 轨迹预训练预测器；[Meta 官方入口](https://github.com/facebookresearch/vjepa2#v-jepa-2-ac) 提供 encoder 和 AC predictor | 候选初始化来源为 V-JEPA 2-AC，仍需核对网络配置、动作/状态语义及权重覆盖率；它不包含训练好的 VeriVLA 进度查询/进度头 |
| 进度标签 | 论文使用 Robometer 产生绝对进度并作时间差；未指定具体公开 checkpoint | [Robometer-4B](https://huggingface.co/robometer/Robometer-4B) 是拟采用的公开起点，需先在本项目任务上检验标签质量；不能认定与作者标签逐位相同 |
| 训练前向 | 官方 `VisionTransformerPredictorAC.forward()` 可计算梯度；`score_candidate()`、`rank_candidates()` 和 rollout 入口装饰了 `no_grad` | 用 predictor 的训练前向实现 latent L1、进度 Huber、同任务同子块位置排序，以及 teacher/self forcing |
| 超参数 | 论文表 7 给出 AdamW、5e-4、weight decay 0.04、batch 32、两项损失权重 1、256×256、M=5，明确属于真机适配设置 | 作为起点，LIBERO 训练轮数和具体适配配置需单列，不能称为论文公布的 LIBERO 完整训练配置 |

**预训练加载需固定版本。** 本轮读取 Meta `src/hub/backbones.py` 的 main 内容（Git blob `435c72ea192361c990baf67e8734f9e4fb64371b`）时，`VJEPA_BASE_URL` 指向 `http://localhost:8300`；[官方 README](https://github.com/facebookresearch/vjepa2#v-jepa-2-ac) 则公布了 `https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`。因此不能把未固定版本的 `torch.hub.load(...main...)` 当成已验证加载流程。后续固定源码和 checkpoint 来源，记录哈希、missing/unexpected keys 与各模块实际加载比例；本轮未下载该大模型。

Meta 编码器入口默认是视频编码器（3D patch embedding、tubelet 2），本地 VeriVLA 编码器默认 `num_frames=1` 时使用 2D embedding；还需对齐 ViT-g 的宽度、深度、RoPE、帧/视角与预处理。仅按同名、同形状筛选参数不足以证明语义兼容。

## 现有轨迹能提供什么

下表列出 LIBERO-Long 的采集格式。服务器文件完整性由数据审计确认。[LIBERO 采集](../scripts/eval_libero.py)

| 内容 | 已实现的记录 | 训练含义 |
| --- | --- | --- |
| 图像 | 每次决策前 NPZ 保存两路 224×224 uint8 图像；每控制步前及终止后写 MP4，左右拼接成 448×224 | 视频预期为 `policy_steps+1` 帧。中间和终止目标图像可来自 MP4；需记录压缩与从 224 调整到训练尺寸的适配 |
| 状态 | 决策开始时的 8D 状态；没有每个控制步的机器人状态 NPZ | 增加独立可训练 8D 状态投影。拟在五个子块中重复当前决策状态，与推理保持一致，避免训练时使用未来真实状态 |
| 动作 | 每次生成 `(10,7)`，实际执行前 `executed_steps` 步，通常为 5，终止时可能更短 | 标签只对应实际执行前缀。后五步未执行，不可拿重规划后的后续视频为其打标签 |
| 任务 | NPZ 保存完整 prompt，回合索引保存 task ID | 检查回合内 prompt 一致，并区分不同 suite 的 task ID；不能混用编号 |
| 恢复 | 最终 `episodes.jsonl` 引用部分旧目录的产物 | 从最终索引读取并去重，保留旧目录；不能递归收集所有历史运行后把保留回合再计一次 |

**首版训练时域。** 一个完整执行前缀为 5 步，分成 `M=5, K=1`。在控制步 t 开始的决策，其输入为当前图像/状态，动作是实际执行的前五步，目标图像取视频帧 `t+1 ... t+5`，标签为对应相邻时刻 Robometer 进度差。短窗口保留实际的 n 步动作、n 个目标帧，并记录长度为 5 的 `valid_subchunk_mask`。训练加载器随后按该 mask 处理填充和损失。逐控制步的 Robometer 差分质量将在打标阶段检验。

**Long 数据规模。** 500 回合包含 459 次成功、41 次超时，共 137,998 个控制步、27,773 次策略决策与同数量的观测 NPZ，以及 138,498 帧视频。500 回合均通过服务器审计，其中 290 回合的产物引用此前运行目录。执行长度为 1、2、3、4、5 的窗口分别为 83、77、105、94、27,414 个。官方模型使用任务专属查询，当前对应 Long 的 10 个任务。[回传汇总](results/pi05_libero_long_seed7.json)、[数据审计](results/pi05_libero_long_verifier_data_audit.json)

## 数据划分先于打标和训练

现有 500 回合已经用于 B3 评测。训练前将这批数据划分为互斥的训练、验证和测试集合，分别用于更新参数、选择配置与触发阈值，以及最终评测。新的公平对照在保留的同一测试初始状态集合上报告 base 与 verifier，B3 汇总作为历史基线引用。

划分以完整回合为单位，按 `(suite, task_id, init_state_id)` 分组。每个任务的 50 个初始状态按 `libero_10|split_seed|task_id|init_state_id` 的 SHA256 排序，依次分配 40 个训练、5 个验证和 5 个测试回合。`split_seed=7` 专用于数据划分，分组结果独立于轨迹结果和输入索引顺序。同一回合的全部窗口归入同一集合，最终索引中重复的任务与初始状态会被报告为错误。服务器回传已确认以下规模。

| 划分 | 回合 | 成功 | 超时 | 动作窗口 | 完整五步 | 短窗口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练 | 400 | 368 | 32 | 22,213 | 21,922 | 291 |
| 验证 | 50 | 44 | 6 | 2,799 | 2,768 | 31 |
| 测试 | 50 | 47 | 3 | 2,761 | 2,724 | 37 |
| 合计 | 500 | 459 | 41 | 27,773 | 27,414 | 359 |

本地核对了各组汇总之和，原始索引保存在服务器，290 个恢复回合仍引用原产物目录。[完整划分回传](results/pi05_libero_long_verifier_split_seed7.json)

同任务、同子块位置上的排序训练样本，并不要求它们来自同一物理状态；但在这种离线排序上表现好，仍需独立的同状态候选比较或闭环实验检验实际选择能力。

## 已完成的服务器数据审计

[audit_verifier_data.py](../scripts/audit_verifier_data.py) 使用 NumPy 和已安装的 imageio-ffmpeg，读取最终索引中的每个回合、全部决策 NPZ，并解码核对视频帧数。检查生成/执行长度、连续时间线、8D 状态、7D 动作、图像尺寸、任务指令及重复索引/产物引用。它不加载模型、不生成标签、不修改源文件，也不需要 GPU；每 50 个回合向终端打印进度，最后输出 JSON。

服务器已使用 B3 的 `.venv-libero` 完成这一步。需要复查源数据时可使用以下命令。

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main &&
env -u PYTHONPATH .venv-libero/bin/python -B -m scripts.audit_verifier_data \
  --run-dir \
  outputs/pi05_libero_parallel/pi05_libero_10_gpu1_seed7_20261112T131428_923173Z
```

本轮回传 `checked_episodes=500`、`episodes_with_errors=0`、`retained_artifact_episodes=290`。各任务均检查 50 回合，汇总与逐任务计数已在本地核对。审计覆盖全部决策 NPZ、视频解码帧数和写入约定的时间线，逐像素对齐和进度标签质量需在后续样本检查中确认。[完整回传 JSON](results/pi05_libero_long_verifier_data_audit.json)

审计工具此前已在 B3 兼容 Python 3.8 环境通过六项测试。

## 已完成的数据划分与窗口索引

`prepare_verifier_long_data.py` 复用 `.venv-libero`，在 CPU 上读取已审计的最终索引、逐回合结果和决策记录，检查时间线与源路径，并生成固定分组和实际执行窗口。所有输入视频及 NPZ 使用源文件路径引用，290 个恢复回合继续引用原始运行目录。[准备脚本](../scripts/prepare_verifier_long_data.py)

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA &&
git pull --ff-only origin main &&
env -u PYTHONPATH .venv-libero/bin/python -B -m scripts.prepare_verifier_long_data \
  --run-dir outputs/pi05_libero_parallel/pi05_libero_10_gpu1_seed7_20261112T131428_923173Z \
  --output-dir outputs/verifier_long/split_seed7 \
  --split-seed 7
```

输出目录包含以下文件。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 来源与脚本哈希、划分方法、逐任务统计、输出索引哈希和数据字段说明 |
| `train_episodes.jsonl`、`validation_episodes.jsonl`、`test_episodes.jsonl` | 回合身份、任务指令、结果、视频与源产物路径 |
| `train_windows.jsonl`、`validation_windows.jsonl`、`test_windows.jsonl` | 决策观测、实际执行动作、上下文与目标帧索引、有效子块 mask |

帧索引从 0 开始。控制步 t、执行 n 步的窗口使用上下文帧 t 和目标帧 `t+1..t+n`，动作数组长度为 n，mask 的前 n 位为 true。索引文件准备完成后统一发布，同一输入重跑会复用字节一致的结果。来源或划分种子改变时使用新的输出目录。

服务器回传已确认三组回合数为 400／50／50、总窗口数为 27,773、完整五步窗口数为 27,414、短窗口数为 359。完整逐任务信息保存在服务器的 `manifest.json`。

准备工具已在 Python 3.8 通过五项本地测试，覆盖分组隔离与顺序稳定性、重复／缺失初始状态、恢复路径与短窗口、命令入口及重复运行、写入失败清理与重试。

## 当前服务器步骤 Robometer 四回合打标

`label_verifier_long_pilot.py` 从训练集中按任务编号选择最先同时包含成功与超时的两个任务，每个任务取初始状态编号最小的一条成功轨迹和一条超时轨迹，共四回合。选择依据固定在运行配置中。试运行用于检查进度标签的方向、数值变化和逐步差分，并测量打标资源开销。2026-11-18 用户要求停止当前进程并改用 GPU 0。当前运行配置包含 GPU 编号并校验目录内容，因此本次使用 `robometer_pilot_seed7_gpu0` 新目录重新打标四回合，复用现有环境和下载缓存。

当前推理配置采用以下固定值。

| 项目 | 配置 |
| --- | --- |
| 模型 | `robometer/Robometer-4B`，权重版本 `beef63bc914c5c189329d49c6d712d96d632aa34` |
| 官方源码 | `robometer/robometer`，提交 `352d160389daa964788de1ec933d1925f3a6de4f` |
| 处理器 | `Qwen/Qwen3-VL-4B-Instruct`，版本 `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 推理 | 官方 `RBM` 模型类，标准 Transformers 加载，BF16，SDPA，单进程，GPU 0，batch 8 |
| 图像 | MP4 左半边的外部视角，224×224，包含初始帧及终止帧 |
| 时间上下文 | 对每个控制步 t，使用 `linspace(0,t,4)` 的四帧前缀，取最后一帧预测值 |
| 标签 | 每个控制步预测一次进度，相邻进度差形成 `Δp`，按窗口执行长度及 mask 对齐 |
| 曲线 | 保存原始进度、成功概率和相邻差分，直接绘制原始值 |

四帧前缀对应官方评测的 `use_frame_steps` 路径。模型的离散进度 logits 通过官方转换函数求连续期望值。标准 Transformers 加载路径及外部视角选择属于本项目的推理适配。权重中的历史 `similarity_head` 额外参数会在加载报告中列出，进度、成功、偏好和主干参数的缺失会终止运行。[官方逐步评估](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/robometer/evals/eval_server.py)、[模型卡](https://huggingface.co/robometer/Robometer-4B/tree/beef63bc914c5c189329d49c6d712d96d632aa34)

安装推理依赖后运行试打标。

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA &&
git pull --ff-only origin main &&
bash scripts/setup_robometer_env.sh &&
env -u PYTHONPATH HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  HF_HUB_VERBOSITY=info HF_HUB_DOWNLOAD_TIMEOUT=60 \
  .venv-robometer/bin/python -u -B -m scripts.label_verifier_long_pilot \
  --split-dir outputs/verifier_long/split_seed7 \
  --output-dir outputs/verifier_long/robometer_pilot_seed7_gpu0 \
  --gpu 0 \
  --batch-size 8
```

安装脚本使用独立的 `.venv-robometer`，Python 3.10、Torch 2.8.0／CUDA 12.6、Transformers 4.57.2，并安装推理适配器所需依赖及固定版本官方模型类。首次运行会下载 Robometer 权重与 Qwen 处理器。基线采集与 CPU 数据整理继续使用 B3 的既有环境。[固定推理依赖](../envs/robometer/requirements.txt)、[上游依赖声明](https://github.com/robometer/robometer/blob/352d160389daa964788de1ec933d1925f3a6de4f/pyproject.toml)

2026-11-17 首次服务器安装已创建 Python 3.10.22 环境，随后在下载 `nvidia-nvjitlink-cu12==12.6.85` 时连接 `pypi.nvidia.com` 超时。原脚本将 PyTorch 源设为整条依赖链的索引，该源的 NVIDIA 包链接指向上述域名。现已使用 `--torch-backend cu126` 指定 PyTorch 后端，并让 NVIDIA 依赖从 PyPI 解析。环境创建增加 `--no-project`，避免套用根项目的 Python 3.11 约束。重新执行上述命令会复用已创建的 `.venv-robometer` 和 uv 下载缓存。[uv 后端选择](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)、[同版 NVIDIA 包](https://pypi.org/project/nvidia-nvjitlink-cu12/12.6.85/)

修复已在本地完成 Python 3.10／Linux x86_64 的 28 项依赖解析，核对全部 NVIDIA 依赖的来源为 PyPI。2026-11-18 收到的服务器日志确认安装及导入成功，版本为 Torch 2.8.0+cu126、Transformers 4.57.2。随后下载 Robometer 权重时出现 SSH `client_loop: send disconnect: Connection reset`，日志显示 3.94 GB 分片已完成，4.97 GB 分片最后显示到 95%。该输出尚不能确认服务器进程是否结束、完整权重是否就绪或打标是否完成。

2026-11-18 随后的第一次诊断采样显示，GPU 0 打标进程 3528920 已运行约十分钟，完整分片为 3,935,034,144 字节，另一个 `.incomplete` 文件为 28,817 字节，最后修改时间早于本次进程启动。官方固定版本元数据确认缺失的 `model-00001-of-00002.safetensors` 应为 4,965,629,698 字节，SHA256 为 `2cfa81761595f6b10bd079736eb32bdeeeb0b49224619411bd511d9d1f1171a4`。当前仅收到第一次采样，具体网络或文件锁原因仍待核实。

本次恢复改用标准 HTTP 下载，并开启 Hugging Face INFO 日志，显示文件下载、重试和锁等待。`HF_HUB_DISABLE_XET=1` 与 `HF_HUB_ENABLE_HF_TRANSFER=0` 在本项目的 `huggingface-hub==0.36.0` 中选择普通 HTTP 路径，`HF_HUB_DOWNLOAD_TIMEOUT=60` 设置每次读取等待的超时。服务器下载恢复和完整模型加载仍待回传。[官方环境变量](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/package_reference/environment_variables)

SSH 重连后执行以下恢复命令。先向当前用户 GPU 0 上的打标进程发送 SIGTERM 并等待退出，再检查是否已有打标进程，最后用 `nohup` 在 GPU 0 后台启动并追加到 `robometer_pilot_seed7_gpu0_http.log`。命令复用现有环境、Hugging Face 缓存及 GPU 0 的打标目录。完整缓存文件会直接复用，未完成文件的下载由 Hugging Face 客户端按缓存状态恢复。后台运行用于跨 SSH 断线继续工作。

```bash
(
  set -e
  cd ~/canyon43079/H-JEPA-VerifierVLA
  old_pilot='[p]ython.*-m scripts[.]label_verifier_long_pilot.*--gpu[[:space:]]+0([[:space:]]|$)'
  pkill -TERM -u "$(id -u)" -f "$old_pilot" || true
  while pgrep -u "$(id -u)" -f "$old_pilot" > /dev/null; do
    sleep 1
  done
  if pgrep -u "$(id -u)" -af '[p]ython.*-m scripts[.]label_verifier_long_pilot'; then
    printf '%s\n' 'Existing pilot process found. Check its progress before restarting.'
    exit 0
  fi
  mkdir -p outputs/verifier_long
  nohup env -u PYTHONPATH HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_VERBOSITY=info HF_HUB_DOWNLOAD_TIMEOUT=60 \
    .venv-robometer/bin/python -u -B -m scripts.label_verifier_long_pilot \
    --split-dir outputs/verifier_long/split_seed7 \
    --output-dir outputs/verifier_long/robometer_pilot_seed7_gpu0 \
    --gpu 0 \
    --batch-size 8 \
    >> outputs/verifier_long/robometer_pilot_seed7_gpu0_http.log 2>&1 < /dev/null &
  printf 'Started GPU 0 pilot PID %s\n' "$!"
)
```

通过新命令启动后，可用下面的命令持续查看日志，按 `Ctrl+C` 结束日志查看。打标完成时读取 `summary.json` 并回传。若恢复命令检测到原进程仍在运行，原前台进程的输出可能没有写入这个日志文件，可以先检查汇总是否已生成。

```bash
tail -n 60 -f ~/canyon43079/H-JEPA-VerifierVLA/outputs/verifier_long/robometer_pilot_seed7_gpu0_http.log
```

```bash
cat ~/canyon43079/H-JEPA-VerifierVLA/outputs/verifier_long/robometer_pilot_seed7_gpu0/summary.json
```

每回合输出 `progress.json`、`window_labels.jsonl`、`progress.png` 与汇总，记录源视频及索引哈希、上下文帧编号、软件与模型版本。完整回合输出统一发布，中断后同一命令会复用已完成的回合。汇总中的 `COMPLETE` 表示打标文件生成完成，`label_quality_status=REVIEW_REQUIRED` 表示需要结合曲线和视频检查语义质量。耗时字段覆盖模型加载后的本轮工作，显存字段为当前进程的 PyTorch 分配／保留峰值。

回传最后的 JSON，重点查看四回合的首末进度、成功／超时之间的差异、负增量与接近零增量比例，以及显存和耗时。进度头允许表达停滞与退步。试打标结果回传后再确定全量标签配置。

## 后续顺序与环境

1. 运行四条训练轨迹的 Robometer 试打标，核对进度曲线和逐步时间对应，测量资源开销。
2. 固定标签配置后生成训练与验证所需的标签，验证侧用于配置选择，测试侧用于最终评测。
3. 固定 V-JEPA 2-AC 源码/权重，验证迁移覆盖率与图像、状态、动作语义；新增 8D 状态投影及任务/进度参数。
4. 使用可求梯度的前向建立小样本训练检查，再训练 verifier；冻结 π₀.₅ 与视觉编码器，保存数据划分、权重来源、配置及训练日志。
5. 在 Long 的验证集上检查候选排序并校准双轴触发阈值，最后在保留的测试初始状态上运行同协议闭环对照。

**环境与验证状态。** 八项打标数据与恢复测试、此前十一项数据审计和划分测试均通过。本地另在 Python 3.10／Torch 2.8.0 CPU／Transformers 4.57.2 上，使用固定的真实 Qwen 处理器和随机初始化的小规格官方 RBM，验证图像预处理、特殊 token、前向、保存与重载，词表为 151,674，输入为 `[2,310]`，进度输出为 `[2,4]`，重载缺失参数为 0。曲线输出也已渲染检查。完整 4B 权重加载、GPU 显存、打标速度与语义质量由本轮服务器试运行验证。当前 Robometer 试打标使用 GPU 0，单进程运行。评测最多两组并行。Verifier 的预训练迁移、8D 状态适配、训练和闭环对照待完成。
