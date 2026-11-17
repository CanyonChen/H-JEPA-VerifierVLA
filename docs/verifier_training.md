# B5 LIBERO-Long verifier 训练

更新日期为 2026-11-17。按用户本轮确认，**当前 B5 使用 LIBERO-Long（`libero_10`）的 10 个任务、500 个回合，推进数据检查、划分、打标、verifier 训练及闭环对照。** π₀.₅ 与视觉编码器冻结，verifier 基于官方结构和论文方法进行本项目复现。

2026-11-17，服务器回传 Long 数据审计 `COMPLETE`，500 回合全部检查通过，错误数为 0。确认 27,414 个完整五步前缀和 359 个短前缀。数据划分与窗口索引工具已通过本地测试，下一步在服务器生成固定清单。进度打标、预训练迁移、训练与闭环对照待完成。[服务器审计记录](results/pi05_libero_long_verifier_data_audit.json)

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

划分以完整回合为单位，按 `(suite, task_id, init_state_id)` 分组。每个任务的 50 个初始状态按 `libero_10|split_seed|task_id|init_state_id` 的 SHA256 排序，依次分配 40 个训练、5 个验证和 5 个测试回合。默认 `split_seed=7` 专用于数据划分，分组结果独立于轨迹结果和输入索引顺序。同一回合的全部窗口归入同一集合，最终索引中重复的任务与初始状态会被报告为错误。预计全量输出为 400／50／50 回合。

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

## 当前服务器步骤 数据划分与窗口索引

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

回传命令最后的 JSON，核对三组回合数为 400／50／50、总窗口数为 27,773、完整五步窗口数为 27,414、短窗口数为 359。各组具体成功／超时数和窗口数由服务器源记录确定。完整逐任务信息保存在 `manifest.json`。

准备工具已在 Python 3.8 通过五项本地测试，覆盖分组隔离与顺序稳定性、重复／缺失初始状态、恢复路径与短窗口、命令入口及重复运行、写入失败清理与重试。服务器分组结果待回传，随后进入进度标签检查。

## 后续顺序与环境

1. 在服务器运行上述准备命令，固定 Long 的训练、验证和测试清单。
2. 用少量训练侧成功／超时视频检查 Robometer 标签与逐步时间对应，验证侧用于配置选择，测试侧保持独立。
3. 固定 V-JEPA 2-AC 源码/权重，验证迁移覆盖率与图像、状态、动作语义；新增 8D 状态投影及任务/进度参数。
4. 使用可求梯度的前向建立小样本训练检查，再训练 verifier；冻结 π₀.₅ 与视觉编码器，保存数据划分、权重来源、配置及训练日志。
5. 在 Long 的验证集上检查候选排序并校准双轴触发阈值，最后在保留的测试初始状态上运行同协议闭环对照。

**环境约束。** 数据检查继续复用 B3 环境。Robometer 的[官方依赖](https://github.com/robometer/robometer/blob/main/pyproject.toml)要求 Python 3.10、Torch 2.8.0、其 robometer extra 要求 Transformers ≥4.57；当前 π₀.₅ 模型环境为 Python 3.11、Torch 2.7.1、Transformers 4.53.2。因此后续标注应通过独立部署的标注服务或其他经过验证的隔离方式接入，不能直接升级现有 B3/B4 环境。本轮未新建环境、安装依赖或开启训练。后续服务器 GPU 工作仍只使用 GPU 1，资源预算按具体标注/训练配置确定，评测仍最多两组并行。
