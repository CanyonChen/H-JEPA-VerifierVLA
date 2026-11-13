# B5：冻结 π₀.₅，训练 VeriVLA 验证器

更新日期：2026-11-13。用户提供了官方资源核查；本轮复核了固定版本 README、空的 Releases 列表、论文相关段落与本地模型实现。**当前没有可直接加载的作者训练好 VeriVLA 权重，后续按官方结构与论文方法自行训练 verifier。** π₀.₅ 策略继续冻结，已完成的 B3/B4 数字作为历史基线保留。新增训练、数据与接口属于本项目复现，不称为作者 checkpoint 的直接测评。

当前只完成了数据检查工具及本地测试；服务器原始轨迹尚未检查，未打标、未迁移预训练权重、未启动训练。VeriVLA 对照仍未运行。

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

代码依据：[LIBERO 采集](../scripts/eval_libero.py)、[Ultra-Long 采集](../scripts/eval_lilo.py)。下表是记录格式，不是服务器文件已完整的结论。

| 内容 | 已实现的记录 | 训练含义 |
| --- | --- | --- |
| 图像 | 每次决策前 NPZ 保存两路 224×224 uint8 图像；每控制步前及终止后写 MP4，左右拼接成 448×224 | 视频预期为 `policy_steps+1` 帧。中间和终止目标图像可来自 MP4；需记录压缩与从 224 调整到训练尺寸的适配 |
| 状态 | 决策开始时的 8D 状态；没有每个控制步的机器人状态 NPZ | 增加独立可训练 8D 状态投影。拟在五个子块中重复当前决策状态，与推理保持一致，避免训练时使用未来真实状态 |
| 动作 | 每次生成 `(10,7)`，实际执行前 `executed_steps` 步，通常为 5，终止时可能更短 | 标签只对应实际执行前缀。后五步未执行，不可拿重规划后的后续视频为其打标签 |
| 任务 | NPZ 保存完整 prompt，回合索引保存 task ID | 检查回合内 prompt 一致，并区分不同 suite 的 task ID；不能混用编号 |
| 恢复 | 最终 `episodes.jsonl` 引用部分旧目录的产物 | 从最终索引读取并去重，保留旧目录；不能递归收集所有历史运行后把保留回合再计一次 |

**拟定的首版训练时域。** 一个完整执行前缀为 5 步，分成 `M=5, K=1`。在控制步 t 开始的决策，其输入为当前图像/状态，动作是已执行的前五步，目标图像取视频帧 `t+1 ... t+5`，标签为对应相邻时刻 Robometer 进度差。此为本项目的五步前缀适配，不自动等同于论文完整候选长度。短于五步的尾段先单独计数，训练时需明确 mask 或变长策略，不能填造未执行动作的真值。逐控制步的 Robometer 差分可能过于平坦或嘈杂，先做标签质量检查再冻结训练配置。

**Long 和 Ultra-Long 的覆盖不同。** 现有 Long 为 459 成功、41 超时；Ultra-Long 为 0 成功、76 乱序失败、14 超时，只有 14 个累计按序完成技能。后者主要覆盖早期状态，不能据此声称具备整条技能链的训练覆盖。官方模型使用任务专属查询；只训练 Long 十任务后，不能直接假定新 Ultra-Long 任务/顺序也已学会评分。先用 Long 验证训练链路，Ultra-Long 另行设计覆盖中后段的训练数据与独立测试。

## 数据划分先于打标和训练

现有 500 回合已经用于 B3 评测。若将其中任何回合用于训练或选择阈值，原 500 回合全量不能继续作为新 verifier 的独立测试集；同样适用于 Ultra-Long 的 90 回合。保留历史记录；新的公平对照应在未参与训练、标签规则调优、阈值和超参数选择的同一测试集合上报告 base 与 verifier。

划分以完整回合为单位。Long 至少按 `(suite, task_id, init_state_id)` 分组，让相同任务/初始状态的重复 rollout 留在同一集合；时间窗口不能跨集合。恢复引用不是新样本。Ultra-Long 的 `trial_id` 不是固定初始状态 seed，应追踪源回合与保存的初始状态，防止重复状态分流。先核对数据，再固定 split manifest；本轮不自动分配训练/验证/测试比例，也不选择有利的回合。

同任务、同子块位置上的排序训练样本，并不要求它们来自同一物理状态；但在这种离线排序上表现好，仍需独立的同状态候选比较或闭环实验检验实际选择能力。

## 当前服务器步骤：只读数据检查

[audit_verifier_data.py](../scripts/audit_verifier_data.py) 使用 NumPy 和已安装的 imageio-ffmpeg，读取最终索引中的每个回合、全部决策 NPZ，并解码核对视频帧数。检查生成/执行长度、连续时间线、8D 状态、7D 动作、图像尺寸、任务指令及重复索引/产物引用。它不加载模型、不生成标签、不修改源文件，也不需要 GPU；每 50 个回合向终端打印进度，最后输出 JSON。

直接复用 B3 的 `.venv-libero`，无需安装或升级依赖：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main &&
env -u PYTHONPATH .venv-libero/bin/python -B -m scripts.audit_verifier_data \
  --run-dir \
  outputs/pi05_libero_parallel/pi05_libero_10_gpu1_seed7_20261112T131428_923173Z \
  outputs/pi05_lilo_formal/pi05_ultra_long_all_gpu1_unseeded_20261113T122040_836162Z
```

回传最后的 JSON。重点查看各运行的 `checked_episodes`、`episodes_with_errors`、`retained_artifact_episodes`、`verified_totals.full_five_step_prefixes`、`short_prefixes` 和 `per_task`。`audit_status=COMPLETE` 只说明本轮文件与时间线检查通过，不表示进度标签可靠或已经可以训练；帧数对齐也不等于逐像素视觉复核。

本地已在 B3 兼容 Python 3.8 环境通过六项测试，覆盖恢复路径只读、完整/短前缀统计、终止帧缺失、NPZ 缺失、未执行后缀误用、重复回合，以及实际 FFmpeg 帧计数。服务器检查结果待回传。

## 后续顺序与环境

1. 完成上述数据检查，确定可用回合、完整前缀和短尾段数量。
2. 固定互斥的数据划分，用少量训练侧成功/失败视频检查 Robometer 标签与逐步时间对应；验证侧用于配置选择，测试侧保持独立。
3. 固定 V-JEPA 2-AC 源码/权重，验证迁移覆盖率与图像、状态、动作语义；新增 8D 状态投影及任务/进度参数。
4. 使用可求梯度的前向建立小样本训练检查，再训练 verifier；冻结 π₀.₅ 与视觉编码器，保存数据划分、权重来源、配置及训练日志。
5. 独立验证候选排序，校准双轴触发阈值，最后运行同协议闭环对照；Ultra-Long 数据覆盖和查询适配单列。

**环境约束。** 数据检查继续复用 B3 环境。Robometer 的[官方依赖](https://github.com/robometer/robometer/blob/main/pyproject.toml)要求 Python 3.10、Torch 2.8.0、其 robometer extra 要求 Transformers ≥4.57；当前 π₀.₅ 模型环境为 Python 3.11、Torch 2.7.1、Transformers 4.53.2。因此后续标注应通过独立部署的标注服务或其他经过验证的隔离方式接入，不能直接升级现有 B3/B4 环境。本轮未新建环境、安装依赖或开启训练。后续服务器 GPU 工作仍只使用 GPU 1，资源预算按具体标注/训练配置确定，评测仍最多两组并行。
