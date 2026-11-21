# π₀.₅ 的 LIBERO 40 任务 VeriVLA 训练

从仓库根目录运行 `train_verifier.sh`。一个 verifier 共享 Long、Spatial、Object、Goal 四套任务的 40 个任务查询。入口独立于 B5 的 Long 试训，初始权重来自 V-JEPA 2-AC。

## 查看计划

```bash
bash train_verifier.sh --gpu 0 --dry-run
```

预演只读取已有索引和标签，列出各套数据状态及将要执行的命令。Long 必须先在项目中以原目录结构提供两个采集运行、`outputs/verifier_long/split_seed7` 和 `outputs/verifier_long/robometer_raw_trainval_seed7_gpu0`。若拆分与标签在其他位置，用 `--long-root` 指定包含这两个子目录的目录。H100 索引中旧的项目根在读取时映射到当前仓库，原始索引及哈希可继续使用。其他来源的归档可用 `--recorded-project-root` 指定原项目路径。

## 正式运行与恢复

```bash
bash train_verifier.sh --gpu 0 --resume=auto
```

`--gpu`（也可写作 `--cuda`）是物理 GPU 编号，PyTorch 在该进程中使用逻辑设备 `cuda:0`。省略时采用 `CUDA_VISIBLE_DEVICES` 的第一个数字编号，未设置时采用 0。正式运行使用 Linux、CUDA、EGL、项目的 `.venv` 策略环境、`.venv-libero` 仿真环境和 `.venv-robometer` 训练环境。`VERIVLA_PYTHON` 可以指定训练解释器。

1. 核对 Long 的 500 个回合、标签和 π₀.₅ 权重指纹。Spatial、Object、Goal 各收集至 500 回合，复用已有的完整采集和标签，部分采集从已完成的回合继续。按每任务 40／5／5 拆分训练、验证、测试回合，只给训练与验证打标。
2. 从既有 V-JEPA 2-AC 权重提取冻结的视觉特征。`--checkpoint` 或 `VJEPA2_AC_CKPT` 可指定文件，默认查找 Torch 缓存及 `/data/models/vjepa2-checkpoints/vjepa2-ac-vitg.pt`。资源准备时优先复用已有 `pi05_libero` 和固定版本 Robometer、Qwen 处理器，必要时显式设置 `PI05_CHECKPOINT`、`OPENPI_DATA_HOME` 和 `HF_HUB_CACHE`。
3. 从预训练预测器开始联合训练，模型在每轮结束时对全部 200 个验证回合产生的窗口计算 Huber 和同任务排序损失，按验证 Huber 选择 `best.pt`。`last.pt` 每 200 次更新以及每轮结束时写入，保存优化器、步数和随机数状态。再次使用相同参数和目录运行会从最近一次完整保存的位置继续。
4. 用最佳权重在 200 个保留测试初始状态上做配对闭环，每个状态各运行一次 π₀.₅ 和一次带 verifier 的策略。校准只用训练轨迹，两个实验臂共享首个候选的噪声，按套件统计成败变化。

默认输出为 `outputs/verivla_pi05_libero/`。`joint40/training.json` 记录离线验证历史及训练耗时，`joint40/last.pt` 用于续训，`joint40/best.pt` 用于闭环，`joint40/closed_loop/summary.json` 记录评测结果。`--resume=required` 要求已有 `last.pt`，`--resume=never` 要求训练输出尚无权重或完成记录。需要另一组配置时用 `--job-dir` 指定独立目录，`--data-root` 和 `--feature-cache` 可继续指向已有的数据与特征。通过 SSH 使用时可在 `tmux` 会话内启动，连接恢复后接回会话。

Long 全量训练与验证的 BF16 视觉张量约需 100.2 GiB，其他套件的用量取决于真实轨迹长度。特征阶段按已有文件和缺失张量计算所需空间。缓存和 checkpoint 可以分别放在不同磁盘。例子中的目录需替换为实际已规划的存储位置。

```bash
bash train_verifier.sh --gpu 1 --resume=auto \
  --epochs 10 --microbatch 8 --feature-cache /path/to/feature-volume/verivla
```

## 与论文的对应关系

沿用冻结的视觉编码器、动作条件预测器、任务查询，以及视觉 L1、进度 Huber 和同任务排序三项损失。AdamW、学习率 `5e-4`、权重衰减 `0.04`、有效 batch 32 和五个子块参考论文真机适配表。10 轮、40／5／5 划分、Robometer-4B 的四帧上下文、单外部视角及 `H=5, M=5` 是本项目设置。每次策略产生十步，只对实际执行的至多五步监督，测试初始状态留给最终闭环。[VeriVLA 论文与附录](https://arxiv.org/html/2610.21985v1#A2.T7)
