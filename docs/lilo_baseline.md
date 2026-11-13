# B4：π₀.₅ / LiLo-VLA Ultra-Long

更新日期：2026-11-13。评测器已实现并通过本地协议测试；服务器预检、9 回合 smoke 和 90 回合正式评测尚未执行。本轮先建立单候选 π₀.₅ baseline，后续 π₀.₅＋原版 VeriVLA 沿用同一协议。

## 固定协议

| 项目 | 本轮设置 |
| --- | --- |
| 协议标识 | `pi05_lilo_ordered_v1` |
| LiLo 来源 | [YY-GX/LiLo-VLA @ 64aa29d](https://github.com/YY-GX/LiLo-VLA/tree/64aa29d77f5457afb3d1642cfc84534dc1e81fad)；[本地副本与许可](../third_party/lilo_vla/UPSTREAM.md) |
| 环境 | 官方三份 Ultra-Long BDDL、技能 JSON、suite 注册和进程内兼容补丁；底层 LIBERO 为 `f78abd6` |
| 客户端 | 独立 `.venv-lilo`：Python 3.10.22、NumPy 1.26.4、robosuite 1.4.1、MuJoCo 3.3.0、PyTorch 2.2.0 CPU；[依赖锁](../envs/lilo/uv.lock) |
| 模型 | 现有 `.venv`、官方 JAX `pi05_libero` 检查点及其归一化统计；每次生成 10 步、执行前 5 步后重新推理 |
| 输入 | 两路图像旋转 180° 并缩放至 224×224、8D 本体状态、整条固定任务指令；不向策略提供实时技能进度或下一子目标 |
| 指令 | 按官方 `tasks_and_skills.json` 的 `skills` 逐项用 `, then ` 连接，保留物体编号；不使用 suite 中不完整的短标题/语言描述 |
| 初始化 | 每回合新建模拟器，调用官方 `make_env()` 的 bare `env.reset()`；不设环境 seed，不加载 `.pruned_init`，不增加静置动作 |
| 预算 | **项目端到端预算**＝200 × 技能数，分别为 1800 / 3200 / 2000 个控制步；不包含 LiLo 运动规划器或恢复模块 |
| 并行 | 默认固定 2 组，每组一个模型服务和一个仿真客户端；GPU/EGL 默认均为 1，沿用自然 CPU/线程调度 |
| 规模 | smoke：9 配置 × 1＝9 回合；正式：9 × 10＝90 回合，分别保存 |

这里复用上游 benchmark 的场景、任务顺序、谓词及 SR/AP 公式。**完整指令构造、端到端总步数预算和下面的时序监测是本项目适配协议**，不是对论文端到端基线逐位一致的复现。上游 `vla_horizon=200` 是 LiLo 单技能交互上限；不能把本项目的 200×技能数称为上游原始端到端预算。论文数字只作有协议差异的参考。

| 本地 task ID | 官方定义 ID | 场景 | 顺序 | 技能数 / 总步数 |
| --- | --- | --- | --- | --- |
| 0 / 3 / 6 | 8 / 19 / 22 | Complete Kitchen Organization | Original / Reorder 1 / Reorder 2 | 9 / 1800 |
| 1 / 4 / 7 | 9 / 20 / 23 | Organize Table | Original / Reorder 1 / Reorder 2 | 16 / 3200 |
| 2 / 5 / 8 | 10 / 21 / 24 | Cooking Preparation Setup | Original / Reorder 1 / Reorder 2 | 10 / 2000 |

### 计分与错误分类

[时序监测器](../scripts/lilo_protocol.py)在每个实际执行的控制步后检查官方技能谓词，累计完成的有序前缀，每步最多推进一个技能；已完成技能不要求在最后一帧仍同时成立。当前技能满足时推进；未来非相邻技能的谓词若从假变真，记录 `out_of_order` 并结束回合。当前技能尚未完成时，下一技能提前发生也属于乱序。当前技能与紧邻下一技能在同一采样步满足时允许继续，下一个技能在下一步检查；与当前谓词完全相同的重复条目不单独触发乱序。

reset 时已经为真的未来条件不作为“新发生”的乱序事件。`pickedup` 的高度基准在每回合清空，在一个 pick 技能变为当前技能时重新采样，避免全局缓存跨回合继承。`env.done` 只记录，不直接作为整条任务成功。上述规则及谓词真值写入 `skills.jsonl`，可审计有序完成数。

- SR：`mean(skills_completed == total_skills)`；同时报告 Wilson 95% 区间。
- AP：`mean(skills_completed / total_skills)`，按回合等权平均，不是把不同长度任务的技能数合并后相除。
- `timeout`、`out_of_order` 是有效任务失败，保留部分进度；加载、通信、仿真、计分或记录异常属于 `infrastructure_error`，中断单列。
- `PASS` 表示预定回合均有有效结果，**不表示每个任务成功**。有缺失或基础设施异常时整体为 `INCOMPLETE`，整体 SR/AP 为 `null`；不能剔除异常后公布正式成功率。

指标依据：[官方复现说明](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/docs/REPRODUCTION.md)、[技能定义](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/configs/tasks_and_skills.json)。上游历史汇总中的 `metrics.*.success_rate` 可能对应瞬时 BDDL 条件，不能直接作为有序 SR。

## 服务器执行

### 1. 更新与环境预检

沿用 B3 的服务器目录和已安装模型环境，不重新下载完整训练数据或模型权重：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main
bash scripts/setup_lilo_env.sh --egl-device 1
```

脚本按锁文件创建独立 `.venv-lilo`，随后逐配置创建环境、绑定全部技能谓词、检查两路相机并保存预检图片。最后应输出 `status: PASS`、9 条 task 报告及 NVIDIA 的 OpenGL vendor/renderer。产物位于 `outputs/lilo_check/run_*/`。此处只有环境检查，没有策略成绩。

### 2. 九回合 smoke

```bash
bash scripts/run_lilo_baseline.sh \
  --gpu 1 --egl-device 1 --max-workers 2 \
  --trials-per-task 1
```

调度器先用实际 π₀.₅ tokenizer 检查完整指令，再启动现有模型服务并复用缓存权重。模型服务会执行原有的 checkpoint 内容指纹与 warmup 检查。期望结束报告 `planned_episodes: 9`、`valid_episodes: 9`、`infrastructure_errors: 0`，并有 SR/AP。先回传预检和 smoke 的末尾 JSON；根据逐步记录和视频核对评分后再开始正式评测。

### 3. 九十回合正式评测

```bash
bash scripts/run_lilo_baseline.sh \
  --gpu 1 --egl-device 1 --max-workers 2 \
  --trials-per-task 10
```

每次启动生成独立运行目录，smoke 不并入正式 90 回合。报告包含整体、9 个配置、3 个顺序组的 SR/AP。不要为追求成功率临时更改总步数、指令、初始化或技能检查规则。

### 4. 中断恢复

复制需要恢复的那次运行实际 `output_dir`，保持原来的 `--trials-per-task`、任务集合和步数预算。例如恢复正式评测：

```bash
bash scripts/run_lilo_baseline.sh \
  --gpu 1 --egl-device 1 --max-workers 2 --trials-per-task 10 \
  --resume-from outputs/pi05_lilo_parallel/<原运行目录名>
```

恢复保留成功、超时、乱序失败的连续有效回合前缀，只补缺失回合；校验 checkpoint 指纹，按每任务已完成模型请求数推进采样 RNG。环境仍使用新的未设 seed reset；`trial_id` 是回合序号，不是固定初始状态编号，也不保证恢复中断回合的同一物理状态。恢复产物写入新目录并引用保留记录。

## 产物与验证边界

每次运行位于 `outputs/pi05_lilo_parallel/pi05_ultra_long_all_gpu1_unseeded_*/`，包含 `run_config.json`、`summary.json`、合并的 `episodes.jsonl`、GPU 日志，以及各 task 的服务和客户端日志。每回合保存初始模拟器状态、双相机视频、各次策略请求的完整观测、动作块及已执行步数、逐步技能真值与事件、最终 `result.json`。checkpoint 内容指纹、服务 ID 和请求编号沿用 B3 实现。

本地已验证：上游 8 个导入文件的 Git blob 哈希全部匹配固定提交；9 配置的数量、技能映射和完整指令；三种场景实际创建、64×64 离屏渲染及九配置全部谓词绑定；13 个 B4 CPU 协议/rollout 测试；15 个并行调度测试（Windows 跳过 1 个 POSIX 端口测试）。完整指令实测为 88 / 156 / 97 tokens，各顺序长度相同，低于当前 `pi05_libero` 的 200-token 上限；该配置 `discrete_state_input=False`。

本地场景检查使用已有 Windows Python 3.8 / MuJoCo 3.2.3 验证环境，只证明场景与对象绑定能够工作。锁定的 Linux Python 3.10 / MuJoCo 3.3.0 环境、H100 EGL 及真实 π₀.₅ 闭环以服务器回传为准；目前没有 B4 成功率结果。原 `.venv-libero`、B3 原始成功谓词和归档 459/500 结果保持独立。
