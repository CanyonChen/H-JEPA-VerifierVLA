# B4：π₀.₅ / LiLo-VLA Ultra-Long

更新日期：2026-11-13。**B4 的 π₀.₅ 单候选正式评测已完成：90 个有效回合，SR 0/90＝0%，AP 1.6049%。** 最终合并集包含 76 次乱序失败、14 次任务超时，基础设施异常与未尝试回合均为 0。保留首段 32 个有效回合，恢复段新增 58 个；此前的心跳异常单独保留在历史记录中。运行复用 B3 环境，固定 v2、**GPU 1 最多两组并行**。本轮基线冻结，下一步为 B5 的原版 VeriVLA 资源与评分接口交接；VeriVLA 对照尚未运行。

## 正式结果

[完整回传汇总](results/pi05_lilo_ultra_long_formal_v2.json)来自用户粘贴的服务器 JSON；已在本地核对九任务、三个顺序组及整体的计数、AP、Wilson 区间和恢复段吞吐量，全部一致。原始逐回合文件与视频尚未下载复核；回传摘要未包含实际运行的项目提交号。最终目录为 `outputs/pi05_lilo_formal/pi05_ultra_long_all_gpu1_unseeded_20261113T122040_836162Z`。

`PASS` 表示计划的有效回合已齐全。成功率为 0%，报告的 Wilson 95% 区间为 **0%–4.0936%**；平均进度 AP 为 **1.6049%**，共记录 14 个按序完成的技能。累计有效回合执行 33451 个控制步、6720 次策略请求。

| task | 场景 / 顺序 | 成功 / 有效回合 | 乱序失败 | 超时 | AP |
| --- | --- | --- | --- | --- | --- |
| 0 | 厨房整理 / Original | 0/10 | 10 | 0 | 0% |
| 1 | 桌面整理 / Original | 0/10 | 10 | 0 | 0% |
| 2 | 烹饪准备 / Original | 0/10 | 0 | 10 | 10.0000% |
| 3 | 厨房整理 / Reorder 1 | 0/10 | 6 | 4 | 4.4444% |
| 4 | 桌面整理 / Reorder 1 | 0/10 | 10 | 0 | 0% |
| 5 | 烹饪准备 / Reorder 1 | 0/10 | 10 | 0 | 0% |
| 6 | 厨房整理 / Reorder 2 | 0/10 | 10 | 0 | 0% |
| 7 | 桌面整理 / Reorder 2 | 0/10 | 10 | 0 | 0% |
| 8 | 烹饪准备 / Reorder 2 | 0/10 | 10 | 0 | 0% |

| 顺序组 | 成功 / 有效回合 | 乱序失败 | 超时 | AP |
| --- | --- | --- | --- | --- |
| Original | 0/30 | 20 | 10 | 3.3333% |
| Reorder 1 | 0/30 | 26 | 4 | 1.4815% |
| Reorder 2 | 0/30 | 30 | 0 | 0% |

只有 task 2、3 出现非零有序进度，分别累计完成 10、4 个技能；其余七配置 AP 为 0。按各配置技能数归一化后，整体 AP 为 `(10 / 10 + 4 / 9) / 90`。这说明当前 π₀.₅ 检查点在本项目 Ultra-Long 协议下几乎无法推进任务；尚无 π₀.₅＋VeriVLA 的对照结果，不能据此评价 verifier 的收益。LIBERO-Long 的 459/500、91.8% 作为另一 benchmark 的基线独立报告。

**恢复与耗时口径。** 最终 90 回合由首段有效 32 回合与恢复段新增 58 回合合并。新增部分为 50 次乱序失败、8 次超时，未报告新基础设施异常；原先 task 2 / trial 2 的心跳断连记录仍保存在旧目录和[异常归档](results/pi05_lilo_formal_segment1_error.json)中。最终汇总的 `infrastructure_errors=0` 描述合并后的有效评测集，并不抹去此前故障。

恢复段耗时 **1308.48 秒（约 21 分 48 秒）**，吞吐 **159.57 个新增有效回合/小时**，GPU 1 整卡显存采样峰值 **18.98 GiB**；这些字段仅覆盖恢复段。两段实际运行时间相加为 2634.70 秒（约 43 分 55 秒），不含两次启动之间的等待；首段整卡采样峰值为 42.08 GiB。

**下一阶段。** 按 2026-11-13 的资源核查，B5 需要冻结 π₀.₅、自行训练 verifier。当前先只读检查已有轨迹，再固定独立数据划分、核对预训练迁移并构造标签；见 [B5 训练准备](verifier_training.md)。Ultra-Long 的现有 90 回合主要覆盖早期失败，需要另行设计训练覆盖；如果将本轮回合用于训练/调参，后续不能仍将其全量作为独立测试。B4 的历史结果和 v2 评分保留，自行补齐部分明确记录为项目复现。

## 固定协议

| 项目 | 本轮设置 |
| --- | --- |
| 协议标识 | `pi05_lilo_ordered_v2`；v2 使用 B3 的既有仿真依赖与 10 步静置，策略控制预算保持不变 |
| LiLo 来源 | [YY-GX/LiLo-VLA @ 64aa29d](https://github.com/YY-GX/LiLo-VLA/tree/64aa29d77f5457afb3d1642cfc84534dc1e81fad)；[本地副本与许可](../third_party/lilo_vla/UPSTREAM.md) |
| 环境 | 官方三份 Ultra-Long BDDL、技能 JSON、suite 注册和进程内兼容补丁；底层 LIBERO 为 `f78abd6` |
| 客户端 | 复用 B3 的 `.venv-libero`：Python 3.8.20、NumPy 1.22.4、robosuite 1.4.1、MuJoCo 3.2.3、PyTorch 1.11.0 CPU；[既有依赖锁](../envs/libero/uv.lock)。预检与运行均不安装或升级包 |
| 模型 | 现有 `.venv`、官方 JAX `pi05_libero` 检查点及其归一化统计；每次生成 10 步、执行前 5 步后重新推理 |
| 输入 | 两路图像旋转 180° 并缩放至 224×224、8D 本体状态、整条固定任务指令；不向策略提供实时技能进度或下一子目标 |
| 指令 | 按官方 `tasks_and_skills.json` 的 `skills` 逐项用 `, then ` 连接，保留物体编号；不使用 suite 中不完整的短标题/语言描述 |
| 初始化 | 每回合新建模拟器，调用官方 `make_env()` 的未设 seed reset；不加载 `.pruned_init`。随后执行 B3 的 10 步静置，再建立技能高度基准；静置步数独立记录，不占策略预算 |
| 预算 | **项目端到端预算**＝200 × 技能数，分别为 1800 / 3200 / 2000 个控制步；不包含 LiLo 运动规划器或恢复模块 |
| 并行 | 本轮 `--gpu 1 --egl-device 1 --max-workers 2`：GPU 1 上最多两组模型服务＋仿真客户端，共用任务队列，沿用自然 CPU/线程调度 |
| 服务通信 | 正式恢复段保留 WebSocket Ping，设置 `ping_timeout=None`；实际设置随 `service.json` 和客户端保存的服务元数据记录。仅调整传输层，不改变推理、执行或评分 |
| 规模 | smoke：9 配置 × 1＝9 回合；正式：9 × 10＝90 回合，分别保存 |

这里复用上游 benchmark 的场景、任务顺序、谓词及 SR/AP 公式。**B3 仿真依赖与静置、完整指令构造、端到端总步数预算和下面的时序监测是本项目适配协议**，不是对论文端到端基线逐位一致的复现。上游声明的环境包含 Python ≥3.10、MuJoCo 3.3.0；本轮按用户要求复用 B3 的 Python 3.8 / MuJoCo 3.2.3，实际版本随结果记录。本地零动作探针发现 Organize Table 的物体会在 reset 后弹起，直接建立 `pickedup` 高度基准会将其误记为提前拾取，因此采用 B3 的 10 步静置后再评分。上游 `vla_horizon=200` 是 LiLo 单技能交互上限；不能把本项目的 200×技能数称为上游原始端到端预算。论文数字只作有协议差异的参考。

| 本地 task ID | 官方定义 ID | 场景 | 顺序 | 技能数 / 总步数 |
| --- | --- | --- | --- | --- |
| 0 / 3 / 6 | 8 / 19 / 22 | Complete Kitchen Organization | Original / Reorder 1 / Reorder 2 | 9 / 1800 |
| 1 / 4 / 7 | 9 / 20 / 23 | Organize Table | Original / Reorder 1 / Reorder 2 | 16 / 3200 |
| 2 / 5 / 8 | 10 / 21 / 24 | Cooking Preparation Setup | Original / Reorder 1 / Reorder 2 | 10 / 2000 |

### 计分与错误分类

[时序监测器](../scripts/lilo_protocol.py)在每个实际执行的控制步后检查官方技能谓词，累计完成的有序前缀，每步最多推进一个技能；已完成技能不要求在最后一帧仍同时成立。当前技能满足时推进；未来非相邻技能的谓词若从假变真，记录 `out_of_order` 并结束回合。当前技能尚未完成时，下一技能提前发生也属于乱序。当前技能与紧邻下一技能在同一采样步满足时允许继续，下一个技能在下一步检查；与当前谓词完全相同的重复条目不单独触发乱序。

静置结束时已经为真的未来条件不作为“新发生”的乱序事件。`pickedup` 的高度基准在每回合静置后清空并重建，在一个 pick 技能变为当前技能时重新采样，避免全局缓存跨回合继承。`env.done` 只记录，不直接作为整条任务成功。上述规则及谓词真值写入 `skills.jsonl`，可审计有序完成数。

- SR：`mean(skills_completed == total_skills)`；同时报告 Wilson 95% 区间。
- AP：`mean(skills_completed / total_skills)`，按回合等权平均，不是把不同长度任务的技能数合并后相除。
- `timeout`、`out_of_order` 是有效任务失败，保留部分进度；加载、通信、仿真、计分或记录异常属于 `infrastructure_error`，中断单列。
- `PASS` 表示预定回合均有有效结果，**不表示每个任务成功**。有缺失或基础设施异常时整体为 `INCOMPLETE`，整体 SR/AP 为 `null`；不能剔除异常后公布正式成功率。

指标依据：[官方复现说明](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/docs/REPRODUCTION.md)、[技能定义](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/configs/tasks_and_skills.json)。上游历史汇总中的 `metrics.*.success_rate` 可能对应瞬时 BDDL 条件，不能直接作为有序 SR。

### 2026-11-13 smoke 回传与待核查项

[回传汇总](results/pi05_lilo_ultra_long_smoke_v2.json)来自用户粘贴的服务器末尾 JSON。随后回传的只读诊断中，完整条目的客户端提交均为 `c8acc5117c9c6d866367623a6fd7373272819201`，模型和 EGL 均为 GPU 1；原始产物与视频尚未下载。

| 项目 | 回传结果 |
| --- | --- |
| 完整性 | 9/9 有效回合，0 基础设施异常、0 中断；`PASS` 仅表示执行和记录完整 |
| v2 指标 | SR 0/9＝0%，Wilson 95% 区间 0%–29.91%；AP 1.11% |
| 终止原因 | 8 次 `out_of_order`；task 2 超时，完成 1/10 技能；其余配置完成 0 技能 |
| 步数 | 乱序回合在 59–113 步结束；task 2 执行 2000 步；合计 2627 步、529 次策略请求 |
| 实际资源 | `gpus: [1]`、`selected_workers: 1`；显存采样峰值约 9.91 GiB，耗时约 35 分 24 秒 |

进一步核对[固定版本 README 的指标定义](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/README.md#results)：SR 要求整条序列按顺序完成，AP 计算首次乱序或失败前完成的技能比例。[上游 LiLo 技能执行器](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/lilo_vla/core/pipeline.py#L949)按既定技能顺序运行并检查当前技能；本项目 v2 为整条指令策略监测未来谓词。因此，不能仅凭 LiLo 执行器没有相同的全局检测循环，就删除乱序终止规则。当前要核查的是事件检测是否对应真实物体操作，v2 计分保持不变。

下一步从已有产物提取期望技能、触发终止的未来技能及谓词真值，再结合视频定位原因。以下诊断仅读取文件、不启动模型或仿真、不占 GPU、不修改结果：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main
env -u PYTHONPATH .venv-libero/bin/python -m scripts.inspect_lilo_run \
  --run-dir outputs/pi05_lilo_parallel/pi05_ultra_long_all_gpu1_unseeded_20261113T081550_589909Z
```

脚本使用运行当时保存的技能定义，索引从 0 开始；`previous_logged_value` 是上一个控制步的记录，技能切换时的内部高度基准重设可能使它与监测器内部前值不同。`inspection_status: COMPLETE` 仅表示所需记录读取完成，不表示评分逻辑已验证。此处为早期 smoke 的读档记录；正式评测的当前状态见下文。

### 逐步回传后的诊断

首份用户诊断粘贴在 task 4 附近有缺段，[片段归档](results/pi05_lilo_smoke_v2_trace_excerpt.json)保留五个完整条目。随后[回放工具的回传](results/pi05_lilo_smoke_v2_replay.json)从原文件重读，补齐九回合摘要如下；这些仍来自用户回传，原始视频尚未下载。

| task | 终止步 | 当时要求 | 触发的未来技能 / 结果 |
| --- | --- | --- | --- |
| 0 | 59 | 拾取黑碗 1 | 黑碗 3 的 `pickedup` 变真 |
| 1 | 80 | 拾取番茄酱 1 | 奶油奶酪 2 的 `pickedup` 变真 |
| 2 | 2000 | 拾取摩卡壶 1 | 已完成开炉灶 1，随后超时，进度 1/10 |
| 3 | 59 | 拾取奶油奶酪 1 | 黑碗 3 的 `pickedup` 变真 |
| 4 | 76 | 拾取字母汤 1 | 奶油奶酪 2 的 `pickedup` 变真 |
| 5 | 113 | 拾取黑碗 1 | 炉灶 1 的 `turnon` 变真 |
| 6 | 61 | 拾取黑碗 2 | 黑碗 3 的 `pickedup` 变真 |
| 7 | 76 | 拾取字母汤 1 | 奶油奶酪 2 的 `pickedup` 变真 |
| 8 | 103 | 拾取黑碗 1 | 炉灶 1 的 `turnon` 变真 |

上游 [`pickedup`](../third_party/lilo_vla/lilo_vla/libero_compat.py) 只要求物体比基准位置升高至少 0.03 m，不检查夹爪接触。上述布尔记录证明谓词触发，尚不能区分主动抓错对象与碰撞抬升。task 8 是另一种关节状态谓词，不能把全部失败都归因于拾取阈值。task 0、3 指令不同却在相同步数触发同一物体，提示需要检查策略对顺序指令的响应，但当前还不能断言模型忽略了语言。

### 回放结论与三回合现场诊断

服务器在 GPU 1 上回放 task 0、3，模型请求数为 0；两者均在第 5 步的第二次观测核验返回 `DIVERGED`，没有到达第 59 步终止现场。重置和静置后的状态误差分别约 `4.92e-15`、`9.79e-16`，但执行动作后，本体观测误差升至约 `1.83e-4`、`2.14e-4`，手腕图像 MAE 为 `0.216`、`0.255`，超过回放容差。因此回传 `CHECK_REQUIRED`，不能据此给原回合下夹爪接触结论。开头的宏配置、Gym 和 NumPy 警告不是此次停止原因，不据此更换 B3 依赖。

保留旧回放记录和容差，转为实际 rollout 现场取证。新增显式 `--record-physics` 开关：在每个已执行的策略步读取 pickedup/turnon 对象的高度、夹爪距离、夹爪任意部位接触、左右指垫接触和开关关节位置，写入 `physics.jsonl`；分别累计任意夹爪接触、双指垫接触的步数和最近接触步，以区分抓取与拨动开关。读取不调用会修改可视化属性的 `turn_on()`；技能真值仍由原评分器记录。该开关默认关闭，v2 任务指令、策略、控制预算与终止规则保持不变。

已执行 task 0、1、8 各一回合，覆盖三类场景，GPU 1 两组并行。这是新初始化的新诊断样本，不是旧回合的重放，不并入原九回合 smoke 或正式九十回合；耗时包含额外记录开销。以下保留当时命令供追溯，当前无需再追加诊断回合：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main
B4_DIAG_ROOT="outputs/pi05_lilo_live_$(date -u +%Y%m%dT%H%M%SZ)"

bash scripts/run_lilo_baseline.sh \
  --gpu 1 --egl-device 1 --max-workers 2 \
  --task-ids 0 1 8 --trials-per-task 1 --record-physics \
  --output-root "$B4_DIAG_ROOT"

env -u PYTHONPATH .venv-libero/bin/python -m scripts.inspect_lilo_run \
  --run-dir "$B4_DIAG_ROOT"/pi05_ultra_long_all_gpu1_unseeded_* --compact
```

`live_physics` 是新回合实际执行现场的终止高度、接触及此前接触步数。接触是物理证据，不作为新成功规则，也不直接证明操作意图。读档工具检查物理记录是否覆盖全部技能记录；旧运行未启用现场记录时该字段为 `null`。

### 现场结果与阶段决定

[归档](results/pi05_lilo_live_diagnostic_v2.json)分别保留用户粘贴的汇总和逐回合诊断，未下载原始文件或视频。三回合均有效，0 成功、2 次乱序终止、1 次超时、AP 0；共 3362 控制步、673 次策略请求。实际 GPU 1、两组并行，耗时约 7 分 20 秒，整卡显存采样峰值约 41.56 GiB；不把该三回合吞吐外推为正式评测耗时。

| task | 现场记录 | 对当前 v2 的解释 |
| --- | --- | --- |
| 0 | 第 60 步：目标黑碗 1 高度几乎不变，接触计数 0；黑碗 3 升高 3.261 cm，夹爪接触记录 13 步，双指垫同时接触计数 0 | 与错误物体接触并发生超过 3 cm 的抬升，按上游 pickedup 定义触发未来技能；不能写成已证实的稳定夹持 |
| 1 | 第 3200 步超时；目标番茄酱 1 高度变化 0、接触计数 0 | 第一技能未完成，未被乱序规则提前截断，仍然失败 |
| 8 | 第 102 步：目标黑碗 1 高度几乎不变、接触计数 0；炉灶夹爪接触 26 步、双指垫接触 21 步，关节值 0.5141，turnon 真值触发 | 与炉灶发生物理交互并提前开启，按当前顺序定义失败 |

这些记录支持在固定 v2 下接受低分，无证据要求放宽计分或修改策略来提高结果。pickedup 的操作性定义本来就是高度阈值，不能仅凭双指垫接触为 0 就将现有实现认定为错误，也不能把该谓词等同于稳定抓握。结束本轮行为诊断，推进完整基线；这三个新样本不证明原九回合每个失败的具体原因。

**回传一致性。** 早期诊断粘贴的 `per_task` 只含 task 0，且写了 102 步、21 次请求；逐回合诊断则给 task 0 为 60 步、task 8 为 102 步。逐回合步数 `60 + 3200 + 102 = 3362` 与整体一致。本地汇总代码按该三组步数生成正确的任务映射，未复现错位；未下载服务器原件，因此归档保留原粘贴值。读档工具已增加整体和按任务的回合数、控制步数、策略请求数及配置任务集合核对，不改变指标或旧产物。此前提供的首段命令在核对通过后启动正式测评；正式首段和恢复段现均已回传，完整结果见文首。

### 正式首段异常与修复

首段目录为 `outputs/pi05_lilo_formal/pi05_ultra_long_all_gpu1_unseeded_20261113T112336_510911Z`。[首次片段归档](results/pi05_lilo_formal_segment1_partial.json)保存用户给出的汇总尾部；随后[异常读档归档](results/pi05_lilo_formal_segment1_error.json)补齐顶层摘要和异常回合，并确认以下整体计数。后者中的日志只保留明确标注的选段，未下载服务器原件。

| 项目 | 首段记录 |
| --- | --- |
| 计划 / 已记录 / 有效 | 90 / 33 / 32 |
| 有效结果构成 | 0 次成功、6 次超时、26 次乱序失败 |
| 基础设施异常 / 未尝试 | 1 / 57 |
| 剩余有效回合目标 | 58＝重跑 1 个异常回合＋57 个未尝试回合 |
| 资源与时间 | GPU 1，两组并行；约 22 分 6 秒，整卡显存采样峰值约 42.08 GiB |

首段结束时正式评测未完成，SR/AP 为 `null`。调度器遇到异常后停止派发新任务，并允许已在运行的任务结束，因此当时后面的配置保持未尝试。恢复后的完整结果见上文。

异常已定位到 `task_id=2, trial_id=2`：`policy_inference` 阶段，客户端下一次 `send` 收到服务端的 `1011 keepalive ping timeout`。前两个回合均达到 2000 步后正常记为任务超时，属于应保留的有效结果。服务日志记录处理过 1150 次推理后关闭；本次未提供证据确定心跳延迟的具体来源。

模型服务原先继承 websockets 的 20 秒心跳等待上限。当前为上游服务类增加可配置的 `ping_timeout`，通用默认值仍为 20 秒；项目 `serve_pi05_libero.py` 显式设置 `None`，继续发送 Ping，但取消 Pong 的短超时断连。该用法用于容忍延迟峰值，见 [websockets 15.0.1 官方说明](https://websockets.readthedocs.io/en/15.0.1/topics/keepalive.html)。设置写入服务元数据 `websocket_ping_timeout_seconds`；没有增加请求重发或改变模型采样、动作和 v2 评分，已有进程退出检查继续生效。

本地使用 websockets 15.0.1 的真实连接与延迟心跳确认测试：有限超时会产生相同的 `1011`；项目设置下延迟后的第二次推理正常返回，请求编号为 1、2，动作值一致。该测试验证传输设置；随后服务器实际恢复的 58 回合也已回传完成。

以下是已经执行并回传的只读排错命令，保留供追溯。它提取异常回合及客户端最后 70 行、服务端最后 40 行日志：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main
env -u PYTHONPATH .venv-libero/bin/python -m scripts.inspect_lilo_run \
  --run-dir outputs/pi05_lilo_formal/pi05_ultra_long_all_gpu1_unseeded_20261113T112336_510911Z \
  --errors-only
```

此次 `inspection_status: COMPLETE` 表示读取完成；该首段当时的正式评测状态是 `INCOMPLETE`。恢复入口保留成功、超时、乱序失败的连续有效前缀，核对 checkpoint 指纹和请求编号，再从缺失回合续跑。

## 已执行的服务器命令（历史记录）

### 1. 更新与环境预检

沿用 B3 的服务器目录、模型环境 `.venv`、仿真环境 `.venv-libero` 和权重缓存：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main
bash scripts/run_lilo_check.sh --egl-device 1
```

脚本直接调用现有 `.venv-libero/bin/python`，在 EGL 1 上依次检查九个配置；逐配置创建环境、绑定全部技能谓词、检查两路相机并保存预检图片。最后应输出 `status: PASS`、`egl_devices: [1]`、9 条 task 报告及 NVIDIA 的 OpenGL vendor/renderer。产物位于 `outputs/lilo_check/run_*/`。此处只有环境检查，没有策略成绩。旧 `setup_lilo_env.sh` 入口也仅转到该预检，不再执行安装。

### 2. 九回合 smoke（核查后需要重跑时）

```bash
bash scripts/run_lilo_baseline.sh \
  --gpu 1 --egl-device 1 --max-workers 2 \
  --trials-per-task 1
```

调度器先用实际 π₀.₅ tokenizer 检查完整指令，再启动现有模型服务并复用缓存权重。模型服务会执行原有的 checkpoint 内容指纹与 warmup 检查。期望结束报告 `planned_episodes: 9`、`valid_episodes: 9`、`infrastructure_errors: 0`，并有 SR/AP。先回传预检和 smoke 的末尾 JSON；根据逐步记录和视频核对评分后再开始正式评测。

两组模型进程均设置 `CUDA_VISIBLE_DEVICES=1`，仿真进程均使用 `MUJOCO_EGL_DEVICE_ID=1`，最多同时执行两个配置，完成一组后领取下一配置，总计仍为 9 回合。这里用单数 `--gpu 1`；复数 `--gpus 1` 属于每卡至多一组的模式，会把并行数限制为 1。现有调度器默认也是 GPU 1、EGL 1、最多两组，命令显式写出以避免混淆。

### 3. 九十回合首段启动命令（已执行，保留供追溯）

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main

env -u PYTHONPATH .venv-libero/bin/python -m scripts.inspect_lilo_run \
  --run-dir outputs/pi05_lilo_live_20261113T105309Z/pi05_ultra_long_all_gpu1_unseeded_20261113T105415_099820Z \
  --compact &&
bash scripts/run_lilo_baseline.sh \
  --gpu 1 --egl-device 1 --max-workers 2 \
  --trials-per-task 10 --steps-per-skill 200 \
  --output-root outputs/pi05_lilo_formal
```

以上是已执行首段的命令。正式规模覆盖全部九个配置、每配置十回合，固定 `pi05_lilo_ordered_v2`；不启用额外物理记录，也不合并此前 smoke/诊断样本。此次完整运行已结束，以下恢复命令保留供追溯。

### 4. 心跳修复后的恢复（已完成）

本次通过以下入口恢复正式首段，保持原来的每配置十次、全部九配置和步数预算，复用现有环境及缓存权重：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main

bash scripts/run_lilo_baseline.sh \
  --gpu 1 --egl-device 1 --max-workers 2 --trials-per-task 10 --steps-per-skill 200 \
  --output-root outputs/pi05_lilo_formal \
  --resume-from outputs/pi05_lilo_formal/pi05_ultra_long_all_gpu1_unseeded_20261113T112336_510911Z
```

恢复保留成功、超时、乱序失败的连续有效回合前缀，只补缺失回合；校验 checkpoint 指纹，按每任务已完成模型请求数推进采样 RNG。环境仍使用新的未设 seed reset；`trial_id` 是回合序号，不是固定初始状态编号，也不保证恢复中断回合的同一物理状态。恢复产物写入新目录并引用保留记录。

最终回传确认 `valid_episodes=90`、`retained_episodes=32`、`new_valid_episodes=58`，GPU 1、两组并行。按既定恢复逻辑，task 2 保留 trial 0、1 的两次超时，按 800 次已完成请求恢复模型 RNG，再从 trial 2 重新开始完整回合。旧目录需继续保留，合并结果和被保留回合的产物仍引用它。

## 产物与验证边界

默认 smoke 位于 `outputs/pi05_lilo_parallel/pi05_ultra_long_all_gpu1_unseeded_*/`；当前正式命令将独立产物写入 `outputs/pi05_lilo_formal/`。各运行包含 `run_config.json`、`summary.json`、合并的 `episodes.jsonl`、GPU 日志，以及各 task 的服务和客户端日志。`task_devices` 记录任务实际分配的模型卡和 EGL 卡，`gpu.csv` 按卡记录；`sampled_peak_vram_gib_by_gpu` 是各卡峰值，本轮仅 GPU 1，`sampled_peak_vram_gib` 对应该卡的总占用采样峰值。每回合保存初始模拟器状态、双相机视频、各次策略请求的完整观测、动作块及已执行步数、逐步技能真值与事件、最终 `result.json`。checkpoint 内容指纹、服务 ID 和请求编号沿用 B3 实现。

本地已验证：上游 8 个导入文件的 Git blob 哈希全部匹配固定提交；9 配置的数量、技能映射和完整指令；三种场景实际创建、64×64 离屏渲染及九配置全部谓词绑定；27 个 B4 CPU 协议/rollout/预检/诊断/回放测试（含按任务汇总错位、异常正文与日志只读提取）；17 个并行调度测试（Windows 跳过 1 个 POSIX 端口测试），覆盖 GPU 1 两组并行、诊断开关传递、双卡每卡一组、smoke/正式回合不重复及恢复流程。完整指令实测为 88 / 156 / 97 tokens，各顺序长度相同，低于当前 `pi05_libero` 的 200-token 上限；该配置 `discrete_state_input=False`。

新增回放工具的本地验证：task 0、3 各执行并恢复 60 步固定动作，12 次请求边界的观测及逐步谓词均通过比对；最大本体观测误差约 `6.38e-6`、最大图像 MAE 约 `0.00324`。这是恢复接口探针，不是服务器动作回放或 π₀.₅ 性能结果。新增四个 CPU 测试覆盖仅执行已记录动作数、输入文件不变、恢复状态不符时停止、观测/谓词不符时拒绝判为匹配。

心跳修复的本地验证：8 项服务与通信测试通过，包含真实 WebSocket 下的超时复现和延迟后继续请求，以及 checkpoint 指纹、预热和 RNG 恢复检查。并行调度与续跑测试共 17 项，Windows 跳过 1 项 POSIX 端口测试。服务器上的 58 个补跑回合已回传完成，合并为正式 90 回合。

现场记录的本地验证：task 0、1、8 各执行五步固定动作，分别读取 4、8、6 个对象。每次读取前后模拟器状态和 pickedup 高度基准均不变，记录可序列化为有限数值 JSON。CPU 测试覆盖接触计数不跨回合继承、读取不调用带副作用的谓词、终止现场与技能记录对应，以及缺失物理记录时返回不完整。该探针没有调用 π₀.₅；服务器三回合现场诊断已另行回传并归档。

本地场景检查使用已有 Windows Python 3.8 / MuJoCo 3.2.3 验证环境；三类场景均通过 10 步静置＋5 步固定动作兼容探针，该探针不使用 π₀.₅ 策略。`reset_sim_state.npy` 保存静置前状态，`initial_sim_state.npy` 保存静置后、首次策略请求前的状态。服务器已分别回传九回合 smoke、三个独立现场诊断回合和正式 90 个有效回合；B4 单候选基线完成，VeriVLA 对照尚未开始，原始视频尚未下载。LiLo 谓词补丁仅在 B4 客户端进程内导入，不改写共享 LIBERO 源文件；B3 的入口与归档 459/500 结果保留。
