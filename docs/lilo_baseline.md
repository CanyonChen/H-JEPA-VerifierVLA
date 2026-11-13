# B4：π₀.₅ / LiLo-VLA Ultra-Long

更新日期：2026-11-13。服务器已回传 v2 协议的 9 回合 smoke：0 次成功、1 次超时、8 次乱序终止，AP 为 1.11%，基础设施异常为 0。旧动作回放在第 5 步观测不符；当前保持 v2，下一步用三回合现场诊断核查终止原因，正式 90 回合待执行。后续运行固定为 **GPU 1，最多两组并行**，复用 B3 环境。本轮先建立单候选 π₀.₅ baseline，后续 π₀.₅＋原版 VeriVLA 沿用同一协议。

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

回传输出 JSON。脚本使用运行当时保存的技能定义，索引从 0 开始；`previous_logged_value` 是上一个控制步的记录，技能切换时的内部高度基准重设可能使它与监测器内部前值不同。`inspection_status: COMPLETE` 仅表示所需记录读取完成，不表示评分逻辑已验证。v2 结果保留，正式 90 回合尚未运行。

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

### 回放结论与当前下一步

服务器在 GPU 1 上回放 task 0、3，模型请求数为 0；两者均在第 5 步的第二次观测核验返回 `DIVERGED`，没有到达第 59 步终止现场。重置和静置后的状态误差分别约 `4.92e-15`、`9.79e-16`，但执行动作后，本体观测误差升至约 `1.83e-4`、`2.14e-4`，手腕图像 MAE 为 `0.216`、`0.255`，超过回放容差。因此回传 `CHECK_REQUIRED`，不能据此给原回合下夹爪接触结论。开头的宏配置、Gym 和 NumPy 警告不是此次停止原因，不据此更换 B3 依赖。

保留旧回放记录和容差，转为实际 rollout 现场取证。新增显式 `--record-physics` 开关：在每个已执行的策略步读取 pickedup/turnon 对象的高度、夹爪距离、夹爪任意部位接触、左右指垫接触和开关关节位置，写入 `physics.jsonl`；分别累计任意夹爪接触、双指垫接触的步数和最近接触步，以区分抓取与拨动开关。读取不调用会修改可视化属性的 `turn_on()`；技能真值仍由原评分器记录。该开关默认关闭，v2 任务指令、策略、控制预算与终止规则保持不变。

下一次只跑 task 0、1、8 各一回合，覆盖三类场景，GPU 1 最多两组并行。这是新初始化的新诊断样本，不是旧回合的重放，不并入原九回合 smoke 或正式九十回合；耗时包含额外记录开销。复用 B3 `.venv`、`.venv-libero` 和权重缓存：

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

回传最后的精简 JSON，重点是每回合的 `live_physics`：终止时的高度差、接触状态及此前双指垫接触步数。它来自新回合的实际执行现场；接触是物理证据，不作为新成功规则，也不直接证明操作意图。读档工具会检查物理记录是否覆盖全部技能记录；旧运行未启用现场记录时该字段为 `null`。正式 90 回合与 VeriVLA 对照尚未运行。

## 服务器执行

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

### 3. 九十回合正式评测（完成评分核查后）

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

本轮单卡运行位于 `outputs/pi05_lilo_parallel/pi05_ultra_long_all_gpu1_unseeded_*/`，包含 `run_config.json`、`summary.json`、合并的 `episodes.jsonl`、GPU 日志，以及各 task 的服务和客户端日志。`task_devices` 记录任务实际分配的模型卡和 EGL 卡，`gpu.csv` 按卡记录；`sampled_peak_vram_gib_by_gpu` 是各卡峰值，本轮仅 GPU 1，`sampled_peak_vram_gib` 对应该卡的总占用采样峰值。每回合保存初始模拟器状态、双相机视频、各次策略请求的完整观测、动作块及已执行步数、逐步技能真值与事件、最终 `result.json`。checkpoint 内容指纹、服务 ID 和请求编号沿用 B3 实现。

本地已验证：上游 8 个导入文件的 Git blob 哈希全部匹配固定提交；9 配置的数量、技能映射和完整指令；三种场景实际创建、64×64 离屏渲染及九配置全部谓词绑定；25 个 B4 CPU 协议/rollout/预检/诊断/回放测试；17 个并行调度测试（Windows 跳过 1 个 POSIX 端口测试），覆盖 GPU 1 两组并行、诊断开关传递、双卡每卡一组、smoke/正式回合不重复及恢复流程。完整指令实测为 88 / 156 / 97 tokens，各顺序长度相同，低于当前 `pi05_libero` 的 200-token 上限；该配置 `discrete_state_input=False`。

新增回放工具的本地验证：task 0、3 各执行并恢复 60 步固定动作，12 次请求边界的观测及逐步谓词均通过比对；最大本体观测误差约 `6.38e-6`、最大图像 MAE 约 `0.00324`。这是恢复接口探针，不是服务器动作回放或 π₀.₅ 性能结果。新增四个 CPU 测试覆盖仅执行已记录动作数、输入文件不变、恢复状态不符时停止、观测/谓词不符时拒绝判为匹配。

现场记录的本地验证：task 0、1、8 各执行五步固定动作，分别读取 4、8、6 个对象。每次读取前后模拟器状态和 pickedup 高度基准均不变，记录可序列化为有限数值 JSON。CPU 测试覆盖接触计数不跨回合继承、读取不调用带副作用的谓词、终止现场与技能记录对应，以及缺失物理记录时返回不完整。该探针没有调用 π₀.₅；三回合现场诊断仍待服务器运行。

本地场景检查使用已有 Windows Python 3.8 / MuJoCo 3.2.3 验证环境；三类场景均通过 10 步静置＋5 步固定动作兼容探针，该探针不使用 π₀.₅ 策略。`reset_sim_state.npy` 保存静置前状态，`initial_sim_state.npy` 保存静置后、首次策略请求前的状态。服务器已回传 GPU 1 单 worker 的九回合结果，原始轨迹与视频待核查；本轮尚无 GPU 1 双 worker 的实测吞吐结果。LiLo 谓词补丁仅在 B4 客户端进程内导入，不改写共享 LIBERO 源文件；B3 的入口与归档 459/500 结果保留。
