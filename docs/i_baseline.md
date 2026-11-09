# 第一阶段：π₀.₅ baseline 复现计划

更新日期：2026-11-09  
所属阶段：[总实验计划](plan.md)中的 P0 / baseline_setup。  
当前状态：已完成论文、官方资源与本地代码静态核对，openpi、ALOHA 与 LIBERO 源码已按固定版本完整准备，并作为本仓库的普通目录管理。统一 uv 模型环境已完成依赖解析与本地 CPU 模型核心检查；服务器依赖安装、模型权重下载、检查点推理、训练与评测尚未执行。  
资源：八卡 A100/H100 服务器、MuJoCo；Piper 真机用于后续阶段。

**本阶段目标。** 使用 Physical Intelligence 官方 openpi 的 π₀.₅-LIBERO 检查点，建立可重复的单候选推理与仿真闭环 baseline。先跑通 LIBERO-Long，再完成四个标准 suite 的评测，保存后续接入 VeriVLA verifier 所需的基线结果和接口约定。

本阶段的“复现”指公开微调检查点的推理与评测复现。首先使用已有权重，不安排从头预训练或重新微调。VeriVLA 的训练、双轴触发器、候选评分和 H-JEPA 改进分别在后续阶段实施；本阶段每次决策只采样一个动作块。

**固定模型与版本。**

| 项目 | 本阶段选择 |
| --- | --- |
| 基础策略 | π₀.₅，使用 flow matching 动作头 |
| 实现 | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)；本地目录为 [openpi/](../openpi/) |
| 源码管理 | 随本项目 `main` 分支管理，`openpi/` 为普通源码目录 |
| 上游代码基准 | `215abfb217dbac7d5f1273282331b9b1866c0479`；2026-11-09 导入前核查时，上游工作副本 HEAD 与远端 main 一致 |
| 配置 | `pi05_libero` |
| 检查点 | `gs://openpi-assets/checkpoints/pi05_libero` |
| 检查点性质 | 官方 π₀.₅ @ 30k LIBERO 微调权重；`pi05_base` 是供微调的基础权重，不作为本轮替代品 |
| 初始推理后端 | JAX，直接使用官方检查点格式 |
| 运行平台 | Linux GPU 服务器；优先沿用官方测试的 Ubuntu 22.04 |
| 仿真环境 | LIBERO / MuJoCo；先 `libero_10`，再扩展 Spatial、Object、Goal |
| 候选数 | 每次决策 1 个；没有 verifier 或 trigger |
| 归一化统计 | 使用当前检查点随附的 `assets/`；不为推理重新计算训练集统计量 |

来源：[固定版本主 README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)、[LIBERO README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)。后续若修改评测代码，记录上游提交、修改内容和实际运行版本。

**源码准备与交接状态。** 以下内容由项目交接记录整合，并于 2026-11-09 在导入前复核各上游工作副本的 Git 状态。发布本项目时移除了 openpi、ALOHA 与 LIBERO 的内部 Git 元数据，源码统一纳入本仓库。

| 项目 | 已完成状态 |
| --- | --- |
| openpi 源码 | 从上述固定提交导入，导入前工作区干净 |
| ALOHA 源码 | 从上游指定提交 `d1dc83afd89ded4379851257fe5d85632d31d5ec` 导入，导入前工作区干净 |
| LIBERO 源码 | 从上游指定提交 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` 导入，导入前工作区干净 |
| LIBERO 准备方式 | 导入前通过固定提交的浅克隆取得完整文件，现随本仓库管理 |
| LIBERO 必需文件 | 已确认 `requirements.txt`、`bddl_files/`、`init_files/` 与 `assets/` 存在 |
| Git 操作入口 | 在本项目根目录执行提交与更新；三个源码目录不再各自维护 Git 历史 |

先前的 LIBERO 检出不完整与 ALOHA 子模块指针差异已在导入前解决。当前已完成的是本地源码准备；下一步在 Linux GPU 服务器克隆本项目，安装模型与仿真依赖，加载 checkpoint 及 norm stats，再进行推理和 EGL 离屏渲染检查。

**执行计划表。**

| 步骤 / 标识 | 要做的工作 | 产物 | 通过条件 | 状态 |
| --- | --- | --- | --- | --- |
| B0 / pi05_setup | openpi、ALOHA 与 LIBERO 本地源码已就绪；在服务器准备相同版本工作副本，记录驱动、GPU、依赖、磁盘缓存位置，建立模型与仿真两个环境 | 环境及版本记录 | 服务器源码、任务 assets 和初始状态齐全；GPU、离屏渲染可用 | 本地源码、uv 配置与 CPU 模型核心检查已完成；服务器环境待执行 |
| B1 / pi05_inference | 加载官方检查点及 norm stats；先用官方 simple client 检查连接，再使用真实 LIBERO 观测检查输入输出 | 推理日志、输入输出样例、冷启动与稳定推理耗时 | 返回有限数值的 `(10,7)` 动作块；真实观测预处理与官方一致 | 待执行 |
| B2 / pi05_libero_smoke | 运行 `libero_10` 每任务 1 个 episode，共 10 个；检查视频、动作执行和终止逻辑 | 十个任务的闭环记录与视频 | 无未解释的加载、通信或渲染错误；成功与失败均有可追溯记录 | 待执行 |
| B3 / pi05_long_baseline | 补齐逐 episode 记录和视频唯一命名；按固定初始化运行 Long 每任务 50 次 | 500 个 episode 的记录、逐任务成功率、Long 总结果 | 预定 episode 全部有有效结果；基础设施异常已单列并处理 | 待执行 |
| B4 / pi05_full_baseline | 用相同模型和执行协议完成 Spatial、Object、Goal，各 500 次；汇总四套 | 共 2000 个 episode 的结果及官方参考比较 | 四套逐任务计数可核对；总指标可从逐 episode 数据重算 | 待执行 |
| B5 / pi05_verifier_handoff | 固定 checkpoint、输入输出、执行长度及采样规则；整理后续候选与 verifier 接口缺口 | 基线交接记录 | 后续能够明确只改变候选生成/选择部分，保留同一基础策略及可比执行协议 | 待执行 |

单 GPU 先建立参考结果。八卡优先用于后续独立评测实例，不需要在本阶段启动分布式训练。分片评测前明确每个实例的 GPU、端口、任务/episode 范围和输出目录，并保证计数不重复；官方 runner 当前没有 task/episode 分片参数。

**环境与启动方式。** 模型侧采用本项目根目录的统一 uv 环境，供 openpi、H-JEPA 模型核心与 VeriVLA 模型共用，安装与验证见 [environment.md](environment.md)。LIBERO 客户端保留独立环境，通过模型服务通信。下方官方 Docker 命令保留为后续仿真部署参考，服务器部署方式待该阶段落实。

| 环节 | 已核对的要求 |
| --- | --- |
| 模型服务 | openpi 要求 Python ≥3.11；官方服务镜像使用 3.11.9；当前依赖包含 `jax[cuda12]==0.5.3` |
| LIBERO 客户端 | 官方镜像使用独立 Python 3.8 环境；不要将其旧依赖直接混入模型环境 |
| GPU | 官方推理显存估计为 >8 GB；实际峰值、编译开销与吞吐在服务器测量 |
| 无显示器运行 | 使用 EGL 离屏渲染，先检查两路相机输出；GLX 仅作为有显示服务时的替代方案 |
| 容器运行 | 确认 NVIDIA Container Toolkit 与驱动可用；为服务和渲染明确 GPU 分配 |
| 权重缓存 | 默认 `~/.cache/openpi`；通过 `OPENPI_DATA_HOME` 指向服务器实际缓存目录，保留权重和归一化 assets |
| 数据准备 | 首轮推理只需 checkpoint、LIBERO 任务与环境资源，不以下载完整训练数据集为前置条件 |

官方 compose 默认使用 host network、8000 端口和各服务单 GPU 申请；它没有自动实现八卡评测分片。后续 VeriVLA 使用 PyTorch 时可以通过独立进程接入，本阶段无需先把 π₀.₅ 转换为 PyTorch。

以下命令是待执行模板，工作目录为服务器 `openpi/` 根目录，缓存路径需替换为实际路径。本项目已包含 ALOHA 与 LIBERO 源码，无需额外初始化子模块。

```bash
export OPENPI_DATA_HOME="/path/to/openpi_assets"

export SERVER_ARGS="--port 8000 policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero"
export CLIENT_ARGS="--args.host 127.0.0.1 --args.port 8000 --args.task-suite-name libero_10 --args.num-trials-per-task 1 --args.replan-steps 5 --args.seed 7 --args.video-out-path data/libero/pi05_libero_smoke/videos"

MUJOCO_GL=egl docker compose -f examples/libero/compose.yml up --build --abort-on-container-exit
```

该命令每个任务运行一次，总计 10 个 episodes。真正的单任务/单 episode 调试需要后续增加筛选入口，不能把 `num-trials-per-task=1` 解释成整套只运行一次。正式运行改为每任务 50 次，并在 B3 前完成下面的记录改动。

若采用已安装好的原生双环境，模型侧的等价入口为：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

在另一个模型环境终端，可使用 `uv run examples/simple_client/main.py --env LIBERO --host 127.0.0.1 --port 8000` 检查通信。随机输入推理不代表闭环评测通过。正式评测前重新启动服务，避免调试请求改变正式采样序列。完整环境命令以[官方 LIBERO 安装说明](../openpi/examples/libero/README.md)为准。

**输入、输出与控制约定。** 保持以下已核对的默认行为，先建立可比较的结果。

| 项目 | 约定 |
| --- | --- |
| 外部图像 | `agentview_image` → `observation/image` |
| 腕部图像 | `robot0_eye_in_hand_image` → `observation/wrist_image` |
| 图像预处理 | 渲染 256×256；两个视角均旋转 180°，保持比例补边缩放到 224×224，转为 uint8 |
| `observation/state` | 8 维：末端位置 3＋四元数转轴角 3＋两指 gripper qpos 2；保留官方字段顺序 |
| `prompt` | 当前任务的原始语言描述 |
| 模型图像适配 | 外部与腕部视角映射到官方输入键；缺失的右腕图像补零并 mask，不自行增加视角 |
| 模型内部动作 | 32 维；当前 action horizon 为 10 |
| 返回动作 | `(10,7)`，适配器截取前 7 个动作维度；六维末端位姿增量控制＋夹爪指令 |
| 动作归一化 | 使用 checkpoint 的 quantile norm stats，输出反归一化后遵循 LIBERO 控制约定 |
| 额外 delta 变换 | `pi05_libero` 的 `extra_delta_transform=False`；不重复减去 state |
| 实际执行 | 每次只执行前 5 步，随后重新观测与推理；后 5 步不执行 |
| Flow matching 采样 | 默认 10 个积分/去噪步；与 10 步 action horizon 是两个不同参数 |

8 维 state 是接口字段，但当前 `pi05_libero` 配置为 `discrete_state_input=False`，且 π₀.₅ 分支不创建 π₀ 式连续 state token。保留这一官方配置，不能把它描述为“默认模型已使用 8 维连续状态条件”，也不为补状态输入擅自修改配置。后续 verifier 可独立使用记录的机器人状态。

动作数值直接交给 `env.step()`；LIBERO 源码已检出，控制器物理单位、坐标轴和缩放仍待结合固定 LIBERO/robosuite 版本核对，不能将原动作数值直接标成米或弧度。

依据：[模型配置](../openpi/src/openpi/training/config.py)、[LIBERO 适配器](../openpi/src/openpi/policies/libero_policy.py)、[评测循环](../openpi/examples/libero/main.py)、[π₀/π₀.₅ 模型](../openpi/src/openpi/models/pi0.py)。

**固定评测协议。**

| 项目 | 首轮参考设置 |
| --- | --- |
| 初始状态 | 使用官方 `get_task_init_states(task_id)`；episode j 对应 `initial_states[j]` |
| 环境 seed | 沿用 runner 默认 7，并写入运行配置 |
| 初始等待 | 10 个仿真步，执行 `[0,0,0,0,0,0,-1]` 等待物体稳定；与策略步数分开记录 |
| 候选 / 执行 | 单候选；生成 10 步、执行 5 步再规划 |
| 任务成功 | 沿用官方 runner 的任务终止/成功判定，并在 B2 对照固定版本环境逻辑核对 |
| 步数预算 | Spatial 220、Object 280、Goal 300、Long 520 个策略控制步；额外 10 步等待另计 |
| 正式规模 | 每套 10 任务 × 每任务 50 次 = 500 episodes；四套共 2000 |

环境 seed 不等于模型采样 seed。当前 JAX policy 默认从 key(0) 开始，每次请求继续 split；客户端 `--args.seed` 不控制这个序列，`client.reset()` 也不会重置服务端 RNG。首轮参考固定单客户端调用顺序、服务重启位置与 episode 顺序，并记录服务请求序号。共享服务的多客户端并发会改变采样分配，不能默认与串行逐动作等价。

后续与 verifier 做严格配对比较前，再实现显式模型噪声/seed 控制。重复评测若使用不同采样种子，记录为“评测种子”，不写成未发生的训练种子复现。

**参考结果与本项目结果分开记录。**

| Suite | 官方 π₀.₅ @ 30k 参考成功率 | 本项目结果 | 计划正式 episodes |
| --- | --- | --- | --- |
| `libero_spatial` | 98.8% | 待测 | 500 |
| `libero_object` | 98.2% | 待测 | 500 |
| `libero_goal` | 98.0% | 待测 | 500 |
| `libero_10` / LIBERO-Long | 92.4% | 待测 | 500 |
| 四套平均 | 96.85% | 待测 | 合计 2000 |

来源：[官方 LIBERO 结果表](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)。[VeriVLA 表 3](../VeriVLA/paper/VeriVLA.md)列出的 π₀.₅ 四套 baseline 数值相同，平均显示为 96.9%。数值相同不证明其所有运行细节一致，后续接入仍需核对执行长度、初始化和采样协议。

只跑 Long 时与 92.4% 对照，不能与四套平均比较。正式报告包含各任务成功数/有效回合数、suite 成功率、四套平均和置信区间。官方值用于定位复现差距，不作为必须精确命中的验收阈值；差距明显时按版本、checkpoint、归一化、图像、动作、初始化、步数与采样顺序排查，不靠挑选种子刷到参考值。

**B3 前需要补齐的最小记录功能。** 当前官方 runner 主要输出控制台统计和视频，存在两个直接影响记录的问题：同任务同结果的视频会覆盖，异常被捕获后仍进入 episode 统计。实施时只补齐结果可追溯所需的记录与汇总，保持动作决策逻辑不变。

| 内容 | 记录要求 |
| --- | --- |
| 运行配置 | 上游 SHA、实际代码差异、LIBERO 版本、依赖/镜像、GPU/驱动、checkpoint 地址及本地权重标识、norm stats、suite、采样与执行参数 |
| 逐 episode 结果 | run/suite/task/init-state/episode ID、任务文本、环境 seed、服务实例及请求区间、成功状态、控制步数、耗时、异常信息 |
| 决策记录 | 实际观测引用、机器人状态、完整 `(10,7)` 动作块、实际执行前缀、请求序号及重规划位置；图像记录先覆盖 smoke、失败和代表性轨迹 |
| 视频 | 文件名包含 suite、task、init-state、episode 和结果，避免覆盖 |
| 结果汇总 | 从逐 episode 数据计算，不手工抄日志；保留预定、有效、成功、失败、超时和基础设施异常计数 |
| 延迟 | 分开记录首次加载/编译、热身与稳定推理；以完成计算后的客户端往返或同步计时为准，另列仿真执行与渲染耗时 |

正式评测中，通信、渲染等基础设施异常单列，修复后按相同 episode ID 定向补跑；任务失败和超时保留原结果。异常不能默默从分母中删除，也不能通过反复重跑任务失败来替换结果。`policy_timing.infer_ms` 的计时点早于结果转 CPU，不能单独当作完整推理延迟。

拟采用以下产物结构，文件由后续实施阶段生成，当前仅定义交付约定：

```text
results/pi05_libero/<run_id>/
  run_config.json
  episodes.jsonl
  summary.json
  decisions/
  videos/
  logs/
```

保存权重和已完成结果，重跑仅针对未完成或基础设施异常的部分。完整同状态分支快照与 1500 条先导续跑属于总计划 P1，不作为本阶段 baseline 的前置条件。

**后续 VeriVLA 接口交接。** 接入点位于 LIBERO runner 获取动作之后、填充执行队列之前。

```text
当前 baseline：观测 → π₀.₅ 单个 chunk → 执行前 5 步 → 新观测
后续 VeriVLA：观测 → 多个候选 chunks → verifier 选择 → 执行固定前缀 → 新观测
```

| 接口事项 | 现状与后续安排 |
| --- | --- |
| 显式动作噪声 | 本地 `Policy.infer(obs, noise=...)` 已支持；当前模型噪声形状是 `(10,32)`，不是环境动作的 `(10,7)` |
| WebSocket 透传 | 当前服务只调用 `infer(obs)`，没有 noise/seed RPC；把 noise 加进观测字典不会自动成为 keyword 参数。后续需要小范围适配 |
| 候选维度 | 对同一观测使用不同噪声生成候选，返回 `K×10×7`；候选轴与时间轴分开 |
| reset 与记录包装器 | 当前 client reset 为空，PolicyRecorder 也没有显式 noise 透传；后续采样接口需一并处理 |
| 本体状态 | 保留当前 8 维原始状态及其定义，供后续 verifier 独立编码；不从模型默认配置推断其状态条件方式 |
| 评分与执行时域 | 基线生成 10 步但只提交 5 步。后续必须明确 verifier 评分覆盖完整 10 步还是实际执行的 5 步 |
| 比较公平性 | 若后续改为执行全部 10 步，必须重跑相同执行协议的纯 π₀.₅ baseline；不能把执行时域变化归因于 verifier |
| 后端 | 可以保留 JAX VLA＋独立 PyTorch verifier；若转换 VLA 后端，单独验证并重新记录 baseline |

这些接口缺口在本阶段记录清楚，实际候选采样与 verifier 接入在下一阶段实施。相关代码：[Policy](../openpi/src/openpi/policies/policy.py)、[WebSocket 服务](../openpi/src/openpi/serving/websocket_policy_server.py)、[客户端](../openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py)。

**本阶段完成条件与待办。**

- [x] 固定本地 openpi 的 `main` 分支与代码提交。
- [x] 完整检出 ALOHA、LIBERO 指定提交，并确认主仓库及两个子模块工作区干净。
- [x] 完成 Windows 用户级精确路径 `safe.directory` 配置，验证 Git 状态检查可用。
- [ ] 在 Linux GPU 服务器准备相同版本工作副本，安装模型与仿真环境。
- [ ] 加载官方 checkpoint 及配套 norm stats，核对服务器 GPU 与 EGL 离屏渲染。
- [ ] 完成真实观测推理和十个任务的 smoke 闭环，核对两路图像及动作约定。
- [ ] 完成逐 episode 记录、唯一视频命名与异常分类，保持策略执行行为一致。
- [ ] 完成 Long 的 500 个有效 episode，保存逐任务结果和失败样例。
- [ ] 完成其余三个 suite，得到可重算的 2000-episode baseline 报告。
- [ ] 保存单候选、10 步预测 / 5 步执行的固定协议及服务端采样记录。
- [ ] 整理 VeriVLA 候选采样、评分时域、状态字段和 seed/noise 适配清单，交接到下一阶段。
