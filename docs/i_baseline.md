# 第一阶段：π₀.₅ baseline 复现计划

更新日期：2026-11-12

所属阶段：[总实验计划](plan.md)中的 P0 / baseline_setup。  
模型环境、CPU/GPU 检查、官方 π₀.₅-LIBERO 检查点与 H100 EGL 均已通过。2026-11-12，服务器在 `~/canyon43079/H-JEPA-VerifierVLA` 完成 B2 的 10/10 成功 smoke，并回传 B3 的 500 个有效 episode，459 次成功、41 次超时，成功率 91.8%，Wilson 95% 区间为 89.06%–93.90%。B3 合并首段 290 回合与恢复段 210 回合，基础设施异常为 0。该次运行使用 3 组并行，运行结束后默认配置改为 2 组。原始产物待拉取复核，B4/B5 按下表继续推进。[B2 汇总](results/pi05_libero_smoke_seed7.json)、[B3 汇总](results/pi05_libero_long_seed7.json)、[B3 记录](libero_baseline.md)

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

先前的 LIBERO 检出不完整与 ALOHA 子模块指针差异已在导入前解决。服务器统一模型环境、官方检查点的合成与真实观测推理、EGL、十回合 smoke 和 500 回合正式 Long 评测均已回传完成记录。后续评测复用现有权重缓存。

**执行计划表。**

| 步骤 / 标识 | 要做的工作 | 产物 | 通过条件 | 状态 |
| --- | --- | --- | --- | --- |
| B0 / pi05_setup | openpi、ALOHA 与 LIBERO 本地源码已就绪；在服务器准备相同版本工作副本，记录驱动、GPU、依赖、磁盘缓存位置，建立模型与仿真两个环境 | 环境及版本记录 | 服务器源码、任务 assets 和初始状态齐全；GPU、离屏渲染可用 | 模型 CPU/GPU 与 H100 NVIDIA EGL 均通过 |
| B1 / pi05_inference | 加载官方检查点及 norm stats；先验证直接调用的合成观测推理，再通过官方 WebSocket 客户端检查真实 LIBERO 观测的输入输出 | 推理日志、输入输出样例、首次及后续调用耗时 | 返回有限数值的 `(10,7)` 动作块；真实观测预处理与官方一致 | 合成与真实观测推理均通过；真实动作执行 5 步 |
| B2 / pi05_libero_smoke | 运行 `libero_10` 每任务 1 个 episode，共 10 个；检查视频、动作执行和终止逻辑 | 十个任务的闭环记录与视频 | 无未解释的加载、通信或渲染错误；成功与失败均有可追溯记录 | 服务器 10/10 成功；原始视频尚未拉取及逐帧复核 |
| B3 / pi05_long_baseline | 补齐逐 episode 记录和视频唯一命名；按固定初始化运行 Long 每任务 50 次 | 500 个 episode 的记录、逐任务成功率、Long 总结果 | 预定 episode 全部有有效结果；基础设施异常已单列并处理 | 服务器回传完成，459/500 成功、41 次超时，91.8%。原始产物待拉取复核。[记录](libero_baseline.md) |
| B4 / pi05_full_baseline | 用相同模型和执行协议完成 Spatial、Object、Goal，各 500 次；汇总四套 | 共 2000 个 episode 的结果及官方参考比较 | 四套逐任务计数可核对；总指标可从逐 episode 数据重算 | 待执行 |
| B5 / pi05_verifier_handoff | 固定 checkpoint、输入输出、执行长度及采样规则；整理后续候选与 verifier 接口缺口 | 基线交接记录 | 后续能够明确只改变候选生成/选择部分，保留同一基础策略及可比执行协议 | 待执行 |

单 GPU 先建立参考结果。八卡优先用于后续独立评测实例，不需要在本阶段启动分布式训练。分片评测前明确每个实例的 GPU、端口、任务/episode 范围和输出目录，并保证计数不重复；官方 runner 当前没有 task/episode 分片参数。

**环境与启动方式。** 模型侧采用本项目根目录的统一 uv 环境，供 openpi、H-JEPA 模型核心与 VeriVLA 模型共用。LIBERO 客户端使用独立的 `.venv-libero`，通过模型服务通信。EGL、真实观测、B2 与 B3 已在服务器完成。下方官方 Docker 命令保留为部署参考。[环境说明](environment.md)、[B3 运行与结果记录](libero_baseline.md)

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

**官方检查点加载与合成观测推理（已通过）。** 复跑时在项目根目录执行：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
git pull --ff-only origin main
CUDA_VISIBLE_DEVICES=1 bash scripts/run_pi05_checkpoint.sh
```

脚本调用上游 `create_trained_policy`，加载 `pi05_libero` 官方 JAX 检查点及其 `assets/physical-intelligence/libero/norm_stats.json`。2026-11-10 读取的[官方 GCS 文件清单](https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints%2Fpi05_libero%2F&maxResults=1000&fields=nextPageToken%2Citems%28name%2Csize%29)包含 16 个文件，共 12,439,085,481 字节，约 11.6 GiB；首次运行还会按上游流程获取 tokenizer。

缓存沿用 `OPENPI_DATA_HOME`，未设置时使用 `~/.cache/openpi`。可在命令前设置该变量以使用容量足够的缓存盘。已下载完整检查点时，可传入 `--checkpoint /absolute/path/to/pi05_libero`。完整缓存会复用；下载失败保留 `.partial` 目录并抛出原异常，避免将未完成缓存发布为正式目录。本项目为上游 `_download_fsspec` 补上了后台任务的 `future.result()` 异常传递。

输入使用官方 `make_libero_example()`：两张 224×224 RGB 合成图像、8 维合成状态以及提示词 `do something`，NumPy seed 默认为 7。相同观测调用两次，JAX 策略随机数沿用官方默认行为；第一次包含 JIT 编译，第二次记录后续调用耗时。这两个耗时用于检查运行过程，不作为吞吐 benchmark。

每次运行创建独立的 `outputs/pi05_checkpoint/run_*/` 目录：`report.json` 记录成功或失败、代码提交、设备与版本；`observation.npz` 保存输入；`actions.npy` 保存两次输出，预期形状为 `(2, 10, 7)`；`run.log` 保存加载和推理日志。通过条件为报告中的 `status: PASS`、`action_shape: [10, 7]`、`all_actions_finite: true`。

本地已通过 Bash/Python 语法、CLI 帮助及新增下载中断回归检查；回归检查覆盖失败后保留临时文件、成功重试及完整缓存复用，运行无需 NVIDIA 显卡。

**服务器结果：pi05_libero_synthetic_seed7。** 2026-11-10，用户在 `h100-6f4b3` 的 GPU 1 上执行提交 `0b9d0ae9bf668e68a088c3d519af31ca275f9129`，报告为 `PASS`。下列数字来自用户回传的终端 JSON，副本保存在 [pi05_libero_synthetic_seed7.json](results/pi05_libero_synthetic_seed7.json)；远端原始输入、动作与日志文件尚未拉取到本地。

| 项目 | 本次结果 |
| --- | --- |
| 检查点下载 | 2135.931 秒，约 35 分 36 秒 |
| 模型与配套统计加载 | 32.339 秒 |
| 第一次推理，含 JIT 编译 | 35.484 秒 |
| 第二次推理 | 69.883 毫秒 |
| 输出 | 两次均为 `(10, 7)`，数值全部有限 |
| 远端产物 | `outputs/pi05_checkpoint/run_20261110T052519_353136Z/` |

权重已缓存到 `/home/jovyan/.cache/openpi/openpi-assets/checkpoints/pi05_libero`，后续启动继续复用。日志中的 `Norm stats not found ... skipping` 对应项目 `assets/` 目录的可选查找；随后已明确从检查点 `assets/physical-intelligence/libero` 加载统计，与 `create_trained_policy` 的实现一致。首次缺少 `gsutil` 后使用的 gcsfs 下载路径已成功完成。

本次结果确认真实官方权重可以接受合成输入并返回动作。第二次调用只有一个样本，69.883 毫秒不是稳定延迟 benchmark，也不能换算为机器人闭环控制频率；任务成功率仍待真实 LIBERO 闭环评测。

以下命令是待执行模板，工作目录为服务器 `openpi/` 根目录，缓存路径需替换为实际路径。本项目已包含 ALOHA 与 LIBERO 源码，无需额外初始化子模块。

```bash
export OPENPI_DATA_HOME="/path/to/openpi_assets"

export SERVER_ARGS="--port 8000 policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero"
export CLIENT_ARGS="--args.host 127.0.0.1 --args.port 8000 --args.task-suite-name libero_10 --args.num-trials-per-task 1 --args.replan-steps 5 --args.seed 7 --args.video-out-path data/libero/pi05_libero_smoke/videos"

MUJOCO_GL=egl docker compose -f examples/libero/compose.yml up --build --abort-on-container-exit
```

该官方模板每个任务运行一次，总计 10 个 episodes。项目自己的入口已支持 `--task-id` 与 `--init-state-id` 筛选。此次 B3 使用单 GPU 任务并行入口，包含每任务 50 次调度、结果记录和中断恢复。[run_libero_parallel.sh](../scripts/run_libero_parallel.sh)

完成上述检查后，原生模型服务从项目根目录使用统一环境启动：

```bash
cd ~/canyon43079/H-JEPA-VerifierVLA
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_pi05_libero.sh
```

在另一个终端的项目根目录，可使用 `env -u PYTHONPATH .venv/bin/python openpi/examples/simple_client/main.py --env LIBERO --host 127.0.0.1 --port 8000` 检查通信。随机输入推理不代表闭环评测通过。正式评测前重新启动服务，避免调试请求改变正式采样序列。完整环境命令以[官方 LIBERO 安装说明](../openpi/examples/libero/README.md)为准。

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
| `libero_10` / LIBERO-Long | 92.4% | 91.8%（459/500，三组并行，含恢复段） | 500 |
| 四套平均 | 96.85% | 待测 | 合计 2000 |

来源：[官方 LIBERO 结果表](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)。[VeriVLA 表 3](../VeriVLA/paper/VeriVLA.md)列出的 π₀.₅ 四套 baseline 数值相同，平均显示为 96.9%。数值相同不证明其所有运行细节一致，后续接入仍需核对执行长度、初始化和采样协议。

本次 Long 比固定官方参考值低 0.6 个百分点。task 8 成功 26/50，24 次超时占全部 41 次超时的 58.5%。运行采用每任务独立模型服务，恢复时重新创建模拟器并按已有请求数推进 RNG。汇总副本来自用户回传，逐回合原始产物待拉取复核。[逐任务结果与恢复段性能](libero_baseline.md)

只跑 Long 时与 92.4% 对照，不能与四套平均比较。正式报告包含各任务成功数/有效回合数、suite 成功率、四套平均和置信区间。官方值用于定位复现差距，不作为必须精确命中的验收阈值；差距明显时按版本、checkpoint、归一化、图像、动作、初始化、步数与采样顺序排查，不靠挑选种子刷到参考值。

**B3 的记录功能。** 官方 runner 的同任务同结果视频会覆盖，异常被捕获后仍进入 episode 统计。本项目 [eval_libero.py](../scripts/eval_libero.py)已补入每回合独立目录、完整观测/动作、实际执行前缀、汇总和异常分类，B2 在服务器运行通过。B3 进一步加入每任务 50 个初始状态的调度、逐任务统计、Wilson 95% 区间，以及[服务端检查点指纹和请求编号](../scripts/serve_pi05_libero.py)。模型加载仍使用上游实现，动作生成与 10 步预测 / 5 步执行协议保持不变。

| 内容 | 记录要求 |
| --- | --- |
| 运行配置 | 上游 SHA、实际代码差异、LIBERO 版本、依赖/镜像、GPU/驱动、checkpoint 地址及本地权重标识、norm stats、suite、采样与执行参数 |
| 逐 episode 结果 | run/suite/task/init-state/episode ID、任务文本、环境 seed、服务实例及请求区间、成功状态、控制步数、耗时、异常信息 |
| 决策记录 | 实际观测引用、机器人状态、完整 `(10,7)` 动作块、实际执行前缀、请求序号及重规划位置；图像记录先覆盖 smoke、失败和代表性轨迹 |
| 视频 | 文件名包含 suite、task、init-state、episode 和结果，避免覆盖 |
| 结果汇总 | 从逐 episode 数据计算，不手工抄日志；保留预定、有效、成功、失败、超时和基础设施异常计数 |
| 延迟 | 分开记录首次加载/编译、热身与稳定推理；以完成计算后的客户端往返或同步计时为准，另列仿真执行与渲染耗时 |

正式评测中，通信、渲染等基础设施异常单列，修复后按相同 episode ID 定向补跑；任务失败和超时保留原结果。异常不能默默从分母中删除，也不能通过反复重跑任务失败来替换结果。`policy_timing.infer_ms` 的计时点早于结果转 CPU，不能单独当作完整推理延迟。

下面保留正式结果归档的交付约定。B2 与串行 B3 使用 `outputs/pi05_libero/<run_id>/`，此次并行 B3 使用 `outputs/pi05_libero_parallel/<run_id>/`。本地已保存用户回传的汇总 JSON，原始产物待按下列结构整理。[B3 运行说明](libero_baseline.md)

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
- [x] 在 Linux GPU 服务器准备相同版本工作副本，安装统一模型环境并通过依赖与 CPU 模型检查。
- [x] 验证 PyTorch 与 JAX 的 GPU 后端，在 GPU 1 上通过四项模型检查。
- [x] 准备独立的 LIBERO 仿真环境，验证 H100 NVIDIA EGL 离屏渲染。
- [x] 加载官方 checkpoint 及配套 norm stats，在服务器 GPU 上通过合成观测推理检查。
- [x] 完成真实观测推理及 5 步动作执行，核对两路图像及动作约定。
- [x] 完成十个任务各一个完整 episode 的 smoke 闭环：10/10 成功。
- [x] 在服务器完成逐 episode 记录与唯一视频路径；异常分类由 CPU 探针验证，策略执行协议不变。
- [ ] 拉取原始视频与记录，完成视觉复核和完整产物归档。
- [x] 完成 Long 的 500 个有效 episode，归档用户回传的逐任务汇总，459/500 成功、41 次超时。
- [ ] 完成其余三个 suite，得到可重算的 2000-episode baseline 报告。
- [ ] 保存单候选、10 步预测 / 5 步执行的固定协议及服务端采样记录。
- [ ] 整理 VeriVLA 候选采样、评分时域、状态字段和 seed/noise 适配清单，交接到下一阶段。
