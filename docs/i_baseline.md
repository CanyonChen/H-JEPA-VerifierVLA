# 第一阶段：π₀.₅ baseline 复现计划

更新日期：2026-11-13

所属阶段：[总实验计划](plan.md)中的 P0 / baseline_setup。  
模型环境、CPU/GPU 检查、官方 π₀.₅-LIBERO 检查点与 H100 EGL 均已通过。2026-11-12，服务器在 `~/canyon43079/H-JEPA-VerifierVLA` 完成 B2 的 10/10 成功 smoke，并回传 B3 的 500 个有效 episode，459 次成功、41 次超时，成功率 91.8%，Wilson 95% 区间为 89.06%–93.90%。B3 合并首段 290 回合与恢复段 210 回合，基础设施异常为 0。该次运行使用 3 组并行，运行结束后默认配置改为 2 组。原始产物待拉取复核，B4/B5 按下表继续推进。[B2 汇总](results/pi05_libero_smoke_seed7.json)、[B3 汇总](results/pi05_libero_long_seed7.json)、[B3 记录](libero_baseline.md)

资源：八卡 A100/H100 服务器、MuJoCo；Piper 真机用于后续阶段。

**本阶段目标。** 使用 Physical Intelligence 官方 openpi 的 π₀.₅-LIBERO 检查点，建立长程任务的单候选推理与仿真闭环 baseline。B3 的 LIBERO-Long 已完成；2026-11-13 按用户要求，将 B4 改为 LiLo-VLA Ultra-Long，原计划的 Spatial、Object、Goal 三套评测后置。随后在相同 benchmark 和执行协议下评测 π₀.₅＋原版 VeriVLA，为长期评分与层级表示实验建立对照。

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
| 仿真环境 | LIBERO / MuJoCo；B3 为 `libero_10`，B4 为 LiLo-VLA 的 `ultra_long_all`；两者复用 `.venv-libero`，LiLo 补丁只在 B4 进程内导入 |
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
| B4 / pi05_lilo_ultra_long | 接入 LiLo-VLA Ultra-Long 的全部 9 个配置，先运行 π₀.₅ 单候选 baseline；首轮每配置 10 次，共 90 回合 | 逐配置、逐顺序组及整体的 SR/AP，逐回合技能完成记录与视频 | 9 个配置均有完整记录；SR/AP 可从有序技能完成记录重算；初始化、控制预算和模型协议已固定 | v2 smoke 0/9、AP 1.11%；独立现场诊断 0/3、AP 0，行为核查收束，保留 v2。核对回传计数后启动正式 90 回合。[指令](lilo_baseline.md) |
| B5 / pi05_verifier_handoff | 固定 checkpoint、输入输出、执行长度及采样规则；落实 VeriVLA 权重、候选采样和评分接口，为 Long 与 Ultra-Long 对照交接 | 基线与 verifier 接口交接记录 | 后续能够明确只改变候选生成/选择部分，保留同一基础策略及各 benchmark 的执行协议 | 待执行；资源与接口核查可与 B4 准备同步推进 |

B3 已在单 GPU 完成参考结果。后续 B4 按用户最新要求仅使用 GPU 1，最多两组模型＋仿真进程并行，共用任务队列；一组完成后领取下一配置，总计仍为 9 回合 smoke 或 90 回合正式评测。参数为 `--gpu 1 --egl-device 1 --max-workers 2`。每项任务的模型卡、EGL 卡、端口、episode 范围和产物均有记录，不涉及分布式训练。

**B4：LiLo-VLA Ultra-Long 评测协议与实施范围。**

本阶段使用 LiLo-VLA 发布的 Ultra-Long benchmark，先评测已有 π₀.₅-LIBERO 检查点。B5 完成接口交接并接通原版 VeriVLA 后，再用相同协议运行 π₀.₅＋VeriVLA 对照。B3 的 459/500 仅属于 LIBERO-Long，两个 benchmark 分别报告。

| 项目 | B4 安排 |
| --- | --- |
| 官方来源 | [YY-GX/LiLo-VLA](https://github.com/YY-GX/LiLo-VLA/tree/64aa29d77f5457afb3d1642cfc84534dc1e81fad)，本次核查固定提交 `64aa29d77f5457afb3d1642cfc84534dc1e81fad` |
| LIBERO 依赖 | 官方固定 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`，与本仓库导入版本一致；LiLo 的兼容补丁会改变部分谓词与阈值，需单独记录和验证 |
| suite | `ultra_long_all`：3 个任务场景 × 3 种技能顺序，共 9 个配置；`ultra_long` 单独只包含 3 个原始顺序配置 |
| 任务与长度 | Complete Kitchen Organization：9 个技能；Organize Table：16 个技能；Cooking Preparation Setup：10 个技能。每项均含 Original、Reorder 1、Reorder 2 |
| 首轮规模 | smoke 每配置 1 次，共 9 回合；正式评测沿用官方每配置 10 次，共 90 回合／方法。smoke 与正式结果分开保存 |
| 初始化 | 官方 `make_env()` 使用未显式设 seed 的 `env.reset()`；本项目随后执行 10 步静置再建立技能基准。固定 `.pruned_init` 扫描属于另一协议，不能直接沿用 B3 的 seed 7＋初始状态 0–49 约定 |
| 成功率 SR | 对每回合计算 `skills_completed == total_skills`，再求均值；技能完成数遵循指定顺序 |
| 平均进度 AP | 对每回合计算 `skills_completed / total_skills`，再求均值；同时报告首个未完成技能及失败位置 |
| 汇总 | 分别报告 9 个配置、Original／Reorder 1／Reorder 2 三组及 Ultra-Long 整体的 SR/AP、样本量与不确定性 |
| 基础策略 | 首轮沿用已有 `pi05_libero` 权重、归一化统计和生成 10 步／执行 5 步约定；记录 checkpoint 指纹、任务指令及模型采样序列 |
| 项目适配协议 | `pi05_lilo_ordered_v2`：复用 B3 仿真依赖及 10 步静置；完整技能指令固定输入；逐步有序前缀监测；策略总预算 200×技能数＝1800/3200/2000，静置单列。与论文原始方法的差异见[运行说明](lilo_baseline.md) |

指标以[官方复现说明](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/docs/REPRODUCTION.md)给出的逐回合技能计数公式为准。该文档指出历史 JSON 的 `metrics.*.success_rate` 可能只是某一时刻的 BDDL 终点判定；读取这一字段或单独调用 `env.check_success()` 都不足以复现其 SR。适配器需要持续记录有序技能完成情况，并用官方 [任务与技能配置](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/configs/tasks_and_skills.json)中的谓词计算 SR/AP。

仅接入 benchmark 的环境、任务和评价逻辑；LiLo-VLA 方法本身的运动规划、物体遮罩与技能恢复模块不属于本次 π₀.₅ baseline。评价使用的技能谓词不自动作为策略输入。现有 π₀.₅ 权重在 Ultra-Long 上的表现属于本项目新测结果，不能把论文中的 LiLo-VLA 或 π₀.₅ 数字写成本项目成绩。

实施顺序：

1. 导入固定版本的 benchmark、任务 assets 和兼容补丁，验证 9 个配置及两路相机；隔离 LiLo 补丁，保留 B3 的原始 LIBERO 环境与成功判定。
2. 编写 π₀.₅ 评测适配，冻结任务指令、顺序评分、步数预算和记录格式。每个任务配置使用独立进程，避免上游 `pickedup` 高度缓存跨任务复用。
3. 核查已回传的 9 回合 v2 smoke，定位 8 次乱序终止的触发技能与谓词；确定评分协议后再运行首轮 90 回合，保存运行配置、逐步技能判定、动作、视频、SR/AP 和模型调用耗时。
4. 原版 VeriVLA 接通后，在相同配置、初始化协议、控制预算与执行长度下追加对照，并记录候选数、触发比例及额外延迟。若另做保存状态的配对实验，单列协议和结果。

2026-11-13 已实现 `eval_lilo.py`、九配置渲染预检和 `run_lilo_baseline.sh`。按用户更正，B4 直接复用 B3 `.venv-libero`，删除另建环境的依赖配置；预检入口为 `run_lilo_check.sh`，仅检查现有环境。共用调度器默认 GPU 1、最多两组，后续命令显式使用 `--gpu 1 --egl-device 1 --max-workers 2`；原 `eval_libero.py` 保持 B2/B3 协议。上游导入文件哈希、完整指令长度、三场景本地渲染/九配置谓词绑定，以及 CPU 计分、动作执行、调度与恢复测试均已核验。服务器九回合 smoke 实际为 GPU 1 单 worker，回传 SR 0/9、AP 1.11%、8 次乱序终止与 1 次超时，基础设施异常为 0。[回传汇总](results/pi05_lilo_ultra_long_smoke_v2.json)仅来自用户粘贴的 JSON；随后只读诊断的完整条目确认客户端提交为 `c8acc51`，原始轨迹与视频尚未下载。上游 README 明确以首次乱序或失败前的有序进度计分，不能仅因 LiLo 技能执行器没有项目的全局检测循环就删除乱序终止。当前保持 v2，先核查物体操作与谓词的对应关系。具体验证边界及逐步命令见[运行说明](lilo_baseline.md)。原版 VeriVLA 对照与 P1 的 1500 条机制诊断分开推进。

最新[服务器回传](results/pi05_lilo_smoke_v2_replay.json)补齐九回合摘要：三个厨房配置都触发黑碗 3，三个桌面整理配置都触发奶油奶酪 2；烹饪原顺序完成开炉灶 1 后超时，两个重排均提前触发开炉灶 1。`pickedup` 仅检测 3 cm 抬升，尚不能直接认定抓错对象。task 0、3 的服务器回放在第 5 步即观测不符，没有到达终止现场，`CHECK_REQUIRED` 不改变原始 smoke 指标。停止通过该回放推断原回合接触情况，保留[早期片段](results/pi05_lilo_smoke_v2_trace_excerpt.json)与回放记录。

三回合[现场诊断已回传](results/pi05_lilo_live_diagnostic_v2.json)：0/3、AP 0、2 次乱序终止、1 次超时，无基础设施异常；GPU 1 实际两组并行，约 7 分 20 秒，整卡显存采样峰值约 41.56 GiB。task 0 在第 60 步接触黑碗 3 并使其升高 3.261 cm，未接触要求的黑碗 1；双指垫接触计数为 0，不能表述为已证实的稳定夹持。task 1 在 3200 步中未记录到目标番茄酱接触并超时；task 8 在第 102 步接触并开启炉灶，要求的黑碗未动。当前证据支持保留 v2，接受低分，结束本轮行为诊断；三回合不合并到旧 smoke 或正式评测。

回传的 `per_task` 存在 task 0 的 102 步与逐回合 60 步不一致，且仅粘贴一个任务条目。归档保留该差异，不补造缺项；本地汇总代码未复现任务错位。下一条命令先用 `inspect_lilo_run.py --compact` 核对服务器原文件中的任务集合、回合数、步数和请求数，通过后由 `&&` 直接启动正式 90 回合，不追加诊断回合。正式仍复用 B3 环境、GPU 1 最多两组，全部九配置每配置十回合；模型、指令、预算和 v2 评分保持固定，原版 VeriVLA 尚未接入。

来源：[官方 benchmark 用法](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/README.md#just-want-the-benchmark)、[suite 注册及 make_env](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/lilo_vla/benchmark/__init__.py)、[复现协议与指标](https://github.com/YY-GX/LiLo-VLA/blob/64aa29d77f5457afb3d1642cfc84534dc1e81fad/docs/REPRODUCTION.md)。

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

**B2/B3 的输入、输出与控制约定。** 以下为已验证的 LIBERO-Long 配置；B4 复用 π₀.₅ 输入输出接口，其任务指令、环境和评分协议按前述 B4 安排单独适配。

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

**B2/B3 的固定评测协议。** B4 的初始化、技能顺序和控制预算以其独立协议为准。

| 项目 | 首轮参考设置 |
| --- | --- |
| 初始状态 | 使用官方 `get_task_init_states(task_id)`；episode j 对应 `initial_states[j]` |
| 环境 seed | 沿用 runner 默认 7，并写入运行配置 |
| 初始等待 | 10 个仿真步，执行 `[0,0,0,0,0,0,-1]` 等待物体稳定；与策略步数分开记录 |
| 候选 / 执行 | 单候选；生成 10 步、执行 5 步再规划 |
| 任务成功 | 沿用官方 runner 的任务终止/成功判定，并在 B2 对照固定版本环境逻辑核对 |
| 步数预算 | Long 520 个策略控制步；额外 10 步等待另计 |
| 正式规模 | B3：Long 10 任务 × 每任务 50 次 = 500 episodes |

环境 seed 不等于模型采样 seed。当前 JAX policy 默认从 key(0) 开始，每次请求继续 split；客户端 `--args.seed` 不控制这个序列，`client.reset()` 也不会重置服务端 RNG。首轮参考固定单客户端调用顺序、服务重启位置与 episode 顺序，并记录服务请求序号。共享服务的多客户端并发会改变采样分配，不能默认与串行逐动作等价。

后续与 verifier 做严格配对比较前，再实现显式模型噪声/seed 控制。重复评测若使用不同采样种子，记录为“评测种子”，不写成未发生的训练种子复现。

**参考结果与本项目结果分开记录。**

| Benchmark | 参考口径 | 本项目结果 | 正式 episodes |
| --- | --- | --- | --- |
| `libero_10` / LIBERO-Long | 官方 π₀.₅ @ 30k 成功率 92.4% | 91.8%（459/500，三组并行，含恢复段） | 已完成 500 |
| LiLo-VLA Ultra-Long / `ultra_long_all` | 上游场景与 SR/AP 公式，项目适配协议 `pi05_lilo_ordered_v2` | smoke 为 0/9、AP 1.11%，乱序计分待核查 | smoke 9；正式计划 9 × 10 = 90／方法 |

Long 参考来源：[官方 LIBERO 结果表](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)。Ultra-Long 的来源及协议见本页 B4；Spatial、Object、Goal 与四套平均不作为当前阶段交付要求。

本次 Long 比固定官方参考值低 0.6 个百分点。task 8 成功 26/50，24 次超时占全部 41 次超时的 58.5%。运行采用每任务独立模型服务，恢复时重新创建模拟器并按已有请求数推进 RNG。汇总副本来自用户回传，逐回合原始产物待拉取复核。[逐任务结果与恢复段性能](libero_baseline.md)

Long 只与 Long 参考比较，Ultra-Long 单独报告 SR/AP，两个 benchmark 不混成一个平均数。正式报告包含各配置成功数/有效回合数、对应指标与置信区间。官方值用于定位复现差距，不作为必须精确命中的验收阈值；差距明显时按版本、checkpoint、归一化、图像、动作、初始化、步数与采样顺序排查，不靠挑选种子刷到参考值。

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
- [x] 接入固定版本 LiLo-VLA Ultra-Long 的环境、9 个配置和有序技能 SR/AP 评分，记录 v2 项目控制预算与任务指令；复用 B3 环境，本地验证完成。
- [x] 回传 Ultra-Long 的 9 回合 v2 smoke：0/9、AP 1.11%，GPU 1 单 worker，原始产物待核查。
- [x] 完成三个新回合的现场取证，结束本轮行为诊断，固定 v2 评分。
- [ ] 核对回传按任务计数后，完成首轮 90 回合 π₀.₅ baseline；仅 GPU 1、最多两组并行，保存可重算的技能记录与视频。
- [ ] 保存单候选、10 步预测 / 5 步执行的固定协议及服务端采样记录。
- [ ] 整理 VeriVLA 候选采样、评分时域、状态字段和 seed/noise 适配清单，交接到下一阶段。
