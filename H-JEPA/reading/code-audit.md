# H-JEPA 代码对照审计

核查日期：2026-11-05。范围：用户提供的 `repository-main/` 快照与 `paper-latex/main_release.tex`。本次只读代码，没有运行仓库训练、规划或环境脚本，没有下载权重/数据或测量运行性能。根目录及祖先、仓库内未发现 AGENTS.md / CLAUDE.md。下文路径均相对项目根。

## 1. 最值得先讲清的六点

1. **这是学世界模型，再用模型优化动作的系统。** 训练时学 encoder、action encoder、predictor；规划时把候选动作张量交给 AdamW 优化。不是行为克隆策略一次前向就直接吐最终动作，也不是在线更新世界模型参数。`stable_worldmodel/solver/gd.py:315–319` 中优化器仅持有 `batch_init`；`:337–358` 对预测代价反传。
2. **层级包括不同时间跨度，也包括不同表示空间。** Ant 三层模型每步分别跨 5、10、20 个环境步，每层 latent 都是 384 维。层数加高不等于 latent 维度缩小。上层状态 encoder 是 learned MLP；HWM 对照模型把它换成 identity。
3. **论文主结果配置的上层状态 window_size=1。** 例如二层 `z²_j = E²(z¹_{2j})`；它没有把两帧平均成一个状态。跨两步预测由 stride=2 和相应两段动作压缩配合实现。当前代码支持 window>1，但不能拿通用功能说成论文主配置。`models/jepa.py:144–158,219–242`；`config/train/ant_hjepa_l3.yaml:35–36,76–77`。
4. **自顶向下传的是 latent 子目标。** 下层先 rollout，再经上层 encoder 投影到子目标的空间比距离。上层宏动作没有一个直接反解器把它变成底层动作。Push-T/Cube 还可加跨层动作匹配代价；Ant 默认权重为 0。`hierarchical_solver.py:155–184,405–427,459–485`。
5. **Ant/仿真联合训练不使用 EMA、默认不 detach prediction target。** 每层 MSE + SIGReg 相加，高层损失沿高层 encoder 回传到低层 encoder；不存在“先固定一个预训练视觉模型再学高层”的主训练流程。`hjepa_utils.py:55–80,104–145,169–174`；论文 `main_release.tex:1853–1863`。代码可配 `detach_pred_target` 或多步训练，这是通用选项，论文当前主配置没有启用。
6. **DROID 的数值是离线开环规划的路径保真度。** 不是真机闭环任务成功率。其配置、action 语义、IDM 和数据管线与仿真不同，不能混成一套统一数字。`README.md:274–277,342–355`。

## 2. 入口与模块地图

| 角色 | 文件及位置 | 做什么 |
|---|---|---|
| 训练入口 | `repository-main/h_jepa/main_hjepa.py:141–152` | Hydra 配置；包装 `hjepa_forward` |
| 模型构造 | `repository-main/h_jepa/hjepa_utils.py:182–337` | 每层独立 encoder、predictor、action encoder/embed；装进 HJEPA |
| 层级编码 | `repository-main/h_jepa/models/hjepa.py:239–294` | 长 L1 流逐层编码，上层动作来自下层动作 embedding；最后各层分别随机裁训练片段 |
| 单层编码/预测 | `repository-main/h_jepa/models/jepa.py:307–360` | `encode` 产出 state/action embeddings；`predict` 预测下一 latent |
| 单层因果 predictor | `repository-main/h_jepa/models/predictors/predictors.py:6–49` | 仿真 ARPredictor；动作作为条件，经 ConditionalBlock/adaLN 调制 |
| DROID predictor | `repository-main/h_jepa/models/predictors/predictors.py:130–165` | CausalTransformerPredictor 内建动作 MLP |
| 宏动作编码 | `repository-main/h_jepa/models/encoders/seq_encoder.py:30–125` | 仿真：单层 transformer + CLS pooling，2个下层动作 embedding → 4维宏动作 |
| DROID 宏动作编码 | `repository-main/h_jepa/models/encoders/seq_encoder.py:145–166` | Flatten 3×7 动作块，经 MLP → 8维宏动作 |
| 训练目标 | `repository-main/h_jepa/hjepa_utils.py:55–179` | teacher-forced MSE、SIGReg，以及配置开启的 IDM/action SIGReg |
| SIGReg / IDM | `repository-main/h_jepa/loss.py:13–47,50–79` | 随机投影正态性约束；相邻 state latent 预测动作 |
| 规划入口 | `repository-main/h_jepa/eval.py` | 分发 simulation planning 与 DROID offline eval |
| 分层规划 | `repository-main/h_jepa/hierarchical_solver.py:90–216` | 顶层到 L1，子目标、horizon schedule、跳过 horizon=1 的高层 |
| 候选动作优化 | `repository-main/stable_worldmodel/solver/gd.py:273–436` | 多候选并行梯度优化、early stop、选择最低代价序列 |
| 闭环执行 | `repository-main/stable_worldmodel/policy.py:387–430` | 动作缓存为空才规划；保留前两个 L1 block 并展开逐步执行 |

代码图中的 `action_encoder` 和 `action_embed` 应分开：前者把一段下层动作表示压成可规划的宏动作；后者把当前层动作变成 predictor 需要的条件向量。仿真 L2 可规划的宏动作是 **4维**，送进 predictor 前变 **384维**；训练输出 `action_2` 存的是后者。队列存的是 action_embed 之前的 4维宏动作（`models/jepa.py:344–351`）。

## 3. Ant 三层：一条训练样本到底是什么

使用 `config/train/ant_hjepa_l3.yaml` 与 `config/train/base/ant.yaml`，不是猜测的统一默认值。

| 字段 | 训练 loader 输出（单样本） | 含义与来源 |
|---|---|---|
| `pixels_level1` | `[13,3,64,64]` | t=0,5,…,60 的 RGB；3通道，ImageNet mean/std 标准化 |
| `action_level1` | `[13,40]` | 13 个块，每块 5×8 个动作数值；依次拼接，**不是把一个8维动作重复5次** |
| `proprio_level1` | `[13,27]` | `qpos[2:]` 与 `qvel` 拼接；去掉全局 xy，保留身体/关节状态 |
| `xy`,`qpos`,`qvel` | 各13个时刻 | 为分析/probing 缓存的辅助列，不等于主视觉模型全部都直接吃这些量 |
| 语言指令 | 无 | 规划目标由 goal 图像给出；Ant 还使用 goal proprio |
| 像素预测标签 | 无重建目标 | 未来图像经同一层 encoder 后形成 latent 监督 |

为什么 13 帧：每层训练目标长度 history3 + rollout1 = 4；L3 4个时间点要求 L2 `(4−1)×2+1 = 7` 个时间点；再要求 L1 `(7−1)×2+1 = 13` 个时间点。HDF5 读入跨度为 `13×5=65` 个原始动作步，图像端点到60，末尾仍读够最后一个5步动作块。依据 `stable_worldmodel/data/dataset.py:128–144,176–202`。

加 batch 后，L1 原始流为 `[B,13,3,64,64]`，动作 `[B,13,40]`。上推时 state 流长度13→7→4；每层为了各自 prediction loss 再裁成4个时刻，因此最终 `embed_1/2/3` 均为 `[B,4,384]`，但它们的时间间隔不同。不能把“每层最后都是4个 latent”解释成同一组时刻。

L1 ViT-Tiny 从图像取 CLS（192维），projector→256；proprio MLP→128；拼接→384。L2/L3 都是384→384 latent MLP，stride2、window1。L1 动作40维→action_embed384维；L2 将两个 L1 action embedding `[2,384]` 压成4维，再嵌入384维；L3同样操作下层动作 embedding。具体出处：`base/ant.yaml:87–129`，`ant_hjepa_l3.yaml:33–114`，`hjepa_utils.py:202–269,291–324`。

图像归一化 `float/255 → 减 ImageNet mean → 除 std` 见 `main_hjepa.py:122–138`；非像素列 z-score 见 `data.py:103–120`。未找到本地 HDF5 样本/ckpt，因此具体动作数值或 latent 数字只能标为“示意”；不能称真实采样。

## 4. 训练：公式和梯度边界

```text
z¹_t = E¹(o_t, proprio_t)                 # Ant L1
z²_j = E²(z¹_{2j})                       # 主配置状态 window=1
u²_j = A²(e¹_{2j}, e¹_{2j+1})            # e是下层动作embedding
e²_j = action_embed²(u²_j)
ẑˡ_{t+1} = Fˡ(zˡ_{≤t}, eˡ_{≤t})
L = Σ_l [ MSE(ẑˡ_{1:4}, zˡ_{1:4}) + λ_l SIGReg(zˡ) ]
```

MSE 的准确代码（`hjepa_utils.py:77–80,104–123`）：

```python
teacher_pred = jepa.predict(emb[:, :history_size], act_emb[:, :history_size])
tgt_component = component_emb.detach() if level_cfg.wm.get("detach_pred_target") else component_emb
teacher_forcing_loss = F.mse_loss(teacher_pred, tgt_component[:, 1:history_size+1])
```

history_size=3、rollout_n=1 时一次并行输入 z0,z1,z2，监督三个下一步输出 z1,z2,z3。这里的 rollout_n=1 **不意味着只训练 z2→z3 一个点**。因果 mask 防止用未来状态作弊。训练时是真实编码上下文的 teacher forcing；规划 rollout 时会使用自己预测的 latent 递推。

数值示意：预测 `[0.8,−0.2]`，目标 `[1.0,0.0]`，MSE=(0.04+0.04)/2=0.04。若整批 SIGReg=0.30，Ant λ=0.09，则该层总项=0.04+0.027=0.067（示意，SIGReg本身不是由此两个数就能计算）。SIGReg 核验全 batch latent 在1024个随机方向上的经验特征函数是否接近标准正态，而不是逐样本“把 latent 归一化”或简单 L2 正则。当前实现输入 `[T,B,D]`，每个时刻跨 batch 统计（`loss.py:28–47`）。

Ant 三层配置：全部从头联合训练；L1 lr=5e−5，上层1e−5，AdamW weight_decay1e−3，batch128，5 epochs，bf16，clip1.0。ViT构造是 `ViTModel(config)`，非 `from_pretrained`（`models/encoders/hf_vit.py:32–37`）。没有固定的“先训L1、再冻L1训练L2”的主训练阶段。论文中的 stage-wise 是比较实验。

### DROID 的额外目标

`L_DROID = Σ_l [L_pred + λ_l SIGReg(z_l) + γ_l MSE(IDM(z_t,z_t+1), action_l)] + 上层动作SIGReg`

L1 λ=.08、IDMγ=100；L2 λ=.16、IDMγ=50、action_sigreg_coeff=.005。IDM 的动作 target detach，state latent不detach，借此强迫表示保留动作相关变化；IDM不是规划时的动作解码器。上层 IDM target 是8维宏动作，在 DROID 中 action_embed为identity。`loss.py:73–79`，`hjepa_utils.py:153–165`，`base/droid.yaml:75–112`，`droid_hjepa_l2.yaml:17–66`。图示/主公式若略去 action SIGReg，完整实现说明须补这一项。

## 5. 规划：两个方向和两种梯度

```text
观测 / goal 图像 ──E1──E2──E3──→ 每层当前latent、每层goal latent（一次编码，固定条件）
L3: 4维宏动作候选 ──F3递推──→ L3未来latents ──到最终goal的距离──→ 优化候选动作
                    第一个预测latent ↓ 子目标
L2: 4维宏动作候选 ──F2递推──→ L2未来latents ──E3──→ 对齐L3子目标
                    第一个预测latent ↓ 子目标
L1: 40维动作块候选 ──F1递推──→ L1未来latents ──E2──→ 对齐L2子目标
                    选最低代价候选 ↓
           2块×5步×8维 → 执行10环境步 → 新的真实观测 → 全部重新规划
```

这里**预测latent→上层encoder**仍保留对动作的梯度；当前观测和真实goal的编码不保留梯度（`hierarchical_solver.py:318–343`）。优化器只更新候选动作，世界模型权重不在优化器里（`gd.py:315–319`）。不要把“训练端到端”与“推理时动作优化”画成同一次模型参数更新。

Ant L3配置每个solve最多90次迭代，early stop 默认最少30步、连续5次相对提升不足1%可停；L1/L2各100个候选，L3 400个。上层候选由训练时最近4000个宏动作拟合逐维高斯初始化，每步截到其2%–98%分位；L1标准高斯初始化。见 `config/eval/ant_l3.yaml:36–103` 和 `gd.py:161–196,217–271,331–394`。

注意：Ant YAML L1 `horizon:4` **不是第一轮实际规划4步**。`pre_decay_horizon:2` 令低层在上层尚未降到1前保持2步；顶层初始5步，之后按整个episode预算递减。初轮实际 H3=5、H2=2、H1=2。每次L1块跨5环境步，L3一步跨20环境步。高层horizon=1会跳过，并按配置将真正goal在更高表示空间传给低层。见 `hierarchical_solver.py:125–153,241–278`。

另一个容易忽略的实现细节：`JEPA.rollout(..., history_size=1)` 默认只取最近一个latent做每步递推（`models/jepa.py:376,397–413`）；目前 `LevelCostModel.rollout`直接调用该默认值（`hierarchical_solver.py:398–399`）。**训练context3 ≠ 规划每步始终输入3个过去state**。不能将训练 history_size 原封不动标在在线推理图上。未发现持久 KV cache。

## 6. 两轮 Ant 示例（数值、位置为示意；时间/形状来自配置）

任务：从迷宫起点走到目标图像所示的房间。模型没有语言输入，下面的中文仅是讲解描述。为使读者看见结构，选一个场景示意：起点靠左，目的地在隔墙右侧，需要先走向门口。

**t=0，真实观测进入。** 当前RGB `[1,1,3,64,64]`、proprio `[1,1,27]` 和goal对应项，各编码得到3个 `[1,1,384]` latent。不能把384维坐标当成物理xy；为了讲图可画“接近门口”对应的解码图，但解码图是事后可视化而非planner必需计算。

**顶层先规划。** 400个候选，张量约 `[1,400,5,4]`。每个4维动作在L3代表20环境步的效果；F3递推5次得到5个预测latent与初始latent。优化最终到goal的距离，选择最低cost候选。把第一个预测latent（约20步后的抽象子目标）传给L2。具体latent数值没有离线模型运行，不能填成实测。

**L2细化。** 100个候选 `[1,100,2,4]`，每个L2步跨10环境步。两次预测后经E3对齐上层子目标。选择后将第一个L2预测latent（约10步后的子目标）传给L1。

**L1落实。** 100个候选 `[1,100,2,40]`，每个40维向量是5条8维电机动作拼起来。两次F1预测后经E2比较L2子目标，选一个 `[1,2,40]` 动作序列，再reshape为 `[1,10,8]` 执行。示意的第一条8维动作可写 `[0.10,−0.05,0.20,0.00,−0.10,0.08,0.03,−0.02]`，必须标“归一化动作示意”；不能宣称是论文真实策略输出。

**t=10，第二轮。** 前一轮动作缓存耗尽，取得新的真实图像和proprio；假设机器人已靠近门口，这是场景示意。以新观测重新编码并重做L3→L2→L1，得到下一批10条真实动作。上一轮想象轨迹不是新观测；候选动作和子目标不作为跨轮持久计划继续执行。horizon可能随预算日程减小，但前两轮顶层在默认350步预算下仍为5（具体由 `decreasing_horizon_schedule` 的分段公式决定；这里不将其写成恒定horizon）。

`采样候选 → latent预测 → 距离代价 → 对动作反传 → 顶层子目标 → 下层细化 → 执行10步 → 新真实观测 → 重规划`。

## 7. DROID 数据/接口不能照搬 Ant

- 训练：256×256、ViT-Small patch16，384维state latent；5 fps，L1一步0.2秒；L2 stride3，一步0.6秒。L1训练context7+target1=8帧，L2context2+target1=3帧（其所需7帧小于L1的8帧，因此加载8帧）。两层均不输入proprio encoder，虽然loader读proprio用于构造动作/监督。
- 7维动作是 `[Δx,Δy,Δz,相对旋转xyz欧拉角,Δgripper]`；位置直接差分，旋转由 `R_next R_currentᵀ` 转Euler，**不等于直接减三个Euler角**，gripper也是差分。代码从采样后的位姿重新算动作（`droid_data.py:30–40,142–170`），不是不加区分读原始DROID控制命令。
- 每个窗口在左右外部相机中选一个，不是同时喂两路图像；不用腕部相机。fps处理专门修正容器标60fps、实际录制15Hz的问题（`droid_data.py:114–140`）。
- 离线评估：L2一次计划12个8维宏动作，把12个子目标全部传给L1；L1计划36个7维动作，覆盖7.2秒。不是仿真“只传首个子目标、执行10步再规划”的模式。`config/eval/droid_l2.yaml:55–66`。
- DROID L1/L2候选16/4，最多90迭代，初始方差放大1.5，上下界mean±2std；这是另一套配置。
- README声明训练74,896 episodes（剔除16个评估clips，64个验证episodes），100 epochs、292 steps/epoch、2GPU×128 batch（`README.md:302–319`）。本次未核查远程下载内容及本机运行时间。

## 8. 论文 / 代码 / 材料的边界

| 容易混淆处 | 准确口径 |
|---|---|
| 上层是否必然丢掉干扰 | 这是部分环境的经验结果，不能写成架构保证。代码只有预测与正则，无显式语义筛选标签。 |
| 更高层是否同时输入多帧 | 主配置window1；stride与state window不同。通用实现支持多帧窗口。 |
| 下层训练target是否停止梯度 | 主配置不detach；可选代码开关不等于论文设置。 |
| action block是否重复同一动作 | 实际reshape 5条不同8维命令；`PlanConfig`注释的“repeated”不宜直接引用。 |
| 文档示例是否真实超参 | `ARCHITECTURE.md:19–22` 的PushT示例frameskip25/stride3，与当前主配置表可能不同；讲稿优先精确YAML。 |
| 在线history是否等于训练history | 当前rollout默认history1，训练Ant history3；不宜讲成相同context。 |
| 可复现范围 | README明确支持Fig6底行、Table1/9 level1列和Fig9b；不要承诺所有paper图表一键重现。 |
| 本地有没有模型和数据 | 当前文件列表未见 .h5/.pt/.ckpt，只有下载说明、DROID manifest/norm stats等。 |
| 是否已实测代码 | 没有，只读核验。报告形状是按代码及配置推导；结果数值来自论文而非本次运行。 |

公开材料入口（README提供）：官方代码 `https://github.com/kevinghst/H-JEPA`；数据 `https://huggingface.co/datasets/jepa-world-models/h-jepa`；模型 `https://huggingface.co/jepa-world-models/h-jepa`。四仿真环境×7模型×3seeds=84个checkpoint，README称9.7GB；DROID3模型×3seeds=9个checkpoint，称2.7GB。此处为本地README所述，未在本次审计中下载验证。

### 数据划分与缺失的进一步核查边界

本地仓库没有 DROID train/validation CSV、解包后的数据目录或 FourRoom HDF5，因此不能从本地文件验证训练/验证的episode重叠计数、磁盘实际episode数量、缺失视频数量；这些若在官方 Hugging Face dataset card 中披露，应引用那张数据卡，并与论文/README声明的名义规模区分。不能仅凭文件名含 `val` 或 `probing_eval` 就断言它们是训练集之外的独立样本。

本地可以直接证实的读取机制：`repository-main/h_jepa/droid_data.py:149–168` 中普通训练/验证样本最多重试10次；读取失败后按原索引和尝试次数构造固定种子，改抽另一个episode。frozen评估clip只尝试1次，失败即报错，不会悄悄换clip。因此名义CSV条目数不自动等于实际每epoch被成功读取的独立episode数。
