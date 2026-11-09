# H-JEPA 论文事实核对

核对对象：用户提供的 `paper-latex/main_release.tex`；`H-JEPA.md` 用于辅助检索，以 TeX 的正文、公式和宏数值为准。本文只读分析论文，未运行官方模型或训练代码。下文 `tex:L` 均指 `paper-latex/main_release.tex:L`。本目录中未发现 AGENTS.md / CLAUDE.md。

## 1. 基本坐标与一句话结论

- 题目：H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning（tex:501）。
- 作者：Wancong Zhang、Basile Terver、Michael Rabbat、Yann LeCun、Randall Balestriero；前两位共同一作，后两位共同指导（tex:503–512）。
- 机构：NYU、Advanced Machine Intelligence、INRIA Paris、Brown University（tex:509–512）。
- 日期/发表地点：本地 TeX 正文未给可靠日期或接收会议。虽然用了 `iclrfinalcopy`（514），518 明确写 `Preprint: remove the conference publication header`，不能据模板宣布 ICLR 录用。日期与版本应另核 arXiv。
- 官方入口：论文给出 https://h-jepa.com/ 和 https://github.com/kevinghst/H-JEPA（tex:522–524）。源码是否完整、权重是否可下载由仓库审计确认；论文声称发布重编码 DROID 语料、manifest、build script（2630）。
- 一句话：**输入当前观测和目标观测，自底向上编码为多个时间尺度的状态；再自顶向下优化动作，让高层预测变成低层子目标，最后输出可执行的原始动作。** 训练从动作标注的轨迹学习动力学；推理按目标图像做规划，不是语言条件策略，也不是 reward-based policy training（tex:803–863、1047–1111）。

## 2. 动机：它改变了两个相互独立的部分

**长时域的时间分解。** 单层模型要连续预测很多小步，误差累积且动作搜索维度大。高层一步跨更久，先给出短期子目标，下层只细化其前缀（579–581、1397–1404）。

**目标距离的表征空间。** 同一 latent 若保留腿姿等快变细节，可以帮助短期动力学，却可能让“已经到达位置、姿势不同”的状态看起来离目标很远。每个高层拥有独立 latent，让规划距离更集中在较慢、较可预测的内容（582–608、1342–1377）。

作者的核心押注不是“深层自动理解语义”，而是：**当数据中不同因素的变化速度有明显分离时，预测更远的训练压力会使高层舍掉快变、难预测的信息，保留慢变信息。** 这有条件：AntMaze、FourRoom 观察明显；Push-T、Cube、DROID 频率比仅约 1.2–1.6，选择性抽象很弱（1181–1207、2288–2296）。

最合适的直觉例子：AntMaze 中目标是沿走廊到达目标位置，腿姿每几步就变化，位置变化较慢。高层规划“下一段到哪”，低层规划“这几步关节施什么力”。腿姿变糊的图来自**后训 decoder 可视化**，decoder 不参与 world-model 训练或规划（1121–1129、1155–1161）。

## 3. 架构与所有主方法公式

### 3.1 每层三个模块

每层有状态编码器 E、动作编码器 A、预测器 F。L1 从 RGB（AntMaze 另加 27D proprio）得到状态；上层从下层 latent 得到自己的 latent（803–815）。

两个时间参数不可混淆：`sℓ` 是相邻高层状态之间相隔几步下层时间；`wℓ` 是构造一个高层状态时吃几个下层状态。**所有实验 wℓ=1**，所以高层状态编码器是逐时刻变换，不是时间平均或多帧聚合；但动作编码器仍需压缩中间的 sℓ 个动作（817–838）。

```text
z⁽ℓ⁾ₜ = E⁽ℓ⁾( z⁽ℓ−1⁾[t·sℓ : t·sℓ+wℓ] )
a⁽ℓ⁾ₜ = A⁽ℓ⁾( a⁽ℓ−1⁾[t·sℓ+wℓ−1 : (t+1)·sℓ+wℓ−1] )
```

原式 tex:826–838，范围左闭右开。w=1、s=2 的数值例子：高层状态取下层 t=0、2、4；第一个高层动作聚合下层 a₀、a₁，第二个聚合 a₂、a₃。状态一次只输入 1 个 latent，而宏动作一次输入 2 个动作 embedding。

### 3.2 一步 teacher forcing 与 SIGReg

原论文式（tex:845–853）：

```text
Lpred⁽ℓ⁾ = (1/cℓ) Στ ‖F⁽ℓ⁾(z⁽ℓ⁾<τ, a⁽ℓ⁾<τ) − z⁽ℓ⁾τ+1‖²
L⁽ℓ⁾ = Lpred⁽ℓ⁾ + λℓ SIGReg(Z⁽ℓ⁾)
Ltotal = Σℓ L⁽ℓ⁾
```

符号：cℓ 是 causal context 长度；F 预测该时间尺度的下一状态；目标由真实未来观测编码产生；Z 是用于正则的 embedding 集合。**注意排版下标：论文写 `<τ` 却把 target 写 `τ+1`，存在一位索引歧义，讲解时采用语义“截至当前状态 + 当前动作 → 下一状态”，以代码的 `[:-1]` / `[1:]` 对齐为准，不应凭这个式子发明两步预测。**

示意数值：预测向量 (0.8, 0.4)，真实未来编码 (1.0, 0.1)，平方欧氏误差 = 0.04+0.09 = 0.13。若实现对坐标也取平均则为 0.065，应明确论文范数与实现 reduction 的区分。

SIGReg（tex:2651–2667）：

```text
hₘ = Z uₘ，uₘ 是随机单位方向
SIGReg(Z) = (1/M) Σₘ T_EP(hₘ)
```

它把高维 latent 投影到 M=1024 个一维方向，用 Epps–Pulley 统计量比较投影分布与标准高斯。不是“把每个样本变成随机噪声”，而是限制整个样本集合的分布，避免所有样本编码为同一点。实现积分区间 [0,3]，17 个梯形积分点，样本数复杂度 O(n)（2661–2667）。

**SIGReg grouping 证据边界：** 主文 857 把 batch/time 展平描述为 n×D；但 DROID 表 1957、1965 与附录 2668、2910 明确 `(T,N,D)`，每个时间位置分别在 N 个序列间检测。应按具体配置说明，不能笼统宣称论文所有训练都把时间展平。

### 3.3 顶层规划

```text
a*⁽ᴸ⁾[0:Hᴸ] = argmin ‖ẑ⁽ᴸ⁾Hᴸ − g⁽ᴸ⁾‖²
(ẑ₁,…,ẑH) = Roll(z₀, a₀,…,aH−1)
```

tex:1056–1070。模型参数固定，对候选动作序列做梯度下降。顶层输出的 latent 轨迹不是环境实际观察，是 world model 的预测；它的第一个预测状态可以成为下层子目标。

### 3.4 下层如何比较高层子目标

```text
Hℓ = Kℓ+1·sℓ+1 + wℓ+1 − 1
z̃⁽ℓ+1⁾ᵢ = E⁽ℓ+1⁾(ẑ⁽ℓ⁾[i·sℓ+1 : i·sℓ+1+wℓ+1])
Jlower = ‖z̃K − ẑ*upper,K‖² + β Σi=1…K−1 ‖z̃ᵢ − ẑ*upper,i‖²
a*lower = argmin Jlower
```

tex:1072–1106。**没有高层状态 decoder 直接还原低层目标；是把低层预测再次通过上层 encoder，再在上层空间比较。** 高层动作也不是直接 inverse-decoded 成原始动作；低层重新优化动作。

闭环 `K=1` 只追踪第一个高层子目标；开环 `K=Hupper`；β=0 仅终点，β>0 还跟踪中间点（1102–1106）。示意：s=2、w=1、K=1，则低层只需预测 2 步来追上 1 个高层子目标；K=3 则低层 H=6。若三个误差为 0.04、0.09、0.01，β=0.5，则 J=0.01+0.5×(0.04+0.09)=0.075。

### 3.5 DROID 的 IDM 与额外正则

```text
Lidm⁽ℓ⁾ = meanₜ [ ‖I⁽ℓ⁾(zₜ,zₜ+1) − sg[a⁽ℓ⁾ₜ]‖² / da ]
Ltotal = Σℓ (Lpred⁽ℓ⁾ + λℓ SIGReg(Z⁽ℓ⁾) + γℓ Lidm⁽ℓ⁾)
         + 0.005 · SIGReg(A⁽²⁾)
```

IDM 主文 tex:1440–1462；action SIGReg 附录 1896–1898、1968。IDM 要从相邻状态恢复连接它们的动作；若 latent 只记录背景，无法恢复变化的动作，就产生损失。**上层 IDM 预测学习得到的宏动作 embedding，不是原始动作，监督目标 sg[a] 切断梯度以防动作编码器迎合 IDM。** 这与预测未来状态 target 是否 stop-gradient 是两件事。

主结果无 latent-target stop-gradient（1950），无 EMA teacher（1855–1856）；不要套用 V-JEPA 的冻结目标编码器图。预测目标 detach 是附录消融。正文 1459 没把 γ 显式写入式子，但表 1959/1967 有实际权重，应在总 loss 解释里补清。

## 4. 训练配方：主结果不是先训底层再训高层

仿真主结果：所有层从零联合训练，上层损失穿过下层编码器；不用预训练权重，不用 EMA，不用 stop-gradient（1733–1737、1853–1863）。DROID 主结果同样 end-to-end；stagewise 先训 L1、冻结后训 L2 只用于数据分层分析、L2 系数搜索和 decoded plans（1909–1917）。

| 项目 | 仿真主设置 | DROID 主设置 |
|---|---|---|
| 图像编码 | ViT-Tiny + CLS projector | ViT-S/16，CLS |
| 时间布局 | L1 每步 5 个环境步；上层 s=2，w=1 | 5 fps；上层 s=3，w=1 |
| sample | 顶层 4 states；每个低层随机 crop 4 states | L1 8 states；L2 取第 0/3/6 个共 3 states |
| teacher forcing | 每层 3 个 targets | L1 7；L2 2 |
| batch | 128 clips | 256 clips |
| 峰值 LR | L1 5e−5；上层 1e−5 | 各层 5e−4 |
| optimizer | AdamW，WD 1e−3 | AdamW，WD 1e−4 |
| warm-up | 1% | 10% |
| clip grad / precision | 1.0 / bfloat16 | 每层 1.0 / bfloat16 |
| epochs | FourRoom 7、Ant 5、Push-T 9、Cube 8 | 100，每 epoch 292 iter |
| 随机种子 | 42/43/44 | 1/1000/10000 |
| 模型训练硬件 | H200，未给统一 GPU 数量或墙钟时间 | 同附录硬件说明，不能据此估时 |

出处：tex:1739–1755、1839–1891、1896–1981；仿真具体值宏 349–397。

仿真 L1 latent：FourRoom/Push-T/Cube 192D；Ant 384D = RGB 投影 256D + proprio 投影 128D。L1 predictor 4 层、8 heads、width 192、FF 1024；upper state encoder 两层 MLP hidden 384；upper action pool 一层 transformer，输出 4D macro action（1839–1851）。上层 predictor 细节随环境变化，不能把 L1 参数照搬所有层。

DROID L2：latent MLP hidden 768；动作把 3×7D 命令拼接，经 hidden 64 MLP + final LayerNorm 输出 8D macro；预测器每层 width 384/depth 12，6 heads，每头 64，FF 1536，dropout 0.1；λ₁=0.08、λ₂=0.16、γ₁=100、γ₂=50，动作正则 0.005（1923–1968）。参数总量论文未列，不编造。

## 5. 一条样本与数据规模

| 环境 | 观测 | 原始动作字段 | 训练来源/规模 |
|---|---|---|---|
| FourRoom | RGB 224×224 | 2D velocity | 3750×400 steps=1.5M transitions；L1 主训练用 1.0M；150 val episodes |
| AntMaze | RGB 64×64 + 27D proprio | 8D joint torque，[-1,1] | explore/stitch 各 12500×400；合计写为 10.05M；1250 held-out stitch episodes |
| Push-T | RGB 224×224 | 2D pusher target position | 18685 train episodes，2.34M frames，均长125；21 val |
| Cube | RGB 224×224 | 5D：末端位移xyz/yaw/gripper | scripted 10000×200；9800 train/200 val |
| DROID | RGB 256×256，单个外视角 | 7D：Δxyz/Δroll-pitch-yaw/gripper | 重编码 74530 episodes、220084 views；16 clips 离线评估 |

出处 tex:1657–1717。**Ant “2×5M=10M” 与正文写 “10.05M transitions” 有口径差异，复述使用论文原数并注明不要自行改成一致。** Proprio 不含全局 xy（1684–1686）。DROID 模型不吃 proprio（1898）；语料包含三相机，但模型每窗从左右外相机中选一个，wrist 不用（2637–2642），不是三视角联合输入。

归一化：仿真 ImageNet 图像标准化，无 augmentation（1742–1743）；DROID center crop square → 256×256 → ImageNet norm（1921–1923），动作按每个坐标的数据统计标准化，避免 gripper 比 Cartesian delta 大 1–2 个量级（1711–1714）。

**三层仿真 sample 的时间数值：** L3 需要4 states → L2需要7 → L1需要13；L1原始帧索引 0,5,…,60，含端点最小长度61环境步。L2时间 0,10,…,60；L3时间0,20,40,60。训练各低层只随机截4 states，而不是把13个状态全部作为该层 loss 的context。loader预留 `5*n1=65` 步，最小几何跨度61与有效采样数量限制不是同一口径（1745–1755、1760–1766）。

各深度最低 L1 frames =4/7/13/25，含端点原始最小长度=16/31/61/121（398–405）。Push-T三层有效clips仅单层58%，四层14%，深度越大不保证效果更好；epoch-matched 不代表 optimizer steps 或训练 FLOPs matched（1754–1755、1861–1863）。

## 6. 真实规划接口与两轮闭环

仿真：**每次只编码当前帧和目标帧，每层 planner rollout context=1；训练 context=3 并不意味着部署时输入最近3个实测帧。** 因果训练第一位置已覆盖 context1（2164–2168）。每一层候选动作通过 AdamW 最多90轮优化，有 early stop；上层 macro actions 初值/限制来自最近4000个训练 embedding 的先验，2–98%分位clip；模型不把无界任意 latent 当成可行技能（2082–2099）。

在仿真最初的三层 Ant 调用中：L3 一步=20环境步；L2一步=10；L1一步=5。配置上限 H=(4,2,5)，但闭环 lower active horizon起初为2，以追踪一个高层子目标；因此不要按配置的 L1 H=4 画“每次执行20步”。所有模型执行两个 L1 blocks=10环境步后重新规划（2101–2111）。

```text
t=0 读当前 o0 与目标 og
    编码 E1→E2→E3
    L3 优化宏动作→预测第一个L3子目标（20步尺度）
    L2 预测2步并经E3与子目标比较→首个L2子目标（10步尺度）
    L1 预测2步并经E2与子目标比较→原始10步动作
    执行这10步
t=10 取得新观测o10；原来的未来预测不当实测状态
     从新状态重复整个top-down优化；剩余预测/动作被丢弃
```

此为流程解释；latent具体坐标/动作值若未取公开数据必须标为示意。H的时程会随剩余预算下降；高层H=1时跳过（2105–2111），并非全程固定H。

DROID：5fps、L1 H=36（7.2秒），L2 H=12（每步0.6秒）；open-loop K=12、β=0.5，对整段只规划一次，不重新观测执行（2173–2208、2478–2488）。它是离线计划评分，不是真机发命令。上层 Gaussian 初始化宽度×1.5，各层clip mean±2std；不加仿真 Cube/Push-T 的动作匹配0.2或noise0.01（2094–2099）。

## 7. 关键结果与可以支持的结论

### 7.1 仿真成功率，mean ± SE，三配对随机种子

表中数字来自 TeX 宏 tex:267–294，最终50个任务评估协议 tex:2057–2065、2113–2124。

**重要：以下四层数值是 TeX 表格宏口径，和同目录图6/仓库转录不一致。1–3层一致。请不要把这张表的四层数值直接贴在图6曲线旁而不注明差异；细见第11节。**

| 环境 | LeWM | H-JEPA 2层 | H-JEPA 3层 | H-JEPA 4层 | HWM 3层 |
|---|---:|---:|---:|---:|---:|
| AntMaze | 18.0±3.5 | 39.3±3.7 | 73.3±3.5 | 63.3±7.7 | 38.7±3.5 |
| FourRoom | 40.7±6.8 | 81.3±6.7 | 96.0±1.1 | 80.0±4.0 | 99.3±0.7 |
| Cube | 32.0±6.1 | 47.3±2.7 | 60.0±5.0 | 55.3±4.4 | 53.3±5.2 |
| Push-T | 40.0±2.3 | 45.3±1.3 | 17.3±3.5 | 0.7±0.7 | 12.0±2.0 |

HWM比较是**同实现、同数据、同训练设置，把所有upper encoders换成identity**，因此用于隔离“时间分层”与“多表征空间”两个因素（1262–1270）；不能直接说跑了另一个作者原始实现。Ant上H-JEPA三层接近HWM两倍；FourRoom反而HWM平均更高，不能说独立 latent 每个任务都带来更高成功率。

### 7.2 固定底层预测，只换cost

| Ant模型深度 | native L1 cost | 最深投影cost | 完整hierarchical |
|---|---:|---:|---:|
| 2 | 23.3±3.3 | 31.3±2.7 | 39.3±3.7 |
| 3 | 16.7±1.8 | 22.0±2.3 | 73.3±3.5 |
| 4 | 10.0±0.0 | 26.7±2.9 | 63.3±7.7 |

tex:226–234、1300–1317。每列是一套不同深度模型；列内flat rows只有目标空间不同。结论：高层cost本身能有帮助，但完整收益还有时间分解。不能把表中每列native L1和独立LeWM=18混为同一个模型。

单调性：Ant full-horizon L1 goal cost的负Spearman 0.48，H-JEPA subgoal tracking 0.94；FourRoom0.56→1.00、Push-T0.75→0.89、Cube0.65→0.99（163–182、1405–1416）。这解释子问题优化更顺，但它不是直接成功率。

### 7.3 DROID 只报告离线保真度

```text
fidelity = [dF(零动作路径,专家路径) − dF(计划路径,专家路径)]
           / dF(零动作路径,专家路径)
```

tex:1425–1438。dF为离散Fréchet距离（米），对累计xyz路径；零动作=0%，专家路径=100%，更差可为负。例：零动作距离0.20m，计划距离0.08m，fidelity=(0.20−0.08)/0.20=60%。不是“60%任务成功率”；没有量到碰撞、抓住/掉落等真机执行结果。

主结果16个选择过的pick-and-place clips，照明好、物体可见且足够大；固定帧索引、抓取为中心7.2秒窗口、移出训练集（2466–2495）。仅xyz评分，不含旋转/gripper（2497–2501）。Frozen baselines因源视频60fps容器标签与实际15Hz差异按原生1fps评估，仅15个clip，因此跨方法口径不完全相同（2483–2488）。DROID图条具体数值需从绘图源数据或figure读，不应从正文编造。

论文结论：加IDM令LeWM成为强baseline，加入第二层继续提升fidelity并减少planner compute；真实机器人闭环验证明确留待后续（1512–1519、1554–1556）。

## 8. 附录的理论贡献与边界（适合备份页）

慢特征坍塌：不同episode编码不同背景，但一个episode内所有时刻latent近乎不变。它可能具有高全局rank且满足SIGReg，却丢掉机器人运动（2800–2810、2838–2854）。

```text
χ(E) = episode内latent方差的平均 / 全语料latent总方差
```

χ=0为完全slow-feature collapse；它不同于所有样本都相同的rank collapse。论文量到DROID 75%方差来自episode间，Cube为28%（1515），但只凭方差占比不直接证明不可规划（2833–2836）。

边缘分布正则不能排除慢坍塌：若背景分布无原子，可用背景构造任意目标边缘分布，F(z,a)=z就有零预测loss；假设是总体理想化，有限场景只能近似（2841–2854）。定理给不可约预测误差 E(E)≥ν(E)≥0；零误差当且仅当表示对动作闭合。它**不保证慢特征一定是任务相关特征**，也不在所有零loss解间做选择（2865–2878）。

```text
SIGReg slow-collapse额外屏障 Δ = λ(T−1)κ   （时间并入样本轴时）
Δ = 0                                   （T作为group轴时）
```

tex:2894–2917。本论文主grouping `(T,N,D)` 是后一种；即使前一种的有限batch屏障，在所试实验也未阻止坍塌。

IDM严格坍塌时的loss下界是给定场景身份仍无法解释的动作方差，按动作维数平均（2983–3002）。若每维动作方差1且场景无法预测动作，Lidm≥1，γ=50就至少付50损失。下界针对**完全坍塌**，近坍塌由实验而非该定理直接覆盖；IDM也不惩罚额外保留distractors（3002）。

仅用负权重 temporal similarity `mean||zt+1−zt||²` 鼓励时间变化（2952–2957）可以防identity predictor，但不能稳健选择可控foreground；属于附录尝试，不是headline recipe（2978）。

## 9. 应在HTML明确列出的局限

1. 选择性抽象依赖数据因素的时间尺度分离；更慢不自动等于任务相关（1198–1207、2876–2878）。
2. 深层长clip减少有效样本，Push-T三层/四层严重退化；训练只epoch matched，steps/FLOPs不同（1254–1280、1751–1755、1861–1863）。
3. DROID是16clips离线开放环，fidelity仅衡量xyz专家路径接近程度；真实闭环、抓取成功尚未验证（2173–2175、2466–2501、1554–1556）。
4. 目前依赖动作监督；无动作视频、语言目标、可变时长层级是未来方向（1558–1562、2643–2645）。
5. 当前目标是观测图像；不含语言条件任务接口。DROID单外相机，wrist和多视角联合编码留给未来（2497–2501、2637–2642）。
6. 主文梯度规划公式省略了附录的候选先验、clip、末两状态cost、两种操作任务额外动作匹配等工程细节（2082–2099），图解应注明主干/实现边界。

## 10. 最有讲解价值的原图与重绘建议

| 内容 | 原始素材 | 精确来源 | 讲解要点 |
|---|---|---|---|
| 动机和开场 | planning_illustrations/teaser | 549–561 | 左侧三层预测解码，右侧compute-success；decoder仅可视化 |
| 训练流 | hjepa_training | 789–798 | w=1/s=2；状态取点与动作pool不同；上层梯度穿过下层encoder |
| 规划流 | hierarchical_planning | 1033–1042 | 上层预测做子目标；下层预测向上投影比较 |
| 抽象到底丢了什么 | abstraction/depth_probes | 1121–1130 | probe+decode，两种证据；别把模糊解释为像素损失 |
| 抽象条件 | freq_abstraction_scatter | 1167–1178 | fast/slow频率gap与probe变化相关，不是保证 |
| 效果/深度 | planning/depth_planning_combined | 1215–1225 | compute sweep和final task集不同；Push-T负例 |
| 目标代价 | heatmaps/composite_levels | 1325–1330 | 高层cost沿走廊更有梯度 |
| 子目标作用 | tmono/instance | 1387–1394 | 全程cost不单调，局部subgoal cost更单调 |
| DROID | droid/scene_diversity_cube_droid、cls_ladder、compute/pareto_real_cls_droid_sharedy | 1498–1509 | IDM防背景坍塌；离线指标与nativefps口径 |

建议自绘4类：① 5/10/20步时间轴；② “状态只取一个latent，动作合并两步”的明确对照；③ 训练参数梯度 vs 规划动作梯度两条路径；④ t=0→执行10步→t=10重规划两轮示例。每页一个机制，公式与数值例拆页。

## 11. 核实发现：四层结果在论文图、TeX表格、仓库转录间不一致

核对了 `repository-main/docs/RESULTS.md:11–21`、`repository-main/assets/depth_planning_sr.png`、`paper-latex/figures/planning/depth_planning_combined.png` 底排，以及 TeX 宏267–294。两幅图已亲眼查看。**论文图与仓库转录互相吻合，论文 TeX 表格宏不同。** 图上的读数只用于核实大致位置，精确值取仓库转录或TeX宏。

| 四层项目 | 本地TeX表格宏 | 仓库RESULTS精确转录，且与图6位置吻合 |
|---|---:|---:|
| Ant H-JEPA | 63.3±7.7（tex:272） | 67.3±2.4（RESULTS:18） |
| Ant HWM | 20.0±2.3（273） | 14.0±2.0（18） |
| FourRoom H-JEPA | 80.0±4.0（279） | 88.0±2.3（19） |
| FourRoom HWM | 92.7±2.7（280） | 92.7±4.4（19） |
| Cube H-JEPA | 55.3±4.4（286） | 62.7±2.4（20） |
| Cube HWM | 39.3±6.4（287） | 34.0±3.1（20） |

四层Push-T一致。主要1–3层headline结果一致。TeX宏注释指向 `figures/planning/depth_planning_sr.csv`（本地未附该CSV），并且正文1362–1366、2057–2065、2113–2124均把深度比较与cost ablation的最终评估关联到同一50-task集。仓库RESULTS自称从原README转录的paper-reported values，而非新验证运行（RESULTS:3–5）。**现有材料不能确定差异来自哪个修订、候选操作点或统计更新；不应自行解释为不同评估设置，也不能选一个值宣称另一个是错的。**

讲解处理：主结果用一致的LeWM/二层/三层数值；四层原图照实显示，用简注说明图表值有差异；详细并列表保留在说明/精读笔记。无需为了教学产物自行重新跑训练或评估。

另有DROID可用的仓库转录数值：LeWM+IDM 33.57±1.52、HWM 35.65±2.22、H-JEPA 38.42±0.94（`repository-main/docs/RESULTS.md:58–69`）；这是仓库paper-reported表，不是本次复测。均是16clip、5fps的Fréchet fidelity，每训练seed先平均planner seeds1/2/3，再跨train seeds1/1000/10000报告均值与SE。
