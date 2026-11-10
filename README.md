# H-JEPA-VerifierVLA

面向长程机器人任务的动作验证研究：以 VeriVLA 的候选动作评分为起点，研究长期成功率监督与 H-JEPA 式多时间尺度预测表示对动作选择的作用。

当前仓库汇集研究计划、上游源码、论文资料和中文讲解页面。第一阶段使用官方 π₀.₅-LIBERO 检查点建立单候选推理与仿真评测 baseline，再逐步开展长期评分与层级表示实验。

## 当前状态

截至 2026-11-10，研究计划与本地源码准备已完成，三者共享的 uv 模型环境已在服务器安装，203 个包的依赖检查及四项 CPU 模型检查均通过；GPU 验证、检查点推理、训练与评测待执行。联合验证模型尚待实现。

- `openpi/` 导入自上游提交 `215abfb217dbac7d5f1273282331b9b1866c0479`，作为本仓库的普通源码目录管理。
- `openpi/third_party/` 包含 ALOHA 与 LIBERO 的完整源码，导入版本分别为 `d1dc83afd89ded4379851257fe5d85632d31d5ec`、`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。三者内部的 Git 元数据已移除。
- 论文、上游 README 和讲解页面中的实验数字属于原作者报告；本项目的实测结果将在执行后另行记录。

## 文档入口

| 内容 | 入口 |
| --- | --- |
| 研究目标、假设与 P0–P5 实验路线 | [长程动作验证实验计划](docs/plan.md) |
| π₀.₅ 检查点、环境、启动命令与评测协议 | [第一阶段 baseline 复现计划](docs/i_baseline.md) |
| 统一 uv 模型环境、安装与兼容性验证 | [环境配置](docs/environment.md) |
| H-JEPA 中英双语论文 | [H-JEPA.md](H-JEPA/paper-latex/H-JEPA.md) |
| H-JEPA 代码与运行说明 | [上游代码 README](H-JEPA/repository-main/README.md) |
| H-JEPA 方法讲解幻灯片 | [deck.html](H-JEPA/repo/deck.html) |
| VeriVLA 中英双语论文 | [VeriVLA.md](VeriVLA/paper/VeriVLA.md) |
| VeriVLA 中文交互报告 | [index.html](VeriVLA/repo/index.html) |
| VeriVLA 验证器交互实验室 | [verifier-lab.html](VeriVLA/repo/verifier-lab.html) |

HTML 页面需克隆到本地后用浏览器打开，并保留配套素材的相对路径。

## 目录结构

```text
.
├── docs/                       研究与 baseline 实验计划
├── H-JEPA/
│   ├── paper-latex/             论文 LaTeX、双语 Markdown 与图表
│   ├── reading/                 论文阅读、代码核对与图表素材
│   ├── repo/                    本地 HTML 方法讲解幻灯片
│   └── repository-main/         H-JEPA 上游代码快照
├── VeriVLA/
│   ├── paper/                   论文源码、双语 Markdown 与图表
│   ├── repo/                    中文交互报告与验证器演示
│   └── repository/              固定版本 VeriVLA 模型源码
└── openpi/                      官方基础策略源码快照，含 ALOHA / LIBERO
```

## 克隆与更新

首次克隆即可取得全部已纳入版本管理的源码与资料：

```bash
git clone https://github.com/CanyonChen/H-JEPA-VerifierVLA.git
cd H-JEPA-VerifierVLA
```

在工作区干净时同步更新：

```bash
git pull --ff-only
```

openpi、ALOHA 与 LIBERO 随本仓库一起更新，无需单独初始化子模块。环境配置与运行命令见 [baseline 计划](docs/i_baseline.md)；计划中的命令面向 Linux GPU 服务器，需按实际环境设置缓存路径。

## 配置模型环境

在 Linux 服务器的项目根目录执行：

```bash
bash scripts/setup_model_env.sh
source .venv/bin/activate
```

统一使用 Python 3.11.13、PyTorch 2.7.1 和根目录 `uv.lock`。该环境包含 openpi、H-JEPA 模型核心与 VeriVLA 模型；兼容范围、验证记录和 GPU 检查命令见 [环境配置](docs/environment.md)。

## 实验路线

1. 跑通 π₀.₅-LIBERO 单候选推理，完成仿真 baseline。
2. 在相同环境状态与固定候选集合下，比较单个动作块的进度排序与后续任务成功率排序。
3. 建立单层长期评分基线，再加入 H-JEPA 启发的多时间尺度预测表示。
4. 在匹配数据、监督和计算预算的条件下完成消融，再推进完整仿真与 Piper 真机验证。

完整协议及阶段通过条件以 [docs/plan.md](docs/plan.md) 为准。

## 来源与许可

- H-JEPA 代码来源：[kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA)，保留其 [MIT 许可证](H-JEPA/repository-main/LICENSE)。
- 基础策略代码来源：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，许可见其 [LICENSE](openpi/LICENSE) 与 [LICENSE_GEMMA.txt](openpi/LICENSE_GEMMA.txt)。
- VeriVLA 资料来源见 [本地阅读报告的来源说明](VeriVLA/repo/README.md#来源与数据口径)。
- 论文原文、图片与第三方前端库保留各自的版权和许可；本项目为独立研究整理。

根目录 `.gitignore` 排除本地环境、缓存、日志、模型权重及实验输出；论文 PDF、图片和离线页面素材纳入版本管理。
