# 统一模型环境

本项目使用根目录的一个 uv 虚拟环境 `.venv/`，在同一 Python 进程中运行 openpi、H-JEPA 模型核心与 VeriVLA 已公开模型。目标平台为 Linux x86_64 GPU 服务器。

## 固定版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.13 |
| PyTorch / torchvision | 2.7.1 / 0.22.1 |
| JAX / jaxlib | 0.5.3 / 0.5.3，Linux 安装 CUDA 12 插件 |
| Flax | 0.10.2 |
| Transformers | 4.53.2 |
| NumPy | 1.26.4 |
| openpi 上游基准 | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LeRobot | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| VeriVLA | `f95ab142d6b5b247b47b36137352eb6289dd86a7` |

完整依赖以根目录 `uv.lock` 为准。它从 `openpi/uv.lock` 的已锁定版本生成，并保留 openpi 的 `ml-dtypes==0.4.1`、`tensorstore==0.1.74` 兼容约束。

## 兼容范围与调整

openpi 固定的 LeRobot 要求 `gymnasium==0.29.1`，H-JEPA 原版完整环境要求 `gymnasium==1.2.3`；两个精确版本不能共存。这里将 H-JEPA 模型核心作为本项目的本地包安装，复用 openpi 已包含的 PyTorch、NumPy、einops 与 Transformers。H-JEPA 原版训练入口、`stable_worldmodel` 和 benchmark 仿真套件不在这个模型环境的安装范围内。

根项目以 editable 方式安装 `openpi`、`openpi-client`、`h_jepa` 和 `verifier`。H-JEPA 中三处依赖工作目录的导入改为包导入，并在其上游包配置中补入 `h_jepa` 模块。网络实现与损失函数保持原有计算。VeriVLA 源码与许可证保留在 `VeriVLA/repository/`，来源见该目录的 `UPSTREAM.md`。

VeriVLA 此版本公开模型结构，尚未提供完整训练代码、LIBERO / RoboTwin 集成。环境配置不代表这些功能已实现。LIBERO 客户端后续沿用独立仿真环境，通过模型服务通信；其旧版 Python / PyTorch 依赖不安装到主模型环境中。

## 服务器安装

在已安装 uv 的服务器上执行：

```bash
cd ~/canyon/H-JEPA-VerifierVLA
git pull --ff-only origin main
bash scripts/setup_model_env.sh
source .venv/bin/activate
```

脚本从根目录执行 `uv sync --frozen`，创建 `.venv/`、安装锁定依赖，随后执行 `uv pip check` 和 CPU 模型检查。`--frozen` 直接使用已提交的 `uv.lock`，不在服务器重新解析或改写锁文件；锁文件与项目配置的一致性在本地通过 `uv lock --check` 验证后随代码提交。

安装命令设置 `UV_PYTHON_DOWNLOADS=automatic`，在缺少 `.python-version` 指定的 Python 3.11.13 时自动下载，覆盖服务器的 `never` 下载设置；该设置仅对本次安装命令生效。`GIT_LFS_SKIP_SMUDGE=1` 遵循 openpi 的 LeRobot 安装要求；`env -u PYTHONPATH` 防止已有 ROS / 其他项目路径混入本次安装和检查。

下载时设置 `UV_HTTP_TIMEOUT=600`（HTTP 读取超时，秒）和 `UV_CONCURRENT_DOWNLOADS=4`，为服务器镜像下载较大的 GPU 安装包留出更长等待时间，并限制并发请求数。若安装因下载超时中断，直接重新执行上述脚本，复用现有 `.venv/` 和 uv 缓存。仓库锁文件使用官方 PyPI；服务器日志中出现的 Sangfor 镜像地址需要结合服务器的下载源或转发配置判断。延长超时后若仍失败，下一步检查服务器到该镜像的连通性与响应。

模型环境统一在根目录管理。后续从根目录使用 `uv run --frozen ...` 或激活 `.venv/` 后使用 `python ...`；在 `openpi/` 子目录执行其自己的 `uv sync` 会转为管理另一套项目环境。

## 验证

安装脚本的 CPU 检查运行：

1. openpi 的两个 Gemma expert，使用上游 `dummy` 配置，执行注意力与 AdaRMSNorm 前向。
2. H-JEPA 两个不同时间步长的层级编码、预测与梯度传播。
3. VeriVLA 图像编码、带 KV cache 的潜在 rollout 与两个候选评分。
4. openpi 策略加载模块导入与 `pi05_libero` 配置读取。

使用一张可用 GPU 再检查 CUDA；按服务器当前分配替换 GPU 编号：

```bash
cd ~/canyon/H-JEPA-VerifierVLA
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH .venv/bin/python scripts/check_model_env.py --device cuda
```

检查脚本关闭本进程的 JAX 显存预分配，明确检查 PyTorch CUDA 和 JAX GPU 后端，失败时返回非零退出码。它使用小规模随机权重，不下载 checkpoint 或数据，也不执行仿真评测。完整检查全部通过时，末尾 JSON 中四个检查项均为 `PASS`。

## 验证记录

- Linux x86_64 / Python 3.11 依赖解析：已通过，锁定 203 个包。
- 本地 Windows CPU 模型核心检查：2026-11-09 已通过；Python 3.11.13、PyTorch 2.7.1+cpu、torchvision 0.22.1+cpu、JAX 0.5.3、Flax 0.10.2、Transformers 4.53.2、NumPy 1.26.4。三个模型在同一进程中的小规模计算均通过。
- 根模型包与 H-JEPA 包构建、模型模块和许可证入包检查、安装脚本语法、锁文件离线一致性检查：已通过。
- Linux x86_64 / glibc 2.39 安装预演：2026-11-10 在本地模拟不同下载索引，`uv sync --frozen --dry-run --offline --python-platform x86_64-manylinux_2_39` 通过，计划安装锁定的 203 个包，锁文件哈希保持不变；此预演未实际下载或安装 Linux 包。
- 服务器安装：2026-11-10 用户日志确认 Python 3.11.13 与 `.venv/` 已创建，LeRobot 固定提交已获取；先遇到 Sangfor 镜像的 PyTorch 下载超时，调整下载参数后，重试停在 `--locked` 一致性校验。服务器 uv 0.13.7 与本地版本相同，工作区干净，glibc 为 2.39。本地锁文件检查通过，服务器触发重新解析的具体配置差异尚未确认；安装脚本已改为 `--frozen`，待服务器重试。
- 服务器完整依赖安装、策略模块导入与 CUDA 检查：尚未完成。

本地 CPU 检查使用同版本模型运行库和 PyTorch CPU 构建，使用 `--core-only` 跳过完整策略加载模块导入。该检查不能代替 Linux CUDA 环境验收或真实检查点推理。

参考：[uv 锁文件与同步](https://docs.astral.sh/uv/concepts/projects/sync/)、[uv 下载参数](https://docs.astral.sh/uv/reference/environment/#uv_http_timeout)、[openpi 安装说明](../openpi/README.md#installation)、[LeRobot 固定依赖](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/pyproject.toml)、[H-JEPA 完整依赖](../H-JEPA/repository-main/pyproject.toml)。
