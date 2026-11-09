# VeriVLA 中文交互解读报告

入口：`index.html`。网页使用原生 HTML、CSS 和 JavaScript，不需要构建、安装依赖或访问 CDN。

## 打开

- 直接用浏览器打开 `index.html`，所有本地交互和图像均可使用。
- 或在本目录运行 `python serve.py`，打开 <http://127.0.0.1:8765>。如果端口被占用，可运行 `python serve.py --port 8766`。
- 页面侧栏的「打印 / 保存 PDF」使用浏览器打印功能。

## 内容

核心直觉、动作采样、双轴触发器、潜在世界模型验证、离线训练、实验对比、延迟分析、贡献与局限、原文引用。

三个交互：

1. **双轴触发演示**：两个百分位与阈值均可调节，严格遵循 `max(τtemp, τcross) > τ0`。输入为教学设定，不代表论文实测。
2. **实验对比图**：切换真实机器人、RoboTwin Clean、Clean2Random 和 LIBERO。柱长统一以 0–100% 为刻度；指标与验证策略在图中标明。
3. **平均延迟计算器**：按论文分项开销推算 `96 + 326r` ms，不表示新的测量结果，也不预测成功率。

图像可点击放大。导航适配桌面和移动设备，交互支持键盘，尊重系统的减少动画设置。

## 文件

```text
repo/
  index.html            报告内容
  styles.css            响应式与打印样式
  app.js                浏览器交互及论文结果数据
  serve.py              可选本地预览服务
  assets/favicon.svg    网页标识
  assets/figures/        使用到的论文原图
  sources/VeriVLA.md     本地双语论文副本
  sources/Figures/      双语原文引用的图片
  sources/Figures_appendix/
```

## 来源与数据口径

- 原论文：[VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models](https://arxiv.org/abs/2610.21985)
- 作者项目页：<https://shirley430316.github.io/projects/verivla/>
- 本地来源：`../paper/VeriVLA.md`、`../paper/main.tex`；对应论文 v1。报告整理日期：2026-10-26。
- 真实任务 50.8% → 65.5% 为按阶段计算的完成得分。
- RoboTwin 表 2 的主结果使用始终开启验证。自适应阈值的计算与成功率权衡见论文图 6。
- 326 ms 为额外采样 80 ms + 验证 246 ms，不含触发机制 14 ms。基础 82 ms、未触发 96 ms、触发 422 ms。
- LIBERO 两种策略分别使用独立验证器权重；每个策略的验证器在其 40 项任务间共享。
- 示意轨迹、阈值默认值和报告判断均与论文实测结论区分。

论文图片及原文版权归原作者所有。本页面是独立中文阅读报告，非作者官方网站。
