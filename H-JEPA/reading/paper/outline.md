# 大纲 · H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning

行号指 `paper.tex`。读某一节：`sed -n '起,止p' paper.tex`（或用 Read 的 offset/limit）。

- L215 摘要
- L252–320 1 Introduction
- L321–767 2 Method
  - L324–533 2.1 Hierarchical predictive learning
  - L534–767 2.2 Hierarchical planning over learned abstractions
- L768–862 3 Learning Hierarchical Abstractions
- L863–1174 4 Planning with Hierarchical Abstractions
  - L893–944 4.1 Planning performance across compute budgets and depths
  - L945–1071 4.2 How hierarchy improves planning
  - L1072–1174 4.3 Scaling to diverse scenes: DROID
- L1175–1198 5 Related Work
- L1199–1217 6 Conclusion
- L1218–1236 Acknowledgments
- L1237–1303 7 Notation
- L1304–1369 8 Environments and datasets
- L1370–1635 9 Architecture and training
  - L1380–1542 9.1 Single-scene environments
  - L1543–1635 9.2 DROID
- L1636–1693 10 Probing and decoding
- L1694–1862 11 Planning evaluation
- L1863–1947 12 Temporal-frequency separation and selective abstraction
  - L1865–1930 12.1 Frequency separation within a trajectory
  - L1931–1947 12.2 Per-entity probes across environments
- L1948–2005 13 Data specialization across levels
- L2006–2071 14 Planning-cost ablations across environments
- L2072–2102 15 Prediction-cost monotonicity
- L2103–2296 16 DROID: corpus, planning evaluation and offline metric
  - L2113–2152 16.1 Planning-evaluation protocol
  - L2153–2226 16.2 The Fr\'echet fidelity offline metric
  - L2227–2263 16.3 Decoded plans
  - L2264–2283 16.4 The re-encoded training corpus
  - L2284–2296 16.5 Future work
- L2297–2427 17 Anti-collapse: SIGReg, inverse dynamics and detachment
  - L2325–2361 17.1 What SIGReg, IDM and detachment each contribute
  - L2362–2389 17.2 Cross-validating the SIGReg and inverse-dynamics coefficients
  - L2390–2427 17.3 SIGReg under data parallelism
- L2428–2653 18 Slow-feature collapse in joint-embedding predictive objectives
  - L2439–2462 18.1 Three kinds of content in a sequence of observations
  - L2463–2487 18.2 Where the latent variance goes
  - L2488–2505 18.3 No regularizer on the marginal law can exclude it
  - L2506–2537 18.4 The objective selects by predictability, and the collapse is an optimum
  - L2538–2598 18.5 Pooling time into the sample axis: a real but small barrier
  - L2599–2629 18.6 Time repulsion: a barrier that needs no action labels
  - L2630–2653 18.7 The inverse-dynamics term is a barrier of a different order
- L2654–2811 19 Extended related work
  - L2657–2686 19.1 World models and planning
  - L2687–2713 19.2 Preventing collapse in non-contrastive SSL
  - L2714–2736 19.3 Hierarchical World Models
  - L2737–2811 19.4 Hierarchical control

## 公式（12 个，按出现顺序；L 后面是所在小节）

- L494 §2.1 Hierarchical predictive learning `eq:level-state`：z^{(\ell)}_t = E^{(\ell)}\!\left(z^{(\ell-1)}_{t\cdot s_\ell\,:\,t\cdot s_\ell+w_\ell}\right), \qquad a^{(\ell…
- L513 §2.1 Hierarchical predictive learning `eq:pred-loss`：\mathcal{L}^{(\ell)}_{\mathrm{pred}} = \frac{1}{c_\ell} \sum_{\tau=1}^{c_\ell} \left\| F^{(\ell)}(z^{(\ell)}_{…
- L527 §2.1 Hierarchical predictive learning `eq:level-obj`：\mathcal{L}^{(\ell)} = \mathcal{L}^{(\ell)}_{\mathrm{pred}} + \lambda_\ell \mathrm{SIGReg}\!\left(Z^{(\ell)}\r…
- L712 §2.2 Hierarchical planning over learned abstr `eq:top-plan`：a^{(L),*}_{0:H_L-1} = \arg\min_{a^{(L)}_{0:H_L-1}} \left\|\hat{z}^{(L)}_{H_L}-g^{(L)}\right\|_2^2.
- L737 §2.2 Hierarchical planning over learned abstr `eq:lift`：\tilde{z}^{(\ell+1)}_i = E^{(\ell+1)}\!\left( \hat{z}^{(\ell)}_{i s_{\ell+1}\,:\,i s_{\ell+1}+w_{\ell+1}}\righ…
- L746 §2.2 Hierarchical planning over learned abstr `eq:lower-plan`：a^{(\ell),*}_{0:H_\ell-1} = \arg\min_{a^{(\ell)}_{0:H_\ell-1}} \left[ \left\|\tilde{z}^{(\ell+1)}_{K_{\ell+1}}…
- L1084 §4.3 Scaling to diverse scenes: DROID `eq:skill`：\text{fidelity} = \frac{d_F(\mathbf{0}, \tilde{p}) - d_F(p, \tilde{p})}{d_F(\mathbf{0}, \tilde{p})}.
- L1104 §4.3 Scaling to diverse scenes: DROID `eq:idm-loss`：\mathcal{L}^{(\ell)}_{\mathrm{idm}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \frac{1}{d_a}\left\| I^{(\ell)}\!\left(z^{…
- L2306 §17 Anti-collapse: SIGReg, inverse dynamics  `eq:sigreg`：\mathrm{SIGReg}\!\left(Z^{(\ell)}\right) = \frac{1}{M}\sum_{m=1}^{M} \mathcal{T}_{\mathrm{EP}}(h_m),
- L2455 §18.1 Three kinds of content in a sequence of  `eq:chi`：\chi(E) = \frac{\mathbb{E}_{b}\!\left[\operatorname{tr}\operatorname{Var}_{t}(z_t \mid b)\right]} {\operatorna…
- L2548 §18.5 Pooling time into the sample axis: a rea `eq:sf-barrier`：\Delta = \lambda\,(T-1)\,\kappa
- L2638 §18.7 The inverse-dynamics term is a barrier o `eq:sf-idm-floor`：\mathcal{L}_{\mathrm{idm}} \;\geq\; \frac{1}{d_a}\,\frac{1}{T-1}\sum_{t=1}^{T-1} \left(\operatorname{tr}\opera…

图表的页码、caption、在哪节被引用：见 `figures/index.md`。
