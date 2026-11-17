# Forward-ablation result versions / 前向消融结果版本

The previous figure mistakenly selected an earlier `w/o reformulation` campaign when assembling the plotting data. This was an archive-selection error: in particular, the earlier U runs used horizon/time scale 25, whereas the current campaign uses 20 and the full model's validation-selection protocol. The current figure uses the completed replacement campaign. Both versions are retained with their actual values and configurations.

前一版绘图在整理数据时误选了旧批次的 `w/o reformulation` 结果，属于归档数据筛选错误。其中旧 U 使用时间上限/时间尺度 25，最新批次改为 20，并采用完整模型的验证选模协议。当前图片使用已完成的新批次；前后两版的真实数值与配置均保留。

| Scene / 场景 | Previous / 前一版 (%) | Current / 最新版 (%) | Full PEAS (%) |
|---|---:|---:|---:|
| U | 173.872 | 201.789 | 0.307 |
| S | 147.807 | 147.806 | 2.238 |
| C | 639.095 | 631.834 | 3.904 |

The update does not change the paper's overall ablation conclusion: full PEAS has lower overall error than the forward variant in every scene in both versions. The feature-mapping and steady-state-factorization results are unchanged.

此次修正不影响论文消融实验的整体结论：两版结果中，完整 PEAS 在三个场景的总体误差均低于前向变体。特征映射与稳态分解两项消融结果保持不变。

| Asset / 内容 | Current / 最新版 | Previous / 前一版 |
|---|---|---|
| Predictions / 预测 | `data/predictions/ablations/<scene>/no_reformulation.npz` | `data/predictions/ablations_previous/<scene>/no_reformulation.npz` |
| Configurations / 配置 | `configs/ablations/<scene>_no_reformulation.json` | `configs/ablations/previous/<scene>_no_reformulation.json` |
| PDF | [Current figure](../results/figures/exp3_ablation_relative_l2_positions_and_mean_1x3.pdf) | [Previous figure](../results/figures/archive/exp3_ablation_previous.pdf) |
| Error tables / 误差表 | [Current](../results/ablation_metrics.csv) | [Version comparison](../results/ablation_version_metrics.csv) |

Reproduce either version from the repository root / 在项目根目录重绘任一版本：

```bash
uv run --locked python scripts/reproduce_figures.py
uv run --locked python scripts/reproduce_figures.py --ablation-version previous --output outputs/figures_previous
```

The version flag switches only the forward-ablation archive; full PEAS, the other two ablations and the comparison figures use the same data in both commands.

版本参数只切换前向消融归档；完整 PEAS、另外两项消融及对比实验数据保持一致。
