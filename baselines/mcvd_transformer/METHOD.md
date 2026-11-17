# 正式 MCvD 方法及实现约定

| 符号 | 物理意义与约定 |
|---|---|
| $\mathbf{x}_0,t$ | 无量纲释放位置向量、经过时间；概率函数以时间在前、位置为条件。 |
| $q(t\mid\mathbf{x}_0)$ | 分子在给定时间前被目标接收器 RX 吸收的累计概率。 |
| $\widehat q_\theta(t\mid\mathbf{x}_0),\theta$ | 神经网络的概率近似及其可训练参数，仅用于 MCvD。 |

所有场景均采用无量纲扩散系数 1、RX 半径 1。几何、边界和测试协议见本仓库根目录 configs/ 下的双语配置说明。

方法参照作者公开提交 c580cca0ef956866e49fbcc84d38e331bcf24de0。保持网络、Glorot 初始化、shape／maximum／raw 乘积监督、Keras epsilon 位置对应的 Adam 更新和实际学习率／损失权重调度。2000 epoch 是补充训练预算；未将其标为原论文 400 epoch 设置。来源与许可证情况见根目录 THIRD_PARTY.md。

Canonical 通过 `author_verified_canonical.py` 保留作者编码与训练；eccentric 和 pipe 通过 `author_adapted_scenes.py` 共用数值例程。发布版统一使用项目根目录的 `scripts/train_mcvd_transformer.py` 装配数据、配置、种子和输出路径。

拓扑实体扩展保留原类别 0–4 的语义，类别 5 为 Robin 球形容器、6 为 Robin 圆柱侧壁、7 为吸收端面、8 为反射端面。训练对完整几何做共同刚体旋转，管轴和端面法向同步旋转，位置均相对 Tx；目标 RX 位于第一行，其余实体及空行随机排序。确定性主预测使用原始方向和正常实体顺序，不平均旋转预测。

所有正式模型使用 160 个训练位置，按增强验证集的 raw CIR 最大归一化绝对误差选择最佳权重。无后处理版本是正式比较入口；配对后处理 API 共用同一权重。实际十个种子见 configs/baselines/mcvd_transformer_<scene>_archive.json，不假定与其他方法逐次配对。已归档预测包含正式 20 个测试位置；完整 160/20/20 标签保留在根目录 data/training/<scene>.npz 中。

实际生效的输入、网络、学习率和损失权重调度见 [中文配置说明](../../configs/README.zh-CN.md#mcvd) / [English configuration guide](../../configs/README.md#mcvd)。特别是 canonical TOML 中的旧调度字段不会覆盖作者适配训练器的调度函数。
