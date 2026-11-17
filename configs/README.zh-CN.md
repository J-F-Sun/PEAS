# 实验配置说明

[English](README.md) | **简体中文** · [项目首页](../README.zh-CN.md)

[坐标与 Table I](#coordinates-and-features) · [PEAS 训练](#peas-training) · [Monte Carlo](#monte-carlo) · [MCvD-Transformer](#mcvd) · [Frozen-PINN](#frozen-pinn) · [参考解求解器](#reference-solvers) · [消融与统计](#protocol)

本文说明**归档论文实验和根目录训练入口实际使用的配置**。U/S/C 分别对应 `canonical`/`eccentric`/`pipe`。JSON/TOML 记录当前图中采用的配置；保留的历史设置和运行时代码覆盖项在下文明确说明。

| 文件 | 作用 / 实际读取入口 |
| --- | --- |
| [scenarios/](scenarios/) | 几何、边界、配点时间范围；`eccentric_reference.json` 记录历史参考解网格 |
| [peas/](peas/) | 完整 PEAS 配置，由 `scripts/train.py` 读取；每份检查点也保存自身推理配置 |
| [ablations/](ablations/) | 当前场景 × 消融变体配置，由消融配置构造器读取 |
| [baselines/](baselines/) 中 `monte_carlo_*.toml` | Julia 仿真设置与数据位置清单 |
| [baselines/](baselines/) 中 `mcvd_transformer_*.toml` | MCvD 输入配置；部分字段会被作者适配训练器覆盖 |
| [baselines/frozen_pinn_archive.json](baselines/frozen_pinn_archive.json) | `train_frozen_pinn.py` **实际读取**的配置与种子；`frozen_pinn_<scene>.json` 是相同设置的逐场景视图 |
| [baselines/](baselines/) 中 `*_archive.json` | 实际种子、历史元数据与预测来源 |
| [paper_protocol.yaml](paper_protocol.yaml) | 实验范围、检查点、种子、模型选择与评价规则 |

`${PEAS_ROOT}` 展开为仓库根目录。Monte Carlo 的 manifest 路径相对 TOML 文件解析。MCvD 的旧 `data.ams_run` 路径仅保留为元数据：发布入口显式加载 `data/training/<scene>.npz`，无需同级 AMS-MOD 仓库。修改配置不会改变归档预测或已经保存的检查点。

<a id="coordinates-and-features"></a>
## 坐标、单位与 Table I 的完整输入特征

长度按 RX 半径 $a$、时间按 $a^2/D$ 无量纲化，因此 $a=D=1$。$d=\|\mathbf{x}_0-\mathbf{c}_{\mathrm{RX}}\|$ 是 TX 到 **RX 球心**的距离，$\tau=t/d^2$。所有对数为自然对数。这里采用固定的物理尺度变换，没有用数据集拟合 z-score 或 min–max 归一化器。

论文将 RX 放在原点；S/C 归档数据将容器中心/轴线放在原点：

| 场景 | 代码坐标中的 RX | 代码 → 论文坐标 | 论文坐标中的外边界 |
| --- | --- | --- | --- |
| U | $(0,0,0)$ | 不变 | 无界 |
| S | $(3,0,0)$ | $\mathbf{x}_{\mathrm{paper}}=\mathbf{x}_{\mathrm{code}}-(3,0,0)$ | 外球中心 $(-3,0,0)$，半径 10 |
| C | $(1,0,0)$ | $\mathbf{x}_{\mathrm{paper}}=\mathbf{x}_{\mathrm{code}}-(1,0,0)$ | 圆柱轴线 $x=-1,y=0$，半径 5 |

模型推理应传入**代码坐标**。S 中令 $\mathbf{e}$ 为从外球中心 $\mathbf{o}$ 指向 RX 的单位向量，则 $s=(\mathbf{x}_0-\mathbf{o})\cdot\mathbf{e}$，$\rho^2=\|\mathbf{x}_0-\mathbf{o}-s\mathbf{e}\|^2$，$r_o=\|\mathbf{x}_0-\mathbf{o}\|$。C 中 $(x_c,y_c,z_c)$ 是圆柱中心坐标，与存储坐标相同，对应论文中的 $(x+1,y,z)$。

### Table I：尚未展开数值缩放的特征组成

| 场景 | 几何特征 | 时间特征 | 扩散相关特征 | 维数 |
| --- | --- | --- | --- | ---: |
| U | — | $t,\ \log(1+t)$ | $d^{-1},\ \tau,\ \log(1+\tau)$ | 5 |
| S | $s,\ \rho^2,\ d^2,\ r_o^2,\ b_{\mathrm{RX}},\ b_{\mathrm{out}}$ | $t$ | $d^{-1},\ \tau,\ A(d,t)$ | 10 |
| C | $x_c,\ \log(1+y_c^2),\ \log(1+z_c^2),\ b_{\mathrm{RX}},\ b_{\mathrm{wall}},\ b_{\mathrm{cap}}$ | $t,\ \sqrt t$ | $d^{-1},\ \log(1+\tau),\ A(d,t)$ | 11 |

其中 $b_{\mathrm{RX}}=d-1$，$b_{\mathrm{out}}=10-r_o$，$b_{\mathrm{wall}}=5-\sqrt{x_c^2+y_c^2}$，$b_{\mathrm{cap}}=\min(z_c+70,70-z_c)$。论文到达特征为 $A(d,t)=\operatorname{erfc}((d-1)/\sqrt{4t})$；实际代码为避免 $t=0$ 的数值问题使用

$$
A_\epsilon(d,t)=\operatorname{erfc}\!\left(\frac{d-1}{\sqrt{4(t+\epsilon)}}\right),\qquad \epsilon=10^{-5}.
$$

该特征不额外乘 $1/d$，只提供输入信息，不将其作为受限域的指定响应。

### U：5 维，按实际张量列顺序

取 $T=20$，$\tau_0=0.01$：

$$
\Phi_U=\left[\frac{t}{T},\ \frac1d,\ \tau,\
\frac{\log(1+t)}{\log(1+T)},\
\frac{\log(1+\tau/\tau_0)}{\log(1+T/\tau_0)}\right].
$$

线性的 $\tau$ 列**不除以** $\tau_0$。对数扩散时间列是 Table I 中特征经过尺度处理后的形式，并非直接 `log1p(tau)`。U 不加入 arrival 特征。见 [canonical.json](peas/canonical.json) 和 [canonical/features.py](../src/peas/scenarios/canonical/features.py)。

### S：10 维，按实际张量列顺序

取 $L=R=10$（`input_scale="outer_radius"`），$T=20$：

$$
\Phi_S=\left[\frac tT,\ \frac sL,\ \frac{\rho^2}{L^2},\ \tau,\
\frac{d^2}{L^2},\ \frac{r_o^2}{L^2},\ \frac{d-1}{L},\
\frac{R-r_o}{L},\ \frac1d,\ A_\epsilon(d,t)\right].
$$

$1/d$ 与 $\tau$ 直接保留；横向距离平方不额外加 epsilon（`outer_eps=0`）。见 [eccentric.json](peas/eccentric.json) 和 [sphere_features.py](../src/peas/common/sphere_features.py)。

### C：11 维，按实际张量列顺序

取 $R=5$、$T=20$、**轴向特征尺度 $L_z=10$**（不是半长 70）、径向平滑 $\epsilon_\rho=10^{-3}$，以及 $\tau_0=1$：

$$
\begin{aligned}
\Phi_C=\bigg[&\frac tT,\ \frac{x_c}{R},\
\log\!\left(1+\frac{y_c^2}{R^2}\right),\
\log\!\left(1+\frac{z_c^2}{L_z^2}\right),\
\frac{d-1}{R},\ \frac{R-\sqrt{x_c^2+y_c^2+\epsilon_\rho^2}}{R},\\
&\frac{\min(z_c+70,70-z_c)}{140},\
\frac{\sqrt{t+\epsilon}-\sqrt\epsilon}{\sqrt T},\
\frac1d,\ \log(1+\tau/\tau_0),\ A_\epsilon(d,t)\bigg].
\end{aligned}
$$

端面距离除以**圆柱全长 140**；平滑平方根时间特征在 $t=0$ 时等于零。对数扩散特征不再除以 `log(1+T)`。$y_c,z_c$ 的平方/对数变换编码镜面对称性；由于 RX 偏离轴线，$x_c$ 保留符号。当前未启用漂移到达特征、Gaussian 特征或 geometry envelope。见 [pipe.json](peas/pipe.json) 和 [pipe/features.py](../src/peas/scenarios/pipe/features.py)。

所有特征运算保留在自动微分图中。MLP 隐藏层采用 `tanh`，输出为线性标量，使用 float32；预测不会截断到 $[0,1]$。

<a id="peas-training"></a>
## PEAS 优化、损失尺度与稳态资产

| 设置 | U | S | C |
| --- | --- | --- | --- |
| 隐藏层数 × 每层宽度 | 4 × 32 | 6 × 64 | 8 × 160 |
| Adam 步数 | 9000 | 7000 | 8500 |
| 配点时间上限 / 输入时间尺度 | 20 / 20 | 20 / 20 | **80 / 20** |
| L-BFGS 最大迭代数 | 禁用 | 500 | 300 |
| Adam 批大小 | PDE 8192；IC/RX/far 各 1024 | PDE 8192；IC/RX/outer 各 2048 | PDE 3072；IC 1536；RX/wall 各 1024；caps 512 |
| 检查点选择 | 3500:250:9000 步上的验证平均 Rel-L2 最小值；相同时取更早检查点 | 物理约束训练，不按响应参考解选模型 | 固定物理验证损失选择并恢复最佳 L-BFGS 状态，验证种子 1073 |

下列调度写作**步数 @ 学习率**。各阶段损失权重见 JSON 中 `training.lr_schedule`，它覆盖初始的 `lr` 与 `loss_weights`：

- U：2500 @ 0.005 → 2500 @ 0.002 → 2500 @ 0.0001 → 1500 @ 0.00002；各阶段保留 Adam 状态。
- S：1800 @ 0.0005 → 2200 @ 0.00008 → 3000 @ 0.00002；保留 Adam 状态，梯度范数上限 1。自适应配点从第 800 步启用，每 200 步刷新，从 16384 个候选点选 2048 个，填充 PDE 批次的 25%。
- C：2000 @ 0.001 → 2500 @ 0.0003 → 2500 @ 0.0001 → 1500 @ 0.00003；每阶段重置 Adam，梯度范数上限 1。`without_late_loss=true` 从目标函数移除晚时刻约束，即便 JSON 中仍保留历史 `late` 批大小/权重字段。L-BFGS 后的 Adam 未启用。

两个启用的 L-BFGS 阶段均使用学习率 0.8、`strong_wolfe` 线搜索；固定批大小及权重见 `training.lbfgs`。

**输入归一化与损失归一化是不同操作。** U 的 PDE 残差先以 $y^3+0.05$ 为分母作 `full` 条件化，其中 $y=1/d$。随后相对目标函数将 PDE/IC 的平方误差按距离分成六组，组边界为 `[1,3,6,10,15,25,200.001]`，每组均方除以 $s_k^2$ 后对各组等权平均。尺度为 $s_k=\operatorname{clip}(\sqrt{\mathrm{EMA}(u^2)},0.02,1)$，EMA 衰减系数 0.99，$u=f_\theta$ 为模型学习的瞬态因子，不使用参考响应幅值。倒数平方权重上限为 2500。分组损失在第 2500 至 3500 步逐渐引入；基于梯度的分量平衡每 100 步更新一次，因子范围 `[0.2,5]`。见 [canonical/physics.py](../src/peas/scenarios/canonical/physics.py)。

S 使用除以稳态因子后的 PDE，稳态场下限为 $10^{-4}$，漂移分量截断幅值为 50，残差截断幅值为 $10^3$。C 使用

$$
\mathcal R_C=\frac{\widehat q_\infty(u_t-\Delta u)-2\nabla\widehat q_\infty\cdot\nabla u}{\widehat q_\infty+0.03}.
$$

论文将边界惩罚统一写为 $\lambda_{\mathrm{BCs}}\mathcal L_{\mathrm{BCs}}$；代码中则分别保留 RX/far、RX/outer 或 RX/wall/caps 损失，并配置各自的阶段权重。S/C 还包含范围惩罚。论文的合并记号不意味着所有边界共用同一个数值权重。

| 固定稳态 FEM 场 | 离散及数值设置 |
| --- | --- |
| [canonical.npz](../data/steady/canonical.npz) | 径向 P1，$[1,200]$ 上 2048 单元/2049 节点；网格幂指数 1.6；外边界 Robin 系数 0.005；下限 $10^{-6}$ |
| [eccentric.npz](../data/steady/eccentric.npz) | 轴对称双球坐标 Q1，96 × 128 节点；外边界 Robin 系数 5 |
| [pipe.npz](../data/steady/pipe.npz) | 四面体 P1，23446 节点/120549 四面体；Gmsh 尺度范围 0.12–0.9，距 RX 的过渡距离 0.35–4；体积加权节点梯度，定位候选数 96 |

U 的 `problem.domain.r_far=25` **不是**实际 FEM 截断半径：保留的采样器和远边界使用 `q_infty.outer_radius=200`。稳态场只计算一次，MLP 拟合时保持固定；它与下文的瞬态参考解用途不同。

<a id="monte-carlo"></a>
## Monte Carlo / AMS-MOD

图例中的 “Monte-Carlo” 对应归档 Julia AMS-MOD 估计量。吸收感知多级分裂方法可参阅 [A-MLS 项目](https://github.com/J-F-Sun/A-MLS)。本仓库在 [baselines/monte_carlo/](../baselines/monte_carlo/) 保留论文所用适配实现，不依赖外部仓库的后续更新。

仿真位置覆盖较大的距离范围，远场分子在有限观察时间内命中 RX 的概率很小。传统直接 Monte Carlo 需要大量独立轨迹才能估计这些小概率，对本实验网格而言耗时过长，因此使用 AMS 的分裂/重采样机制集中计算资源。

| 仿真设置 | U | S | C |
| --- | ---: | ---: | ---: |
| 每次 AMS 粒子群的 `num_replicas` | **250** | **250** | **250** |
| `n_worst` 选择参数 | 100 | 1 | 1 |
| 每个位置的独立 `repeats` | **128** | **128** | **128** |
| 内部扩散步长 `dt` | 0.00025 | 0.0002 | 0.0002 |
| 仿真时间上限 | 25 | 20 | 20 |
| 输出时间采样数 | 801 | 801 | 801 |
| 最大分裂迭代数 | 100000 | 100000 | 100000 |
| 主种子 | 40 | 1234 | 20260908 |
| 数据位置种子 | 2026100501 | 2026100503 | 2026100504 |
| 允许的位置距离范围 | 1.5–21.5 | 1.25–10.8 | 1.25–10 |

配置文件：[U](baselines/monte_carlo_canonical.toml)、[S](baselines/monte_carlo_eccentric.toml)、[C](baselines/monte_carlo_pipe.toml)。

**250 个副本与 128 次重复是两个层次。** 每个位置的独立运行合计初始化 $250\times128=32000$ 个副本实例；同一次 AMS 中的副本经分支后具有谱系相关性。这不等价于 32000 条独立传统 MC 轨迹，也不是重采样过程产生的全部轨迹片段数。每场景 200 个位置合计 **25600 次独立 AMS 运行**，三场景共 76800 次；其中每场景 20 个测试位置对应 2560 次。

每个 MCvD 标签和图中 MC 估计量都是 128 条曲线的平均。原始测试运行保存在 `data/predictions/comparison/<scene>/monte_carlo.npz`。U 标签保留 $[0,25]$ 的 801 个采样，论文图只评价前 641 个、即 $[0,20]$ 的采样。Smoke 模式会替换正式参数：16 个副本、4 次重复、时间上限 0.2、`dt=0.002`、41 个输出时间点、最大迭代 2000；它不能用于论文精度评价。

<a id="mcvd"></a>
## MCvD-Transformer

使用根环境中的 [scripts/train_mcvd_transformer.py](../scripts/train_mcvd_transformer.py)。实现基于上游提交 `c580cca0ef956866e49fbcc84d38e331bcf24de0` 的 PyTorch 适配，增加 S/C 几何与边界描述。每场景有独立的[训练数据](../data/training/)、160/20/20 划分与十个归档种子。

| 设置 | 论文实验实际实现 |
| --- | --- |
| 输入 | 15 个拓扑实体 × 13 个字段，以及 7 维数值特征 |
| 坐标编码 | 以 TX 为原点；RX 放第一行；数据增强对完整几何共同刚体旋转；其余实体/填充行随机排序 |
| 数值特征 | 扩散系数、实际输出时间间隔、五个无遮挡视线条目 |
| 几何类别 | 保留原 0–4 类；5 为 Robin 外球，6 为 Robin 圆柱，7 为吸收端面，8 为反射端面（论文 C 使用 7） |
| Transformer | 2 层；16 个注意力头；**每头 13 维**；FF 宽度 32；dropout 0.1 |
| 全连接分支 | 共享 400→800→1600；shape 隐藏宽度 1622；maximum 分支 1600→800→400 |
| 输出 | 801 个曲线采样加 10 个前置零填充；shape × maximum 得到 raw 曲线 |
| 训练 | 2000 epochs；batch 64（160 位置分为 64、64、32）；Adam，betas (0.9,0.999)，Keras epsilon 约定 $10^{-7}$；Glorot 初始化；无梯度裁剪 |
| 模型选择 | 增强验证集 raw CIR 的最大归一化绝对误差 MxNAE 最小 |
| 论文预测 | 原始曲线，不做平滑、单调性修正或旋转预测平均 |

**实际学习率调度：**三个作者适配训练器都使用从零开始的 epoch $e=0,\ldots,1999$：在 $e=0\ldots25$ 从 $10^{-5}$ 线性升至 $10^{-3}$，到 $e=200$ 线性降至 $5\times10^{-5}$，之后保持。U TOML 中旧的 `20/280/0.0003` 调度不由这个入口执行。见 [lr_for_epoch 与 executed_recipe](../baselines/mcvd_transformer/scripts/author_verified_canonical.py) 和 [author_config](../baselines/mcvd_transformer/src/mcvd_transformer/author_scene_adapter.py)；只修改被忽略的 TOML 字段不会改变该实验。

目标函数从 shape 与 maximum 监督逐步转向 raw 曲线监督：$\alpha(\mathcal L_{\mathrm{shape}}+\mathcal L_{\mathrm{max}})+(1-\alpha)\mathcal L_{\mathrm{raw}}$。Shape 损失包含 $10^{-4}$ 的单调惩罚；maximum 损失的归一化分母为 `max(abs(target_max),1)`，raw 曲线项按目标最终值归一化。权重在 epoch 结束时更新：从零计数到第 31 个 epoch 仍为 $\alpha=1$，随后每 epoch 下降 $1/200$，第 231 个 epoch 达到零。验证分数先求每个位置跨时间的最大归一化绝对误差，再对位置平均。U 标签覆盖 $[0,25]$，S/C 覆盖 $[0,20]$；论文指标统一评价 $[0,20]$。

发布入口从 NPZ 元数据读取历史 profile：U 为 `experiment`，S/C 为 `formal`。2000 epoch 是本文适配实验预算，不代表上游论文原始预算。种子见 `mcvd_transformer_<scene>_archive.json`。

<a id="frozen-pinn"></a>
## Frozen-PINN

使用 [scripts/train_frozen_pinn.py](../scripts/train_frozen_pinn.py) 和独立锁定的 [Frozen-PINN 环境](../baselines/frozen_pinn/pyproject.toml)：Python 3.12.12、NumPy 1.26.4、SciPy 1.12.0、scikit-learn 1.4.2、swimnetworks 0.0.2。方法使用上游 boundary-compliant `tanh` 基函数、uniform-pair SWIM 采样、SVD 降维与系数时间演化，不采用 Adam epoch 调度。每个释放位置/种子独立拟合，再积分 RX 通量。

| 设置 | U | S | C |
| --- | ---: | ---: | ---: |
| 空间表示 | 径向 | 三维笛卡尔 | 三维笛卡尔 |
| inner / outer 基函数数 | 1536 / 768 | 4096 / 512 | 4096 / 512 |
| 内部配点数 | 4096 | 12288 | 12288 |
| `surface_points` 设置 | 384（径向实际使用端点） | 4096 | 4096 |
| 源附近采样比例 / 扩散宽度倍数 | 0.25 / 6 | 0.15 / 4 | 0.15 / 4 |
| Gaussian `sigma_max` / 边界裕量因子 | 0.15 / 4 | 0.5 / 1 | 0.5 / 1 |
| 最小二乘 `rcond` | $10^{-10}$ | $10^{-5}$ | $10^{-5}$ |
| SVD 截断阈值 | $10^{-7}$ | 0.01 | 0.01 |
| 求解时间上限 / 输出点数 | 25 / 801 | 20 / 801 | 20 / 801 |

共同设置：边界权重 100；LSODA 相对/绝对容差 $10^{-6}$/$10^{-8}$；三维 RX 表面通量积分使用 2048 点；时间积分使用 4001 个密集采样点并加入请求的输出时刻，采用累计梯形积分；4 个 CPU 线程。U 使用精确径向表面积分，外截断半径 30。S/C 虽保留 `outer_cutoff=60` 字段，实际区域由物理外球/圆柱边界确定。

Dirac 源用在区域内归一化的 Gaussian 近似，$\sigma=\min(\sigma_{\max},\text{到最近边界的距离}/\text{裕量因子})$；U 使用其解析球面平均。空间输入为半径或笛卡尔坐标，不使用 PEAS 特征向量。负值与非单调预测不作修正。逐场景视图见 [U](baselines/frozen_pinn_canonical.json)、[S](baselines/frozen_pinn_eccentric.json)、[C](baselines/frozen_pinn_pipe.json)；入口实际读取 [frozen_pinn_archive.json](baselines/frozen_pinn_archive.json) 中的 `configs` 和 `seeds`。

每场景使用三个与报告测试集重叠的位置选择超参数：U `002/009/020`，S `005/009/016`，C `004/010/017`（加上对应场景的 `_test_` 前缀）。[调参位置记录](../data/validation/frozen_pinn_tuning_points.json) 保留这一选择范围。每场景十种子 × 20 位置，共 200 次独立源拟合。上游 MIT 声明及适配说明见 [LICENSE.upstream](../baselines/frozen_pinn/LICENSE.upstream) 和 [FIDELITY.md](../baselines/frozen_pinn/FIDELITY.md)。

<a id="reference-solvers"></a>
## Reference solution：独立瞬态参考解

[参考解生成入口](../scripts/generate_reference.py) 从 `data/reference/<scene>.npz` 读取测试位置和输出时刻，不使用 MC 曲线或训练后的神经网络。`--dt` 可覆盖积分步长，默认设置对应下列归档求解方案。

| 设置 | U | S | C |
| --- | --- | --- | --- |
| 参考解 | 解析解 | 瞬态四面体 P1 FEM | 瞬态四面体 P1 FEM |
| 公式 / 网格 | $t>0$ 时 $q=d^{-1}\operatorname{erfc}((d-1)/\sqrt{4t})$；内部 $q(0)=0$ | 历史网格 98883 节点 / 585438 四面体 | 随附网格 23446 节点 / 120549 四面体 |
| 质量矩阵 | — | 按行求和集总 | 一致质量矩阵 |
| 时间格式 | — | 隐式 Euler | Crank–Nicolson，先做两个后向 Euler 半步 |
| 内部积分步长 | — | 0.005；到 20 共 4000 步 | 0.01；到 20 共 2000 个完整时间区间 |
| 输出网格 | [0,20] 上 641 个时刻 | [0,20] 上 801 个时刻 | [0,20] 上 801 个时刻 |
| 外边界 | 无穷远衰减 | Robin 系数 5 | 侧壁 Robin 系数 1；端面 $q=0$ |

两个受限参考解都在 RX 上施加 $q=1$，流体内部初始概率为零，使用稀疏线性系统并复用矩阵分解。S 使用 [eccentric/reference.py](../src/peas/scenarios/eccentric/reference.py)：Gmsh `mesh_size=0.4`，一阶单元，RX 附近尺度 0.1、远处 0.4，过渡距离 0.2–1.2。它独立于神经模型的双球坐标稳态 FEM；重新生成的网格可能受 Gmsh 版本/平台影响，原始来源见 [eccentric_reference.json](scenarios/eccentric_reference.json)。

C 使用 [common/references.py](../src/peas/common/references.py)，从 `data/steady/pipe.npz` 仅读取网格、连接关系和边界信息，**不把其中存储的稳态 `q` 当作瞬态参考解**。空间 P1 插值与时间插值将解映射到所需观测位置/时刻。两个 FEM 参考解都是有限分辨率的数值解，不是精确解析真值；启动阶段可能存在的极小负值保留不截断。

可选 `fem` 环境依赖固定 Gmsh 4.15.2，供 S 使用。参考解的 `--smoke` 只保留两个位置、三个输出时刻，并将 S 网格改粗为 `mesh_size=1.5`，仅用于检查求解流程。新结果写入 `outputs/reference/`，不会替换论文归档参考解。

<a id="protocol"></a>
## 消融、模型选择与报告统计

| 变体 | 改动内容 | 每场景完整实验量 |
| --- | --- | --- |
| `no_reformulation` | 每个 TX 单独拟合前向密度，再由 RX 通量积分得到概率 | 20 位置 × 10 种子 = 200 次拟合 |
| `no_steady` | 后向网络直接预测 $q$，不使用固定 $q_\infty$ 分解 | 10 次拟合，每个模型查询全部位置 |
| `no_features` | 以归一化空间/时间替代设计特征，保留稳态分解 | 10 次拟合，每个模型查询全部位置 |

[ablations/](ablations/) 保存当前三项消融的实际配置。U 使用时间上限/时间尺度 20、响应归一化下限 0.02，并在 3500:250:9000 步按验证误差选模；前向变体先聚合每个种子的 20 个独立验证源模型，再确定测试权重。S 使用 7000 步 Adam 加 L-BFGS。C 使用训练上限 80、时间尺度 20、8500 步 Adam、无 late loss，以及固定物理验证损失的 L-BFGS 选模。所有误差均在 $[0,20]$ 计算。权重哈希和环境记录见 [run_provenance.json](ablations/run_provenance.json) 与 [forward_run_provenance.json](ablations/forward_run_provenance.json)，旧版保留方式见[版本说明](../docs/ablation_versions.md)。

神经模型第 $r$ 次运行、第 $i$ 个位置的误差为 $e_{r,i}=100\|\widehat{\mathbf q}_{r,i}-\mathbf q_i\|_2/\|\mathbf q_i\|_2$。位置均值对十次运行平均，场景均值对 20 位置和十次运行等权平均。场景 SD 为先对位置求平均、再跨运行计算 `std(mean(e, axis=positions), ddof=1)`，不是各位置 SD 的平均。MC 先平均 128 条曲线再算误差；`evaluate.py` 用 2048 次 bootstrap、种子 20261114 估计其不确定性。相对误差图不画阴影，概率图画均值的逐时刻 95% 置信区间。

U 的十个 PEAS 种子含用于选择归一化下限的三次筛选（40、18、1414717112）及七次确认运行。S/C 不按瞬态参考标签选择模型。Frozen-PINN 的调参/测试位置重叠情况已在上文说明。应使用各方法自己的归档种子列表，不能假设所有方法种子严格配对。完整 PEAS 在对比图和消融图中使用同一组模型。完整来源与统计约定见 [protocol.md](../docs/protocol.md) 和 [data_format.md](../docs/data_format.md)。
