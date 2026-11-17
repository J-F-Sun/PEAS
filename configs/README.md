# Experiment configurations

**English** | [简体中文](README.zh-CN.md) · [Project home](../README.md)

[Coordinates and Table I](#coordinates-and-features) · [PEAS training](#peas-training) · [Monte Carlo](#monte-carlo) · [MCvD-Transformer](#mcvd) · [Frozen-PINN](#frozen-pinn) · [Reference solvers](#reference-solvers) · [Ablations and statistics](#protocol)

This guide describes the configurations **actually used by the archived paper experiments and the root training entry points**. U/S/C correspond to `canonical`/`eccentric`/`pipe`. JSON/TOML files record the current plotted contracts; retained historical settings and explicit runtime overrides are identified below.

| Files | Role / effective reader |
| --- | --- |
| [scenarios/](scenarios/) | Geometry, boundaries and collocation time ranges; `eccentric_reference.json` records the historical reference mesh |
| [peas/](peas/) | Full PEAS; `scripts/train.py` loads these configs and each checkpoint retains its own inference config |
| [ablations/](ablations/) | Current scene × variant settings, read by the ablation config builder |
| [baselines/](baselines/) `monte_carlo_*.toml` | Julia simulation settings and dataset manifests |
| [baselines/](baselines/) `mcvd_transformer_*.toml` | MCvD input configuration; author trainers override some settings |
| [baselines/frozen_pinn_archive.json](baselines/frozen_pinn_archive.json) | **Effective** Frozen-PINN configs and seeds read by `train_frozen_pinn.py`; `frozen_pinn_<scene>.json` files are matching per-scene views |
| [baselines/](baselines/) `*_archive.json` | Actual run seeds, historical metadata and prediction provenance |
| [paper_protocol.yaml](paper_protocol.yaml) | Experiment scope, checkpoints, seeds, selection and evaluation rules |

`${PEAS_ROOT}` resolves to the repository root. Monte Carlo manifest paths are relative to their TOML file. MCvD's legacy `data.ams_run` paths are metadata: the released wrapper explicitly loads `data/training/<scene>.npz`, with no dependency on a sibling AMS-MOD checkout. Editing a config does not change an archived prediction or an already saved checkpoint.

<a id="coordinates-and-features"></a>
## Coordinates, units and the complete Table I features

Length is scaled by RX radius $a$, time by $a^2/D$; consequently $a=D=1$. Let $d=\|\mathbf{x}_0-\mathbf{c}_{\mathrm{RX}}\|$ be the **center-to-center** TX–RX distance and $\tau=t/d^2$. Natural logarithms are used. These are fixed, physically chosen transformations, not fitted z-scores or dataset min–max normalizers.

The manuscript places RX at the origin. The archived S/C coordinates place the container center/axis at the origin:

| Scene | Code-frame RX | Code → paper position | Outer geometry in paper coordinates |
| --- | --- | --- | --- |
| U | $(0,0,0)$ | unchanged | unbounded |
| S | $(3,0,0)$ | $\mathbf{x}_{\mathrm{paper}}=\mathbf{x}_{\mathrm{code}}-(3,0,0)$ | sphere center $(-3,0,0)$, radius 10 |
| C | $(1,0,0)$ | $\mathbf{x}_{\mathrm{paper}}=\mathbf{x}_{\mathrm{code}}-(1,0,0)$ | cylinder axis $x=-1,y=0$, radius 5 |

Pass **code-frame coordinates** to model inference. In S, $\mathbf{e}$ points from outer center $\mathbf{o}$ to RX, $s=(\mathbf{x}_0-\mathbf{o})\cdot\mathbf{e}$, $\rho^2=\|\mathbf{x}_0-\mathbf{o}-s\mathbf{e}\|^2$, and $r_o=\|\mathbf{x}_0-\mathbf{o}\|$. In C, $(x_c,y_c,z_c)$ are cylinder-centered coordinates, equal to the stored coordinates and to $(x+1,y,z)$ in the paper frame.

### Table I: feature composition before numerical scaling

| Scene | Geometric features | Temporal features | Diffusion-related features | Dimension |
| --- | --- | --- | --- | ---: |
| U | — | $t,\ \log(1+t)$ | $d^{-1},\ \tau,\ \log(1+\tau)$ | 5 |
| S | $s,\ \rho^2,\ d^2,\ r_o^2,\ b_{\mathrm{RX}},\ b_{\mathrm{out}}$ | $t$ | $d^{-1},\ \tau,\ A(d,t)$ | 10 |
| C | $x_c,\ \log(1+y_c^2),\ \log(1+z_c^2),\ b_{\mathrm{RX}},\ b_{\mathrm{wall}},\ b_{\mathrm{cap}}$ | $t,\ \sqrt t$ | $d^{-1},\ \log(1+\tau),\ A(d,t)$ | 11 |

Here $b_{\mathrm{RX}}=d-1$, $b_{\mathrm{out}}=10-r_o$, $b_{\mathrm{wall}}=5-\sqrt{x_c^2+y_c^2}$, and $b_{\mathrm{cap}}=\min(z_c+70,70-z_c)$. The paper's arrival descriptor is $A(d,t)=\operatorname{erfc}((d-1)/\sqrt{4t})$. The exact implementation regularizes it as

$$
A_\epsilon(d,t)=\operatorname{erfc}\!\left(\frac{d-1}{\sqrt{4(t+\epsilon)}}\right),\qquad \epsilon=10^{-5}.
$$

It is an input descriptor, without an extra $1/d$ prefactor; it does not prescribe the confined-domain response.

### U: 5 inputs, in the actual tensor order

With $T=20$ and $\tau_0=0.01$:

$$
\Phi_U=\left[\frac{t}{T},\ \frac1d,\ \tau,\
\frac{\log(1+t)}{\log(1+T)},\
\frac{\log(1+\tau/\tau_0)}{\log(1+T/\tau_0)}\right].
$$

The linear $\tau$ column is **not** divided by $\tau_0$. The logarithmic diffusion column therefore means a scaled version of the Table I feature, not literally `log1p(tau)`. No arrival descriptor is added. See [canonical.json](peas/canonical.json) and [canonical/features.py](../src/peas/scenarios/canonical/features.py).

### S: 10 inputs, in the actual tensor order

With $L=R=10$ (`input_scale="outer_radius"`) and $T=20$:

$$
\Phi_S=\left[\frac tT,\ \frac sL,\ \frac{\rho^2}{L^2},\ \tau,\
\frac{d^2}{L^2},\ \frac{r_o^2}{L^2},\ \frac{d-1}{L},\
\frac{R-r_o}{L},\ \frac1d,\ A_\epsilon(d,t)\right].
$$

$1/d$ and $\tau$ are retained directly; the transverse square has no added epsilon (`outer_eps=0`). See [eccentric.json](peas/eccentric.json) and [sphere_features.py](../src/peas/common/sphere_features.py).

### C: 11 inputs, in the actual tensor order

With $R=5$, $T=20$, **axial feature scale $L_z=10$** (not the half-length 70), radial smoothing $\epsilon_\rho=10^{-3}$, and $\tau_0=1$:

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

The cap descriptor is divided by the **full cylinder length 140**. The smooth square-root feature equals zero at $t=0$. The logarithmic diffusion feature has no additional division by `log(1+T)`. Squared/logarithmic $y_c,z_c$ features encode reflection symmetry; $x_c$ remains signed because RX is off-axis. No drift-arrival, Gaussian or geometry-envelope feature is enabled. See [pipe.json](peas/pipe.json) and [pipe/features.py](../src/peas/scenarios/pipe/features.py).

All feature operations remain in the differentiation graph. MLPs use `tanh` hidden activations, a linear scalar output and float32 arithmetic; predictions are not clipped to $[0,1]$.

<a id="peas-training"></a>
## PEAS optimization, loss scales and steady assets

| Setting | U | S | C |
| --- | --- | --- | --- |
| Hidden layers × width | 4 × 32 | 6 × 64 | 8 × 160 |
| Adam steps | 9000 | 7000 | 8500 |
| Collocation horizon / feature time scale | 20 / 20 | 20 / 20 | **80 / 20** |
| L-BFGS maximum iterations | disabled | 500 | 300 |
| Adam batches | PDE 8192; IC/RX/far 1024 each | PDE 8192; IC/RX/outer 2048 each | PDE 3072; IC 1536; RX/wall 1024 each; caps 512 |
| Checkpoint selection | Minimum mean validation Rel-L2, steps 3500:250:9000; earlier wins ties | Physics-only training; no response-reference selection | Restore best L-BFGS state by fixed physics-validation loss, seed 1073 |

Stage schedules below give **steps @ learning rate**. The corresponding loss weights are in each JSON's `training.lr_schedule`, overriding the initial `lr` and `loss_weights`:

- U: 2500 @ 0.005 → 2500 @ 0.002 → 2500 @ 0.0001 → 1500 @ 0.00002; Adam state retained across stages.
- S: 1800 @ 0.0005 → 2200 @ 0.00008 → 3000 @ 0.00002; Adam state retained; gradient norm clipped at 1. Adaptive collocation starts at step 800, refreshes every 200, selects 2048 of 16384 candidates, and fills 25% of PDE batches.
- C: 2000 @ 0.001 → 2500 @ 0.0003 → 2500 @ 0.0001 → 1500 @ 0.00003; Adam reset at each stage; gradient norm clipped at 1. `without_late_loss=true` removes late-time targets from the objective, even though historical `late` batch/weight fields remain in the JSON. Post-L-BFGS Adam is disabled.

Both enabled L-BFGS stages use learning rate 0.8 and `strong_wolfe` line search; their fixed batch sizes and weights are in `training.lbfgs`.

**Input normalization and loss normalization are separate.** For U, the PDE residual first uses the `full` conditioning denominator $y^3+0.05$, with $y=1/d$. The relative objective then averages PDE/IC squared errors equally over six distance groups with edges `[1,3,6,10,15,25,200.001]`, dividing each group's mean square by $s_k^2$. The scale is $s_k=\operatorname{clip}(\sqrt{\mathrm{EMA}(u^2)},0.02,1)$, EMA decay 0.99, where $u=f_\theta$ is the learned transient factor. It uses no reference-response amplitude. The inverse-square cap is 2500. Grouped loss blends in from steps 2500 to 3500; gradient-based component balancing updates every 100 steps, with factors in `[0.2,5]`. See [canonical/physics.py](../src/peas/scenarios/canonical/physics.py).

S uses the divided steady-factor PDE with steady-field floor $10^{-4}$, drift-component clipping at 50 and residual clipping at $10^3$. C uses

$$
\mathcal R_C=\frac{\widehat q_\infty(u_t-\Delta u)-2\nabla\widehat q_\infty\cdot\nabla u}{\widehat q_\infty+0.03}.
$$

The paper groups boundary penalties as $\lambda_{\mathrm{BCs}}\mathcal L_{\mathrm{BCs}}$; the implementation retains separate RX/far, RX/outer, or RX/wall/cap losses with their own stage weights. S/C also include range penalties. This grouped notation does not imply one shared numerical boundary weight.

| Fixed steady FEM field | Discretization and numerical settings |
| --- | --- |
| [canonical.npz](../data/steady/canonical.npz) | Radial P1, 2048 elements/2049 nodes on $[1,200]$; mesh power 1.6; outer Robin coefficient 0.005; floor $10^{-6}$ |
| [eccentric.npz](../data/steady/eccentric.npz) | Axisymmetric bispherical Q1, 96 × 128 nodes; outer Robin coefficient 5 |
| [pipe.npz](../data/steady/pipe.npz) | Tetrahedral P1, 23446 nodes/120549 tetrahedra; Gmsh size range 0.12–0.9, RX-distance transition 0.35–4; volume-weighted nodal gradients; locator candidates 96 |

For U, `problem.domain.r_far=25` is **not** the actual FEM truncation radius: the retained sampler and far boundary use `q_infty.outer_radius=200`. The steady field is computed once and remains fixed during MLP fitting. It is distinct from the transient reference below.

<a id="monte-carlo"></a>
## Monte Carlo / AMS-MOD

The figure label “Monte-Carlo” refers to the archived Julia AMS-MOD estimator. See the related [A-MLS project](https://github.com/J-F-Sun/A-MLS) for absorption-aware multilevel splitting. This release keeps the paper's adapted implementation in [baselines/monte_carlo/](../baselines/monte_carlo/), rather than depending on a moving external checkout.

The release-position range extends deep into the far field, where RX absorption within a finite observation window becomes rare. Direct conventional Monte Carlo would require many independent trajectories to resolve these small probabilities and is too slow for this experiment grid. AMS focuses sampling through splitting/resampling; this motivates its use, without asserting an unmeasured speedup factor.

| Simulation setting | U | S | C |
| --- | ---: | ---: | ---: |
| `num_replicas` per AMS population | **250** | **250** | **250** |
| `n_worst` selection parameter | 100 | 1 | 1 |
| Independent `repeats` per position | **128** | **128** | **128** |
| Internal diffusion `dt` | 0.00025 | 0.0002 | 0.0002 |
| Simulated horizon | 25 | 20 | 20 |
| Output time samples | 801 | 801 | 801 |
| Maximum splitting iterations | 100000 | 100000 | 100000 |
| Master seed | 40 | 1234 | 20260908 |
| Dataset position seed | 2026100501 | 2026100503 | 2026100504 |
| Allowed position-distance range | 1.5–21.5 | 1.25–10.8 | 1.25–10 |

Files: [U](baselines/monte_carlo_canonical.toml), [S](baselines/monte_carlo_eccentric.toml), [C](baselines/monte_carlo_pipe.toml).

**250 replicas and 128 repeats are different levels.** Each position starts $250\times128=32000$ replica instances across its independent runs; replicas within an AMS run are genealogically dependent after branching. This is not equivalent to 32000 independent conventional-MC paths, nor a count of all segments generated by resampling. With 200 positions, there are **25600 independent AMS runs per scene**, or 76800 across three scenes. The 20-position test subset contains 2560 runs per scene.

The mean of 128 curves supplies each MCvD target and the plotted MC estimator. Raw test-run curves remain in `data/predictions/comparison/<scene>/monte_carlo.npz`. U labels retain $[0,25]$ and 801 times; the figures evaluate only the first 641 samples over $[0,20]$. Smoke mode deliberately replaces these settings (16 replicas, 4 repeats, horizon 0.2, `dt=0.002`, 41 times, 2000 maximum iterations); it cannot produce paper-quality estimates.

<a id="mcvd"></a>
## MCvD-Transformer

Use [scripts/train_mcvd_transformer.py](../scripts/train_mcvd_transformer.py) in the root environment. This is the retained PyTorch adaptation of upstream commit `c580cca0ef956866e49fbcc84d38e331bcf24de0`, with geometry/boundary extensions for S/C. Each scene has its own [training dataset](../data/training/), 160/20/20 split and ten archived seeds.

| Setting | Effective paper implementation |
| --- | --- |
| Input | 15 topology entities × 13 fields, plus 7 numerical features |
| Coordinate encoding | TX-centered geometry; RX is first entity; joint rigid rotation of all geometry during augmentation; remaining entity/padding rows shuffled |
| Numerical features | Diffusivity, actual output time spacing, five unobstructed line-of-sight entries |
| Geometry classes | Existing 0–4 retained; 5 Robin shell, 6 Robin cylinder, 7 absorbing cap, 8 reflecting cap (the paper C uses 7) |
| Transformer | 2 blocks; 16 heads; **13 dimensions per head**; FF width 32; dropout 0.1 |
| Dense branches | Shared 400→800→1600; shape hidden 1622; maximum branch 1600→800→400 |
| Output | 801 curve samples plus 10 leading zero-padding entries; shape × maximum predicts the raw curve |
| Training | 2000 epochs; batch 64 (160 positions → batches 64,64,32); Adam, betas (0.9,0.999), Keras epsilon convention $10^{-7}$; Glorot initialization; no gradient clipping |
| Selection | Lowest augmented-validation raw-CIR maximum-normalized absolute error (MxNAE) |
| Reported prediction | Raw curve without smoothing, monotonic correction or averaging over rotated predictions |

**Effective learning-rate schedule:** all three author trainers use epoch index $e=0,\ldots,1999$, linear warmup from $10^{-5}$ to $10^{-3}$ over $e=0\ldots25$, linear decay to $5\times10^{-5}$ by $e=200$, then hold. The U TOML's legacy `20/280/0.0003` schedule is not executed by this entry point. See [lr_for_epoch and executed_recipe](../baselines/mcvd_transformer/scripts/author_verified_canonical.py) and [author_config](../baselines/mcvd_transformer/src/mcvd_transformer/author_scene_adapter.py); modifying ignored TOML fields alone will not change this experiment.

The objective transitions from shape-plus-maximum supervision to raw-curve supervision: $\alpha(\mathcal L_{\mathrm{shape}}+\mathcal L_{\mathrm{max}})+(1-\alpha)\mathcal L_{\mathrm{raw}}$. Shape loss includes a $10^{-4}$ monotonic penalty. Maximum-loss normalization uses `max(abs(target_max),1)`; raw-curve terms normalize by the target final value. The epoch-end schedule has $\alpha=1$ through zero-based epoch 31, decreases by $1/200$ per epoch thereafter, and reaches zero at epoch 231. The validation score is the mean over positions of the maximum-in-time normalized absolute error. U labels span $[0,25]$, S/C $[0,20]$; all paper metrics use $[0,20]$.

The released wrapper reads NPZ metadata to retain the historical `experiment` profile for U and `formal` profiles for S/C. The 2000-epoch budget is this paper's adapted experiment, not a claim about the upstream paper's original budget. Seed lists come from `mcvd_transformer_<scene>_archive.json`.

<a id="frozen-pinn"></a>
## Frozen-PINN

Use [scripts/train_frozen_pinn.py](../scripts/train_frozen_pinn.py) with the separately locked [Frozen-PINN environment](../baselines/frozen_pinn/pyproject.toml): Python 3.12.12, NumPy 1.26.4, SciPy 1.12.0, scikit-learn 1.4.2, swimnetworks 0.0.2. It uses the upstream boundary-compliant `tanh` basis, uniform-pair SWIM sampling, SVD reduction and time evolution of coefficients. There is no Adam epoch schedule. Each source/seed is fitted independently, followed by RX-flux integration.

| Setting | U | S | C |
| --- | ---: | ---: | ---: |
| Spatial representation | radial | Cartesian 3D | Cartesian 3D |
| Inner / outer basis count | 1536 / 768 | 4096 / 512 | 4096 / 512 |
| Interior collocation count | 4096 | 12288 | 12288 |
| `surface_points` setting | 384 (radial uses endpoints) | 4096 | 4096 |
| Source-local sample fraction / spread multiplier | 0.25 / 6 | 0.15 / 4 | 0.15 / 4 |
| Gaussian `sigma_max` / boundary margin factor | 0.15 / 4 | 0.5 / 1 | 0.5 / 1 |
| Least-squares `rcond` | $10^{-10}$ | $10^{-5}$ | $10^{-5}$ |
| SVD cutoff | $10^{-7}$ | 0.01 | 0.01 |
| Solve horizon / output points | 25 / 801 | 20 / 801 | 20 / 801 |

Common settings: boundary weight 100; LSODA with relative/absolute tolerances $10^{-6}$/$10^{-8}$; 2048 RX surface-quadrature points for 3-D flux; 4001 dense time-integration points augmented with requested output times; cumulative trapezoidal integration; 4 CPU threads. U uses exact radial surface integration and an outer cutoff of 30. The S/C configs retain `outer_cutoff=60`, but their physical sphere/cylinder bounds determine the domain.

The Dirac source is approximated by a domain-normalized Gaussian with $\sigma=\min(\sigma_{\max},\text{nearest-boundary gap}/\text{margin factor})$. U uses its analytical spherical average. Spatial inputs are radius or Cartesian coordinates, not the PEAS feature vector. Negative/nonmonotone predictions are retained without correction. The per-scene views are [U](baselines/frozen_pinn_canonical.json), [S](baselines/frozen_pinn_eccentric.json), [C](baselines/frozen_pinn_pipe.json); the wrapper reads `configs` and `seeds` from [frozen_pinn_archive.json](baselines/frozen_pinn_archive.json).

Hyperparameters were selected on three tuning positions per scene that overlap the reported test set: U `002/009/020`, S `005/009/016`, C `004/010/017` (with the scene's `_test_` prefix). [Tuning-point records](../data/validation/frozen_pinn_tuning_points.json) preserve this selection scope. Ten seeds × 20 positions give 200 independent source fits per scene. The upstream MIT notice and adaptation details are retained in [LICENSE.upstream](../baselines/frozen_pinn/LICENSE.upstream) and [FIDELITY.md](../baselines/frozen_pinn/FIDELITY.md).

<a id="reference-solvers"></a>
## Reference solutions: independent transient evaluation

The [reference generation entry point](../scripts/generate_reference.py) loads test positions and requested times from `data/reference/<scene>.npz`. It does not use MC curves or a trained neural model. `--dt` can override the integration step; defaults below reproduce the archived recipe.

| Setting | U | S | C |
| --- | --- | --- | --- |
| Reference | Analytical | Transient tetrahedral P1 FEM | Transient tetrahedral P1 FEM |
| Formula / mesh | $q=d^{-1}\operatorname{erfc}((d-1)/\sqrt{4t})$ for $t>0$; interior $q(0)=0$ | Historical 98883 nodes / 585438 tetrahedra | Bundled 23446 nodes / 120549 tetrahedra |
| Mass matrix | — | Row-sum lumped | Consistent |
| Time scheme | — | Implicit Euler | Crank–Nicolson, with two initial backward-Euler half steps |
| Integration step | — | 0.005; 4000 steps to 20 | 0.01; 2000 full intervals to 20 |
| Output grid | 641 times on [0,20] | 801 times on [0,20] | 801 times on [0,20] |
| Outer condition | decay at infinity | Robin coefficient 5 | Robin wall coefficient 1; caps $q=0$ |

Both confined references impose $q=1$ on RX and zero initial probability in the fluid interior, and solve sparse linear systems with factorization reuse. S uses [eccentric/reference.py](../src/peas/scenarios/eccentric/reference.py): Gmsh `mesh_size=0.4`, first-order elements, RX-adjacent size 0.1, far size 0.4, distance transition 0.2–1.2. Its solver is separate from the neural model's bispherical steady FEM. Mesh regeneration can depend on Gmsh/platform; [eccentric_reference.json](scenarios/eccentric_reference.json) records the original provenance.

C uses [common/references.py](../src/peas/common/references.py) and reads only mesh/connectivity/boundary data from `data/steady/pipe.npz`; it **does not use the stored steady `q` as the transient reference**. Spatial P1 interpolation and time interpolation map the FEM solution to requested observations. Both FEM references are finite-resolution numerical references, not exact analytical ground truth; tiny negative startup values are retained.

The optional `fem` environment extra pins Gmsh 4.15.2 for S. Reference `--smoke` keeps two positions and three output times, with a deliberately coarser S mesh (`mesh_size=1.5`); it is only a solver pipeline test. Fresh results go to `outputs/reference/` and do not replace paper reference archives.

<a id="protocol"></a>
## Ablations, selection and reported statistics

| Variant | Changed component | Full ensemble per scene |
| --- | --- | --- |
| `no_reformulation` | Forward density fitted per TX; recover probability by RX-flux integration | 20 positions × 10 seeds = 200 fits |
| `no_steady` | Direct backward prediction of $q$, without fixed $q_\infty$ factorization | 10 fits, each querying all positions |
| `no_features` | Replace designed inputs by normalized space/time; retain factorization | 10 fits, each querying all positions |

[ablations/](ablations/) records all three variants under the current scenario-specific protocols. U uses horizon/time scale 20, response floor 0.02 and validation selection at 3500:250:9000; the forward variant aggregates 20 independent validation-source models per seed before selecting its test checkpoints. S uses 7000 Adam steps plus L-BFGS. C uses horizon 80, time scale 20, 8500 Adam steps and physics-validation L-BFGS selection without late loss. All errors use $[0,20]$. Source hashes and environments are retained in [run_provenance.json](ablations/run_provenance.json) and [forward_run_provenance.json](ablations/forward_run_provenance.json). See [version notes](../docs/ablation_versions.md) for the previous archive.

For neural run $r$ and test position $i$, $e_{r,i}=100\|\widehat{\mathbf q}_{r,i}-\mathbf q_i\|_2/\|\mathbf q_i\|_2$. Position means average the ten runs; the scene mean averages all 20 position errors and ten runs equally. The scene SD is `std(mean(e, axis=positions), ddof=1)` across runs, not the average of positionwise SDs. Monte Carlo first averages 128 curves and then computes its error; its uncertainty in `evaluate.py` uses 2048 bootstrap samples, seed 20261114. The relative-error figures omit uncertainty bands; the probability figure uses pointwise 95% CIs for mean curves.

The ten U PEAS seeds include three screening runs (40, 18, 1414717112) used to choose the normalization floor and seven confirmation runs. S/C do not select on transient reference labels. Frozen-PINN's overlapping tuning positions are disclosed above. Use method-specific archived seed lists rather than assuming paired seeds across methods. The full PEAS ensemble is shared by comparison and ablation figures. See [protocol.md](../docs/protocol.md) and [data_format.md](../docs/data_format.md) for the complete provenance and statistical contract.
