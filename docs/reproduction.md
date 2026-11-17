# Reproduction and inference

Run these commands from the release root. The distributed directory contains no virtual environments, caches or generated `outputs/`; installation and execution create them locally. This checkout does not require the original workspace, external drives or baseline repositories.

## Environment

Python **3.12.12** is selected by `.python-version`. Root dependencies are locked, including NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1, Matplotlib 3.11.2 and PyYAML 6.0.3. The supplied environment was verified on macOS arm64 with CPU inference/training smoke tests. The forward campaign ran on Linux NVIDIA GPUs; its training environments are recorded in `configs/ablations/forward_run_provenance.json`. The root lock provides the portable reproduction environment.

```bash
uv cache dir
uv sync --locked --offline
# If the cache lacks a required wheel/interpreter, allow a normal verified download:
uv sync --locked
```

Do not disable certificate verification. The environment is project-local `.venv`; existing projects are untouched. The lock includes optional Gmsh 4.15.2 for reference mesh generation (`uv sync --locked --extra fem`). It is not needed to reproduce archived figures or run PEAS inference. On Linux, Gmsh may require system OpenGL/GLU runtime libraries.

## Rebuild and verify the paper

```bash
uv run --locked python scripts/verify_release.py --inference
uv run --locked python scripts/evaluate.py
uv run --locked python scripts/reproduce_figures.py
uv run --locked python -m unittest discover -s tests -v
```

The verification checks every manifest hash, 504 recomputed mean error values, the 480 current plotting values, and all 30 PEAS inference weights. It writes `outputs/verification.json`. Evaluation additionally reports model-run SDs and Student-t confidence intervals. For MC, it reports bootstrap SD and resampling error quantiles under the legacy `ci95_*` columns and CI label; their interpretation is documented in [data_format.md](data_format.md#monte-carlo-bootstrap-diagnostics). File and point-estimate verification does not establish statistical coverage of those bounds. New figures go to `outputs/figures/` (PDF/PNG/SVG); the current paper PDFs are stored one each in `results/figures/`; the ablation PDF incorporates the completed complementary and forward campaigns. The two relative-L2 figures have no uncertainty bands and use three-decimal mean annotations.

## PEAS inference and training

```bash
uv run --locked python scripts/predict.py --scene canonical --seed 40
uv run --locked python scripts/train.py --scene canonical --smoke
uv run --locked python scripts/train.py --scene eccentric --smoke
uv run --locked python scripts/train.py --scene pipe --smoke
# One full independent training run; select --device cuda:0 on a suitable GPU host:
uv run --locked python scripts/train.py --scene canonical --seed 40 --device cpu
```

Replace the scene with `eccentric` or `pipe` as needed. Canonical full training saves `best_validation.pt` using the actual independent validation grid. The supplied inference checkpoints omit optimizer/RNG state; retraining starts afresh. Existing output directories are not overwritten.

```bash
uv run --locked python scripts/train.py --scene canonical --variant no_features --smoke
uv run --locked python scripts/train.py --scene eccentric --variant no_steady --smoke
uv run --locked python scripts/train.py --scene pipe --variant no_reformulation --source-id pipe_test_001 --smoke
```

Remove `--smoke` for the formal training budget. U backward variants save `selected.pt` using the independent validation grid and candidates 3500:250:9000. U forward training first fits validation sources and aggregates their scores, then trains test sources with the chosen epoch:

```bash
# Repeat for every ID in data/validation/canonical.npz (20 sources per seed):
uv run --locked python scripts/train.py --scene canonical --variant no_reformulation --seed 40 --split validation --source-id canonical_validation_001 --output outputs/forward
uv run --locked python scripts/select_forward.py --output outputs/forward --seeds 40
# Repeat for every test-source ID and seed:
uv run --locked python scripts/train.py --scene canonical --variant no_reformulation --seed 40 --split test --source-id canonical_test_001 --selection outputs/forward/selection.json --output outputs/forward
```

The archived per-seed choices are in `configs/ablations/canonical_forward_selection.json`. To retrain test sources with those archived choices, pass that path to `--selection`. Smoke training uses steps 1 and 2 and separate `_smoke` output directories.

中文：U 前向消融先完成每个种子的 20 个验证源训练，汇总验证误差选出步数，再训练该种子的 20 个测试源。也可以通过 `--selection configs/ablations/canonical_forward_selection.json` 使用归档选模结果重新训练测试源。S/C 沿用各场景的优化及选模规则。

前后两版预测和重绘命令见 [ablation_versions.md](ablation_versions.md)。默认绘图使用最新批次；`--ablation-version previous` 重绘此前误选旧批次数据的版本。

## Baseline training

MCvD's entry below uses the author-adapted trainer, not the generic convenience trainer. No TensorFlow environment is needed for this retained PyTorch port.

```bash
uv run --locked python scripts/train_mcvd_transformer.py --scene canonical --epochs 1 --output outputs/mcvd_smoke
uv run --locked python scripts/train_mcvd_transformer.py --scene eccentric --seed 40 --epochs 2000
```

Choose each method's actual seeds from `configs/baselines/*_archive.json`. The datasets include all 160/20/20 labels and uncertainty arrays. Baseline checkpoints are not bundled: per-run archived curves suffice for exact figure reproduction, and the training code/datasets are supplied for new fits.

Frozen-PINN retains its separately locked NumPy 1.x environment. Use its Python explicitly; do not mix its dependencies into the root environment.

```bash
uv sync --project baselines/frozen_pinn --locked --offline
# Or omit --offline if its cache is incomplete.
baselines/frozen_pinn/.venv/bin/python scripts/train_frozen_pinn.py --scene canonical --smoke
baselines/frozen_pinn/.venv/bin/python scripts/train_frozen_pinn.py --scene eccentric --source-id eccentric_test_001
```

## Reference and Monte-Carlo generation

```bash
uv run --locked python scripts/generate_reference.py --scene canonical
uv run --locked python scripts/generate_reference.py --scene pipe
uv sync --locked --extra fem
uv run --locked --extra fem python scripts/generate_reference.py --scene eccentric
```

The spherical reference uses the exact historical solver source and parameters; this is a relatively expensive 3-D FEM solve. Gmsh mesh regeneration can depend on version/platform, so archived references remain the source of the paper plots. `--smoke` uses a short time grid and a deliberately coarser eccentric mesh, and is not a paper reference.

AMS-MOD is a Julia project using standard libraries only. Julia 1.11.9 was used for the packaging smoke test; the code declares Julia >=1.10. Bitwise stochastic replay also depends on the historical Julia RNG version, not only the master seed.

```bash
julia --project=baselines/monte_carlo -e 'using Pkg; Pkg.instantiate()'
bash scripts/generate_mc_data.sh validate --config configs/baselines/monte_carlo_canonical.toml
bash scripts/generate_mc_data.sh smoke --config configs/baselines/monte_carlo_canonical.toml --split test --limit 1 --run-id smoke --output-root outputs/monte_carlo
bash scripts/generate_mc_data.sh run --config configs/baselines/monte_carlo_canonical.toml --run-id formal --output-root outputs/monte_carlo
```

Use the eccentric/pipe configs for the other scenes. `JULIA=/path/to/julia` selects an explicit Julia binary. All paper figure reproduction uses supplied numerical arrays and needs neither fresh MC simulation nor model fitting.

## Release status

`LICENSE` is pending by the owner's decision. MCvD-Transformer's pinned upstream source did not provide a LICENSE in the retained archive; its redistribution status is recorded in `THIRD_PARTY.md`. This prepared folder has not been published. `CITATION.cff` contains verified manuscript title/authors and no invented DOI or acceptance claim.

