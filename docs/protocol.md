# Experimental protocol

PEAS learns cumulative absorption probabilities through backward reformulation, steady-state factorization and feature mapping. The experiments evaluate prediction accuracy across release positions and assess the contribution of these three components in unbounded, spherical and cylindrical domains.

## Training and selection

| Scene | Training horizon | Time feature scale | Optimization | Model selection |
|---|---:|---:|---|---|
| U: unbounded domain | 20 | 20 | 9000 Adam steps | Lowest mean validation Rel-L2 at 3500:250:9000; earliest tie |
| S: spherical confinement | 20 | 20 | 7000 Adam steps + L-BFGS, up to 500 iterations | Final optimization state |
| C: cylindrical confinement | 80 | 20 | 8500 Adam steps + L-BFGS, up to 300 iterations; no late loss | Best fixed physics-validation L-BFGS state |

Each scene uses ten model initialization seeds, paired between full PEAS and its ablations. U uses response normalization floor 0.02 and an independent 20-position response validation set. Its ten seeds comprise three screening seeds (40, 18, 1414717112) used in normalization tuning and seven confirmation seeds. S/C use physics-based training and selection. Seed lists, geometry and configuration files are indexed in [paper_protocol.yaml](../configs/paper_protocol.yaml) and [configs/](../configs/README.md).

## Ablation design and results

The ablations remove backward reformulation (`no_reformulation`), steady-state factorization (`no_steady`) or feature mapping (`no_features`). Full PEAS uses exactly the same model ensemble as the comparison figures. All variants share the principal training settings and selection protocol within each scene.

Removing backward reformulation produces a fixed-source forward density model. Its RX surface flux is integrated over time to obtain the absorption probability; one model is trained for each source position. Backward models share weights across source positions. This difference in model count and computation follows from the forward/backward formulations being compared.

For U forward models, each seed first trains 20 independent validation-source models. Candidate errors are averaged across those sources to select one epoch, which is then used for all 20 test-source models of that seed. The completed campaign contains 200 validation-source fits and 600 test-source fits across U/S/C. [Validation scores](../data/validation/canonical_forward_scores.json), [selected epochs](../configs/ablations/canonical_forward_selection.json), and [checkpoint provenance](../configs/ablations/forward_run_provenance.json) are included for reproduction.

Mean relative L2 error (%):

| Scene | Full PEAS | w/o feature mapping | w/o steady-state factorization | w/o reformulation |
|---|---:|---:|---:|---:|
| U | 0.307 | 2.575 | 5.571 | 201.789 |
| S | 2.238 | 23.592 | 46.028 | 147.806 |
| C | 3.904 | 11.650 | 46.547 | 631.834 |

Full PEAS achieves the lowest overall error in all three scenes, supporting the contribution of the three components under the evaluated settings. Both the previous and current forward-ablation results are retained in [version notes](ablation_versions.md). The current campaign supplies the default figure data.

## Evaluation and statistics

All methods use identical paper reference arrays and [0,20] evaluation grids: 20 positions × 641 times for U, and 20 × 801 for S/C. For each run and position, the metric is `100 * ||prediction-reference||₂ / ||reference||₂` over the full time grid. Position and scene summaries equally weight ten runs and twenty positions. The run-level scene standard deviation is computed after averaging positions within each run. The relative-L2 plots show three-decimal means without uncertainty bands.

Raw predictions and signed forward flux integrals are retained. Relative errors can exceed 100% when a predicted curve departs substantially from the reference. Monte-Carlo is evaluated as the error of the archived 128-simulation mean curve. Its bootstrap SD and percentile bounds describe the conditional resampling error distribution. The MC table retains legacy `ci95_*` column names; these bounds are not reported as confidence intervals with established 95% coverage. For U, the observed overall error is 2.724138%, while the bootstrap mean is 3.466467% and the 2.5th–97.5th percentile range is [2.731196%, 4.309465%]. The plotted value remains the observed error; the bootstrap mean is not a replacement estimate of typical independent-batch performance. Definitions and diagnostics are given in [data_format.md](data_format.md#monte-carlo-bootstrap-diagnostics).

## Baselines and references

MCvD-Transformer uses three 200-position datasets with 160/20/20 training/validation/test splits. Labels are averages of 128 AMS-MOD simulations. The adapted trainer runs up to 2000 epochs and selects by augmented validation raw-CIR MxNAE; reported errors use raw output curves. The canonical dataset retains its `experiment` metadata tag, and S/C retain `formal`. Frozen-PINN uses its fixed hyperparameters and declared tuning positions; the overlapping tuning/reported positions are listed in [frozen_pinn_tuning_points.json](../data/validation/frozen_pinn_tuning_points.json). Each baseline retains its own seed list.

U uses the analytical response `erfc((r-1)/sqrt(4t))/r`, with zero response at t=0. S uses tetrahedral P1 transient FEM with lumped mass and implicit Euler, mesh size 0.4 and dt=0.005 (98,883 nodes; 585,438 elements). C uses tetrahedral P1 FEM with consistent mass, Crank–Nicolson and two initial backward-Euler half steps, dt=0.01. The neural models' steady FEM assets are stored separately from transient references. All quantities are nondimensional, with unit radius scale and diffusivity; archived FEM values retain their numerical startup values.

## Reproduction records

[run_provenance.json](../configs/ablations/run_provenance.json) and [forward_run_provenance.json](../configs/ablations/forward_run_provenance.json) record source checkpoints, seeds, hashes and training environments. Historical source paths identify provenance; reproduction uses the files within this repository. Root Python and dependencies are pinned by `.python-version` and `uv.lock`. See [reproduction.md](reproduction.md) for figure reconstruction, inference and training commands.
