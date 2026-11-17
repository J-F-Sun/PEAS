# Data formats and numerical definitions

All NPZ files can be opened with `numpy.load(path, allow_pickle=False)`. Identifiers must be aligned by string, not just array length.

| Path / key | Shape | Meaning |
| --- | --- | --- |
| `data/reference/<scene>.npz`: `samples`, `ids`, `times`, `reference` | (20,3), (20,), (T,), (20,T) | Absolute Cartesian positions in the scenario frame; nondimensional elapsed time; cumulative RX absorption probability |
| `data/reference/*_training_grid.npz` | same keys; canonical T=801 up to 25 | Retained historical source grids; current U variants use horizon 20 and the separate validation archive |
| `data/predictions/{comparison,ablations}/<scene>/*.npz`: `prediction` | (R,20,T) | Raw per-run curves, R=10 except MC R=1 (the mean of 128) |
| MC `run_predictions` | (128,20,T) | Individual simulation curves before averaging |
| model `seeds` | (10,) | Run-axis seed order, when supplied by the original archive |
| `data/training/<scene>.npz`: `values`, `splits` | (200,801), (200,) | Supervised Monte-Carlo labels; explicit train/validation/test membership |
| `data/validation/canonical.npz` | samples, times, reference and IDs | Actual independent response validation grid used for canonical checkpoint selection |
| `data/validation/mcvd_transformer_*.npz` | 20 validation positions | Extracted convenience views of the original datasets, not additional samples |
| `data/steady/*.npz` | scenario-dependent FEM arrays | Unchanged FEM provider assets, including the pipe mesh |
| `checkpoints/peas/<scene>/seed_<seed>.pt` | PyTorch weights-only compatible dict | state_dict, numerical configuration, relative steady asset path; no optimizer state |

T=641 for canonical paper evaluation, T=801 for spherical/cylindrical confinement. The reference and every prediction have exactly equal time/position arrays within each release scene. Historical `*_training_grid` can have a longer horizon. Training datasets retain their original 801-point grids, including canonical [0,25]. Reference coordinates are absolute; plotted distances are `norm(samples-rx_center)`.

For run r and position i, `e[r,i]=100*norm(prediction[r,i,:]-reference[i,:])/norm(reference[i,:])`. The figure's position value is `mean_r e[r,i]`; its scene value is `mean_r mean_i e[r,i]`. This is not the error of the model ensemble mean curve. For Monte-Carlo only, the error is evaluated on its 128-simulation mean estimator.

The scene standard deviation for model runs is `std_r(mean_i(e[r,i]), ddof=1)`, rather than the mean of per-position standard deviations. Model-run rows report Student-t 95% confidence intervals for their mean error. Monte-Carlo rows use a different resampling calculation, described below.

### Monte-Carlo bootstrap diagnostics

The MC point estimate is the relative-L2 error of the archived 128-simulation mean curve. Bootstrap analysis resamples complete simulation rows jointly across positions and times, using 2048 multinomial replicates and seed 20261114. Each replicate averages 128 draws and recomputes the error. The bootstrap SD and the 2.5th/97.5th percentiles describe this conditional resampling error distribution.

The existing CSV schema and evaluator retain the legacy field names `ci95_low_percent` / `ci95_high_percent` and the MC `interval` label `Percentile bootstrap CI of error of 128-run mean`. **For MC rows, interpret these values as bootstrap error quantile bounds, not as a confidence interval with established 95% coverage.** The point estimate, bootstrap SD and quantile bounds have the following meanings:

| Existing field | Meaning for MC rows |
|---|---|
| `mean_percent` | Error of the archived 128-simulation mean curve |
| `std_percent` | Sample SD of the 2048 bootstrap error replicates |
| `ci95_low_percent`, `ci95_high_percent` | 2.5th and 97.5th percentiles of those error replicates; legacy column names |
| `n_runs` | One archived aggregate curve; its constituent simulation count is 128 |

Recomputing the existing arrays gives:

| Scene / position | Archived error (%) | Bootstrap mean (%) | Bootstrap SD (percentage points) | Bootstrap mean minus archived error (percentage points) | Bootstrap 2.5%–97.5% range (%) |
|---|---:|---:|---:|---:|---|
| U overall | 2.724138 | 3.466467 | 0.404816 | +0.742329 | [2.731196, 4.309465] |
| U `canonical_test_010` | 0.657395 | 2.021319 | 1.121588 | +1.363923 | [0.667585, 4.722080] |
| S overall | 2.480071 | 2.553198 | 0.203048 | +0.073127 | [2.174572, 2.962815] |
| C overall | 3.858235 | 3.930817 | 0.221205 | +0.072582 | [3.493128, 4.373721] |

For U overall and `canonical_test_010`, the archived error lies just below the resampling 2.5th percentile. This exposes a substantial shift in the bootstrap error distribution; the current calculation does not establish coverage for inference about an underlying population error. A percentile interval excluding its point estimate is not, by itself, a general test of interval validity.

The bootstrap mean is not a replacement for the plotted point estimate. Resampled mean curves fluctuate around the observed mean curve. Because the L2 norm is convex, this added variation can increase their average error even without evidence that the observed batch was unusually favorable. Thus `3.466467 ± 0.404816` describes the conditional bootstrap distribution, not an independently established typical error of fresh 128-simulation batches. The paper retains the measured U aggregate error of 2.724138%; estimating the performance distribution of fresh batches requires corresponding independent-batch evidence.

These diagnostics concern the nonlinear relative-L2 error statistic. They are separate from the pointwise mean-probability bands in the 3×4 figure. This documentation correction preserves the archived predictions, metric values and figures.

### Metric tables and figures

`results/expected_metrics.csv` has 504 rows: 2 experiment groups × 3 scenes × 4 methods × (20 positions + overall). Error units are **percent**. `__mean__` identifies overall rows. `results/paper_plotted_metrics.csv` contains 480 plotting-audit position rows: 300 retained rows (240 comparison rows and 60 full-PEAS ablation rows) and 180 updated ablation rows (three variants × three scenes × 20 positions) from the complementary and forward checkpoint audits. It is checked against the current prediction archives. `verify_release.py` verifies file hashes and recomputed point estimates; it does not validate confidence-interval coverage.

The 3×4 figure plots the mean probability curve with pointwise Student-t 95% confidence bands (10 model runs or 128 simulations); its error is `abs(mean_prediction-reference)`. The two 1×3 figures have no bands/error bars and label the mean to three decimal places. Their x-axis is ordinal positions sorted by `(distance,id)`, with selected distances as tick labels; it is not a uniformly sampled continuous distance axis.

`manifest.json` enumerates SHA-256, size, source SHA-256 where available, derivation notes, config/seed records and verification scope. Paths in configs are relative to the config file or expanded using `${PEAS_ROOT}`. Embedded historical provenance metadata may describe the original training environment; it is not used to load external files.

The current `no_features`/`no_steady` archives retain ten runs per group, with source records in `configs/ablations/run_provenance.json`. Current `no_reformulation` predictions come from the completed forward campaign, with records in `configs/ablations/forward_run_provenance.json`. The previous forward predictions remain separately in `data/predictions/ablations_previous/`; see [ablation_versions.md](ablation_versions.md).
