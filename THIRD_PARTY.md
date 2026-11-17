# Third-party provenance and licensing

The original PEAS license is **pending**, as requested by the project owner. `LICENSE` is a status notice, not a license grant. Nothing here changes third-party rights.

| Component | Source / retained version | License status and scope |
| --- | --- | --- |
| MCvD-Transformer adaptation | https://github.com/nrgBoun/MCvD-Transformer, commit `c580cca0ef956866e49fbcc84d38e331bcf24de0` | No upstream LICENSE at the pinned commit was present in the local source/provenance archive. Redistribution permission remains unresolved; do not describe this component as MIT-licensed. |
| Frozen-PINN / swimpde | https://gitlab.com/felix.dietrich/swimpde-paper, commit `0090e67a87ae31197229521d5a2d4134b96ef629` | MIT; verbatim notice in `baselines/frozen_pinn/LICENSE.upstream`. File hashes and the upstream patch are retained under its `protocol/`. The application adapter is separate from upstream code. |
| AMS-MOD | Local three-scene adaptation; individual source SHA-256 values are in `manifest.json` | No standalone license found in the supplied AMS-MOD project; pending owner confirmation. Julia standard-library dependencies remain under their own licenses. |
| PEAS / finite-element solvers | Local paper implementation and historical eccentric reference solver | Original project license pending. |

MCvD-Transformer keeps the author-method training routines, topology adapter, Keras-style Adam implementation and raw prediction convention. Its 2000-epoch budget is the adaptation used here, not a claim about the original paper's training budget. See `baselines/mcvd_transformer/METHOD.md`.

Frozen-PINN retains the original source notice, 20-file provenance list, and patch documenting application changes. Its NumPy 1.x environment has its own uv lock because it is incompatible with the plotting project's NumPy 2.x environment. Do not install one environment over the other.

External Python/Julia packages are installed through their package managers; their source is not relicensed by this project. Gmsh is an optional dependency for generating a new eccentric reference mesh; it is not needed to use the archived FEM assets, infer, or draw the figures.
