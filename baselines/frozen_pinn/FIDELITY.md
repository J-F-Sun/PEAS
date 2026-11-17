# Frozen-PINN fidelity and application adaptations

Upstream: https://gitlab.com/felix.dietrich/swimpde-paper, commit `0090e67a87ae31197229521d5a2d4134b96ef629`. MIT notice: LICENSE.upstream. Exact original and vendored hashes: protocol/upstream.json. All upstream changes are shown in protocol/upstream.patch. Eighteen of twenty Python files are unchanged.

The executable path is the author's BoundaryCompliantAnsatz, uniform SWIM point-pair sampler, least-squares linear boundary layer, native truncated SVD/pseudoinverse, least-squares initial coefficients and Diffusion_Solver.fit with scipy.solve_ivp. Single tanh hidden layer; fixed weights/biases; linear output; no gradient descent. Sampling and derivatives come from the original code/swimnetworks 0.0.2. The original clipped tanh derivatives are preserved, including their small nonzero tails at large preactivations.

Two minimal upstream edits plus instrumentation:
1. Permit an existing BoundaryCondition-compatible object in place of a homogeneous boundary-name string. The same native least-squares boundary layer handles identity and D*normal_derivative+kappa rows. Its constant column is constrained as well.
2. Fix the upstream outer_basis=True IC row mismatch: _A has interior rows, so its least-squares right-hand side must evaluate u0 on interior_points, rather than all_points.
3. Record elapsed ODE time and nfev/njev/nlu. No ODE formula, integration strategy or architecture is changed.

Application adapters:
- Solve FORWARD density independently for each source, p_t=D*Laplacian(p), D=1.
- Canonical/concentric use physical radial density with Laplacian p_rr+2*p_r/r. No reciprocal coordinates, u=r*p transformation or backward equation.
- Eccentric/pipe retain three Cartesian coordinates and actual full domains. Native pair weights act directly on physical coordinates.
- Homogeneous receiver Dirichlet, Robin outer loss, pipe absorbing end caps. Robin is a necessary application extension of the native linear B operator, not an upstream demonstrated benchmark.
- A truncated, normalized displaced Gaussian approximates the delta initial source. Radial scenes use its exact angular average. sigma=min(sigma_max, minimum_boundary_gap/boundary_sigma_margin), chosen by configuration, never fitted independently to each test curve.
- Fixed collocation combines global domain points and initial-source-local points to resolve the specified IC. It contains no reference-solution labels. Native SWIM pair selection remains uniform over these spatial points.
- RX probability is the unmodified signed boundary flux integrated in time. It is not 1 minus remaining mass because other exits absorb molecules.
- The portable checkpoint contains the native sampled weights, native boundary linear layer, native SVD projection and original SciPy dense ODE interpolant. Reloading never retrains or repairs predictions.

No PEAS model, steady field, q_infinity*f ansatz, arrival feature, inverse-distance feature, backward reformulation or prediction repair is imported. Reference NPZ files are independent analytic/transient-FEM evaluation data copied from the existing experiment; only their coordinates and time grids are used before fitting finishes. These selected points are tuning points, not held-out generalization evidence.

The historical tuning budget and selection procedure are documented in the root `docs/protocol.md`. This distribution retains the solver and upstream provenance, with training exposed through the root `scripts/train_frozen_pinn.py`. Historical tuning launchers, budget files and duplicate datasets are omitted.
