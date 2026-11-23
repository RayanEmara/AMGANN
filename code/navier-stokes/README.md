# Navier–Stokes + ANN + AMG

A runnable 2D end-to-end experiment: generate Navier–Stokes systems, measure
AMG costs at several strong thresholds, train a matrix-image CNN, and use its
predicted optimum inside a time-dependent solve. The fixed baseline is theta=0.5.

The new testcase uses **scikit-fem Q2/Q1 finite elements**, **PETSc CG with hypre
BoomerAMG**, and **PyTorch for CNN training**. Model inference uses exported NumPy
weights, so the numerical-solver container does not need PyTorch. This is a
self-contained addition to the repository; it does not depend on the legacy
deal.II/TensorFlow experiment scripts or their pretrained models. Existing HDF5
models are not compatible with this new CNN.

## Polimi GPU cluster

[VPN access, PBS smoke test and full GPU run](CLUSTER.md).

## Small local run

Requires Python 3.10+ and a running Docker engine. From the repository root:

```bash
bash code/navier-stokes/run_local.sh my-first-run
```

The script creates/uses `.venv`, installs local Python dependencies if absent,
builds `environment/Dockerfile.navier-stokes` for the host architecture, collects
timings in Docker, trains on the host CPU, and runs ANN inference and actual
BoomerAMG solves back in Docker. On Apple Silicon this uses ARM Linux, not x86
emulation. The first environment build downloads dependencies; subsequent runs
reuse the image. Choose a new run name each time: existing outputs are never
deleted or overwritten.

Outputs under `data/navier-stokes/my-first-run/` (gitignored):

- `dataset/metadata.json`: mesh splits, matrix identities, runtime versions,
  candidate thresholds and every measured setup/solve timing.
- `dataset/*.matrix.npz`, `*.rhs.npz`: sparse systems and RHS sequences.
- `dataset/training.npz`: pooled images, scalar features, labels and group IDs.
- `model.npz`, `model.training.json`: portable CNN weights, metadata, training
  history and a PyTorch-versus-NumPy inference check.
- `comparison/comparison.json`: complete fixed/ANN runs, selections, solver
  residuals, errors and timings.
- `comparison/{ann,fixed}.fields.npz`: velocity/pressure coefficients and DOF
  coordinates for visualization.
- `convergence/convergence.json`: manufactured-solution refinement results.

For a fixed-threshold solve alone, after the image is built:

```bash
docker run --rm -v "$PWD":/workspace amg-ann-ns run \
  --backend petsc --cells 12 --steps 5 --dt 0.002 --viscosity 0.05 \
  --output data/navier-stokes/fixed-example
```

Add `--model data/navier-stokes/my-first-run/model.npz` to enable ANN selection.
`python -m ns_amg --help` lists the separate `collect`, `train`, `run`,
`compare`, and `convergence` commands. For host commands set
`PYTHONPATH=code/navier-stokes` and use `.venv/bin/python`.

A Docker-free testing backend is also available as `--backend pyamg`, using
classical Ruge–Stuben AMG. Train a separate model with data collected from that
backend. The program rejects mismatched model/backend combinations: PyAMG and
BoomerAMG timings and threshold meanings must not be treated as interchangeable.

## Numerical method

The starting point is Quarteroni, *Numerical Models for Differential Problems*,
second edition, section 16.7.3, equations (16.66)–(16.69). This implementation uses
explicit convection, backward-Euler diffusion and a nonincremental pressure
projection, with a mass-lumped discrete velocity inner product.

On the unit square the PDE is

```text
du/dt + (u . grad)u - nu Laplacian(u) + grad(p) = f
div(u) = 0
u = 0 on the boundary
```

The manufactured solution is defined by

```text
psi(x,y,t) = sin(pi*x)^2 sin(pi*y)^2 exp(-t)
u = (dpsi/dy, -dpsi/dx)
p = cos(pi*x) cos(pi*y) exp(-t)
```

The forcing includes the exact time derivative, **nonzero nonlinear convection**,
diffusion and pressure gradient. The time-step RHS evaluates convection using
the current numerical velocity. Thus this is a Navier–Stokes test, not a Stokes
test with convection omitted.

Let D be the diagonal row-sum-lumped Q2 velocity mass matrix, K the stiffness
matrix, and B=(Bx,By) the Q2-to-Q1 weak divergence matrix. Boundary velocity DOFs
are eliminated before forming the following operators:

```text
A = D/dt + nu K
A u* = D u^n/dt + F(t_next) - N(u^n)      (one solve per velocity component)
S = B D^-1 B^T
S p = -B u*/dt
u_next = u* + dt D^-1 B^T p
```

Here the weak gradient is `-B^T`, explaining the correction's plus sign. S is
the mass-lumped discrete pressure-Poisson/Schur operator. It is deliberately
formed from the same divergence and mass operators used in the correction;
substituting an independently assembled pressure Laplacian would lose the
exact discrete projection property. This is a discrete realization of the
fractional-step idea, not a verbatim transcription of the book's continuum
boundary-value problem.

One pressure DOF is pinned symmetrically for the solve; pressure is then shifted
to zero integral mean. The compatible RHS and constant nullspace are checked.
The sampled initial velocity is projected once before the time loop. Both
operators use CG with symmetric AMG smoothing, a zero initial guess for each
RHS, relative tolerance 1e-9 and absolute tolerance 1e-13. Failed solves raise
an error; there is no silent direct-solver fallback. A conservative
explicit-convection CFL guard rejects overly large time steps.

For fixed mesh, viscosity and dt, A and S are constant. The ANN is called once
per operator, and two AMG hierarchies are reused throughout the simulation
(including both velocity components). The code reports both `weak_divergence`
and the physical `divergence_l2`; the former is near solver tolerance, while
the latter decreases with mesh refinement. Weak incompressibility does not
mean pointwise divergence is zero.

## Learning and timing

The CNN receives a 16x16 four-channel matrix image (sum, positive maximum,
negative magnitude maximum, stored-nonzero count), each channel signed-log
normalized. The balanced contiguous pooling bins follow the original
`common/view_maker.h` idea; this implementation eliminates explicit zero entries.
Scalar features are scaled log2(number of DOFs), theta, operator role, and
log1p(number of anticipated RHS solves). The network predicts normalized cost
for candidates `[0.1, 0.3, 0.5, 0.7, 0.9]`; inference selects the minimum.

Training labels measure **one AMG setup plus the entire captured RHS sequence**,
with hierarchy reuse. Candidate order is randomized and three repetitions are
reduced to their median. Labels are min–max normalized per matrix. This
accounts for reuse rather than charging setup at every step. Repeated pressure
matrices are deduplicated; the first captured trajectory represents each one.
The matrix-only representation does not model arbitrary RHS-dependent effects.

The small preset generates 18 distinct matrix groups / 90 candidate samples:
training meshes have 6 and 8 cells per side, validation uses 10, and the default
comparison uses a held-out 12-cell mesh and viscosity 0.05. Split membership is
by mesh, with matrix fingerprints checked for cross-split duplicates. Checkpoint
selection uses validation MSE, not the held-out comparison. This is an integration
dataset, far too small to establish generalization or a performance advantage.

Comparison alternates fixed and ANN run order, takes five repetitions, verifies
the solutions agree, and reports median full-simulation wall time. It includes
assembly, diagnostics, model loading, pooling/inference, AMG setup and solves.
Python imports/backend initialization are warmed equally; process startup,
Docker startup, output writing and offline training/data collection are excluded.
The report also includes separate setup, solve and selection timings. A speedup
below 1 means the ANN path was slower; no positive speedup is assumed.

## Validation and scope

```bash
PYTHONPATH=code/navier-stokes OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest code/navier-stokes/tests -q

docker run --rm -v "$PWD":/workspace -e NS_TEST_BACKEND=petsc \
  --entrypoint python3 amg-ann-ns -m pytest code/navier-stokes/tests -q
```

Tests cover pressure nullspace and projection, wall constraints, AMG agreement
with a direct reference, hierarchy reuse, threshold-selector integration,
manufactured-solution refinement, CFL rejection, pooling, portable inference
equivalence and output preservation. The torch-specific export test runs on the
host and is skipped in the small solver-only container.

This first implementation supports a 2D structured square, homogeneous no-slip
walls, constant viscosity, fixed dt and a manufactured forcing. It is serial;
it does not yet implement arbitrary geometries, inlet/outlet conditions, 3D,
adaptive meshes or turbulent flows. Broader training, RHS diversity, longer
trajectories and larger held-out systems are required before claiming an
ANN speedup or using this as a general-purpose flow solver.

References: [scikit-fem](https://scikit-fem.readthedocs.io/en/latest/extended.html),
[PETSc hypre options](https://petsc.org/release/manualpages/PC/PCHYPRE/),
[PyAMG classical AMG](https://pyamg.readthedocs.io/en/latest/generated/pyamg.classical.html).
