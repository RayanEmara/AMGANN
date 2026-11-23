"""Collect real solve timings, train cost predictions, and compare full runs."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import save_npz

from .fem import Config, simulate
from .linear import AMGSolver
from .network import Predictor, matrix_view, parameters

CANDIDATES = [0.1, 0.3, 0.5, 0.7, 0.9]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new path.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fingerprint(matrix):
    matrix = matrix.copy().tocsr()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64).tobytes())
    digest.update(matrix.indices.astype(np.int64).tobytes())
    digest.update(matrix.data.astype(np.float64).tobytes())
    return digest.hexdigest()


def versions(backend):
    import skfem
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "scipy": scipy.__version__, "scikit_fem": skfem.__version__}
    if backend in ("petsc", "petsc-cuda"):
        from petsc4py import PETSc
        result["petsc"] = list(PETSc.Sys.getVersion())
    else:
        import pyamg
        result["pyamg"] = pyamg.__version__
    return result


def collect(output, backend="petsc", repeats=3, steps=5, candidates=None, profile="local"):
    output = new_directory(output)
    candidates = list(CANDIDATES if candidates is None else candidates)
    if repeats < 1 or len(candidates) < 2 or not all(0 < x < 1 for x in candidates):
        raise ValueError("Require repeats >= 1 and at least two thresholds in (0, 1)")
    runtime = versions(backend)  # Import/initialize the backend before timing labels.
    rng = np.random.default_rng(7)
    samples, groups, seen = [], [], {}
    # Split by mesh, not by theta or time step: no repeated matrix across splits.
    train_meshes, validation_meshes, dts = {
        "local": ((6, 8), (10,), (0.001, 0.002)),
        "smoke": ((6,), (8,), (0.002,)),
        "full": ((24, 40, 64), (80,), (0.0002, 0.0004)),
    }[profile]
    cases = [("train", Config(cells=n, viscosity=nu, dt=dt, steps=steps))
             for n in train_meshes for nu in (0.01, 0.1, 1.0) for dt in dts]
    cases += [("validation", Config(cells=n, viscosity=nu, dt=dts[-1], steps=steps))
              for n in validation_meshes for nu in (0.01, 0.1, 1.0)]
    for split, config in cases:
        print(f"Collect {split}: {asdict(config)}", flush=True)
        _, _, (matrices, rhs_sets) = simulate(config, backend, capture=True)
        for role, matrix in matrices.items():
            key = fingerprint(matrix)
            if key in seen:
                if seen[key] != split:
                    raise RuntimeError("Identical matrix leaked across dataset splits")
                continue  # Constant pressure operator only needs one trajectory per mesh.
            seen[key] = split
            rhs_set = rhs_sets[role]
            gid = f"{role}-{key[:16]}"
            save_npz(output / f"{gid}.matrix.npz", matrix)
            np.savez_compressed(output / f"{gid}.rhs.npz", rhs=np.stack(rhs_set))
            trials = {t: [] for t in candidates}
            # Warm up this matrix path, including the first solve/coarse solver.
            warm = AMGSolver(matrix, 0.5, backend, config.rtol)
            try:
                warm.solve(rhs_set[0])
            finally:
                warm.close()
            for _ in range(repeats):
                for theta in rng.permutation(candidates):
                    solver = AMGSolver(matrix, float(theta), backend, config.rtol)
                    try:
                        for rhs in rhs_set:
                            solver.solve(rhs)
                        trials[float(theta)].append(solver.summary())
                    finally:
                        solver.close()
            costs = np.array([np.median([r["setup_seconds"] + r["solve_seconds"]
                                         for r in trials[t]]) for t in candidates])
            spread = np.ptp(costs)
            target = (costs - costs.min()) / spread if spread > 0 else np.zeros_like(costs)
            view = matrix_view(matrix)
            for theta, value in zip(candidates, target):
                samples.append((view, parameters(matrix.shape[0], theta, role, len(rhs_set)),
                                value, split, gid))
            groups.append({"id": gid, "fingerprint": key, "split": split, "role": role,
                           "config": asdict(config), "ndof": matrix.shape[0],
                           "horizon": len(rhs_set), "median_cost_seconds": costs.tolist(),
                           "trials": {str(t): trials[t] for t in candidates}})
    np.savez_compressed(output / "training.npz",
                        images=np.stack([s[0] for s in samples]),
                        params=np.stack([s[1] for s in samples]),
                        targets=np.array([s[2] for s in samples], dtype=np.float32),
                        split=np.array([s[3] for s in samples]),
                        group=np.array([s[4] for s in samples]))
    metadata = {"backend": backend, "candidates": candidates, "repeats": repeats,
                "label": "median(setup + sum of cold-start RHS solves with hierarchy reuse)",
                "profile": profile, "training_meshes": train_meshes, "validation_meshes": validation_meshes, "steps": steps,
                "versions": runtime, "groups": groups}
    write_json(output / "metadata.json", metadata)
    return {"matrix_groups": len(groups), "samples": len(samples), "backend": backend}


def compare(config, model, output, backend="petsc", repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output = new_directory(output)
    runtime = versions(backend)
    # Equal warm-up for imports, assembly and AMG. Not included in either timing.
    simulate(config, backend)
    runs = {"fixed": [], "ann": []}
    fields, chosen = {}, []
    for repeat in range(repeats):
        for mode in (("fixed", "ann") if repeat % 2 == 0 else ("ann", "fixed")):
            load_start = perf_counter()
            selector = Predictor(model, backend) if mode == "ann" else None
            load_seconds = perf_counter() - load_start if selector else 0.0
            result, field, _ = simulate(config, backend, selector=selector)
            result["model_load_seconds"] = load_seconds
            result["wall_seconds"] += load_seconds
            runs[mode].append(result)
            fields[mode] = field
            if selector:
                chosen.append(selector.selections)
    velocity_difference = max(float(np.max(np.abs(fields["ann"][c] - fields["fixed"][c])))
                              for c in ("u", "v"))
    pressure_difference = float(np.max(np.abs(fields["ann"]["p"] - fields["fixed"]["p"])))
    if velocity_difference > 1e-6 or pressure_difference > 1e-5:
        raise RuntimeError("ANN and baseline solutions differ beyond linear solver tolerance")
    fixed = float(np.median([r["wall_seconds"] for r in runs["fixed"]]))
    ann = float(np.median([r["wall_seconds"] for r in runs["ann"]]))
    for mode in runs:
        np.savez_compressed(output / f"{mode}.fields.npz", **fields[mode])
    report = {"backend": backend, "versions": runtime, "config": asdict(config),
              "model": str(Path(model).resolve()), "repeats": repeats,
              "timing_scope": "Warm-process full simulation including assembly, diagnostics, pooling, inference, model loading, AMG setup and all solves; excludes file writes and process startup",
              "fixed_median_seconds": fixed, "ann_median_seconds": ann,
              "speedup": fixed / ann, "velocity_max_difference": velocity_difference,
              "pressure_max_difference": pressure_difference,
              "selected_thresholds": chosen, "runs": runs}
    write_json(output / "comparison.json", report)
    return {key: report[key] for key in ("backend", "fixed_median_seconds", "ann_median_seconds",
             "speedup", "velocity_max_difference", "pressure_max_difference")}


def convergence(output, backend="petsc"):
    output = new_directory(output)
    results = []
    for cells in (4, 8, 16):
        # Same final time, with dt reduced proportional to h^2.
        steps = (cells // 4)**2
        config = Config(cells=cells, steps=steps, dt=0.008 / steps, viscosity=0.1)
        result, _, _ = simulate(config, backend)
        results.append({"config": asdict(config), **result["diagnostics"][-1]})
    for previous, current in zip(results, results[1:]):
        for key in ("velocity_l2_error", "pressure_l2_error", "divergence_l2"):
            if not current[key] < previous[key]:
                raise RuntimeError(f"Refinement did not reduce {key}")
    write_json(output / "convergence.json", results)
    return results
