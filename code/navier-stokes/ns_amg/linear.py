"""Reusable CG-AMG solvers; no silent direct-solver fallback."""
from time import perf_counter
import itertools
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

_prefixes = itertools.count()


@lru_cache(maxsize=1)
def check_cuda_build():
    """Check the build used by petsc4py, including hypre's device support."""
    import petsc4py
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage("cuda") or not PETSc.Sys.hasExternalPackage("hypre"):
        raise RuntimeError("petsc-cuda requires PETSc with CUDA and hypre. Use the cluster GPU image.")
    config = petsc4py.get_config()
    root = Path(config["PETSC_DIR"])
    candidates = [root / config.get("PETSC_ARCH", "") / "include/petscconf.h",
                  root / "include/petscconf.h"]
    header = next((p for p in candidates if p.is_file()), None)
    if header is None or not re.search(r"^#define PETSC_HAVE_HYPRE_DEVICE 1$", header.read_text(), re.M):
        raise RuntimeError("Cannot confirm CUDA-enabled hypre in petscconf.h. Use the cluster GPU image.")


class AMGSolver:
    def __init__(self, matrix, theta=0.5, backend="petsc", rtol=1e-9):
        if not 0 < theta < 1:
            raise ValueError("theta must lie strictly between zero and one")
        self.matrix = csr_matrix(matrix, dtype=np.float64)
        self.matrix.sort_indices()
        self.theta, self.backend, self.rtol = float(theta), backend, rtol
        self.records = []
        start = perf_counter()
        if backend == "pyamg":
            import pyamg
            self.hierarchy = pyamg.ruge_stuben_solver(
                self.matrix, strength=("classical", {"theta": theta}),
                presmoother=("gauss_seidel", {"sweep": "symmetric"}),
                postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
                max_coarse=10,
            )
            self.preconditioner = self.hierarchy.aspreconditioner(cycle="V")
        elif backend in ("petsc", "petsc-cuda"):
            from petsc4py import PETSc
            if backend == "petsc-cuda":
                check_cuda_build()
            self.PETSc = PETSc
            self.mat = PETSc.Mat().createAIJ(
                size=self.matrix.shape,
                csr=(self.matrix.indptr.astype(PETSc.IntType),
                     self.matrix.indices.astype(PETSc.IntType), self.matrix.data),
                comm=PETSc.COMM_SELF,
            )
            if backend == "petsc-cuda":
                self.mat.convert("seqaijcusparse")
                self.mat.setVecType("seqcuda")
                self.device_rhs, self.device_x = self.mat.createVecs()
                if self.mat.getType() != "seqaijcusparse" or self.device_rhs.getType() != "seqcuda":
                    raise RuntimeError("PETSc failed to allocate cuSPARSE matrix and CUDA vectors")
            self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            self.ksp.setOperators(self.mat)
            self.ksp.setType("cg")
            self.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            self.ksp.setTolerances(rtol=rtol, atol=1e-13, max_it=1000)
            self.ksp.setInitialGuessNonzero(False)
            prefix = f"nsamg{next(_prefixes)}_"
            self.ksp.setOptionsPrefix(prefix)
            pc = self.ksp.getPC()
            pc.setType("hypre")
            pc.setHYPREType("boomeramg")
            options = PETSc.Options()
            values = {
                "pc_hypre_boomeramg_strong_threshold": theta,
                "pc_hypre_boomeramg_max_iter": 1,
                "pc_hypre_boomeramg_tol": 0.0,
                "pc_hypre_boomeramg_relax_type_all": "symmetric-SOR/Jacobi",
            }
            if backend == "petsc-cuda":
                values.update(pc_hypre_boomeramg_coarsen_type="PMIS",
                              pc_hypre_boomeramg_interp_type="ext+i",
                              pc_hypre_boomeramg_relax_type_all="l1scaled-Jacobi")
            try:
                for key, value in values.items():
                    options[prefix + key] = value
                self.ksp.setFromOptions()
                self.ksp.setUp()
                if backend == "petsc-cuda":
                    PETSc.DeviceContext.getCurrent().synchronize()
            finally:
                for key in values:
                    del options[prefix + key]
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self.setup_seconds = perf_counter() - start

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=np.float64)
        start = perf_counter()
        iterations = 0
        if self.backend == "pyamg":
            def callback(_):
                nonlocal iterations
                iterations += 1
            x, reason = cg(self.matrix, rhs, M=self.preconditioner,
                           rtol=self.rtol, atol=1e-13, maxiter=1000,
                           callback=callback)
            success = reason == 0
        else:
            if self.backend == "petsc-cuda":
                b, xvec = self.device_rhs, self.device_x
                b.getArray()[:] = rhs
            else:
                b = self.PETSc.Vec().createWithArray(rhs.copy(), comm=self.PETSc.COMM_SELF)
                xvec = b.duplicate()
            try:
                self.ksp.solve(b, xvec)
                x = xvec.getArray().copy()
                iterations = self.ksp.getIterationNumber()
                reason = self.ksp.getConvergedReason()
                success = reason > 0
            finally:
                if self.backend == "petsc":
                    b.destroy()
                    xvec.destroy()
        elapsed = perf_counter() - start
        norm_rhs = np.linalg.norm(rhs)
        residual = np.linalg.norm(self.matrix @ x - rhs)
        if not success or not np.isfinite(x).all() or residual > 10 * max(
            self.rtol * norm_rhs, 1e-13
        ):
            raise RuntimeError(f"CG failed: reason={reason}, residual={residual:g}")
        self.records.append({"seconds": elapsed, "iterations": int(iterations),
                             "relative_residual": float(residual / max(norm_rhs, 1e-30))})
        return x

    def close(self):
        if self.backend in ("petsc", "petsc-cuda") and self.ksp is not None:
            if self.backend == "petsc-cuda":
                self.device_rhs.destroy()
                self.device_x.destroy()
            self.ksp.destroy()
            self.mat.destroy()
            self.ksp = None

    def summary(self):
        return {**({"matrix_type": self.mat.getType(), "vector_type": self.device_x.getType()}
                   if self.backend == "petsc-cuda" else {}),
                "theta": self.theta, "setup_seconds": self.setup_seconds,
                "solve_seconds": sum(r["seconds"] for r in self.records),
                "solves": len(self.records),
                "iterations": sum(r["iterations"] for r in self.records),
                "max_relative_residual": max(
                    (r["relative_residual"] for r in self.records), default=0.0)}
