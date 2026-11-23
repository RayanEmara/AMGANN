"""Q2/Q1, first-order, mass-lumped Chorin projection on a no-slip square.

With B = (q, div v) and diagonal lumped velocity mass D, the discrete
pressure operator is S = B D^-1 B^T.  Using this SAME operator in the
correction guarantees B U_new = 0 to the linear solve tolerance.  It is
the discrete pressure-Poisson operator, not an independently assembled
Laplacian followed by an inconsistent gradient correction.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import diags
from skfem import Basis, BilinearForm, LinearForm, MeshQuad, ElementQuad1, ElementQuad2, asm
from skfem.models.poisson import mass, laplace

from .linear import AMGSolver


@dataclass(frozen=True)
class Config:
    cells: int = 12
    dt: float = 0.002
    steps: int = 5
    viscosity: float = 0.05
    rtol: float = 1e-9

    def __post_init__(self):
        if (self.cells < 2 or self.steps < 1 or self.dt <= 0 or self.viscosity <= 0
                or not np.isfinite([self.dt, self.viscosity]).all()):
            raise ValueError("Require cells >= 2, steps >= 1, dt > 0, viscosity > 0")
        if not 0 < self.rtol < 1:
            raise ValueError("rtol must lie between zero and one")


def exact(x, t):
    px, py = np.pi * x[0], np.pi * x[1]
    decay = np.exp(-t)
    u = np.pi * np.sin(px)**2 * np.sin(2 * py) * decay
    v = -np.pi * np.sin(2 * px) * np.sin(py)**2 * decay
    p = np.cos(px) * np.cos(py) * decay
    return u, v, p


def forcing(x, t, nu):
    px, py = np.pi * x[0], np.pi * x[1]
    sx, sy = np.sin(px), np.sin(py)
    ax, ay = np.sin(2 * px), np.sin(2 * py)
    bx, by = np.cos(2 * px), np.cos(2 * py)
    e = np.exp(-t)
    u, v, _ = exact(x, t)
    ux, uy = np.pi**2 * ax * ay * e, 2 * np.pi**2 * sx**2 * by * e
    vx, vy = -2 * np.pi**2 * bx * sy**2 * e, -np.pi**2 * ax * ay * e
    lap_u = 2 * np.pi**3 * (bx - 2 * sx**2) * ay * e
    lap_v = 2 * np.pi**3 * ax * (2 * sy**2 - by) * e
    return (-u + u * ux + v * uy - nu * lap_u - np.pi * sx * np.cos(py) * e,
            -v + u * vx + v * vy - nu * lap_v - np.pi * np.cos(px) * sy * e)


@BilinearForm
def divergence_x(u, q, w):
    return u.grad[0] * q


@BilinearForm
def divergence_y(u, q, w):
    return u.grad[1] * q


@LinearForm
def load(v, w):
    return w.value * v


class Discretization:
    def __init__(self, config):
        self.config = config
        nodes = np.linspace(0, 1, config.cells + 1)
        mesh = MeshQuad.init_tensor(nodes, nodes)
        self.velocity = Basis(mesh, ElementQuad2(), intorder=6)
        self.pressure = Basis(mesh, ElementQuad1(), intorder=6)
        self.boundary = self.velocity.get_dofs().all()
        self.free = np.setdiff1d(np.arange(self.velocity.N), self.boundary)
        consistent_mass = asm(mass, self.velocity)
        self.lump = np.asarray(consistent_mass.sum(axis=1)).ravel()[self.free]
        if np.any(self.lump <= 0):
            raise ValueError("Lumped mass must be positive")
        stiffness = asm(laplace, self.velocity)[self.free][:, self.free]
        self.predictor = (diags(self.lump / config.dt) + config.viscosity * stiffness).tocsr()
        self.bx = asm(divergence_x, self.velocity, self.pressure)[:, self.free].tocsr()
        self.by = asm(divergence_y, self.velocity, self.pressure)[:, self.free].tocsr()
        self.schur = (self.bx @ diags(1 / self.lump) @ self.bx.T
                      + self.by @ diags(1 / self.lump) @ self.by.T).tocsr()
        # Pressure p_0 = 0 removes its constant nullspace, preserving symmetry.
        self.pressure_matrix = self.schur[1:, 1:].tocsr()
        self.pressure_weights = np.asarray(asm(mass, self.pressure).sum(axis=1)).ravel()

    def expand(self, values):
        result = np.zeros(self.velocity.N)
        result[self.free] = values
        return result

    def project(self, u, v, solver):
        divergence = self.bx @ u + self.by @ v
        # Compatibility is inherited from zero normal boundary velocity.
        if abs(divergence.sum()) > 1e-10 * max(1.0, np.linalg.norm(divergence)):
            raise RuntimeError("Incompatible pressure right-hand side")
        rhs = -divergence[1:] / self.config.dt
        p = np.zeros(self.pressure.N)
        p[1:] = solver(rhs)
        u = u + self.config.dt * (self.bx.T @ p) / self.lump
        v = v + self.config.dt * (self.by.T @ p) / self.lump
        p -= np.dot(self.pressure_weights, p) / self.pressure_weights.sum()
        return u, v, p

    def diagnostics(self, u, v, p, t):
        uf = self.velocity.interpolate(self.expand(u))
        vf = self.velocity.interpolate(self.expand(v))
        ue, ve, _ = exact(self.velocity.global_coordinates(), t)
        ph = self.pressure.interpolate(p)
        _, _, pe = exact(self.pressure.global_coordinates(), t)
        return {
            "time": t,
            "velocity_l2_error": float(np.sqrt(np.sum(((uf - ue)**2 + (vf - ve)**2) * self.velocity.dx))),
            "pressure_l2_error": float(np.sqrt(np.sum((ph - pe)**2 * self.pressure.dx))),
            "weak_divergence": float(np.linalg.norm(self.bx @ u + self.by @ v)),
            "divergence_l2": float(np.sqrt(np.sum((uf.grad[0] + vf.grad[1])**2 * self.velocity.dx))),
            "kinetic_energy": float(0.5 * np.sum((uf**2 + vf**2) * self.velocity.dx)),
        }


def simulate(config, backend="petsc", selector=None, capture=False):
    start = perf_counter()
    d = Discretization(config)
    assembly_seconds = perf_counter() - start
    matrices = {"velocity": d.predictor, "pressure": d.pressure_matrix}
    selection_seconds = {}
    solvers = {}
    snapshots = {role: [] for role in matrices}
    try:
        for role, matrix in matrices.items():
            ts = perf_counter()
            horizon = 2 * config.steps if role == "velocity" else config.steps + 1
            theta = selector(matrix, role, horizon) if selector else 0.5
            selection_seconds[role] = perf_counter() - ts
            solvers[role] = AMGSolver(matrix, theta, backend, config.rtol)

        def solve(role, rhs):
            if capture:
                snapshots[role].append(rhs.copy())
            return solvers[role].solve(rhs)

        u, v, _ = exact(d.velocity.doflocs[:, d.free], 0)
        # Make the sampled initial condition discretely divergence-free.
        u, v, _ = d.project(u, v, lambda rhs: solve("pressure", rhs))
        diagnostics = []
        for step in range(1, config.steps + 1):
            cfl = 2 * config.cells * config.dt * (np.max(np.abs(u)) + np.max(np.abs(v)))
            if cfl > 0.5:
                raise ValueError(f"Explicit-convection CFL {cfl:.3g} exceeds 0.5; reduce dt")
            t = step * config.dt
            uf = d.velocity.interpolate(d.expand(u))
            vf = d.velocity.interpolate(d.expand(v))
            fu, fv = forcing(d.velocity.global_coordinates(), t, config.viscosity)
            cu = uf * uf.grad[0] + vf * uf.grad[1]
            cv = uf * vf.grad[0] + vf * vf.grad[1]
            rhs_u = d.lump * u / config.dt + asm(load, d.velocity, value=fu - cu)[d.free]
            rhs_v = d.lump * v / config.dt + asm(load, d.velocity, value=fv - cv)[d.free]
            us, vs = solve("velocity", rhs_u), solve("velocity", rhs_v)
            u, v, p = d.project(us, vs, lambda rhs: solve("pressure", rhs))
            diagnostics.append({"step": step, "cfl": float(cfl), **d.diagnostics(u, v, p, t)})
        elapsed = perf_counter() - start
        result = {
            "config": asdict(config), "backend": backend,
            "wall_seconds": elapsed, "assembly_seconds": assembly_seconds,
            "selection_seconds": selection_seconds,
            "linear": {role: solver.summary() for role, solver in solvers.items()},
            "diagnostics": diagnostics,
            "dofs": {"velocity_per_component": int(d.velocity.N), "pressure": int(d.pressure.N)},
        }
        fields = {"velocity_coordinates": d.velocity.doflocs,
                  "pressure_coordinates": d.pressure.doflocs,
                  "u": d.expand(u), "v": d.expand(v), "p": p}
        return result, fields, (matrices, snapshots)
    finally:
        for solver in solvers.values():
            solver.close()
