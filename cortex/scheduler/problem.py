"""
cortex.scheduler.problem
========================
Encodes the quantum job scheduling problem as a QUBO (Quadratic Unconstrained
Binary Optimization) and then maps it to a QAOA cost Hamiltonian.

The scheduling problem
----------------------
Given:
  - N jobs with priority weights w_i and estimated QPU time t_i
  - M backends (QPU nodes) with capacity c_j and error rate e_j

Find: optimal assignment of jobs to backends that maximizes:
  sum(w_i * x_ij) − penalty * overload(backend_j)

This is a weighted bin-packing variant, NP-hard classically for large N.
QAOA finds approximate solutions in polynomial circuit depth.

QUBO encoding
-------------
Binary variable x_ij = 1 if job i is assigned to backend j.

Objective (maximize → negate for minimization):
  −sum_ij (w_i / e_j) * x_ij          ← high-priority jobs to low-error QPUs

Constraints (as penalty terms):
  +A * sum_j(x_ij − 1)²               ← each job assigned exactly once
  +B * sum_i(t_i * x_ij − c_j)²       ← backend capacity not exceeded

The full QUBO matrix Q encodes both objective and constraints.
The cost Hamiltonian H_C = sum_ij Q_ij * Z_i ⊗ Z_j is what QAOA optimizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple


# ── Problem data classes ──────────────────────────────────────────────────────

@dataclass
class SchedulingJob:
    """A job to be scheduled onto a backend."""
    id: str
    priority: float          # 1–10
    estimated_shots: int     # proxy for QPU time cost
    num_qubits: int = 2

    @property
    def weight(self) -> float:
        """Normalized weight: higher priority = more valuable to schedule."""
        return self.priority / 10.0

    @property
    def time_cost(self) -> float:
        """Normalized time cost for capacity constraints."""
        return math.log1p(self.estimated_shots) / math.log1p(100_000)


@dataclass
class QPUBackend:
    """A QPU backend available for job execution."""
    id: str
    name: str
    capacity: float          # max concurrent normalized load (0–1)
    error_rate: float        # two-qubit gate error rate (lower = better)
    available: bool = True

    @property
    def quality(self) -> float:
        """Backend quality: inverse of error rate, clipped."""
        return 1.0 / max(self.error_rate, 1e-4)


# ── QUBO formulation ──────────────────────────────────────────────────────────

class QUBOProblem(NamedTuple):
    """
    QUBO matrix Q where the objective is min x^T Q x.
    Variables x are binary, indexed as x[i*M + j] for job i, backend j.
    """
    Q: list[list[float]]     # n_vars × n_vars matrix
    n_jobs: int
    n_backends: int
    var_names: list[str]     # human-readable label per variable

    @property
    def n_vars(self) -> int:
        return self.n_jobs * self.n_backends

    def var_index(self, job_i: int, backend_j: int) -> int:
        return job_i * self.n_backends + backend_j

    def decode(self, bitstring: str) -> dict[str, str]:
        """
        Decode a QAOA result bitstring into a job→backend assignment.

        Args:
            bitstring: e.g. "100010" for 2 jobs × 3 backends

        Returns:
            dict mapping job_id → backend_id (or "unassigned")
        """
        bits = [int(b) for b in bitstring]
        assignment: dict[str, str] = {}
        for i in range(self.n_jobs):
            job_label = self.var_names[i * self.n_backends].split("→")[0].strip()
            assigned = "unassigned"
            for j in range(self.n_backends):
                idx = i * self.n_backends + j
                if idx < len(bits) and bits[idx] == 1:
                    backend_label = self.var_names[idx].split("→")[1].strip()
                    assigned = backend_label
                    break
            assignment[job_label] = assigned
        return assignment


def build_qubo(
    jobs: list[SchedulingJob],
    backends: list[QPUBackend],
    penalty_assignment: float = 3.0,
    penalty_capacity: float = 2.0,
) -> QUBOProblem:
    """
    Build the QUBO matrix for the job scheduling problem.

    Args:
        jobs:               Jobs to schedule
        backends:           Available QPU backends
        penalty_assignment: Lagrange multiplier for "one backend per job"
        penalty_capacity:   Lagrange multiplier for capacity constraints

    Returns:
        QUBOProblem with Q matrix and metadata.
    """
    backends = [b for b in backends if b.available]
    N = len(jobs)
    M = len(backends)
    n = N * M

    def idx(i: int, j: int) -> int:
        return i * M + j

    # Initialize Q as zero matrix
    Q = [[0.0] * n for _ in range(n)]

    # ── Objective: maximize sum_ij (w_i * quality_j) * x_ij ──────────────────
    # Negate because QUBO minimizes
    for i, job in enumerate(jobs):
        for j, backend in enumerate(backends):
            if not backend.available:
                continue
            score = job.weight * backend.quality
            k = idx(i, j)
            Q[k][k] -= score

    # ── Constraint 1: each job assigned to exactly one backend ────────────────
    # Penalize sum_j(x_ij) ≠ 1  →  penalty * (sum_j x_ij - 1)²
    # = penalty * (sum_j x_ij² + 2*sum_{j<j'} x_ij*x_ij' - 2*sum_j x_ij + 1)
    # (constant term dropped; diagonal from x² = x for binary vars)
    for i in range(N):
        for j in range(M):
            k = idx(i, j)
            Q[k][k] += penalty_assignment * (1 - 2)  # diagonal: coeff of x_ij

        for j in range(M):
            for jp in range(j + 1, M):
                k1, k2 = idx(i, j), idx(i, jp)
                Q[k1][k2] += 2 * penalty_assignment   # cross terms

    # ── Constraint 2: backend capacity ───────────────────────────────────────
    # For each backend j: sum_i(t_i * x_ij) ≤ c_j
    # Soft penalty: penalty * max(0, sum_i t_i * x_ij - c_j)²
    # Linearized as quadratic penalty over pairs
    for j, backend in enumerate(backends):
        cap = backend.capacity
        for i, job in enumerate(jobs):
            k = idx(i, j)
            t = job.time_cost
            # Diagonal: t²x_ij - 2*cap*t*x_ij  (from expansion)
            Q[k][k] += penalty_capacity * t * (t - 2 * cap)

        for i in range(N):
            for ip in range(i + 1, N):
                k1, k2 = idx(i, j), idx(ip, j)
                ti, tip = jobs[i].time_cost, jobs[ip].time_cost
                Q[k1][k2] += 2 * penalty_capacity * ti * tip

    # Variable names for readability
    var_names = [
        f"job_{jobs[i].id[:6]} → {backends[j].name}"
        for i in range(N) for j in range(M)
    ]

    return QUBOProblem(Q=Q, n_jobs=N, n_backends=M, var_names=var_names)


# ── Cost evaluation ───────────────────────────────────────────────────────────

def evaluate_assignment(
    bitstring: str,
    qubo: QUBOProblem,
) -> float:
    """
    Evaluate the QUBO cost x^T Q x for a given bitstring assignment.
    Lower is better.
    """
    x = [int(b) for b in bitstring]
    n = qubo.n_vars
    cost = 0.0
    for i in range(n):
        for j in range(n):
            cost += qubo.Q[i][j] * x[i] * x[j]
    return cost


def best_bitstring(
    counts: dict[str, int],
    qubo: QUBOProblem,
) -> str:
    """
    Given QAOA measurement counts, return the bitstring with the lowest QUBO cost.
    This is the quantum approximate solution to the scheduling problem.
    """
    best, best_cost = "", float("inf")
    for bs, count in counts.items():
        if len(bs) != qubo.n_vars:
            continue
        cost = evaluate_assignment(bs, qubo)
        if cost < best_cost:
            best_cost = cost
            best = bs
    return best
