"""
cortex.scheduler.optimizer
==========================
Runs the QAOA circuit on a Cortex backend and extracts the optimal
job-to-QPU assignment from the measurement results.

Pipeline:
    SchedulingJob list + QPUBackend list
        │
        ▼
    build_qubo()          ← formulate as QUBO matrix
        │
        ▼
    build_qaoa_circuit()  ← encode QUBO into OpenQASM 3.0 QAOA circuit
        │
        ▼
    Cortex.run()          ← execute on QPU or Aer simulator
        │
        ▼
    best_bitstring()      ← find lowest-cost measurement outcome
        │
        ▼
    SchedulingResult      ← job_id → backend_id mapping + metadata

Classical fallback
------------------
If n_vars > 20 or the QPU is unavailable, falls back to a greedy
priority-weighted assignment (polynomial time, deterministic).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from cortex.scheduler.problem import (
    SchedulingJob, QPUBackend, QUBOProblem,
    build_qubo, best_bitstring, evaluate_assignment,
)
from cortex.scheduler.qaoa import (
    build_qaoa_circuit, circuit_stats, suggest_parameters,
)

logger = logging.getLogger(__name__)

MAX_QAOA_VARS = 20   # above this, use classical fallback


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class SchedulingResult:
    """Output of the QAOA scheduler."""
    assignment: dict[str, str]           # job.id → backend.id
    method: str                          # "qaoa" | "classical"
    qubo_cost: float                     # QUBO objective value (lower = better)
    n_qubits: int                        # circuit size used
    shots: int
    execution_time_ms: float
    top_bitstring: str
    counts: dict[str, int] = field(default_factory=dict)
    circuit_stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"SchedulingResult ({self.method}, {self.n_qubits}q)"]
        for job_id, backend_id in self.assignment.items():
            lines.append(f"  {job_id[:8]} → {backend_id}")
        lines.append(f"  cost={self.qubo_cost:.4f}  time={self.execution_time_ms:.0f}ms")
        return "\n".join(lines)


# ── Main optimizer ────────────────────────────────────────────────────────────

class QAOAScheduler:
    """
    Quantum-native job scheduler using QAOA.

    Usage:
        scheduler = QAOAScheduler(backend="aer")

        jobs = [SchedulingJob("j1", priority=8, estimated_shots=1024), ...]
        backends = [QPUBackend("b1", "ibm_brisbane", capacity=0.8, error_rate=0.01)]

        result = scheduler.schedule(jobs, backends)
        print(result)
    """

    def __init__(
        self,
        backend: str = "aer",
        p: int = 1,
        shots: int = 2048,
        gamma: float | None = None,
        beta: float | None = None,
        force_classical: bool = False,
    ):
        self.backend = backend
        self.p = p
        self.shots = shots
        self._gamma = gamma
        self._beta  = beta
        self.force_classical = force_classical

    def schedule(
        self,
        jobs: list[SchedulingJob],
        backends: list[QPUBackend],
    ) -> SchedulingResult:
        """
        Find the optimal job-to-backend assignment.

        Automatically uses QAOA if the problem fits within qubit budget,
        otherwise falls back to classical greedy assignment.
        """
        if not jobs:
            return SchedulingResult(
                assignment={}, method="classical", qubo_cost=0.0,
                n_qubits=0, shots=0, execution_time_ms=0.0,
                top_bitstring="",
            )

        available = [b for b in backends if b.available]
        if not available:
            logger.warning("No available backends — all jobs unassigned")
            return SchedulingResult(
                assignment={j.id: "unassigned" for j in jobs},
                method="classical", qubo_cost=float("inf"),
                n_qubits=0, shots=0, execution_time_ms=0.0,
                top_bitstring="", warnings=["No backends available"],
            )

        qubo = build_qubo(jobs, available)
        stats = circuit_stats(qubo, self.p)
        n_vars = qubo.n_vars

        logger.info(
            f"Scheduling {len(jobs)} jobs × {len(available)} backends = "
            f"{n_vars} QUBO vars, ~{stats['total_gates']} gates"
        )

        if self.force_classical or n_vars > MAX_QAOA_VARS:
            warn = f"Problem size {n_vars} > {MAX_QAOA_VARS} vars — using classical fallback"
            logger.warning(warn)
            return self._classical_schedule(jobs, available, qubo, [warn])

        return self._qaoa_schedule(jobs, available, qubo, stats)

    # ── QAOA path ─────────────────────────────────────────────────────────────

    def _qaoa_schedule(
        self,
        jobs: list[SchedulingJob],
        backends: list[QPUBackend],
        qubo: QUBOProblem,
        stats: dict,
    ) -> SchedulingResult:
        from cortex.core import Cortex
        from cortex.models import CircuitIntent

        n = qubo.n_vars
        gamma, beta = self._gamma, self._beta
        if gamma is None or beta is None:
            gamma, beta = suggest_parameters(n, self.p)

        logger.info(f"Building QAOA circuit: {n}q p={self.p} γ={gamma:.4f} β={beta:.4f}")
        qasm = build_qaoa_circuit(qubo, gamma=gamma, beta=beta, p=self.p, shots=self.shots)

        # Build a synthetic CircuitIntent for the connector
        intent = CircuitIntent(
            raw_text=f"QAOA scheduler: {qubo.n_jobs}j × {qubo.n_backends}b",
            num_qubits=n,
            circuit_type="qaoa",
            shots=self.shots,
            metadata={"qaoa_scheduler": True, "p": self.p},
        )

        t0 = time.monotonic()
        try:
            cx = Cortex(backend=self.backend)
            result = cx._connector.execute(intent, qasm)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if not result.success:
                raise RuntimeError(result.error)

            counts = result.counts
            top_bs = best_bitstring(counts, qubo)

            if not top_bs:
                raise RuntimeError("No valid bitstring found in measurement results")

            cost = evaluate_assignment(top_bs, qubo)
            assignment = _decode_assignment(top_bs, jobs, backends)

            logger.info(
                f"QAOA done in {elapsed_ms:.0f}ms — "
                f"best: {top_bs} (cost={cost:.4f})"
            )

            return SchedulingResult(
                assignment=assignment,
                method="qaoa",
                qubo_cost=cost,
                n_qubits=n,
                shots=self.shots,
                execution_time_ms=elapsed_ms,
                top_bitstring=top_bs,
                counts=counts,
                circuit_stats=stats,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            warn = f"QAOA execution failed ({exc}), falling back to classical"
            logger.warning(warn)
            return self._classical_schedule(jobs, backends, qubo, [warn])

    # ── Classical greedy fallback ──────────────────────────────────────────────

    def _classical_schedule(
        self,
        jobs: list[SchedulingJob],
        backends: list[QPUBackend],
        qubo: QUBOProblem,
        warnings: list[str],
    ) -> SchedulingResult:
        """
        Greedy priority-weighted assignment.
        Assigns each job (highest priority first) to the best available backend
        (lowest error rate, most remaining capacity).
        """
        t0 = time.monotonic()

        # Track remaining capacity per backend
        remaining = {b.id: b.capacity for b in backends}
        assignment: dict[str, str] = {}

        # Sort jobs by descending priority
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)

        for job in sorted_jobs:
            # Pick backend: minimize error_rate, subject to capacity
            candidates = [
                b for b in backends
                if b.available and remaining[b.id] >= job.time_cost
            ]
            if not candidates:
                # Relax capacity constraint if no backend fits
                candidates = [b for b in backends if b.available]

            if not candidates:
                assignment[job.id] = "unassigned"
                continue

            best_backend = min(candidates, key=lambda b: b.error_rate)
            assignment[job.id] = best_backend.id
            remaining[best_backend.id] = max(0, remaining[best_backend.id] - job.time_cost)

        # Synthesize a "bitstring" for cost calculation
        n = qubo.n_vars
        bits = ["0"] * n
        backend_ids = [b.id for b in backends]
        for i, job in enumerate(jobs):
            bk_id = assignment.get(job.id)
            if bk_id and bk_id in backend_ids:
                j = backend_ids.index(bk_id)
                bits[i * len(backends) + j] = "1"
        top_bs = "".join(bits)
        cost = evaluate_assignment(top_bs, qubo)

        elapsed_ms = (time.monotonic() - t0) * 1000
        return SchedulingResult(
            assignment=assignment,
            method="classical",
            qubo_cost=cost,
            n_qubits=0,
            shots=0,
            execution_time_ms=elapsed_ms,
            top_bitstring=top_bs,
            warnings=warnings,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_assignment(
    bitstring: str,
    jobs: list[SchedulingJob],
    backends: list[QPUBackend],
) -> dict[str, str]:
    """
    Map a QAOA measurement bitstring to a job→backend assignment dict.
    Variables are ordered x[i*M + j] for job i, backend j.
    """
    M = len(backends)
    bits = [int(b) for b in bitstring]
    result: dict[str, str] = {}

    for i, job in enumerate(jobs):
        assigned = "unassigned"
        for j, backend in enumerate(backends):
            idx = i * M + j
            if idx < len(bits) and bits[idx] == 1:
                assigned = backend.id
                break
        result[job.id] = assigned

    return result
