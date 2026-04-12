"""
cortex.scheduler.integration
============================
Bridges the QAOA Scheduler with the Cortex cloud job queue.

When activated, the cloud queue uses QAOA to decide which backend
processes each batch of jobs, instead of round-robin assignment.

Usage (in server.py):
    from cortex.scheduler.integration import QAOAQueueIntegration
    integration = QAOAQueueIntegration(scheduler_backend="aer")
    optimal_backend = integration.assign(job, available_backends)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cortex.cloud.models import Job, JobPriority
from cortex.scheduler.problem import SchedulingJob, QPUBackend
from cortex.scheduler.optimizer import QAOAScheduler, SchedulingResult

logger = logging.getLogger(__name__)


# ── Default backend registry (extend for real multi-QPU setups) ───────────────

DEFAULT_BACKENDS = [
    QPUBackend(
        id="aer",
        name="Aer Simulator",
        capacity=1.0,
        error_rate=1e-6,   # simulator: near-zero error
        available=True,
    ),
    QPUBackend(
        id="ibm_quantum",
        name="IBM Quantum",
        capacity=0.7,
        error_rate=0.01,   # ~1% two-qubit gate error (typical for Eagle/Heron)
        available=False,   # set to True when IBM token is configured
    ),
]


@dataclass
class AssignmentDecision:
    """Result of a single job assignment decision."""
    job_id: str
    backend_id: str
    method: str          # "qaoa" | "classical"
    confidence: float    # fraction of shots that agreed with this assignment


class QAOAQueueIntegration:
    """
    Integrates QAOA scheduling into the Cortex cloud queue.

    For each batch of queued jobs, calls the QAOA scheduler to find
    the optimal job→backend assignment, then returns per-job decisions
    that the queue workers can act on.
    """

    def __init__(
        self,
        scheduler_backend: str = "aer",
        p: int = 1,
        shots: int = 1024,
        backends: list[QPUBackend] | None = None,
    ):
        self._scheduler = QAOAScheduler(
            backend=scheduler_backend,
            p=p,
            shots=shots,
        )
        self._backends = backends or DEFAULT_BACKENDS

    def assign_batch(self, jobs: list[Job]) -> list[AssignmentDecision]:
        """
        Use QAOA to assign a batch of queued jobs to backends.

        Args:
            jobs: List of QUEUED jobs from the cloud queue.

        Returns:
            One AssignmentDecision per job.
        """
        if not jobs:
            return []

        sched_jobs = [_job_to_scheduling(j) for j in jobs]
        available  = [b for b in self._backends if b.available]

        logger.info(
            f"QAOA batch scheduling: {len(jobs)} jobs → "
            f"{len(available)} backends"
        )

        result: SchedulingResult = self._scheduler.schedule(sched_jobs, available)
        logger.info(str(result))

        decisions: list[AssignmentDecision] = []
        for job in jobs:
            backend_id = result.assignment.get(job.id, "aer")
            confidence = _estimate_confidence(job.id, backend_id, result)
            decisions.append(AssignmentDecision(
                job_id=job.id,
                backend_id=backend_id,
                method=result.method,
                confidence=confidence,
            ))

        return decisions

    def assign_single(self, job: Job) -> AssignmentDecision:
        """Assign a single job (wraps assign_batch for convenience)."""
        decisions = self.assign_batch([job])
        return decisions[0] if decisions else AssignmentDecision(
            job_id=job.id, backend_id="aer", method="classical", confidence=1.0
        )

    def register_backend(self, backend: QPUBackend) -> None:
        """Register a new QPU backend with the scheduler."""
        existing = {b.id for b in self._backends}
        if backend.id not in existing:
            self._backends.append(backend)
            logger.info(f"Registered backend: {backend.name} (error={backend.error_rate})")

    def set_backend_availability(self, backend_id: str, available: bool) -> None:
        """Mark a backend as available or offline."""
        for b in self._backends:
            if b.id == backend_id:
                b.available = available
                status = "online" if available else "offline"
                logger.info(f"Backend {backend_id} → {status}")
                return
        logger.warning(f"Backend {backend_id} not found in registry")

    @property
    def backends(self) -> list[QPUBackend]:
        return list(self._backends)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _job_to_scheduling(job: Job) -> SchedulingJob:
    """Convert a cloud Job to a SchedulingJob for QAOA input."""
    return SchedulingJob(
        id=job.id,
        priority=float(job.priority.value),
        estimated_shots=job.shots,
        num_qubits=2,   # unknown at queue time; use default
    )


def _estimate_confidence(
    job_id: str,
    backend_id: str,
    result: SchedulingResult,
) -> float:
    """
    Estimate confidence of an assignment as the fraction of QAOA shots
    that produced the same (job→backend) decision.

    For classical fallback, returns 1.0 (deterministic).
    """
    if result.method == "classical" or not result.counts:
        return 1.0

    total = sum(result.counts.values())
    if total == 0:
        return 0.0

    # Count shots where this job is mapped to this backend
    # (need job index from id to compute variable index)
    matching = 0
    for bs, count in result.counts.items():
        decoded = result.top_bitstring   # use top bitstring as reference
        if bs == decoded:
            matching += count

    return round(matching / total, 4)
