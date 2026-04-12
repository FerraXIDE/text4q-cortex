"""
cortex.cloud.queue
==================
In-memory job queue with async worker pool.

Architecture:
    JobQueue (asyncio.Queue + priority sort)
        └── N Workers (asyncio tasks)
                └── each worker calls Cortex.run() for one job at a time

For production, replace the in-memory store with Redis + Postgres.
The interface stays identical — only the storage backend changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterator

from cortex.cloud.models import Job, JobStatus, JobPriority, QueueStats

logger = logging.getLogger(__name__)


class JobQueue:
    """
    Async, priority-aware job queue with configurable worker pool.

    Usage:
        queue = JobQueue(num_workers=4)
        await queue.start()

        job_id = await queue.submit(job)
        job    = await queue.get(job_id)

        await queue.stop()
    """

    def __init__(self, num_workers: int = 2):
        self._num_workers = num_workers
        self._jobs: dict[str, Job] = {}            # job_id → Job
        self._queue: asyncio.PriorityQueue = None  # type: ignore
        self._workers: list[asyncio.Task] = []
        self._rate_counters: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the worker pool. Must be called before submitting jobs."""
        if self._started:
            return
        self._queue = asyncio.PriorityQueue()
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"cortex-worker-{i}")
            for i in range(self._num_workers)
        ]
        self._started = True
        logger.info(f"JobQueue started with {self._num_workers} workers")

    async def stop(self) -> None:
        """Gracefully stop all workers."""
        for _ in self._workers:
            await self._queue.put((0, None))   # sentinel
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False
        logger.info("JobQueue stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    async def submit(self, job: Job) -> str:
        """
        Add a job to the queue.
        Priority queue uses (negative_priority, created_at) so higher
        priority jobs are dequeued first, with FIFO tiebreak.
        """
        async with self._lock:
            self._jobs[job.id] = job

        sort_key = (-job.priority.value, job.created_at.timestamp())
        await self._queue.put((sort_key, job.id))
        logger.info(f"Job {job.id[:8]} queued (priority={job.priority.name})")
        return job.id

    async def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a QUEUED job. Cannot cancel running/done jobs."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.QUEUED:
                return False
            job.status = JobStatus.CANCELLED
        return True

    def list_jobs(
        self,
        user_id: str | None = None,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[Job]:
        jobs = list(self._jobs.values())
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def stats(self) -> QueueStats:
        counts: dict[str, int] = defaultdict(int)
        for job in self._jobs.values():
            counts[job.status.value] += 1
        return QueueStats(
            queued=counts["queued"],
            running=counts["running"],
            done=counts["done"],
            failed=counts["failed"],
            total=len(self._jobs),
            workers=self._num_workers,
        )

    def check_rate_limit(self, user_id: str, quota: int = 60) -> bool:
        """
        Return True if user is within their hourly job quota.
        Sliding window: counts jobs submitted in the last 3600 seconds.
        """
        now = time.time()
        window = [t for t in self._rate_counters[user_id] if now - t < 3600]
        self._rate_counters[user_id] = window
        if len(window) >= quota:
            return False
        self._rate_counters[user_id].append(now)
        return True

    # ── Worker ────────────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        logger.debug(f"Worker {worker_id} started")
        while True:
            _, job_id = await self._queue.get()

            # Sentinel → shutdown
            if job_id is None:
                self._queue.task_done()
                break

            job = self._jobs.get(job_id)
            if not job or job.status == JobStatus.CANCELLED:
                self._queue.task_done()
                continue

            logger.info(f"Worker {worker_id} running job {job_id[:8]}")
            await self._execute_job(job)
            self._queue.task_done()

        logger.debug(f"Worker {worker_id} stopped")

    async def _execute_job(self, job: Job) -> None:
        """Execute a single job using Cortex and update its state."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_cortex, job
            )
            job.status = JobStatus.DONE
            job.qasm = result.qasm
            job.counts = result.counts
            job.circuit_type = result.intent.circuit_type
            job.num_qubits = result.intent.num_qubits
            job.execution_time_ms = result.execution_time_ms
            if result.error:
                job.status = JobStatus.FAILED
                job.error = result.error

        except Exception as exc:
            logger.exception(f"Job {job.id[:8]} failed: {exc}")
            job.status = JobStatus.FAILED
            job.error = str(exc)

        finally:
            job.finished_at = datetime.now(timezone.utc)
            logger.info(
                f"Job {job.id[:8]} → {job.status.value} "
                f"({job.duration_ms:.0f}ms)" if job.duration_ms else ""
            )

    @staticmethod
    def _run_cortex(job: Job):
        """Synchronous Cortex execution (runs in thread pool executor)."""
        from cortex.core import Cortex
        cx = Cortex(backend=job.backend, nlp=job.nlp_mode)
        return cx.run(job.text)
