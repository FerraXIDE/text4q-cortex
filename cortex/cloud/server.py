"""
cortex.cloud.server
===================
FastAPI server exposing the Cortex job queue as a REST API.

Endpoints:
    POST   /jobs              Submit a new quantum job
    GET    /jobs              List jobs (filterable by status)
    GET    /jobs/{id}         Get job details + results
    DELETE /jobs/{id}         Cancel a queued job
    GET    /queue/stats       Queue and worker statistics
    GET    /health            Health check

Auth: API key via X-API-Key header (or ?api_key= query param for testing).

Run with:
    uvicorn cortex.cloud.server:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from cortex.cloud.models import (
    Job, JobSubmit, JobStatus, JobPriority,
    JobResponse, QueueStats, User,
)
from cortex.cloud.queue import JobQueue

logger = logging.getLogger(__name__)

# ── Global queue instance ─────────────────────────────────────────────────────

_queue: JobQueue | None = None

def get_queue() -> JobQueue:
    if _queue is None:
        raise RuntimeError("Queue not initialized")
    return _queue


# ── Seed users (replace with DB in production) ────────────────────────────────

_USERS: dict[str, User] = {
    "dev-key-0000": User(
        id="user-dev",
        username="dev",
        api_key="dev-key-0000",
        is_admin=True,
        quota_jobs_per_hour=1000,
    ),
    "demo-key-1111": User(
        id="user-demo",
        username="demo",
        api_key="demo-key-1111",
        quota_jobs_per_hour=20,
    ),
}

def get_user(api_key: str) -> User | None:
    return _USERS.get(api_key)


# ── Auth dependency ───────────────────────────────────────────────────────────

def require_user(
    x_api_key: Annotated[str | None, Header()] = None,
    api_key: Annotated[str | None, Query()] = None,
) -> User:
    key = x_api_key or api_key
    if not key:
        raise HTTPException(status_code=401, detail="API key required")
    user = get_user(key)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return user

def require_admin(user: User = Depends(require_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ── Lifespan: start/stop the queue ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queue
    num_workers = int(os.environ.get("CORTEX_WORKERS", "2"))
    _queue = JobQueue(num_workers=num_workers)
    await _queue.start()
    logger.info(f"Cortex cloud server started (workers={num_workers})")
    yield
    await _queue.stop()
    logger.info("Cortex cloud server stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="text4q Cortex Cloud API",
    description="Natural language quantum computing job queue",
    version="0.2.0-alpha",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    q = get_queue()
    stats = q.stats()
    return {
        "status": "ok",
        "version": "0.2.0-alpha",
        "queue": stats.model_dump(),
    }


@app.post("/jobs", response_model=JobResponse, status_code=202)
async def submit_job(
    payload: JobSubmit,
    user: User = Depends(require_user),
    queue: JobQueue = Depends(get_queue),
):
    """Submit a quantum job to the execution queue."""
    # Rate limiting
    if not queue.check_rate_limit(user.id, user.quota_jobs_per_hour):
        raise HTTPException(
            status_code=429,
            detail=f"Hourly quota of {user.quota_jobs_per_hour} jobs exceeded",
        )

    # Validate backend
    allowed_backends = {"aer", "ibm_quantum"}
    if payload.backend not in allowed_backends:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend '{payload.backend}'. Allowed: {allowed_backends}",
        )

    job = Job(
        user_id=user.id,
        text=payload.text,
        shots=payload.shots,
        backend=payload.backend,
        nlp_mode=payload.nlp_mode,
        priority=payload.priority,
        tags=payload.tags,
    )

    job_id = await queue.submit(job)
    return JobResponse(job_id=job_id, status=JobStatus.QUEUED, message="Job accepted")


@app.get("/jobs")
async def list_jobs(
    status: JobStatus | None = None,
    limit: int = Query(default=20, le=100),
    user: User = Depends(require_user),
    queue: JobQueue = Depends(get_queue),
):
    """List jobs. Admins see all jobs; regular users see only their own."""
    user_filter = None if user.is_admin else user.id
    jobs = queue.list_jobs(user_id=user_filter, status=status, limit=limit)
    return {"jobs": [j.summary() for j in jobs], "total": len(jobs)}


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    user: User = Depends(require_user),
    queue: JobQueue = Depends(get_queue),
):
    """Get full job details, including QASM and measurement results."""
    job = await queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not user.is_admin and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your job")

    return {
        **job.summary(),
        "qasm": job.qasm,
        "counts": job.counts,
        "error": job.error,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@app.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    user: User = Depends(require_user),
    queue: JobQueue = Depends(get_queue),
):
    """Cancel a queued job (cannot cancel running or finished jobs)."""
    job = await queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not user.is_admin and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your job")

    cancelled = await queue.cancel(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in status '{job.status.value}'",
        )
    return JobResponse(job_id=job_id, status=JobStatus.CANCELLED, message="Job cancelled")


@app.get("/queue/stats", response_model=QueueStats)
async def queue_stats(
    _user: User = Depends(require_admin),
    queue: JobQueue = Depends(get_queue),
):
    """Queue statistics (admin only)."""
    return queue.stats()


@app.get("/users/me")
async def me(user: User = Depends(require_user)):
    """Return current user info."""
    return {
        "id": user.id,
        "username": user.username,
        "is_admin": user.is_admin,
        "quota_jobs_per_hour": user.quota_jobs_per_hour,
    }

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the web dashboard (no auth needed)."""
    html_file = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(content=html_file.read_text())

