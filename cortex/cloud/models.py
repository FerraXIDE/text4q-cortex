"""
cortex.cloud.models
===================
Data models for the Cortex cloud layer (job queue, users, results).
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


def _now() -> datetime:
    return datetime.now(timezone.utc)

def _uid() -> str:
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


class JobPriority(int, Enum):
    LOW    = 1
    NORMAL = 5
    HIGH   = 10


# ── Job models ────────────────────────────────────────────────────────────────

class JobSubmit(BaseModel):
    """Payload sent by the client to submit a new job."""
    text: str = Field(..., description="Natural language circuit description")
    shots: int = Field(1024, ge=1, le=100_000)
    backend: str = Field("aer", description="Target backend: aer | ibm_quantum")
    nlp_mode: str = Field("pattern", description="NLP engine: pattern | llm")
    priority: JobPriority = JobPriority.NORMAL
    tags: list[str] = Field(default_factory=list)


class Job(BaseModel):
    """A job stored in the queue."""
    id: str = Field(default_factory=_uid)
    user_id: str
    status: JobStatus = JobStatus.QUEUED
    priority: JobPriority = JobPriority.NORMAL

    # Input
    text: str
    shots: int = 1024
    backend: str = "aer"
    nlp_mode: str = "pattern"
    tags: list[str] = Field(default_factory=list)

    # Output (filled on completion)
    qasm: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    circuit_type: str | None = None
    num_qubits: int | None = None
    execution_time_ms: float | None = None
    error: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "text": self.text[:80] + ("…" if len(self.text) > 80 else ""),
            "circuit_type": self.circuit_type,
            "num_qubits": self.num_qubits,
            "shots": self.shots,
            "backend": self.backend,
            "created_at": self.created_at.isoformat(),
            "duration_ms": self.duration_ms,
        }


# ── User models ───────────────────────────────────────────────────────────────

class User(BaseModel):
    id: str = Field(default_factory=_uid)
    username: str
    api_key: str = Field(default_factory=_uid)
    is_admin: bool = False
    quota_jobs_per_hour: int = 60
    created_at: datetime = Field(default_factory=_now)


# ── API response models ───────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = ""


class QueueStats(BaseModel):
    queued: int
    running: int
    done: int
    failed: int
    total: int
    workers: int
