"""Data models for text4q Cortex."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Backend(str, Enum):
    IBM_QUANTUM = "ibm_quantum"
    AER = "aer"
    PENNYLANE = "pennylane"
    GOOGLE = "google"


@dataclass
class CircuitIntent:
    """
    Parsed intent extracted from natural language input.
    This is what the NLP engine produces before generating QASM.
    """
    raw_text: str
    num_qubits: int
    circuit_type: str                        # e.g. "bell_state", "ghz", "qft", "custom"
    shots: int = 1024
    noise_model: dict[str, Any] | None = None
    custom_gates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CortexResult:
    """
    Result returned after executing a quantum job through Cortex.
    """
    intent: CircuitIntent
    qasm: str                                # generated OpenQASM 3.0 circuit
    counts: dict[str, int]                  # measurement results
    backend: str
    shots: int
    execution_time_ms: float
    job_id: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def most_probable(self) -> str:
        """Return the bitstring with the highest count."""
        return max(self.counts, key=self.counts.get)  # type: ignore

    def __repr__(self) -> str:
        top = self.most_probable() if self.counts else "—"
        return (
            f"CortexResult(backend={self.backend!r}, shots={self.shots}, "
            f"top={top!r}, time={self.execution_time_ms:.0f}ms)"
        )
