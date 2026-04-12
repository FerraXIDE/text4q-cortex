"""Data models for text4q Cortex."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Backend(str, Enum):
    IBM_QUANTUM = "ibm_quantum"
    AER         = "aer"
    PENNYLANE   = "pennylane"
    GOOGLE      = "google"


@dataclass
class CircuitIntent:
    """
    Parsed intent extracted from natural language input.
    This is what the NLP engine produces before generating QASM.
    """
    raw_text:     str
    num_qubits:   int
    circuit_type: str
    shots:        int = 1024
    noise_model:  dict[str, Any] | None = None
    custom_gates: list[str] = field(default_factory=list)
    metadata:     dict[str, Any] = field(default_factory=dict)


@dataclass
class CortexResult:
    """
    Result returned after executing a quantum job through Cortex.
    """
    intent:            CircuitIntent
    qasm:              str
    counts:            dict[str, int]
    backend:           str
    shots:             int
    execution_time_ms: float
    job_id:            str | None = None
    error:             str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def most_probable(self) -> str:
        """Return the bitstring with the highest count."""
        return max(self.counts, key=self.counts.get)  # type: ignore

    def diagram(self, output: str = "text") -> str:
        """
        Return a visual diagram of the quantum circuit.

        Args:
            output: "text" for ASCII art (default), "latex_source" for LaTeX.

        Returns:
            String representation of the circuit diagram.

        Example:
            result = cx.run("Bell state")
            print(result.diagram())

                 ┌───┐     ┌─┐
            q_0: ┤ H ├──■──┤M├
                 └───┘┌─┴─┐└╥┘
            q_1: ─────┤ X ├─╫─
                      └───┘ ║
            c: 2/═══════════╩═
        """
        if not self.qasm:
            return "(no circuit available)"
        try:
            from qiskit import qasm3
            circuit = qasm3.loads(self.qasm)
            return str(circuit.draw(output=output))
        except Exception as e:
            return f"(diagram unavailable: {e})"

    def save_diagram(self, filename: str = "circuit.png", dpi: int = 150) -> str:
        """
        Save the circuit diagram as a PNG image.

        Args:
            filename: Output file path (default: circuit.png).
            dpi:      Image resolution (default: 150).

        Returns:
            The filename saved.
        """
        if not self.qasm:
            raise ValueError("No QASM circuit to visualize")
        try:
            from qiskit import qasm3
            import matplotlib
            matplotlib.use("Agg")
            circuit = qasm3.loads(self.qasm)
            fig = circuit.draw(output="mpl", style="iqp")
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")
            fig.clf()
            return filename
        except ImportError:
            raise ImportError("pip install matplotlib to save circuit diagrams")

    def fidelity(self) -> float | None:
        """
        Estimate circuit fidelity for 2-qubit circuits (Bell-type).
        Returns the fraction of measurements in the expected states.
        Only meaningful for Bell and GHZ circuits.
        """
        if not self.counts or self.intent.circuit_type not in ("bell_state", "ghz"):
            return None
        n = self.intent.num_qubits
        expected = {"0" * n, "1" * n}
        correct = sum(v for k, v in self.counts.items() if k in expected)
        return round(correct / self.shots, 4)

    def __repr__(self) -> str:
        top = self.most_probable() if self.counts else "—"
        return (
            f"CortexResult(backend={self.backend!r}, shots={self.shots}, "
            f"top={top!r}, time={self.execution_time_ms:.0f}ms)"
        )