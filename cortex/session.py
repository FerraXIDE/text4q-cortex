"""
cortex.session
==============
Conversational session for text4q Cortex.

Allows building quantum circuits incrementally through natural language,
accumulating gates across multiple run() calls before executing.

Usage:
    from cortex import Cortex

    cx = Cortex(backend="aer")

    with cx.session():
        cx.run("Create 3 qubits")
        cx.run("Apply H to qubit 0")
        cx.run("CNOT from 0 to 1")
        cx.run("CNOT from 0 to 2")
        result = cx.run("measure all")   # executes accumulated circuit

    print(result.counts)
    # {'000': 512, '111': 512}

Special commands during a session:
    "show circuit"   — print the accumulated circuit diagram
    "undo"           — remove the last gate added
    "reset"          — clear all accumulated gates
    "how many gates" — show current gate count
    "measure [all]"  — execute the accumulated circuit
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortex.models import CortexResult


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """Mutable state accumulated during a conversational session."""
    num_qubits:  int = 0
    gate_lines:  list[str] = field(default_factory=list)
    history:     list[str] = field(default_factory=list)   # commands so far
    executed:    bool = False

    def add_gate(self, qasm_line: str) -> None:
        self.gate_lines.append(qasm_line.strip().rstrip(";") + ";")

    def undo(self) -> str | None:
        """Remove and return the last gate added."""
        if self.gate_lines:
            removed = self.gate_lines.pop()
            if self.history:
                self.history.pop()
            return removed
        return None

    def reset(self) -> None:
        """Clear all accumulated gates (keep qubit count)."""
        self.gate_lines.clear()
        self.history.clear()
        self.executed = False

    def build_qasm(self, include_measure: bool = True) -> str:
        """Build a complete OpenQASM 3.0 string from accumulated gates."""
        n = self.num_qubits or 2
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            "",
            f"qubit[{n}] q;",
            f"bit[{n}]   c;",
            "",
        ]
        lines.extend(self.gate_lines)
        if include_measure:
            lines.append("")
            lines.append("c = measure q;")
        return "\n".join(lines)

    @property
    def gate_count(self) -> int:
        return len(self.gate_lines)

    def diagram(self) -> str:
        """Return ASCII diagram of the current accumulated circuit."""
        if not self.gate_lines and self.num_qubits == 0:
            return "(empty circuit — add gates first)"
        try:
            from qiskit import qasm3
            circuit = qasm3.loads(self.build_qasm(include_measure=False))
            return str(circuit.draw(output="text"))
        except Exception as e:
            return f"(diagram unavailable: {e})"


# ── Special command detection ─────────────────────────────────────────────────

_QUBIT_COUNT_RE = re.compile(r"(?:create|use|with|set|initialize)\s+(\d+)\s+qubit", re.I)
_MEASURE_RE     = re.compile(r"^\s*measure(?:\s+all)?\s*$", re.I)
_SHOW_RE        = re.compile(r"\b(show|display|print|view)\s+(?:the\s+)?circuit\b", re.I)
_UNDO_RE        = re.compile(r"^\s*undo\s*$", re.I)
_RESET_RE       = re.compile(r"^\s*reset\s*$", re.I)
_GATE_COUNT_RE  = re.compile(r"\bhow\s+many\s+(?:gates?|operations?)\b", re.I)
_STATUS_RE      = re.compile(r"\b(status|info|state)\b", re.I)


class SessionCommand:
    """Identifies the type of command during a session."""
    SET_QUBITS  = "set_qubits"
    ADD_GATE    = "add_gate"
    MEASURE     = "measure"
    SHOW        = "show"
    UNDO        = "undo"
    RESET       = "reset"
    GATE_COUNT  = "gate_count"
    STATUS      = "status"
    UNKNOWN     = "unknown"


def classify_session_command(text: str) -> tuple[str, int | None]:
    """
    Classify a natural language command within a session context.

    Returns:
        (command_type, optional_qubit_count)
    """
    text_stripped = text.strip()

    if m := _QUBIT_COUNT_RE.search(text_stripped):
        return SessionCommand.SET_QUBITS, int(m.group(1))
    if _MEASURE_RE.match(text_stripped):
        return SessionCommand.MEASURE, None
    if _SHOW_RE.search(text_stripped):
        return SessionCommand.SHOW, None
    if _UNDO_RE.match(text_stripped):
        return SessionCommand.UNDO, None
    if _RESET_RE.match(text_stripped):
        return SessionCommand.RESET, None
    if _GATE_COUNT_RE.search(text_stripped):
        return SessionCommand.GATE_COUNT, None
    if _STATUS_RE.search(text_stripped):
        return SessionCommand.STATUS, None

    return SessionCommand.ADD_GATE, None


# ── Context manager ───────────────────────────────────────────────────────────

class CortexSession:
    """
    Context manager for a conversational Cortex session.

    Usage:
        with cx.session() as s:
            cx.run("Create 3 qubits")
            cx.run("Apply H to qubit 0")
            result = cx.run("measure all")
    """

    def __init__(self, cortex_instance):
        self._cx = cortex_instance

    def __enter__(self):
        self._cx._start_session()
        return self

    def __exit__(self, *args):
        self._cx._end_session()


# ── Session result ────────────────────────────────────────────────────────────

@dataclass
class SessionResponse:
    """
    Response from a run() call during a session.
    May be a gate-added acknowledgment or a full execution result.
    """
    kind:        str                    # "ack" | "result" | "info" | "error"
    message:     str                    # human-readable response
    result:      CortexResult | None = None
    gate_added:  str | None = None
    gate_count:  int = 0

    @property
    def is_result(self) -> bool:
        return self.kind == "result" and self.result is not None

    @property
    def counts(self) -> dict:
        return self.result.counts if self.result else {}

    def __repr__(self) -> str:
        if self.is_result:
            return f"SessionResponse(result={self.result})"
        return f"SessionResponse(kind={self.kind!r}, msg={self.message!r})"
