"""
cortex.nlp.optimizer
====================
Quantum circuit optimizer for text4q Cortex.

Reduces gate count before sending circuits to real QPUs,
which directly improves fidelity by minimizing decoherence exposure.

Two optimization strategies:
    1. Rule-based (fast, no backend needed)
       Cancels known redundant gate pairs: H H, X X, Z Z, S S S S, etc.
       Runs in O(n) time on the gate list.

    2. Qiskit transpiler (thorough, requires backend)
       Uses Qiskit's preset pass manager at configurable optimization level.
       Level 0 = minimal, Level 3 = aggressive (may change qubit layout).

Usage:
    from cortex.nlp.optimizer import optimize_qasm, OptimizationResult

    result = optimize_qasm(qasm, backend="aer", level=2)
    print(result.summary())
    print(result.optimized_qasm)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


# ── Known self-inverse gate pairs (gate applied twice = identity) ─────────────

SELF_INVERSE = {"h", "x", "y", "z", "cx", "cy", "cz", "swap", "ccx"}

# Gates that cancel after N applications (N-periodic)
PERIODIC: dict[str, int] = {
    "s":   4,    # S^4 = I
    "t":   8,    # T^8 = I
    "sdg": 4,
    "tdg": 8,
}


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    original_qasm:    str
    optimized_qasm:   str
    original_gates:   int
    optimized_gates:  int
    gates_removed:    int
    optimization_ms:  float
    level:            int
    method:           str   # "rule_based" | "qiskit" | "combined"
    rules_applied:    list[str] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        if self.original_gates == 0:
            return 0.0
        return round((self.gates_removed / self.original_gates) * 100, 1)

    def summary(self) -> str:
        return (
            f"Optimization level {self.level} ({self.method})\n"
            f"  Gates: {self.original_gates} → {self.optimized_gates} "
            f"(-{self.gates_removed}, -{self.reduction_pct}%)\n"
            f"  Time:  {self.optimization_ms:.1f} ms\n"
            f"  Rules: {', '.join(self.rules_applied) if self.rules_applied else 'none'}"
        )

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(gates={self.original_gates}→{self.optimized_gates}, "
            f"-{self.reduction_pct}%, method={self.method!r})"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def optimize_qasm(
    qasm:    str,
    backend: str = "aer",
    level:   int = 1,
) -> OptimizationResult:
    """
    Optimize an OpenQASM 3.0 circuit.

    Args:
        qasm:    OpenQASM 3.0 circuit string.
        backend: Target backend — affects gate set for transpiler.
        level:   Optimization level:
                   0 = no optimization (passthrough)
                   1 = rule-based only (fast, no backend needed)
                   2 = rule-based + Qiskit transpiler level 1
                   3 = rule-based + Qiskit transpiler level 2 (aggressive)

    Returns:
        OptimizationResult with optimized QASM and statistics.
    """
    t0 = time.monotonic()
    original_count = _count_gates(qasm)

    if level == 0:
        return OptimizationResult(
            original_qasm=qasm, optimized_qasm=qasm,
            original_gates=original_count, optimized_gates=original_count,
            gates_removed=0, optimization_ms=0.0,
            level=0, method="none",
        )

    # Level 1: rule-based only
    optimized, rules = _rule_based_optimize(qasm)
    method = "rule_based"

    # Levels 2-3: additionally run Qiskit transpiler
    if level >= 2:
        qiskit_level = 1 if level == 2 else 2
        try:
            optimized = _qiskit_optimize(optimized, backend, qiskit_level)
            method = "combined"
        except Exception:
            pass  # fall back to rule-based result silently

    elapsed_ms = (time.monotonic() - t0) * 1000
    optimized_count = _count_gates(optimized)

    return OptimizationResult(
        original_qasm=qasm,
        optimized_qasm=optimized,
        original_gates=original_count,
        optimized_gates=optimized_count,
        gates_removed=original_count - optimized_count,
        optimization_ms=elapsed_ms,
        level=level,
        method=method,
        rules_applied=rules,
    )


# ── Rule-based optimizer ──────────────────────────────────────────────────────

def _rule_based_optimize(qasm: str) -> tuple[str, list[str]]:
    """
    Apply algebraic simplification rules to the gate list.

    Rules applied:
    - Self-inverse cancellation: H H → I, X X → I, CX CX → I, etc.
    - Periodic gate reduction: S S S S → I, T^8 → I
    - Adjacent identity removal

    Returns:
        (optimized_qasm, list_of_rule_names_applied)
    """
    header, gates, footer = _split_qasm(qasm)
    rules_applied: list[str] = []

    # Self-inverse cancellation (multiple passes until stable)
    prev_len = len(gates) + 1
    while len(gates) < prev_len:
        prev_len = len(gates)
        gates, applied = _cancel_self_inverse(gates)
        if applied:
            rules_applied.append("self_inverse_cancellation")

    # Periodic gate reduction
    gates, applied = _reduce_periodic(gates)
    if applied:
        rules_applied.append("periodic_gate_reduction")

    optimized = _join_qasm(header, gates, footer)
    return optimized, list(set(rules_applied))


def _cancel_self_inverse(gates: list[str]) -> tuple[list[str], bool]:
    """
    Cancel consecutive identical self-inverse gates on the same qubits.
    H q[0]; H q[0]; → removed
    CX q[0], q[1]; CX q[0], q[1]; → removed
    """
    if len(gates) < 2:
        return gates, False

    result: list[str] = []
    i = 0
    applied = False

    while i < len(gates):
        if i + 1 < len(gates):
            g1 = _parse_gate_line(gates[i])
            g2 = _parse_gate_line(gates[i + 1])
            if (g1 and g2
                    and g1["name"] == g2["name"]
                    and g1["name"] in SELF_INVERSE
                    and g1["qubits"] == g2["qubits"]):
                # Cancel both — skip them
                i += 2
                applied = True
                continue
        result.append(gates[i])
        i += 1

    return result, applied


def _reduce_periodic(gates: list[str]) -> tuple[list[str], bool]:
    """
    Reduce sequences of periodic gates.
    S S S S → identity (removed)
    T T T T T T T T → identity (removed)
    """
    if not gates:
        return gates, False

    applied = False
    for gate_name, period in PERIODIC.items():
        result: list[str] = []
        i = 0
        while i < len(gates):
            # Collect consecutive same gates on same qubit
            run = [i]
            j = i + 1
            while j < len(gates):
                g_curr = _parse_gate_line(gates[i])
                g_next = _parse_gate_line(gates[j])
                if (g_curr and g_next
                        and g_next["name"] == gate_name
                        and g_next["qubits"] == g_curr["qubits"]):
                    run.append(j)
                    j += 1
                else:
                    break

            if len(run) >= period:
                # Remove full periods, keep remainder
                remainder = len(run) % period
                for k in range(remainder):
                    result.append(gates[run[k]])
                i = j
                if len(run) - remainder > 0:
                    applied = True
            else:
                result.append(gates[i])
                i += 1
        gates = result

    return gates, applied


# ── Qiskit transpiler optimization ────────────────────────────────────────────

def _qiskit_optimize(qasm: str, backend: str, level: int) -> str:
    """
    Run Qiskit's preset pass manager on the circuit.
    Converts QASM 3.0 → QuantumCircuit → optimize → QASM 3.0.
    """
    from qiskit import qasm3
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    circuit = qasm3.loads(qasm)
    sim = AerSimulator()
    pm = generate_preset_pass_manager(optimization_level=level, backend=sim)
    optimized_circuit = pm.run(circuit)

    # Convert back to QASM 3.0
    return qasm3.dumps(optimized_circuit)


# ── QASM parsing helpers ──────────────────────────────────────────────────────

_GATE_LINE_RE = re.compile(
    r"^([\w]+)\s+((?:q\[\d+\](?:,\s*q\[\d+\])*))(?:\s*;)?$"
)
_PARAM_GATE_RE = re.compile(
    r"^([\w]+)\([^)]*\)\s+((?:q\[\d+\](?:,\s*q\[\d+\])*))(?:\s*;)?$"
)


def _parse_gate_line(line: str) -> dict | None:
    """Parse a QASM gate line into name + qubit list. Returns None if not a gate."""
    line = line.strip().rstrip(";")
    if not line or line.startswith("//") or line.startswith("OPENQASM") \
            or line.startswith("include") or line.startswith("qubit") \
            or line.startswith("bit") or "measure" in line:
        return None

    # Parametric gate — extract name and qubits
    m = _PARAM_GATE_RE.match(line)
    if m:
        qubits = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", m.group(2)))
        return {"name": m.group(1).lower(), "qubits": qubits, "parametric": True}

    # Simple gate
    m = _GATE_LINE_RE.match(line)
    if m:
        qubits = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", m.group(2)))
        return {"name": m.group(1).lower(), "qubits": qubits, "parametric": False}

    return None


def _split_qasm(qasm: str) -> tuple[list[str], list[str], list[str]]:
    """Split QASM into header lines, gate lines, and footer (measurements)."""
    lines = [l.strip() for l in qasm.splitlines()]
    header, gates, footer = [], [], []

    in_gates = False
    for line in lines:
        if not line:
            continue
        is_decl = (
            line.startswith("OPENQASM")
            or line.startswith("include")
            or line.startswith("qubit")
            or line.startswith("bit")
            or line.startswith("//")
        )
        is_measure = "measure" in line

        if is_decl:
            header.append(line)
        elif is_measure:
            footer.append(line)
            in_gates = False
        else:
            gates.append(line)
            in_gates = True

    return header, gates, footer


def _join_qasm(header: list[str], gates: list[str], footer: list[str]) -> str:
    """Reassemble QASM from header, gate list, and footer."""
    parts = header + [""] + gates + [""] + footer
    return "\n".join(p for p in parts if p is not None)


def _count_gates(qasm: str) -> int:
    """Count non-header, non-measurement gate lines in a QASM string."""
    _, gates, _ = _split_qasm(qasm)
    return len([g for g in gates if g.strip()])
