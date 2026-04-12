"""
cortex.scheduler.qaoa
=====================
Builds the QAOA circuit from a QUBO scheduling problem.

QAOA (Quantum Approximate Optimization Algorithm) — Farhi et al. 2014
----------------------------------------------------------------------
Given a QUBO cost function C(x) = x^T Q x, QAOA prepares the state:

    |ψ(γ,β)⟩ = U_B(β_p) U_C(γ_p) … U_B(β_1) U_C(γ_1) |+⟩^n

where:
  U_C(γ) = exp(−iγ H_C) = product of RZZ gates encoding Q
  U_B(β) = exp(−iβ H_B) = product of RX gates (mixer)
  |+⟩^n  = H^⊗n |0⟩^n   (uniform superposition, equal-weight starting point)

Measuring |ψ(γ,β)⟩ gives bitstrings; the one with lowest C(x) is
the approximate optimal job assignment.

For p=1 (one QAOA layer), the circuit is:
  H on all qubits  →  RZZ(2γ*Q_ij) for each pair  →  RX(2β) on all qubits  →  measure

OpenQASM 3.0 output — runs on any backend that Cortex supports.

Parameter optimization
----------------------
v0.4 uses fixed parameters γ=0.5, β=0.3 (good heuristic for p=1).
v0.5 will add classical optimization loop (SciPy minimize on <C>).
"""

from __future__ import annotations

import math
from cortex.scheduler.problem import QUBOProblem


def build_qaoa_circuit(
    qubo: QUBOProblem,
    gamma: float = 0.5,
    beta: float = 0.3,
    p: int = 1,
    shots: int = 2048,
) -> str:
    """
    Build a QAOA OpenQASM 3.0 circuit for the scheduling QUBO.

    Args:
        qubo:   The scheduling problem encoded as a QUBO matrix.
        gamma:  Phase-separation angle (controls cost function encoding).
        beta:   Mixing angle (controls exploration vs exploitation).
        p:      Number of QAOA layers (depth). Higher p → better approximation.
        shots:  Number of measurement shots (passed as comment for metadata).

    Returns:
        OpenQASM 3.0 string ready to execute on any Cortex backend.
    """
    n = qubo.n_vars

    if n == 0:
        raise ValueError("QUBO has no variables — nothing to schedule")
    if n > 20:
        raise ValueError(
            f"QUBO has {n} variables (max 20 for this prototype). "
            "Reduce jobs or backends, or use the classical fallback."
        )

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        f"// QAOA Scheduler — text4q Cortex",
        f"// Problem: {qubo.n_jobs} jobs × {qubo.n_backends} backends = {n} qubits",
        f"// Layers: p={p}  γ={gamma:.4f}  β={beta:.4f}  shots={shots}",
        "",
        f"qubit[{n}] q;",
        f"bit[{n}]   c;",
        "",
    ]

    # ── Initial superposition: H on all qubits ────────────────────────────────
    lines.append("// Initial superposition |+>^n")
    for i in range(n):
        lines.append(f"h q[{i}];")
    lines.append("")

    # ── p QAOA layers ─────────────────────────────────────────────────────────
    for layer in range(p):
        g = gamma * (layer + 1) / p   # scale gamma across layers
        b = beta  * (layer + 1) / p

        lines.append(f"// ── Layer {layer + 1}/{p} ──────────────────")

        # Cost unitary U_C(γ): RZZ gates encoding the QUBO Q matrix
        lines.append(f"// U_C(γ={g:.4f}): phase-separation — encodes QUBO cost")
        _append_cost_unitary(lines, qubo.Q, n, g)

        # Mixer unitary U_B(β): RX gates on all qubits
        lines.append(f"// U_B(β={b:.4f}): mixer — explores solution space")
        for i in range(n):
            angle = 2.0 * b
            lines.append(f"rx({angle:.6f}) q[{i}];")
        lines.append("")

    # ── Measurement ───────────────────────────────────────────────────────────
    lines.append("// Measure all qubits")
    lines.append("c = measure q;")

    return "\n".join(lines)


def _append_cost_unitary(
    lines: list[str],
    Q: list[list[float]],
    n: int,
    gamma: float,
) -> None:
    """
    Append RZZ gates for the cost Hamiltonian H_C = Σ_ij Q_ij Z_i Z_j.

    U_C(γ) = exp(−iγ H_C)
           = Π_ij exp(−iγ Q_ij Z_i Z_j)
           = Π_ij  CNOT · RZ(2γ Q_ij) · CNOT   (for off-diagonal)
           = Π_i   RZ(2γ Q_ii)                  (for diagonal)

    RZZ(θ) = exp(−i θ/2 Z⊗Z)
    """
    threshold = 1e-9   # skip negligible terms

    # Diagonal terms: RZ on single qubits
    for i in range(n):
        qii = Q[i][i]
        if abs(qii) < threshold:
            continue
        angle = 2.0 * gamma * qii
        lines.append(f"rz({angle:.6f}) q[{i}];")

    # Off-diagonal terms: RZZ (CNOT–RZ–CNOT pattern)
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i][j] + Q[j][i]   # symmetrize
            if abs(qij) < threshold:
                continue
            angle = 2.0 * gamma * qij
            # RZZ(θ) decomposition:
            lines.append(f"cx q[{i}], q[{j}];")
            lines.append(f"rz({angle:.6f}) q[{j}];")
            lines.append(f"cx q[{i}], q[{j}];")

    lines.append("")


# ── Parameter suggestions ─────────────────────────────────────────────────────

def suggest_parameters(n_vars: int, p: int = 1) -> tuple[float, float]:
    """
    Heuristic parameter initialization for QAOA.
    Based on: Brandao et al. (2018) fixed-angle conjecture for p=1.

    Returns (gamma, beta) tuple.
    """
    # For p=1, optimal angles are near π/8 and π/4 respectively
    # Slight scaling for larger problem sizes
    scale = 1.0 / math.sqrt(max(n_vars, 1))
    gamma = (math.pi / 8) * (1 + 0.1 * scale)
    beta  = (math.pi / 4) * (1 - 0.1 * scale)
    return round(gamma, 6), round(beta, 6)


# ── Circuit metadata ──────────────────────────────────────────────────────────

def circuit_stats(qubo: QUBOProblem, p: int = 1) -> dict:
    """
    Estimate circuit statistics before execution.
    Useful for deciding whether to use real QPU or simulator.
    """
    n = qubo.n_vars
    Q = qubo.Q
    threshold = 1e-9

    n_diag = sum(1 for i in range(n) if abs(Q[i][i]) > threshold)
    n_offdiag = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if abs(Q[i][j] + Q[j][i]) > threshold
    )

    # Each off-diagonal term → 2 CNOT + 1 RZ = 3 gates
    # Each diagonal term    → 1 RZ
    # Mixer: n RX gates per layer
    gates_per_layer = n_diag + 3 * n_offdiag + n
    total_gates     = p * gates_per_layer + n   # +n for initial H

    # Circuit depth estimate (assuming serial execution)
    depth_estimate = p * (1 + 3 * n_offdiag + 1) + 1

    return {
        "n_qubits":      n,
        "p_layers":      p,
        "n_variables":   n,
        "n_diag_terms":  n_diag,
        "n_offdiag_terms": n_offdiag,
        "total_gates":   total_gates,
        "depth_estimate": depth_estimate,
        "recommend_real_qpu": n <= 10 and total_gates <= 200,
    }
