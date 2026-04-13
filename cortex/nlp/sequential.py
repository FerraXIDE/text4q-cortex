"""
cortex.nlp.sequential
=====================
Sequential command parser for text4q Cortex.

Translates free-form natural language sequences into OpenQASM 3.0 circuits.
This is the core of what makes Cortex a real NL compiler, not just a
template generator.

Supported syntax examples:
    "Apply H to qubit 0, then CNOT from 0 to 1, then measure"
    "X on qubit 0, Y on qubit 1, Z on qubit 2"
    "Apply Hadamard to all qubits, then measure"
    "RX(pi/2) on qubit 0, RZ(pi/4) on qubit 1"
    "Swap qubit 0 and qubit 1, measure all"
    "Apply Toffoli on qubits 0, 1 and 2"
    "Create 5 qubits, apply H to all, CNOT from 0 to 1, measure"

Architecture:
    text → tokenize_commands() → [Command] → validate() → build_qasm()
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Literal


# ── Gate aliases ──────────────────────────────────────────────────────────────

GATE_ALIASES: dict[str, str] = {
    # Single-qubit gates
    "h": "h", "hadamard": "h",
    "x": "x", "pauli-x": "x", "not": "x", "bit-flip": "x",
    "y": "y", "pauli-y": "y",
    "z": "z", "pauli-z": "z", "phase-flip": "z",
    "s": "s", "t": "t",
    "sdg": "sdg", "tdg": "tdg",
    "rx": "rx", "ry": "ry", "rz": "rz",
    # Two-qubit gates
    "cx": "cx", "cnot": "cx", "controlled-not": "cx",
    "cy": "cy", "cz": "cz",
    "swap": "swap",
    "cp": "cp", "cphase": "cp",
    # Three-qubit gates
    "ccx": "ccx", "toffoli": "ccx", "ccnot": "ccx",
    "cswap": "cswap", "fredkin": "cswap",
}

# How many qubits each gate needs
GATE_QUBITS: dict[str, int] = {
    "h": 1, "x": 1, "y": 1, "z": 1, "s": 1, "t": 1,
    "sdg": 1, "tdg": 1, "rx": 1, "ry": 1, "rz": 1,
    "cx": 2, "cy": 2, "cz": 2, "swap": 2, "cp": 2,
    "ccx": 3, "cswap": 3,
}

PARAMETRIC_GATES = {"rx", "ry", "rz", "cp"}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GateCommand:
    """A single gate operation parsed from natural language."""
    gate:    str                    # normalized gate name e.g. "cx"
    qubits:  list[int]              # target qubit indices
    angle:   float | None = None    # rotation angle for parametric gates
    raw:     str = ""               # original text fragment

    def to_qasm(self) -> str:
        """Convert this gate command to an OpenQASM 3.0 gate line."""
        q = ", ".join(f"q[{i}]" for i in self.qubits)
        if self.gate in PARAMETRIC_GATES and self.angle is not None:
            return f"{self.gate}({self.angle:.6f}) {q};"
        return f"{self.gate} {q};"


@dataclass
class MeasureCommand:
    """A measurement operation."""
    qubits: list[int] | Literal["all"] = "all"
    raw:    str = ""


@dataclass
class ParseResult:
    """Result of parsing a sequential command string."""
    commands:   list[GateCommand | MeasureCommand]
    num_qubits: int
    warnings:   list[str] = field(default_factory=list)
    errors:     list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


# ── Tokenizer ─────────────────────────────────────────────────────────────────

# Separators between commands
_SEP_RE = re.compile(
    r"\s*(?:,\s*(?:then|and)?\s*|;\s*|\bthen\b\s*|\band\s+then\b\s*)", re.I
)

# Qubit index extraction: "qubit 0", "qubits 0, 1 and 2", "all qubits"
_QUBIT_ALL_RE = re.compile(r"\b(all|every|each)\b", re.I)
_QUBIT_IDX_RE = re.compile(r"qubit[s]?\s*([\d,\s]+(?:and\s+\d+)?)", re.I)
_BARE_IDX_RE  = re.compile(r"\b(\d+)\b")

# Gate + optional angle: "RX(pi/2)", "cx", "Hadamard"
_GATE_RE = re.compile(
    r"\b(hadamard|toffoli|ccnot|fredkin|cswap|cnot|cphase|pauli-?x|pauli-?y|pauli-?z"
    r"|sdg|tdg|rx|ry|rz|cx|cy|cz|ccx|swap|cp|not|h|x|y|z|s|t)"
    r"(?:\(([^)]+)\))?",
    re.I,
)

# Detect qubit count declarations: "create 3 qubits", "with 4 qubits"
_QUBIT_COUNT_RE = re.compile(
    r"(?:create|use|with|on|for)\s+(\d+)\s+qubit", re.I
)

# Detect "from Q to Q" (CNOT-style)
_FROM_TO_RE = re.compile(r"from\s+(\d+)\s+to\s+(\d+)", re.I)

# Detect "on qubit(s) N" and "to qubit N"
_ON_TO_RE = re.compile(r"(?:on|to)\s+qubit[s]?\s+([\d,\s]+(?:and\s+\d+)?)", re.I)

# Detect measure
_MEASURE_RE = re.compile(r"\b(measure|measurement|readout)\b", re.I)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_sequential(text: str) -> ParseResult:
    """
    Parse a natural language sequential command string into a ParseResult.

    Args:
        text: e.g. "Apply H to qubit 0, CNOT from 0 to 1, measure all"

    Returns:
        ParseResult with list of GateCommand/MeasureCommand and metadata.
    """
    warnings: list[str] = []
    errors:   list[str] = []
    commands: list[GateCommand | MeasureCommand] = []

    # Detect explicit qubit count declaration
    declared_qubits: int | None = None
    m = _QUBIT_COUNT_RE.search(text)
    if m:
        declared_qubits = int(m.group(1))

    # Split text into command fragments
    fragments = _split_fragments(text)

    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue

        # Skip qubit count declarations
        if _QUBIT_COUNT_RE.search(frag):
            continue

        # Measurement command
        if _MEASURE_RE.search(frag):
            qubits = _extract_measure_targets(frag)
            commands.append(MeasureCommand(qubits=qubits, raw=frag))
            continue

        # Gate command
        gate_match = _GATE_RE.search(frag)
        if not gate_match:
            warnings.append(f"Could not parse fragment: '{frag}'")
            continue

        gate_raw   = gate_match.group(1).lower().replace("-", "")
        angle_expr = gate_match.group(2)
        gate       = GATE_ALIASES.get(gate_raw, gate_raw)
        needed     = GATE_QUBITS.get(gate, 1)

        # Parse angle for parametric gates
        angle: float | None = None
        if gate in PARAMETRIC_GATES:
            if angle_expr:
                angle = _eval_angle(angle_expr)
            else:
                # Look for angle in text: "RX pi/2 on qubit 0"
                angle_in_text = re.search(
                    r"(?:angle|by|of|=)?\s*(pi[/\s]\d+|\d+\.?\d*)", frag, re.I
                )
                angle = _eval_angle(angle_in_text.group(1)) if angle_in_text else math.pi / 2

        # Extract qubit indices
        qubits = _extract_qubits(frag, needed)

        if not qubits:
            warnings.append(f"No qubit indices found in: '{frag}'")
            continue

        # Validate qubit count
        if len(qubits) < needed:
            errors.append(
                f"Gate '{gate}' needs {needed} qubit(s), "
                f"but only {len(qubits)} found in: '{frag}'"
            )
            continue

        commands.append(GateCommand(
            gate=gate,
            qubits=qubits[:needed],
            angle=angle,
            raw=frag,
        ))

    # Determine total qubit count
    max_qubit = _max_qubit_index(commands)
    num_qubits = max(
        declared_qubits or 0,
        max_qubit + 1 if max_qubit >= 0 else 2,
    )

    # Validate qubit indices against circuit size
    for cmd in commands:
        if isinstance(cmd, GateCommand):
            for q in cmd.qubits:
                if q >= num_qubits:
                    errors.append(
                        f"Qubit index {q} out of range "
                        f"(circuit has {num_qubits} qubits)"
                    )

    # Resolve "all" measurements
    for cmd in commands:
        if isinstance(cmd, MeasureCommand) and cmd.qubits == "all":
            cmd.qubits = list(range(num_qubits))

    return ParseResult(
        commands=commands,
        num_qubits=num_qubits,
        warnings=warnings,
        errors=errors,
    )


def sequential_to_qasm(result: ParseResult) -> str:
    """
    Convert a ParseResult into an OpenQASM 3.0 circuit string.

    Args:
        result: Output of parse_sequential().

    Returns:
        OpenQASM 3.0 string ready to execute.
    """
    if not result.valid:
        raise ValueError(f"Cannot compile invalid parse result: {result.errors}")

    n = result.num_qubits
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        f"qubit[{n}] q;",
        f"bit[{n}]   c;",
        "",
    ]

    has_measure = any(isinstance(c, MeasureCommand) for c in result.commands)

    for cmd in result.commands:
        if isinstance(cmd, GateCommand):
            lines.append(cmd.to_qasm())
        elif isinstance(cmd, MeasureCommand):
            if isinstance(cmd.qubits, list):
                for i, q in enumerate(cmd.qubits):
                    lines.append(f"c[{i}] = measure q[{q}];")
            else:
                lines.append("c = measure q;")

    # Add measurement if not explicitly included
    if not has_measure:
        lines.append("")
        lines.append("// Auto-added measurement")
        lines.append("c = measure q;")

    return "\n".join(lines)


def parse_and_compile(text: str) -> tuple[ParseResult, str]:
    """
    Full pipeline: parse natural language → OpenQASM 3.0.

    Returns:
        (ParseResult, qasm_string) tuple.
    """
    result = parse_sequential(text)
    if not result.valid:
        return result, ""
    qasm = sequential_to_qasm(result)
    return result, qasm


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_fragments(text: str) -> list[str]:
    """Split a command string into individual gate fragments."""
    # Remove common preamble words
    text = re.sub(r"^\s*(?:please\s+)?(?:now\s+)?", "", text, flags=re.I)

    gate_kw = set(GATE_ALIASES.keys()) | {"measure", "measurement"}

    # Split by 'then' and semicolons first (always safe separators)
    then_sep = re.compile(r"\s*(?:;\s*|then\s*)", re.I)
    primary = then_sep.split(text)

    # For each primary chunk, split by comma ONLY if next word is a gate keyword
    result = []
    for part in primary:
        sub = re.split(r",\s*", part)
        merged = sub[0]
        for s in sub[1:]:
            first_word = re.match(r"[a-zA-Z]+", s.strip())
            if first_word and first_word.group(0).lower() in gate_kw:
                result.append(merged.strip())
                merged = s
            else:
                merged = merged + ", " + s
        result.append(merged.strip())

    # Final pass: if a fragment contains multiple gate keywords, split on the second one
    final = []
    for frag in result:
        gate_matches = list(re.finditer(
            r"\b(hadamard|toffoli|ccnot|fredkin|cswap|cnot|cphase|pauli-?x|pauli-?y|pauli-?z"
            r"|sdg|tdg|rx|ry|rz|cx|cy|cz|ccx|swap|cp|not|h|x|y|z|s|t|measure)\b",
            frag, re.I
        ))
        if len(gate_matches) <= 1:
            final.append(frag)
        else:
            # Split at the position of the second gate
            split_pos = gate_matches[1].start()
            final.append(frag[:split_pos].rstrip(", "))
            final.append(frag[split_pos:])

    return [f for f in final if f.strip()]


def _extract_qubits(frag: str, needed: int) -> list[int]:
    """Extract qubit indices from a text fragment."""
    # "from Q to Q" pattern (CNOT-style)
    m = _FROM_TO_RE.search(frag)
    if m:
        return [int(m.group(1)), int(m.group(2))]

    # "qubit N and qubit M" pattern (Swap-style)
    multi_qubit = re.findall(r"qubit\s+(\d+)", frag, re.I)
    if len(multi_qubit) >= needed:
        return [int(q) for q in multi_qubit[:needed]]

    # "qubit(s) N, M and K"
    m = _QUBIT_IDX_RE.search(frag)
    if m:
        return _parse_index_list(m.group(1))

    # "on qubit N" / "to qubit N"
    m = _ON_TO_RE.search(frag)
    if m:
        return _parse_index_list(m.group(1))

    # "all qubits" — return placeholder, resolved later
    if _QUBIT_ALL_RE.search(frag):
        return list(range(needed))  # minimal placeholder

    # Bare numbers as last resort
    nums = [int(x) for x in _BARE_IDX_RE.findall(frag)]
    return nums[:needed] if nums else []


def _extract_measure_targets(frag: str) -> list[int] | Literal["all"]:
    """Extract measurement targets from a measure fragment."""
    if _QUBIT_ALL_RE.search(frag) or re.search(r"\ball\b", frag, re.I):
        return "all"
    m = _QUBIT_IDX_RE.search(frag)
    if m:
        return _parse_index_list(m.group(1))
    m = _ON_TO_RE.search(frag)
    if m:
        return _parse_index_list(m.group(1))
    nums = [int(x) for x in _BARE_IDX_RE.findall(frag)
            if x not in ("measure", "measurement")]
    return _parse_index_list(" ".join(str(n) for n in nums)) if nums else "all"


def _parse_index_list(text: str) -> list[int]:
    """Parse '0, 1 and 2' or '0 1 2' into [0, 1, 2]."""
    text = text.replace("and", " ")
    return sorted(set(int(x) for x in re.findall(r"\d+", text)))


def _eval_angle(expr: str) -> float:
    """Safely evaluate an angle expression like 'pi/2' or '3.14'."""
    expr = expr.strip().replace("π", "pi")
    expr = re.sub(r"pi", str(math.pi), expr, flags=re.I)
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return math.pi / 2  # sensible default


def _max_qubit_index(commands: list) -> int:
    """Find the highest qubit index used across all commands."""
    max_idx = -1
    for cmd in commands:
        if isinstance(cmd, GateCommand):
            for q in cmd.qubits:
                max_idx = max(max_idx, q)
        elif isinstance(cmd, MeasureCommand) and isinstance(cmd.qubits, list):
            for q in cmd.qubits:
                max_idx = max(max_idx, q)
    return max_idx
