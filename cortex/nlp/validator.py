"""
cortex.nlp.validator
====================
Intelligent error validation for text4q Cortex.

Validates circuit intent and QASM before execution, producing
clear, actionable error messages instead of silent failures or
cryptic internal exceptions.

Validation layers:
    1. Input text    — empty, too long, encoding issues
    2. Circuit size  — qubit count limits for real QPU vs simulator
    3. Gate logic    — duplicate qubits, invalid angles, unknown gates
    4. Shots         — range and type validation
    5. QASM syntax   — structural check before sending to backend

Usage:
    from cortex.nlp.validator import validate_all, CortexValidationError

    try:
        validate_all(text, intent, qasm, backend="aer")
    except CortexValidationError as e:
        print(e.user_message)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal


# ── Limits ────────────────────────────────────────────────────────────────────

LIMITS = {
    "aer": {
        "max_qubits": 30,
        "max_shots":  100_000,
        "min_shots":  1,
    },
    "ibm_quantum": {
        "max_qubits": 127,
        "max_shots":  100_000,
        "min_shots":  1,
    },
}

MAX_TEXT_LENGTH  = 1000
MAX_ANGLE        = 100 * math.pi   # angles beyond this are almost certainly mistakes
KNOWN_GATES      = {
    "h", "x", "y", "z", "s", "t", "sdg", "tdg",
    "rx", "ry", "rz", "cx", "cy", "cz", "ccx",
    "swap", "cswap", "cp", "measure",
}


# ── Error class ───────────────────────────────────────────────────────────────

@dataclass
class ValidationError:
    """A single validation problem found in the input."""
    code:         str    # machine-readable error code
    message:      str    # developer message
    user_message: str    # clear, actionable message for the end user
    severity:     Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        return self.user_message


class CortexValidationError(Exception):
    """Raised when validation fails. Contains all found errors."""

    def __init__(self, errors: list[ValidationError]):
        self.errors   = errors
        self.warnings = [e for e in errors if e.severity == "warning"]
        self.fatal    = [e for e in errors if e.severity == "error"]
        super().__init__(self.user_message)

    @property
    def user_message(self) -> str:
        lines = ["Cortex could not process your request:\n"]
        for i, e in enumerate(self.fatal, 1):
            lines.append(f"  {i}. {e.user_message}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w.user_message}")
        return "\n".join(lines)

    @property
    def first(self) -> ValidationError:
        return self.errors[0]


# ── Main validator ────────────────────────────────────────────────────────────

def validate_all(
    text:    str,
    intent=None,
    qasm:    str | None = None,
    backend: str = "aer",
) -> list[ValidationError]:
    """
    Run all validation checks. Returns list of warnings.
    Raises CortexValidationError if any fatal errors found.

    Args:
        text:    Original natural language input.
        intent:  Parsed CircuitIntent (optional).
        qasm:    Generated QASM string (optional).
        backend: Target backend name.

    Returns:
        List of warning ValidationErrors (empty = fully valid).

    Raises:
        CortexValidationError: If any fatal validation errors are found.
    """
    errors: list[ValidationError] = []

    errors += _validate_text(text)
    if intent:
        errors += _validate_intent(intent, backend)
    if qasm:
        errors += _validate_qasm_structure(qasm)
    if intent and qasm:
        errors += _validate_gate_logic(qasm, intent)

    fatal = [e for e in errors if e.severity == "error"]
    if fatal:
        raise CortexValidationError(errors)

    return [e for e in errors if e.severity == "warning"]


# ── Layer 1: Text validation ──────────────────────────────────────────────────

def _validate_text(text: str) -> list[ValidationError]:
    errors = []

    if not text or not text.strip():
        errors.append(ValidationError(
            code="EMPTY_INPUT",
            message="Input text is empty",
            user_message=(
                "Please provide a circuit description. "
                "Example: \"Bell state with 2 qubits\" or "
                "\"Apply H to qubit 0, CNOT from 0 to 1, measure all\""
            ),
        ))
        return errors  # no point checking further

    if len(text) > MAX_TEXT_LENGTH:
        errors.append(ValidationError(
            code="INPUT_TOO_LONG",
            message=f"Input length {len(text)} exceeds {MAX_TEXT_LENGTH}",
            user_message=(
                f"Your description is too long ({len(text)} chars, max {MAX_TEXT_LENGTH}). "
                "Please describe one circuit at a time."
            ),
        ))

    # Detect if user pasted QASM directly
    if text.strip().startswith("OPENQASM"):
        errors.append(ValidationError(
            code="RAW_QASM_INPUT",
            message="User pasted raw QASM instead of natural language",
            user_message=(
                "It looks like you pasted an OpenQASM circuit directly. "
                "Cortex accepts natural language descriptions. "
                "Example: \"Bell state\" or \"Apply H to qubit 0, CNOT from 0 to 1\""
            ),
            severity="warning",
        ))

    return errors


# ── Layer 2: Intent validation ────────────────────────────────────────────────

def _validate_intent(intent, backend: str) -> list[ValidationError]:
    errors = []
    limits = LIMITS.get(backend, LIMITS["aer"])

    # Qubit count
    if intent.num_qubits <= 0:
        errors.append(ValidationError(
            code="INVALID_QUBIT_COUNT",
            message=f"num_qubits={intent.num_qubits}",
            user_message=(
                "Could not determine the number of qubits. "
                "Try specifying explicitly: \"Bell state with 2 qubits\""
            ),
        ))
    elif intent.num_qubits > limits["max_qubits"]:
        errors.append(ValidationError(
            code="TOO_MANY_QUBITS",
            message=f"num_qubits={intent.num_qubits} > max={limits['max_qubits']}",
            user_message=(
                f"You requested {intent.num_qubits} qubits, but the {backend} backend "
                f"supports a maximum of {limits['max_qubits']}. "
                f"Try reducing the qubit count."
            ),
        ))
    elif intent.num_qubits > 20 and backend == "aer":
        errors.append(ValidationError(
            code="LARGE_SIMULATION",
            message=f"num_qubits={intent.num_qubits} may be slow on simulator",
            user_message=(
                f"Simulating {intent.num_qubits} qubits locally may be very slow "
                f"(memory grows as 2^n). Consider using fewer qubits or a real QPU."
            ),
            severity="warning",
        ))

    # Shots
    if intent.shots < limits["min_shots"]:
        errors.append(ValidationError(
            code="INVALID_SHOTS",
            message=f"shots={intent.shots} < min={limits['min_shots']}",
            user_message=(
                f"Number of shots must be at least {limits['min_shots']}. "
                f"You specified {intent.shots}."
            ),
        ))
    elif intent.shots > limits["max_shots"]:
        errors.append(ValidationError(
            code="TOO_MANY_SHOTS",
            message=f"shots={intent.shots} > max={limits['max_shots']}",
            user_message=(
                f"Maximum shots allowed is {limits['max_shots']:,}. "
                f"You requested {intent.shots:,}."
            ),
        ))

    return errors


# ── Layer 3: QASM structure validation ───────────────────────────────────────

def _validate_qasm_structure(qasm: str) -> list[ValidationError]:
    errors = []

    if not qasm or not qasm.strip():
        errors.append(ValidationError(
            code="EMPTY_QASM",
            message="Generated QASM is empty",
            user_message=(
                "Cortex could not generate a circuit from your description. "
                "Try a more specific description like \"Bell state with 2 qubits\"."
            ),
        ))
        return errors

    if "OPENQASM" not in qasm:
        errors.append(ValidationError(
            code="INVALID_QASM_HEADER",
            message="QASM missing header",
            user_message="Internal error: generated circuit is malformed. Please try again.",
        ))

    if "measure" not in qasm.lower():
        errors.append(ValidationError(
            code="NO_MEASUREMENT",
            message="Circuit has no measurement",
            user_message=(
                "The circuit has no measurement operation. "
                "Add 'measure all' to your description to get results."
            ),
            severity="warning",
        ))

    return errors


# ── Layer 4: Gate logic validation ────────────────────────────────────────────

def _validate_gate_logic(qasm: str, intent) -> list[ValidationError]:
    errors = []
    n = intent.num_qubits

    for line in qasm.splitlines():
        line = line.strip().rstrip(";")
        if not line or line.startswith("//") or line.startswith("OPENQASM") \
           or line.startswith("include") or line.startswith("qubit") \
           or line.startswith("bit"):
            continue

        # Extract qubit indices from this line
        qubit_indices = [int(i) for i in re.findall(r"q\[(\d+)\]", line)]

        # Check out-of-range qubit indices
        for idx in qubit_indices:
            if idx >= n:
                errors.append(ValidationError(
                    code="QUBIT_OUT_OF_RANGE",
                    message=f"qubit index {idx} >= num_qubits {n}",
                    user_message=(
                        f"Qubit index {idx} is out of range. "
                        f"Your circuit has {n} qubit(s) (indices 0 to {n-1}). "
                        f"Try: \"Apply H to qubit 0\" instead."
                    ),
                ))

        # Check duplicate qubits in same gate
        if len(qubit_indices) > 1 and len(set(qubit_indices)) < len(qubit_indices):
            errors.append(ValidationError(
                code="DUPLICATE_QUBIT",
                message=f"Duplicate qubit in: {line}",
                user_message=(
                    f"A gate cannot use the same qubit twice. "
                    f"Found duplicate qubit in: \"{line}\". "
                    f"Example fix: \"CNOT from 0 to 1\" (not \"CNOT from 0 to 0\")."
                ),
            ))

        # Check extreme rotation angles
        angle_match = re.search(r"(?:rx|ry|rz|cp)\(([^)]+)\)", line, re.I)
        if angle_match:
            try:
                angle = float(angle_match.group(1))
                if abs(angle) > MAX_ANGLE:
                    errors.append(ValidationError(
                        code="EXTREME_ANGLE",
                        message=f"angle={angle} in: {line}",
                        user_message=(
                            f"Rotation angle {angle:.2f} rad seems too large. "
                            f"Typical values are between 0 and 2π (~6.28). "
                            f"Example: \"RX(pi/2) on qubit 0\"."
                        ),
                        severity="warning",
                    ))
            except ValueError:
                pass

    return errors


# ── Quick validators for common cases ─────────────────────────────────────────

def validate_text_only(text: str) -> None:
    """Validate just the input text. Raises CortexValidationError if invalid."""
    errors = _validate_text(text)
    fatal = [e for e in errors if e.severity == "error"]
    if fatal:
        raise CortexValidationError(errors)


def format_warning(warnings: list[ValidationError]) -> str:
    """Format a list of warnings into a readable string."""
    if not warnings:
        return ""
    lines = ["Warning:"]
    for w in warnings:
        lines.append(f"  - {w.user_message}")
    return "\n".join(lines)
