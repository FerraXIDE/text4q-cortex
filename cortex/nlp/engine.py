"""
cortex.nlp.engine
=================
NLP engine: translates natural language descriptions into OpenQASM 3.0 circuits.

v0.1 uses pattern-matching for common circuit types (Bell, GHZ, QFT, random).
v0.2 will plug in an LLM backend (OpenAI / local model) for arbitrary circuits.
"""

from __future__ import annotations
import re
from cortex.models import CircuitIntent


# ── Pattern library ──────────────────────────────────────────────────────────

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(bell|epr|entangl)\b", re.I),             "bell_state"),
    (re.compile(r"\bghz\b",               re.I),              "ghz"),
    (re.compile(r"\bqft|fourier\b",       re.I),              "qft"),
    (re.compile(r"\b(random|haar|unitary)\b", re.I),           "random_unitary"),
    (re.compile(r"\b(grover|search)\b",   re.I),              "grover"),
    (re.compile(r"\b(vqe|variational)\b", re.I),              "vqe"),
    (re.compile(r"\b(qaoa)\b",            re.I),              "qaoa"),
    (re.compile(r"\bteleport",             re.I),              "teleportation"),
]

_QUBIT_RE  = re.compile(r"(\d+)\s*qubit", re.I)
_SHOTS_RE  = re.compile(r"(\d[\d_]*)\s*shot", re.I)
_T1_RE     = re.compile(r"t1\s*[=:]\s*([\d.]+)\s*(us|µs|ms)?", re.I)
_T2_RE     = re.compile(r"t2\s*[=:]\s*([\d.]+)\s*(us|µs|ms)?", re.I)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_intent(text: str) -> CircuitIntent:
    """
    Parse a natural language description into a CircuitIntent.

    Args:
        text: e.g. "Bell state with 2 qubits, noise T1=50us, 1024 shots"

    Returns:
        CircuitIntent with extracted parameters.
    """
    circuit_type = _detect_circuit_type(text)
    num_qubits   = _extract_qubits(text, circuit_type)
    shots        = _extract_shots(text)
    noise_model  = _extract_noise(text)

    return CircuitIntent(
        raw_text=text,
        num_qubits=num_qubits,
        circuit_type=circuit_type,
        shots=shots,
        noise_model=noise_model,
    )


def intent_to_qasm(intent: CircuitIntent) -> str:
    """
    Convert a CircuitIntent into an OpenQASM 3.0 circuit string.
    Dispatches to the appropriate circuit builder.
    """
    builders = {
        "bell_state":      _build_bell,
        "ghz":             _build_ghz,
        "qft":             _build_qft,
        "random_unitary":  _build_random,
        "grover":          _build_grover_stub,
        "teleportation":   _build_teleportation,
    }
    builder = builders.get(intent.circuit_type, _build_generic_stub)
    return builder(intent)


# ── Pattern detection ─────────────────────────────────────────────────────────

def _detect_circuit_type(text: str) -> str:
    for pattern, name in _PATTERNS:
        if pattern.search(text):
            return name
    return "custom"


def _extract_qubits(text: str, circuit_type: str) -> int:
    m = _QUBIT_RE.search(text)
    if m:
        return int(m.group(1))
    # sensible defaults per circuit type
    defaults = {"bell_state": 2, "ghz": 3, "qft": 4, "teleportation": 3}
    return defaults.get(circuit_type, 4)


def _extract_shots(text: str) -> int:
    m = _SHOTS_RE.search(text)
    if m:
        return int(m.group(1).replace("_", ""))
    return 1024


def _extract_noise(text: str) -> dict | None:
    noise: dict = {}
    m1 = _T1_RE.search(text)
    m2 = _T2_RE.search(text)
    if m1:
        val = float(m1.group(1))
        unit = (m1.group(2) or "us").lower()
        noise["T1_us"] = val * 1000 if "ms" in unit else val
    if m2:
        val = float(m2.group(1))
        unit = (m2.group(2) or "us").lower()
        noise["T2_us"] = val * 1000 if "ms" in unit else val
    return noise if noise else None


# ── Circuit builders (OpenQASM 3.0) ──────────────────────────────────────────

def _build_bell(intent: CircuitIntent) -> str:
    return """\
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];

c = measure q;
"""


def _build_ghz(intent: CircuitIntent) -> str:
    n = intent.num_qubits
    cx_lines = "\n".join(f"cx q[0], q[{i}];" for i in range(1, n))
    return f"""\
OPENQASM 3.0;
include "stdgates.inc";

qubit[{n}] q;
bit[{n}] c;

h q[0];
{cx_lines}

c = measure q;
"""


def _build_qft(intent: CircuitIntent) -> str:
    n = intent.num_qubits
    gates = []
    for i in range(n):
        gates.append(f"h q[{i}];")
        for j in range(i + 1, n):
            k = j - i + 1
            gates.append(f"cp(pi/{2**(k-1)}) q[{j}], q[{i}];")
    gates_str = "\n".join(gates)
    return f"""\
OPENQASM 3.0;
include "stdgates.inc";

qubit[{n}] q;
bit[{n}] c;

{gates_str}

c = measure q;
"""


def _build_random(intent: CircuitIntent) -> str:
    n = intent.num_qubits
    layer = "\n".join(f"h q[{i}];\nrz(pi/4) q[{i}];" for i in range(n))
    return f"""\
OPENQASM 3.0;
include "stdgates.inc";

qubit[{n}] q;
bit[{n}] c;

{layer}

c = measure q;
"""


def _build_grover_stub(intent: CircuitIntent) -> str:
    n = intent.num_qubits
    return f"""\
OPENQASM 3.0;
include "stdgates.inc";

// Grover's algorithm stub — oracle TBD
qubit[{n}] q;
bit[{n}] c;

// Uniform superposition
{"".join(f"h q[{i}];" + chr(10) for i in range(n))}
// TODO: oracle + diffusion operator

c = measure q;
"""


def _build_teleportation(intent: CircuitIntent) -> str:
    return """\
OPENQASM 3.0;
include "stdgates.inc";

// Quantum teleportation: q[0]=message, q[1..2]=Bell pair
qubit[3] q;
bit[3] c;

// Prepare Bell pair
h q[1];
cx q[1], q[2];

// Alice's operations
cx q[0], q[1];
h q[0];

// Measure Alice's qubits
c[0] = measure q[0];
c[1] = measure q[1];

// Bob's corrections (classically controlled)
if (c[1]) x q[2];
if (c[0]) z q[2];

c[2] = measure q[2];
"""


def _build_generic_stub(intent: CircuitIntent) -> str:
    n = intent.num_qubits
    return f"""\
OPENQASM 3.0;
include "stdgates.inc";

// Auto-generated stub for: {intent.raw_text}
qubit[{n}] q;
bit[{n}] c;

// TODO: implement circuit for "{intent.circuit_type}"
h q[0];

c = measure q;
"""
