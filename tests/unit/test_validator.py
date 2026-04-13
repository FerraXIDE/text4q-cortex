"""
Unit tests for the intelligent error validator.
"""

import pytest
from cortex.nlp.validator import (
    validate_all, validate_text_only,
    CortexValidationError, ValidationError,
    _validate_text, _validate_intent, _validate_qasm_structure, _validate_gate_logic,
)
from cortex.models import CircuitIntent


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_intent(num_qubits=2, shots=1024, circuit_type="bell_state"):
    return CircuitIntent(
        raw_text="Bell state",
        num_qubits=num_qubits,
        circuit_type=circuit_type,
        shots=shots,
    )

VALID_BELL_QASM = """\
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


# ── Text validation ───────────────────────────────────────────────────────────

class TestTextValidation:

    def test_empty_string_raises(self):
        with pytest.raises(CortexValidationError) as exc:
            validate_text_only("")
        assert exc.value.fatal[0].code == "EMPTY_INPUT"

    def test_whitespace_only_raises(self):
        with pytest.raises(CortexValidationError):
            validate_text_only("   ")

    def test_too_long_raises(self):
        with pytest.raises(CortexValidationError) as exc:
            validate_text_only("A" * 1001)
        assert exc.value.fatal[0].code == "INPUT_TOO_LONG"

    def test_valid_text_passes(self):
        validate_text_only("Bell state with 2 qubits")  # should not raise

    def test_raw_qasm_gives_warning(self):
        errors = _validate_text("OPENQASM 3.0;\nqubit[2] q;")
        warnings = [e for e in errors if e.severity == "warning"]
        assert any(e.code == "RAW_QASM_INPUT" for e in warnings)

    def test_error_message_is_helpful(self):
        with pytest.raises(CortexValidationError) as exc:
            validate_text_only("")
        assert "Example" in exc.value.user_message


# ── Intent validation ─────────────────────────────────────────────────────────

class TestIntentValidation:

    def test_too_many_qubits_aer(self):
        intent = make_intent(num_qubits=31)
        errors = _validate_intent(intent, "aer")
        assert any(e.code == "TOO_MANY_QUBITS" for e in errors)

    def test_max_qubits_aer_passes(self):
        intent = make_intent(num_qubits=30)
        errors = _validate_intent(intent, "aer")
        fatal = [e for e in errors if e.severity == "error"]
        assert not any(e.code == "TOO_MANY_QUBITS" for e in fatal)

    def test_zero_qubits_raises(self):
        intent = make_intent(num_qubits=0)
        errors = _validate_intent(intent, "aer")
        assert any(e.code == "INVALID_QUBIT_COUNT" for e in errors)

    def test_negative_shots_raises(self):
        intent = make_intent(shots=-1)
        errors = _validate_intent(intent, "aer")
        assert any(e.code == "INVALID_SHOTS" for e in errors)

    def test_zero_shots_raises(self):
        intent = make_intent(shots=0)
        errors = _validate_intent(intent, "aer")
        assert any(e.code == "INVALID_SHOTS" for e in errors)

    def test_too_many_shots_raises(self):
        intent = make_intent(shots=200_000)
        errors = _validate_intent(intent, "aer")
        assert any(e.code == "TOO_MANY_SHOTS" for e in errors)

    def test_valid_shots_pass(self):
        intent = make_intent(shots=1024)
        errors = _validate_intent(intent, "aer")
        fatal = [e for e in errors if e.severity == "error"]
        assert not fatal

    def test_large_simulation_warning(self):
        intent = make_intent(num_qubits=25)
        errors = _validate_intent(intent, "aer")
        warnings = [e for e in errors if e.severity == "warning"]
        assert any(e.code == "LARGE_SIMULATION" for e in warnings)

    def test_ibm_allows_more_qubits(self):
        intent = make_intent(num_qubits=50)
        errors = _validate_intent(intent, "ibm_quantum")
        fatal = [e for e in errors if e.severity == "error"]
        assert not any(e.code == "TOO_MANY_QUBITS" for e in fatal)


# ── QASM structure validation ─────────────────────────────────────────────────

class TestQASMValidation:

    def test_empty_qasm_raises(self):
        errors = _validate_qasm_structure("")
        assert any(e.code == "EMPTY_QASM" for e in errors)

    def test_valid_qasm_passes(self):
        errors = _validate_qasm_structure(VALID_BELL_QASM)
        fatal = [e for e in errors if e.severity == "error"]
        assert not fatal

    def test_missing_header_raises(self):
        errors = _validate_qasm_structure("qubit[2] q; h q[0];")
        assert any(e.code == "INVALID_QASM_HEADER" for e in errors)

    def test_no_measurement_warning(self):
        qasm = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];"
        errors = _validate_qasm_structure(qasm)
        warnings = [e for e in errors if e.severity == "warning"]
        assert any(e.code == "NO_MEASUREMENT" for e in warnings)


# ── Gate logic validation ─────────────────────────────────────────────────────

class TestGateLogicValidation:

    def test_duplicate_qubit_cnot(self):
        qasm = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\ncx q[0], q[0];\nc = measure q;"
        intent = make_intent(num_qubits=2)
        errors = _validate_gate_logic(qasm, intent)
        assert any(e.code == "DUPLICATE_QUBIT" for e in errors)

    def test_valid_cnot_passes(self):
        intent = make_intent(num_qubits=2)
        errors = _validate_gate_logic(VALID_BELL_QASM, intent)
        fatal = [e for e in errors if e.severity == "error"]
        assert not any(e.code == "DUPLICATE_QUBIT" for e in fatal)

    def test_extreme_angle_warning(self):
        import math
        qasm = f"OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nrx({1000 * math.pi}) q[0];\nc = measure q;"
        intent = make_intent(num_qubits=1)
        errors = _validate_gate_logic(qasm, intent)
        warnings = [e for e in errors if e.severity == "warning"]
        assert any(e.code == "EXTREME_ANGLE" for e in warnings)

    def test_normal_angle_passes(self):
        import math
        qasm = f"OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nrx({math.pi/2}) q[0];\nc = measure q;"
        intent = make_intent(num_qubits=1)
        errors = _validate_gate_logic(qasm, intent)
        warnings = [e for e in errors if e.severity == "warning"]
        assert not any(e.code == "EXTREME_ANGLE" for e in warnings)


# ── Integration: validate_all ─────────────────────────────────────────────────

class TestValidateAll:

    def test_valid_circuit_passes(self):
        intent = make_intent()
        warnings = validate_all("Bell state", intent, VALID_BELL_QASM, backend="aer")
        assert isinstance(warnings, list)

    def test_empty_text_raises(self):
        intent = make_intent()
        with pytest.raises(CortexValidationError) as exc:
            validate_all("", intent, VALID_BELL_QASM)
        assert exc.value.fatal[0].code == "EMPTY_INPUT"

    def test_too_many_qubits_raises(self):
        intent = make_intent(num_qubits=200)
        with pytest.raises(CortexValidationError) as exc:
            validate_all("Bell state with 200 qubits", intent, VALID_BELL_QASM)
        assert exc.value.fatal[0].code == "TOO_MANY_QUBITS"

    def test_error_message_is_user_friendly(self):
        with pytest.raises(CortexValidationError) as exc:
            validate_all("", make_intent(), VALID_BELL_QASM)
        msg = str(exc.value)
        assert "Please provide" in msg or "Example" in msg


# ── Integration: Cortex.run() with validation ─────────────────────────────────

class TestCortexRunValidation:

    def test_empty_input_raises(self):
        from cortex import Cortex
        cx = Cortex(backend="aer")
        with pytest.raises(CortexValidationError):
            cx.run("")

    def test_valid_input_works(self):
        from cortex import Cortex
        cx = Cortex(backend="aer")
        result = cx.run("Bell state with 2 qubits")
        assert result.success
        assert result.counts

    def test_duplicate_qubit_raises(self):
        from cortex import Cortex
        cx = Cortex(backend="aer")
        with pytest.raises(CortexValidationError) as exc:
            cx.run("CNOT from 0 to 0, measure all")
        assert exc.value.fatal[0].code == "DUPLICATE_QUBIT"

    def test_too_many_qubits_raises(self):
        from cortex import Cortex
        cx = Cortex(backend="aer")
        with pytest.raises(CortexValidationError) as exc:
            cx.run("Bell state with 200 qubits")
        assert exc.value.fatal[0].code == "TOO_MANY_QUBITS"
