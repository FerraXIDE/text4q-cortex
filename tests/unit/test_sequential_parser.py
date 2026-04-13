"""Unit tests for the sequential command parser."""

import pytest
from cortex.nlp.sequential import (
    parse_sequential, sequential_to_qasm, parse_and_compile,
    GateCommand, MeasureCommand,
)


def is_sequential_text(text):
    from cortex.nlp.sequential import _GATE_RE, _QUBIT_IDX_RE, _FROM_TO_RE
    has_gate   = bool(_GATE_RE.search(text))
    has_qubits = bool(_QUBIT_IDX_RE.search(text) or _FROM_TO_RE.search(text))
    return has_gate and has_qubits


class TestParseSequential:

    def test_bell_via_sequential(self):
        result = parse_sequential(
            "Apply H to qubit 0, then CNOT from 0 to 1, measure all"
        )
        assert result.valid
        assert result.num_qubits == 2
        assert len(result.commands) == 3

    def test_ghz_via_sequential(self):
        result = parse_sequential(
            "Apply H to qubit 0, CNOT from 0 to 1, CNOT from 1 to 2, measure all"
        )
        assert result.valid
        assert result.num_qubits == 3

    def test_toffoli_three_qubits(self):
        result = parse_sequential("Apply Toffoli on qubits 0, 1 and 2, measure all")
        assert result.valid
        gates = [c for c in result.commands if isinstance(c, GateCommand)]
        assert any(g.gate == "ccx" for g in gates)
        toffoli = next(g for g in gates if g.gate == "ccx")
        assert len(toffoli.qubits) == 3

    def test_swap_two_qubits(self):
        result = parse_sequential("Swap qubit 0 and qubit 1, measure all")
        assert result.valid
        gates = [c for c in result.commands if isinstance(c, GateCommand)]
        assert any(g.gate == "swap" for g in gates)

    def test_parametric_rx(self):
        result = parse_sequential("RX(pi/2) on qubit 0, measure all")
        assert result.valid
        gates = [c for c in result.commands if isinstance(c, GateCommand)]
        rx = next(g for g in gates if g.gate == "rx")
        assert abs(rx.angle - 1.5707963) < 0.001

    def test_parametric_rz(self):
        result = parse_sequential("RZ(pi/4) on qubit 1, measure all")
        assert result.valid
        gates = [c for c in result.commands if isinstance(c, GateCommand)]
        rz = next(g for g in gates if g.gate == "rz")
        assert abs(rz.angle - 0.7853981) < 0.001

    def test_multiple_single_qubit_gates(self):
        result = parse_sequential("X on qubit 0, Y on qubit 1, Z on qubit 2, measure all")
        assert result.valid
        assert result.num_qubits == 3
        gates = [c for c in result.commands if isinstance(c, GateCommand)]
        gate_names = [g.gate for g in gates]
        assert "x" in gate_names
        assert "y" in gate_names
        assert "z" in gate_names

    def test_measure_all_detected(self):
        result = parse_sequential("H on qubit 0, measure all")
        measures = [c for c in result.commands if isinstance(c, MeasureCommand)]
        assert len(measures) == 1
        assert measures[0].qubits == list(range(result.num_qubits))

    def test_measure_specific_qubit(self):
        result = parse_sequential("H on qubit 0, measure qubit 0")
        measures = [c for c in result.commands if isinstance(c, MeasureCommand)]
        assert 0 in measures[0].qubits

    def test_qubit_out_of_range_error(self):
        result = parse_sequential("H on qubit 0, CNOT from 0 to 5, measure all")
        # qubit 5 with only 2 declared → should either expand or error
        assert result.num_qubits >= 6 or len(result.errors) > 0

    def test_auto_measure_added(self):
        result = parse_sequential("H on qubit 0, CNOT from 0 to 1")
        qasm = sequential_to_qasm(result)
        assert "measure" in qasm

    def test_hadamard_alias(self):
        result = parse_sequential("Apply Hadamard to qubit 0, measure all")
        gates = [c for c in result.commands if isinstance(c, GateCommand)]
        assert any(g.gate == "h" for g in gates)

    def test_cnot_alias(self):
        result = parse_sequential("CNOT from 0 to 1, measure all")
        gates = [c for c in result.commands if isinstance(c, GateCommand)]
        assert any(g.gate == "cx" for g in gates)


class TestSequentialToQasm:

    def test_bell_qasm_valid(self):
        result = parse_sequential(
            "Apply H to qubit 0, CNOT from 0 to 1, measure all"
        )
        qasm = sequential_to_qasm(result)
        assert "OPENQASM 3.0" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0], q[1]" in qasm
        assert "measure" in qasm

    def test_qasm_has_correct_qubit_count(self):
        result = parse_sequential(
            "H on qubit 0, CNOT from 0 to 2, measure all"
        )
        qasm = sequential_to_qasm(result)
        assert "qubit[3]" in qasm

    def test_parametric_gate_in_qasm(self):
        result = parse_sequential("RX(pi/2) on qubit 0, measure all")
        qasm = sequential_to_qasm(result)
        assert "rx(" in qasm

    def test_toffoli_in_qasm(self):
        result = parse_sequential("Toffoli on qubits 0, 1 and 2, measure all")
        qasm = sequential_to_qasm(result)
        assert "ccx q[0], q[1], q[2]" in qasm

    def test_invalid_result_raises(self):
        from cortex.nlp.sequential import ParseResult
        bad = ParseResult(commands=[], num_qubits=2, errors=["forced error"])
        with pytest.raises(ValueError):
            sequential_to_qasm(bad)


class TestParseAndCompile:

    def test_returns_tuple(self):
        result, qasm = parse_and_compile("H on qubit 0, measure all")
        assert result.valid
        assert "OPENQASM 3.0" in qasm

    def test_empty_on_error(self):
        # Force an error by passing completely invalid text
        result, qasm = parse_and_compile("qubit 999 overflow error gate")
        if not result.valid:
            assert qasm == ""


class TestIsSequential:

    def test_bell_sequential_detected(self):
        assert is_sequential_text(
            "Apply H to qubit 0, CNOT from 0 to 1"
        )

    def test_named_circuit_not_sequential(self):
        # "Bell state" should not be detected as sequential
        # (no explicit gate name + qubit index combo)
        text = "Bell state with 2 qubits"
        from cortex.nlp.sequential import _GATE_RE, _FROM_TO_RE, _QUBIT_IDX_RE
        has_gate   = bool(_GATE_RE.search(text))
        has_qubits = bool(_QUBIT_IDX_RE.search(text) or _FROM_TO_RE.search(text))
        assert not (has_gate and has_qubits)

    def test_rx_sequential_detected(self):
        assert is_sequential_text("RX(pi/2) on qubit 0")
