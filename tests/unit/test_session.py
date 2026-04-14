"""
Tests for the conversational session mode.
"""

import pytest
from cortex import Cortex
from cortex.session import (
    SessionState, SessionResponse,
    classify_session_command, SessionCommand,
)


# ── SessionState unit tests ───────────────────────────────────────────────────

class TestSessionState:

    def test_initial_state(self):
        s = SessionState()
        assert s.num_qubits == 0
        assert s.gate_count == 0
        assert not s.executed

    def test_add_gate(self):
        s = SessionState(num_qubits=2)
        s.add_gate("h q[0];")
        assert s.gate_count == 1

    def test_undo_removes_last_gate(self):
        s = SessionState(num_qubits=2)
        s.add_gate("h q[0];")
        s.add_gate("cx q[0], q[1];")
        removed = s.undo()
        assert "cx" in removed
        assert s.gate_count == 1

    def test_undo_empty_returns_none(self):
        s = SessionState()
        assert s.undo() is None

    def test_reset_clears_gates(self):
        s = SessionState(num_qubits=3)
        s.add_gate("h q[0];")
        s.add_gate("cx q[0], q[1];")
        s.reset()
        assert s.gate_count == 0
        assert s.num_qubits == 3  # qubit count preserved

    def test_build_qasm_includes_header(self):
        s = SessionState(num_qubits=2)
        s.add_gate("h q[0];")
        qasm = s.build_qasm()
        assert "OPENQASM 3.0" in qasm
        assert "qubit[2]" in qasm

    def test_build_qasm_includes_measurement(self):
        s = SessionState(num_qubits=2)
        s.add_gate("h q[0];")
        qasm = s.build_qasm(include_measure=True)
        assert "measure" in qasm

    def test_build_qasm_without_measure(self):
        s = SessionState(num_qubits=2)
        s.add_gate("h q[0];")
        qasm = s.build_qasm(include_measure=False)
        assert "measure" not in qasm

    def test_diagram_returns_string(self):
        s = SessionState(num_qubits=2)
        s.add_gate("h q[0];")
        diagram = s.diagram()
        assert isinstance(diagram, str)
        assert len(diagram) > 0

    def test_empty_diagram_message(self):
        s = SessionState()
        diagram = s.diagram()
        assert "empty" in diagram.lower()


# ── Command classification tests ──────────────────────────────────────────────

class TestClassifySessionCommand:

    def test_create_qubits(self):
        cmd, n = classify_session_command("Create 3 qubits")
        assert cmd == SessionCommand.SET_QUBITS
        assert n == 3

    def test_use_qubits(self):
        cmd, n = classify_session_command("Use 4 qubits")
        assert cmd == SessionCommand.SET_QUBITS
        assert n == 4

    def test_measure_all(self):
        cmd, _ = classify_session_command("measure all")
        assert cmd == SessionCommand.MEASURE

    def test_measure_alone(self):
        cmd, _ = classify_session_command("measure")
        assert cmd == SessionCommand.MEASURE

    def test_show_circuit(self):
        cmd, _ = classify_session_command("show circuit")
        assert cmd == SessionCommand.SHOW

    def test_display_circuit(self):
        cmd, _ = classify_session_command("display the circuit")
        assert cmd == SessionCommand.SHOW

    def test_undo(self):
        cmd, _ = classify_session_command("undo")
        assert cmd == SessionCommand.UNDO

    def test_reset(self):
        cmd, _ = classify_session_command("reset")
        assert cmd == SessionCommand.RESET

    def test_gate_count(self):
        cmd, _ = classify_session_command("how many gates")
        assert cmd == SessionCommand.GATE_COUNT

    def test_status(self):
        cmd, _ = classify_session_command("status")
        assert cmd == SessionCommand.STATUS

    def test_add_gate_h(self):
        cmd, _ = classify_session_command("Apply H to qubit 0")
        assert cmd == SessionCommand.ADD_GATE

    def test_add_gate_cnot(self):
        cmd, _ = classify_session_command("CNOT from 0 to 1")
        assert cmd == SessionCommand.ADD_GATE


# ── Integration: Cortex session tests ────────────────────────────────────────

class TestCortexSession:

    def test_session_context_manager(self):
        cx = Cortex(backend="aer")
        assert not cx._in_session
        with cx.session():
            assert cx._in_session
        assert not cx._in_session

    def test_session_set_qubits(self):
        cx = Cortex(backend="aer")
        with cx.session():
            r = cx.run("Create 2 qubits")
            assert r.kind == "ack"
            assert cx._session_state.num_qubits == 2

    def test_session_add_gate(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            r = cx.run("Apply H to qubit 0")
            assert r.kind == "ack"
            assert r.gate_count == 1

    def test_session_undo(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            cx.run("Apply H to qubit 0")
            cx.run("CNOT from 0 to 1")
            r = cx.run("undo")
            assert r.kind == "ack"
            assert cx._session_state.gate_count == 1

    def test_session_reset(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            cx.run("Apply H to qubit 0")
            r = cx.run("reset")
            assert cx._session_state.gate_count == 0

    def test_session_show_circuit(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            cx.run("Apply H to qubit 0")
            r = cx.run("show circuit")
            assert r.kind == "info"
            assert "circuit" in r.message.lower()

    def test_session_gate_count(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            cx.run("Apply H to qubit 0")
            cx.run("CNOT from 0 to 1")
            r = cx.run("how many gates")
            assert "2" in r.message

    def test_session_measure_executes(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            cx.run("Apply H to qubit 0")
            cx.run("CNOT from 0 to 1")
            r = cx.run("measure all")
            assert r.kind == "result"
            assert r.result is not None
            assert r.counts

    def test_session_bell_state_counts(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            cx.run("Apply H to qubit 0")
            cx.run("CNOT from 0 to 1")
            r = cx.run("measure all")
        counts = r.counts
        # Bell state: only 00 and 11 should appear
        assert all(k in ("00", "11") for k in counts)

    def test_session_ghz_state(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 3 qubits")
            cx.run("Apply H to qubit 0")
            cx.run("CNOT from 0 to 1")
            cx.run("CNOT from 0 to 2")
            r = cx.run("measure all")
        counts = r.counts
        assert all(k in ("000", "111") for k in counts)

    def test_measure_empty_circuit_returns_error(self):
        cx = Cortex(backend="aer")
        with cx.session():
            r = cx.run("measure all")
            assert r.kind == "error"

    def test_session_does_not_affect_normal_run(self):
        cx = Cortex(backend="aer")
        with cx.session():
            cx.run("Create 2 qubits")
            cx.run("Apply H to qubit 0")

        # After session ends, normal run() works
        r = cx.run("Bell state with 2 qubits")
        assert r.counts  # normal CortexResult, not SessionResponse

    def test_session_auto_detects_qubits(self):
        """Session should auto-expand qubit count when gates reference new qubits."""
        cx = Cortex(backend="aer")
        with cx.session():
            # No explicit qubit count set
            cx.run("Apply H to qubit 0")
            cx.run("CNOT from 0 to 1")
            assert cx._session_state.num_qubits >= 2
