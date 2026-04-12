"""Unit tests for the NLP engine."""

import pytest
from cortex.nlp.engine import parse_intent, intent_to_qasm


class TestParseIntent:

    def test_bell_state_detected(self):
        intent = parse_intent("Create a Bell state with 2 qubits")
        assert intent.circuit_type == "bell_state"
        assert intent.num_qubits == 2

    def test_ghz_state_detected(self):
        intent = parse_intent("GHZ state with 5 qubits")
        assert intent.circuit_type == "ghz"
        assert intent.num_qubits == 5

    def test_qft_detected(self):
        intent = parse_intent("Apply quantum Fourier transform to 4 qubits")
        assert intent.circuit_type == "qft"

    def test_shots_extracted(self):
        intent = parse_intent("Bell state, 2048 shots")
        assert intent.shots == 2048

    def test_default_shots(self):
        intent = parse_intent("Bell state")
        assert intent.shots == 1024

    def test_noise_t1_extracted(self):
        intent = parse_intent("Bell state T1=50us")
        assert intent.noise_model is not None
        assert intent.noise_model["T1_us"] == 50.0

    def test_noise_t2_extracted(self):
        intent = parse_intent("GHZ with T1=100us T2=80us")
        assert intent.noise_model["T2_us"] == 80.0

    def test_grover_detected(self):
        intent = parse_intent("Run Grover search on 4 qubits")
        assert intent.circuit_type == "grover"

    def test_teleportation_detected(self):
        intent = parse_intent("Quantum teleportation protocol")
        assert intent.circuit_type == "teleportation"


class TestIntentToQasm:

    def test_bell_qasm_valid(self):
        intent = parse_intent("Bell state")
        qasm = intent_to_qasm(intent)
        assert "OPENQASM 3.0" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0], q[1]" in qasm

    def test_ghz_qasm_has_correct_qubits(self):
        intent = parse_intent("GHZ state 5 qubits")
        qasm = intent_to_qasm(intent)
        assert "qubit[5]" in qasm
        assert "cx q[0], q[4]" in qasm

    def test_qft_qasm_generated(self):
        intent = parse_intent("QFT on 3 qubits")
        qasm = intent_to_qasm(intent)
        assert "OPENQASM 3.0" in qasm
        assert "qubit[3]" in qasm

    def test_teleportation_qasm_has_three_qubits(self):
        intent = parse_intent("Quantum teleportation")
        qasm = intent_to_qasm(intent)
        assert "qubit[3]" in qasm

    def test_custom_circuit_generates_stub(self):
        intent = parse_intent("Something completely custom")
        qasm = intent_to_qasm(intent)
        assert "OPENQASM 3.0" in qasm
        assert "TODO" in qasm
