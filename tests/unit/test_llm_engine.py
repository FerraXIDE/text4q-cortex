"""
Unit tests for the LLM engine (v0.2).
Uses mock backends so no API credentials are needed.
"""

import pytest
from unittest.mock import MagicMock
from cortex.nlp.llm_engine import (
    LLMEngine,
    LLMBackend,
    validate_qasm,
    extract_qasm_and_meta,
    ValidationResult,
)
from cortex.models import CircuitIntent


# ── Mock backend ──────────────────────────────────────────────────────────────

VALID_BELL_RESPONSE = """\
OPENQASM 3.0;
include "stdgates.inc";

// Bell state: maximally entangled 2-qubit state
qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];

c = measure q;

---METADATA---
{"num_qubits": 2, "circuit_type": "bell_state", "shots": 1024, "description": "Bell state"}
---END---
"""

VALID_GHZ_RESPONSE = """\
OPENQASM 3.0;
include "stdgates.inc";

// GHZ state with 5 qubits
qubit[5] q;
bit[5] c;

h q[0];
cx q[0], q[1];
cx q[0], q[2];
cx q[0], q[3];
cx q[0], q[4];

c = measure q;

---METADATA---
{"num_qubits": 5, "circuit_type": "ghz", "shots": 2048, "description": "5-qubit GHZ state"}
---END---
"""

INVALID_QASM_RESPONSE = """\
This is not valid QASM at all.
Just some random text.
---METADATA---
{"num_qubits": 2, "circuit_type": "custom", "shots": 1024, "description": "broken"}
---END---
"""


def make_mock_backend(response: str) -> LLMBackend:
    mock = MagicMock(spec=LLMBackend)
    mock.complete.return_value = response
    mock.name = "mock/test-model"
    return mock


# ── Tests: extract_qasm_and_meta ──────────────────────────────────────────────

class TestExtractQasmAndMeta:

    def test_extracts_qasm_correctly(self):
        qasm, meta = extract_qasm_and_meta(VALID_BELL_RESPONSE)
        assert "OPENQASM 3.0" in qasm
        assert "h q[0]" in qasm
        assert "METADATA" not in qasm

    def test_extracts_metadata_correctly(self):
        _, meta = extract_qasm_and_meta(VALID_BELL_RESPONSE)
        assert meta["circuit_type"] == "bell_state"
        assert meta["num_qubits"] == 2

    def test_strips_markdown_fences(self):
        raw = "```qasm\nOPENQASM 3.0;\n```"
        qasm, _ = extract_qasm_and_meta(raw)
        assert "```" not in qasm

    def test_no_metadata_block_returns_empty_dict(self):
        raw = "OPENQASM 3.0;\nqubit[2] q;"
        _, meta = extract_qasm_and_meta(raw)
        assert meta == {}


# ── Tests: validate_qasm ──────────────────────────────────────────────────────

class TestValidateQasm:

    def test_valid_bell_passes(self):
        qasm = """\
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""
        result = validate_qasm(qasm)
        assert result.valid is True
        assert result.num_qubits == 2

    def test_invalid_qasm_fails(self):
        result = validate_qasm("this is not qasm")
        assert result.valid is False
        assert result.error is not None

    def test_empty_string_fails(self):
        result = validate_qasm("")
        assert result.valid is False


# ── Tests: LLMEngine ─────────────────────────────────────────────────────────

class TestLLMEngine:

    def test_successful_translation_bell(self):
        engine = LLMEngine(backend=make_mock_backend(VALID_BELL_RESPONSE))
        intent, qasm = engine.translate("Create a Bell state")
        assert intent.circuit_type == "bell_state"
        assert intent.num_qubits == 2
        assert "OPENQASM 3.0" in qasm
        assert intent.metadata.get("llm_generated") is True

    def test_successful_translation_ghz(self):
        engine = LLMEngine(backend=make_mock_backend(VALID_GHZ_RESPONSE))
        intent, qasm = engine.translate("GHZ state 5 qubits 2048 shots")
        assert intent.circuit_type == "ghz"
        assert intent.num_qubits == 5
        assert intent.shots == 2048

    def test_fallback_on_invalid_qasm(self):
        """When LLM returns invalid QASM, should fall back to pattern engine."""
        engine = LLMEngine(
            backend=make_mock_backend(INVALID_QASM_RESPONSE),
            fallback=True,
        )
        intent, qasm = engine.translate("Bell state")
        # Fallback produces valid QASM
        assert "OPENQASM 3.0" in qasm
        assert intent.metadata.get("llm_fallback") is True

    def test_no_fallback_raises_on_invalid(self):
        """With fallback=False, invalid QASM should raise RuntimeError."""
        engine = LLMEngine(
            backend=make_mock_backend(INVALID_QASM_RESPONSE),
            fallback=False,
            max_retries=1,
        )
        with pytest.raises(RuntimeError, match="LLM translation failed"):
            engine.translate("Bell state")

    def test_llm_backend_called_with_correct_text(self):
        mock_backend = make_mock_backend(VALID_BELL_RESPONSE)
        engine = LLMEngine(backend=mock_backend)
        engine.translate("My custom circuit description")
        call_args = mock_backend.complete.call_args
        assert "My custom circuit description" in call_args[1]["user"]

    def test_retries_on_failure(self):
        """Should retry max_retries times before falling back."""
        mock_backend = make_mock_backend(INVALID_QASM_RESPONSE)
        engine = LLMEngine(backend=mock_backend, max_retries=3, fallback=True)
        engine.translate("Bell state")
        assert mock_backend.complete.call_count == 3

    def test_noise_extracted_from_text(self):
        engine = LLMEngine(backend=make_mock_backend(VALID_BELL_RESPONSE))
        intent, _ = engine.translate("Bell state T1=50us T2=30us")
        assert intent.noise_model is not None
        assert intent.noise_model["T1_us"] == 50.0
        assert intent.noise_model["T2_us"] == 30.0


# ── Tests: Cortex integration with LLM mode ───────────────────────────────────

class TestCortexLLMIntegration:

    def test_cortex_llm_mode_parse(self):
        from cortex.core import Cortex
        from cortex.nlp.llm_engine import LLMEngine
        mock_llm = LLMEngine(backend=make_mock_backend(VALID_BELL_RESPONSE))

        cx = Cortex(backend="aer", nlp="llm")
        cx._llm_engine = mock_llm

        intent = cx.parse("Create a Bell state")
        assert intent.circuit_type == "bell_state"

    def test_cortex_info_shows_llm_mode(self):
        from cortex.core import Cortex
        cx = Cortex(backend="aer", nlp="pattern")
        info = cx.info()
        assert info["nlp_mode"] == "pattern"
        assert info["llm_backend"] is None
