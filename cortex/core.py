"""
cortex.core
===========
Main Cortex class — the single entry point for users.

v0.1: pattern-based NLP engine
v0.2: LLM-powered engine with v0.1 fallback
"""

from __future__ import annotations
from typing import Literal

from cortex.models import CortexResult, CircuitIntent, Backend
from cortex.nlp.engine import parse_intent, intent_to_qasm


class Cortex:
    """
    text4q Cortex — natural language quantum computing interface.

    Examples:
        # Local simulation, pattern engine (no credentials needed)
        cx = Cortex(backend="aer")
        result = cx.run("Bell state with 2 qubits")

        # LLM engine: accepts ANY circuit description
        cx = Cortex(backend="aer", nlp="llm", llm_backend="anthropic")
        result = cx.run("Simulate H2 molecule ground state with 4 qubits using VQE")

        # IBM Quantum
        cx = Cortex(backend="ibm_quantum", token="YOUR_TOKEN")
        result = cx.run("GHZ state, 5 qubits, 2048 shots")
    """

    def __init__(
        self,
        backend: str | Backend = Backend.AER,
        token: str | None = None,
        backend_name: str = "ibm_brisbane",
        connector=None,
        nlp: Literal["pattern", "llm"] = "pattern",
        llm_backend: Literal["anthropic", "openai"] = "anthropic",
        llm_model: str | None = None,
    ):
        self.backend = Backend(backend)
        self.nlp_mode = nlp
        self._connector = connector or self._build_connector(token, backend_name)
        self._llm_engine = None

        if nlp == "llm":
            self._llm_engine = self._build_llm_engine(llm_backend, llm_model)

    def _build_connector(self, token: str | None, backend_name: str):
        if self.backend == Backend.AER:
            from cortex.connectors.ibm import AerConnector
            return AerConnector()
        elif self.backend == Backend.IBM_QUANTUM:
            from cortex.connectors.ibm import IBMQuantumConnector
            return IBMQuantumConnector(token=token, backend_name=backend_name)
        else:
            raise NotImplementedError(f"Backend {self.backend} not yet supported.")

    def _build_llm_engine(self, llm_backend: str, llm_model: str | None):
        from cortex.nlp.llm_engine import LLMEngine, AnthropicBackend, OpenAIBackend
        if llm_backend == "anthropic":
            b = AnthropicBackend(model=llm_model or "claude-haiku-4-5-20251001")
        else:
            b = OpenAIBackend(model=llm_model or "gpt-4o-mini")
        return LLMEngine(backend=b, fallback=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, text: str) -> CircuitIntent:
        """Parse natural language into a CircuitIntent (without executing)."""
        if self._llm_engine:
            intent, _ = self._llm_engine.translate(text)
            return intent
        return parse_intent(text)

    def compile(self, text_or_intent: str | CircuitIntent) -> str:
        """Translate natural language (or intent) into OpenQASM 3.0."""
        if isinstance(text_or_intent, CircuitIntent):
            return intent_to_qasm(text_or_intent)
        if self._llm_engine:
            _, qasm = self._llm_engine.translate(text_or_intent)
            return qasm
        intent = parse_intent(text_or_intent)
        return intent_to_qasm(intent)

    def run(self, text: str) -> CortexResult:
        """
        Full pipeline: parse → compile → execute.

        Args:
            text: Natural language circuit description (any complexity in llm mode).

        Returns:
            CortexResult with counts, QASM, and metadata.
        """
        if self._llm_engine:
            intent, qasm = self._llm_engine.translate(text)
        else:
            intent = parse_intent(text)
            qasm   = intent_to_qasm(intent)

        return self._connector.execute(intent, qasm)

    def info(self) -> dict:
        """Return current configuration as a dict."""
        return {
            "version": "0.2.0-alpha",
            "backend": self.backend.value,
            "nlp_mode": self.nlp_mode,
            "llm_backend": self._llm_engine.backend_name if self._llm_engine else None,
        }

    def __repr__(self) -> str:
        nlp = f", nlp={self.nlp_mode!r}" if self.nlp_mode != "pattern" else ""
        return f"Cortex(backend={self.backend.value!r}{nlp})"
