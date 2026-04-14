"""
cortex.core
===========
Main Cortex class — the single entry point for users.

v0.1: pattern-based NLP engine
v0.2: LLM-powered engine with v0.1 fallback
v0.8: circuit optimization + intelligent error validation
"""

from __future__ import annotations
from typing import Literal

from cortex.models import CortexResult, CircuitIntent, Backend
from cortex.nlp.engine import parse_intent, intent_to_qasm


class Cortex:
    """
    text4q Cortex — natural language quantum computing interface.

    Examples:
        # Local simulation (no credentials needed)
        cx = Cortex(backend="aer")
        result = cx.run("Bell state with 2 qubits")

        # With circuit optimization
        cx = Cortex(backend="aer", optimize=2)
        result = cx.run("Apply H H to qubit 0, CNOT from 0 to 1, measure all")
        print(result.intent.metadata["optimization"])

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
        optimize: int = 1,
    ):
        self.backend = Backend(backend)
        self.nlp_mode = nlp
        self._optimize_level = optimize
        self._connector = connector or self._build_connector(token, backend_name)
        self._llm_engine = None
        self._in_session = False
        self._session_state = None

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

    def optimize(self, qasm: str, level: int | None = None) -> str:
        """
        Optimize a QASM circuit string, removing redundant gates.

        Args:
            qasm:  OpenQASM 3.0 string to optimize.
            level: Override the instance optimize level (0-3).

        Returns:
            Optimized OpenQASM 3.0 string.
        """
        from cortex.nlp.optimizer import optimize_qasm
        lvl = level if level is not None else self._optimize_level
        result = optimize_qasm(qasm, backend=self.backend.value, level=lvl)
        return result.optimized_qasm

    def run(self, text: str) -> CortexResult:
        """
        Full pipeline: validate → parse → compile → optimize → execute.

        Automatically detects the input type:
        - Named circuits (Bell, GHZ, QFT...) → pattern engine
        - Sequential commands (Apply H to qubit 0, CNOT from 0 to 1...) → sequential parser
        - Arbitrary descriptions with llm mode → LLM engine

        Args:
            text: Natural language circuit description.

        Returns:
            CortexResult with counts, QASM, and metadata.

        Raises:
            CortexValidationError: If the input is invalid.
        """
        # If inside a conversational session, delegate to session handler
        if self._in_session:
            return self._run_in_session(text)

        from cortex.nlp.validator import validate_all, validate_text_only, CortexValidationError

        # Layer 1: validate text before anything else
        validate_text_only(text)

        try:
            if self._llm_engine:
                intent, qasm = self._llm_engine.translate(text)
            else:
                from cortex.nlp.engine import is_sequential, parse_sequential_intent
                if is_sequential(text):
                    intent, qasm = parse_sequential_intent(text)
                else:
                    intent = parse_intent(text)
                    qasm   = intent_to_qasm(intent)

            # Layers 2-4: validate intent + QASM
            warnings = validate_all(
                text, intent, qasm,
                backend=self.backend.value,
            )

            # Circuit optimization
            if self._optimize_level > 0:
                from cortex.nlp.optimizer import optimize_qasm
                opt = optimize_qasm(qasm, backend=self.backend.value, level=self._optimize_level)
                qasm = opt.optimized_qasm
                intent.metadata["optimization"] = {
                    "gates_before":   opt.original_gates,
                    "gates_after":    opt.optimized_gates,
                    "removed":        opt.gates_removed,
                    "reduction_pct":  opt.reduction_pct,
                    "method":         opt.method,
                }

            result = self._connector.execute(intent, qasm)

            # Attach warnings to result metadata
            if warnings:
                intent.metadata["warnings"] = [w.user_message for w in warnings]

            return result

        except CortexValidationError:
            raise
        except Exception as exc:
            fallback_intent = CircuitIntent(
                raw_text=text, num_qubits=0,
                circuit_type="unknown", shots=0,
            )
            return CortexResult(
                intent=fallback_intent,
                qasm="",
                counts={},
                backend=self.backend.value,
                shots=0,
                execution_time_ms=0.0,
                error=str(exc),
            )

    def info(self) -> dict:
        """Return current configuration as a dict."""
        return {
            "version":        "0.1.5",
            "backend":        self.backend.value,
            "nlp_mode":       self.nlp_mode,
            "optimize_level": self._optimize_level,
            "llm_backend":    self._llm_engine.backend_name if self._llm_engine else None,
        }

    def __repr__(self) -> str:
        nlp = f", nlp={self.nlp_mode!r}" if self.nlp_mode != "pattern" else ""
        opt = f", optimize={self._optimize_level}" if self._optimize_level != 1 else ""
        return f"Cortex(backend={self.backend.value!r}{nlp}{opt})"


    # ── Session / conversational mode ─────────────────────────────────────────

    def session(self):
        """
        Start a conversational session. Use as context manager:

            with cx.session():
                cx.run("Create 3 qubits")
                cx.run("Apply H to qubit 0")
                cx.run("CNOT from 0 to 1")
                result = cx.run("measure all")
        """
        from cortex.session import CortexSession
        return CortexSession(self)

    def _start_session(self) -> None:
        from cortex.session import SessionState
        self._session_state = SessionState()
        self._in_session = True

    def _end_session(self) -> None:
        self._session_state = None
        self._in_session = False

    def _run_in_session(self, text: str):
        """Handle a run() call while inside a conversational session."""
        from cortex.session import (
            classify_session_command, SessionCommand,
            SessionResponse, SessionState,
        )
        from cortex.nlp.sequential import parse_sequential

        state: SessionState = self._session_state
        cmd_type, extra = classify_session_command(text)

        # ── Set qubit count ───────────────────────────────────────────────────
        if cmd_type == SessionCommand.SET_QUBITS:
            state.num_qubits = extra
            state.history.append(text)
            return SessionResponse(
                kind="ack",
                message=f"Circuit initialized with {extra} qubits. "
                        f"Start adding gates.",
                gate_count=state.gate_count,
            )

        # ── Show diagram ──────────────────────────────────────────────────────
        if cmd_type == SessionCommand.SHOW:
            diagram = state.diagram()
            return SessionResponse(
                kind="info",
                message=f"Current circuit ({state.gate_count} gates, "
                        f"{state.num_qubits} qubits):\n\n{diagram}",
                gate_count=state.gate_count,
            )

        # ── Undo last gate ────────────────────────────────────────────────────
        if cmd_type == SessionCommand.UNDO:
            removed = state.undo()
            if removed:
                return SessionResponse(
                    kind="ack",
                    message=f"Removed: {removed}. "
                            f"Circuit now has {state.gate_count} gates.",
                    gate_count=state.gate_count,
                )
            return SessionResponse(
                kind="ack",
                message="Nothing to undo.",
                gate_count=0,
            )

        # ── Reset ─────────────────────────────────────────────────────────────
        if cmd_type == SessionCommand.RESET:
            n = state.num_qubits
            state.reset()
            return SessionResponse(
                kind="ack",
                message=f"Circuit reset. {n} qubits ready for new gates.",
                gate_count=0,
            )

        # ── Gate count ────────────────────────────────────────────────────────
        if cmd_type == SessionCommand.GATE_COUNT:
            return SessionResponse(
                kind="info",
                message=f"The circuit has {state.gate_count} gate(s) "
                        f"on {state.num_qubits} qubit(s).",
                gate_count=state.gate_count,
            )

        # ── Status ────────────────────────────────────────────────────────────
        if cmd_type == SessionCommand.STATUS:
            return SessionResponse(
                kind="info",
                message=f"Session active — {state.gate_count} gate(s), "
                        f"{state.num_qubits} qubit(s). "
                        f"Say 'measure all' to execute.",
                gate_count=state.gate_count,
            )

        # ── Measure — execute accumulated circuit ─────────────────────────────
        if cmd_type == SessionCommand.MEASURE:
            if state.gate_count == 0:
                return SessionResponse(
                    kind="error",
                    message="No gates to execute. Add gates first.",
                    gate_count=0,
                )
            qasm = state.build_qasm(include_measure=True)
            from cortex.models import CircuitIntent
            intent = CircuitIntent(
                raw_text="session",
                num_qubits=state.num_qubits or 2,
                circuit_type="session",
                shots=1024,
                metadata={"session": True, "history": list(state.history)},
            )
            result = self._connector.execute(intent, qasm)
            state.executed = True
            return SessionResponse(
                kind="result",
                message=f"Executed {state.gate_count} gates. "
                        f"Top result: |{result.most_probable()}⟩",
                result=result,
                gate_count=state.gate_count,
            )

        # ── Add gate ──────────────────────────────────────────────────────────
        if cmd_type == SessionCommand.ADD_GATE:
            # Auto-detect qubit count from gate if not set
            parsed = parse_sequential(text)
            if not parsed.valid:
                return SessionResponse(
                    kind="error",
                    message=f"Could not parse: {parsed.errors[0] if parsed.errors else text}",
                    gate_count=state.gate_count,
                )

            # Update qubit count if needed
            if parsed.num_qubits > state.num_qubits:
                state.num_qubits = parsed.num_qubits

            # Extract just the gate lines from parsed result
            from cortex.session import SessionState
            from cortex.nlp.sequential import sequential_to_qasm, GateCommand
            for cmd in parsed.commands:
                if isinstance(cmd, GateCommand):
                    state.add_gate(cmd.to_qasm())

            state.history.append(text)
            added = len([c for c in parsed.commands
                        if hasattr(c, 'gate')])
            return SessionResponse(
                kind="ack",
                message=f"Added {added} gate(s). "
                        f"Circuit has {state.gate_count} gate(s) total. "
                        f"Say 'measure all' to execute or keep adding gates.",
                gate_added=text,
                gate_count=state.gate_count,
            )

        return SessionResponse(
            kind="error",
            message=f"Command not recognized in session context: '{text}'",
            gate_count=state.gate_count,
        )
