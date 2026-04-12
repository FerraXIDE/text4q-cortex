"""
cortex.nlp.llm_engine
=====================
LLM-powered NLP engine for text4q Cortex v0.2.

Translates arbitrary natural language circuit descriptions into valid
OpenQASM 3.0 using an LLM backend (Anthropic Claude by default, OpenAI fallback).

The engine:
  1. Sends the user's text to the LLM with a strict quantum-expert system prompt
  2. Validates the returned QASM syntax using Qiskit's parser
  3. Falls back to the pattern-based engine (v0.1) if validation fails
  4. Extracts CircuitIntent metadata from the LLM response

Architecture:
    User text ──► LLMEngine.translate()
                      │
                      ├── LLMBackend (Anthropic / OpenAI / Local)
                      │       └── returns raw QASM string
                      │
                      ├── QASMValidator.validate()
                      │       └── parse with Qiskit, raise on error
                      │
                      └── CortexResult / CircuitIntent
"""

from __future__ import annotations

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from cortex.models import CircuitIntent
from cortex.nlp.engine import parse_intent, intent_to_qasm  # v0.1 fallback

logger = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert quantum computing engineer and OpenQASM 3.0 compiler.
Your task: translate natural language circuit descriptions into valid OpenQASM 3.0 code.

RULES:
- Always output valid OpenQASM 3.0 (not 2.0).
- Start with: OPENQASM 3.0;
- Include: include "stdgates.inc";
- Declare all qubits as: qubit[N] q;
- Declare classical bits as: bit[N] c;
- End with measurement: c = measure q;
- Use standard gates: h, x, y, z, cx, ccx, rx, ry, rz, cp, swap, cz, t, s
- For parametric gates use exact values: rx(pi/2), rz(pi/4), cp(pi/8)
- Add a brief comment at the top explaining the circuit.
- Do NOT include markdown fences (```), just raw QASM.

ALSO output a JSON block at the end (after the QASM) with this exact format:
---METADATA---
{"num_qubits": N, "circuit_type": "type", "shots": 1024, "description": "..."}
---END---

Where circuit_type is one of: bell_state, ghz, qft, grover, vqe, qaoa,
teleportation, random_unitary, hamiltonian_sim, custom.
"""

USER_TEMPLATE = """\
Translate this quantum circuit description to OpenQASM 3.0:

\"{text}\"

Output only the QASM code followed by the ---METADATA--- block. No explanations.
"""


# ── LLM Backend abstraction ───────────────────────────────────────────────────

class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send system + user message and return the raw text response."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class AnthropicBackend(LLMBackend):
    """Uses Claude (claude-haiku for speed, claude-sonnet for quality)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text


class OpenAIBackend(LLMBackend):
    """Uses GPT-4o-mini by default."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def complete(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


# ── QASM Validator ────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    num_qubits: int = 0
    num_classical: int = 0
    error: str | None = None


def validate_qasm(qasm: str) -> ValidationResult:
    """
    Validate an OpenQASM 3.0 string using structural checks.

    Qiskit's from_qasm_str only supports QASM 2.0, so we validate via
    a structural parser that checks the required sections are present and
    extracts qubit/classical bit counts from declarations.
    """
    if not qasm or not qasm.strip():
        return ValidationResult(valid=False, error="Empty QASM string")

    stripped = qasm.strip()

    # Must start with OPENQASM header
    if not re.search(r"OPENQASM\s+(2\.0|3\.0)\s*;", stripped):
        return ValidationResult(valid=False, error="Missing OPENQASM version header")

    # Must declare qubits (QASM 3.0: "qubit[N] name" or QASM 2.0: "qreg name[N]")
    q3 = re.search(r"qubit\[(\d+)\]\s+\w+\s*;", stripped)
    q2 = re.search(r"qreg\s+\w+\[(\d+)\]\s*;", stripped)
    qubit_match = q3 or q2
    if not qubit_match:
        return ValidationResult(valid=False, error="No qubit declaration found")

    # Must declare classical bits
    c3 = re.search(r"bit\[(\d+)\]\s+\w+\s*;", stripped)
    c2 = re.search(r"creg\s+\w+\[(\d+)\]\s*;", stripped)
    classical_match = c3 or c2
    if not classical_match:
        return ValidationResult(valid=False, error="No classical bit declaration found")

    # Must contain a measurement
    if "measure" not in stripped:
        return ValidationResult(valid=False, error="No measurement found")

    # Must contain at least one gate instruction
    gate_re = re.compile(
        r"^\s*(h|x|y|z|cx|cy|cz|ccx|rx|ry|rz|cp|swap|t|s|tdg|sdg|u)\s",
        re.M | re.I,
    )
    if not gate_re.search(stripped):
        return ValidationResult(valid=False, error="No gate instructions found")

    num_qubits   = int(qubit_match.group(1))
    num_classical = int(classical_match.group(1))

    return ValidationResult(valid=True, num_qubits=num_qubits, num_classical=num_classical)


# ── Metadata extractor ────────────────────────────────────────────────────────

_META_RE = re.compile(
    r"---METADATA---\s*(\{.*?\})\s*---END---",
    re.DOTALL,
)

def extract_qasm_and_meta(raw: str) -> tuple[str, dict]:
    """
    Split LLM response into QASM code and metadata JSON.
    Returns (qasm_string, metadata_dict).
    """
    meta: dict = {}
    m = _META_RE.search(raw)
    if m:
        try:
            meta = json.loads(m.group(1))
        except json.JSONDecodeError:
            logger.warning("Could not parse metadata JSON from LLM response")
        qasm = raw[: m.start()].strip()
    else:
        qasm = raw.strip()

    # Strip any accidental markdown fences
    qasm = re.sub(r"^```[a-z]*\n?", "", qasm, flags=re.M).strip("`").strip()

    return qasm, meta


# ── Main LLM Engine ───────────────────────────────────────────────────────────

class LLMEngine:
    """
    LLM-powered circuit compiler for text4q Cortex v0.2.

    Falls back transparently to the v0.1 pattern engine if:
      - LLM API is unavailable / no key configured
      - LLM output fails QASM validation after max_retries attempts
    """

    def __init__(
        self,
        backend: LLMBackend | Literal["anthropic", "openai"] = "anthropic",
        max_retries: int = 2,
        fallback: bool = True,
    ):
        if isinstance(backend, str):
            self._backend = (
                AnthropicBackend() if backend == "anthropic" else OpenAIBackend()
            )
        else:
            self._backend = backend

        self.max_retries = max_retries
        self.fallback = fallback

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def translate(self, text: str) -> tuple[CircuitIntent, str]:
        """
        Translate natural language to (CircuitIntent, OpenQASM 3.0 string).

        Tries the LLM up to max_retries times. If all attempts fail and
        fallback=True, delegates to the v0.1 pattern engine.

        Returns:
            (intent, qasm) tuple ready to pass to a connector.
        """
        last_error: str = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"LLM attempt {attempt}/{self.max_retries} via {self.backend_name}")
                raw = self._backend.complete(
                    system=SYSTEM_PROMPT,
                    user=USER_TEMPLATE.format(text=text),
                )
                qasm, meta = extract_qasm_and_meta(raw)
                validation = validate_qasm(qasm)

                if validation.valid:
                    intent = self._build_intent(text, qasm, meta, validation)
                    logger.info(f"LLM compiled circuit: {intent.circuit_type} ({intent.num_qubits}q)")
                    return intent, qasm

                last_error = validation.error or "Unknown QASM error"
                logger.warning(f"QASM validation failed (attempt {attempt}): {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM backend error (attempt {attempt}): {e}")

        # All retries exhausted
        if self.fallback:
            logger.info("Falling back to pattern-based engine (v0.1)")
            intent = parse_intent(text)
            qasm = intent_to_qasm(intent)
            intent.metadata["llm_fallback"] = True
            intent.metadata["llm_error"] = last_error
            return intent, qasm

        raise RuntimeError(
            f"LLM translation failed after {self.max_retries} attempts: {last_error}"
        )

    def _build_intent(
        self,
        text: str,
        qasm: str,
        meta: dict,
        validation: ValidationResult,
    ) -> CircuitIntent:
        """Build a CircuitIntent from LLM metadata + validated circuit."""
        # Extract shots from the original text if not in metadata
        from cortex.nlp.engine import _extract_shots, _extract_noise
        shots = meta.get("shots") or _extract_shots(text)
        noise = _extract_noise(text)

        return CircuitIntent(
            raw_text=text,
            num_qubits=validation.num_qubits or meta.get("num_qubits", 2),
            circuit_type=meta.get("circuit_type", "custom"),
            shots=shots,
            noise_model=noise,
            metadata={
                "llm_backend": self.backend_name,
                "description": meta.get("description", ""),
                "llm_generated": True,
            },
        )
