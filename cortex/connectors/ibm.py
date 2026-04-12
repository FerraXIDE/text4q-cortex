"""
cortex.connectors.ibm
=====================
IBM Quantum connector using qiskit-ibm-runtime.
Supports both real QPU execution and local Aer simulation.
"""

from __future__ import annotations
import re
import time
import os

from cortex.models import CortexResult, CircuitIntent


class IBMQuantumConnector:
    """
    Connects text4q Cortex to IBM Quantum backends.

    Usage:
        connector = IBMQuantumConnector(token="YOUR_TOKEN")
        # or set env var IBM_QUANTUM_TOKEN
    """

    def __init__(
        self,
        token: str | None = None,
        backend_name: str = "ibm_brisbane",
        use_simulator: bool = False,
    ):
        self.token = token or os.environ.get("IBM_QUANTUM_TOKEN", "")
        self.backend_name = backend_name
        self.use_simulator = use_simulator
        self._service = None
        self._backend = None

    def _ensure_connected(self) -> None:
        if self._backend is not None:
            return
        if self.use_simulator:
            self._connect_aer()
        else:
            self._connect_ibm()

    def _connect_ibm(self) -> None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.token,
            )
            self._backend = self._service.backend(self.backend_name)
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to IBM Quantum: {e}\n"
                "Set IBM_QUANTUM_TOKEN env var or pass token= to IBMQuantumConnector."
            ) from e

    def _connect_aer(self) -> None:
        try:
            from qiskit_aer import AerSimulator
            self._backend = AerSimulator()
        except ImportError as e:
            raise ImportError(
                "qiskit-aer not installed. Run: pip install qiskit-aer"
            ) from e

    def execute(self, intent: CircuitIntent, qasm: str) -> CortexResult:
        self._ensure_connected()

        t0 = time.monotonic()
        try:
            counts, job_id = self._run_qasm(qasm, intent.shots)
            elapsed_ms = (time.monotonic() - t0) * 1000
            return CortexResult(
                intent=intent,
                qasm=qasm,
                counts=counts,
                backend=str(self._backend),
                shots=intent.shots,
                execution_time_ms=elapsed_ms,
                job_id=job_id,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            return CortexResult(
                intent=intent,
                qasm=qasm,
                counts={},
                backend=str(self._backend),
                shots=intent.shots,
                execution_time_ms=elapsed_ms,
                error=str(e),
            )

    def _run_qasm(self, qasm: str, shots: int) -> tuple[dict[str, int], str | None]:
        from qiskit import QuantumCircuit

        # Aer only supports QASM 2.0 via from_qasm_str.
        # We build the circuit directly from the QASM 3.0 string
        # by parsing qubit count and gate list ourselves.
        circuit = self._qasm3_to_circuit(qasm)

        if self.use_simulator:
            from qiskit_aer import AerSimulator
            sim: AerSimulator = self._backend  # type: ignore
            result = sim.run(circuit, shots=shots).result()
            counts = result.get_counts()
            return {k: v for k, v in counts.items()}, None

        # Real IBM Quantum execution via Sampler primitive
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=self._backend,
        )
        isa_circuit = pm.run(circuit)
        sampler = Sampler(backend=self._backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts_raw = pub_result.data.c.get_counts()
        return counts_raw, job.job_id()

    def _qasm3_to_circuit(self, qasm: str):
        """
        Convert an OpenQASM 3.0 string to a Qiskit QuantumCircuit.

        Qiskit's from_qasm_str only handles QASM 2.0. This method builds
        the circuit programmatically by parsing the QASM 3.0 gate list.
        Supports: h, x, y, z, cx, cy, cz, ccx, rx, ry, rz, cp, swap, t, s,
                  tdg, sdg, measure.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import (
            HGate, XGate, YGate, ZGate, CXGate, CYGate, CZGate, CCXGate,
            RXGate, RYGate, RZGate, CPhaseGate, SwapGate, TGate, SGate,
            TdgGate, SdgGate,
        )
        import math

        # Extract qubit count
        m = re.search(r'qubit\[(\d+)\]', qasm)
        n = int(m.group(1)) if m else 2
        circuit = QuantumCircuit(n, n)

        # Parse gate lines
        for line in qasm.splitlines():
            line = line.strip().rstrip(';')
            if not line or line.startswith('//') or line.startswith('OPENQASM') \
               or line.startswith('include') or line.startswith('qubit') \
               or line.startswith('bit'):
                continue

            # Measurement: c = measure q
            if 'measure' in line:
                circuit.measure(list(range(n)), list(range(n)))
                continue

            # Classically controlled (teleportation etc) — skip for now
            if line.startswith('if'):
                continue

            # Parse parametric gates: gate(angle) q[i], q[j]
            param_match = re.match(
                r'(\w+)\(([^)]+)\)\s+q\[(\d+)\](?:,\s*q\[(\d+)\])?', line
            )
            if param_match:
                gate_name = param_match.group(1).lower()
                angle_str = param_match.group(2)
                q0 = int(param_match.group(3))
                q1 = int(param_match.group(4)) if param_match.group(4) else None

                # Evaluate angle expression safely
                angle = self._eval_angle(angle_str)

                if gate_name == 'rx':
                    circuit.rx(angle, q0)
                elif gate_name == 'ry':
                    circuit.ry(angle, q0)
                elif gate_name == 'rz':
                    circuit.rz(angle, q0)
                elif gate_name in ('cp', 'cu1', 'p'):
                    circuit.cp(angle, q0, q1)
                continue

            # Parse simple gates: gate q[i] or gate q[i], q[j]
            simple_match = re.match(
                r'(\w+)\s+q\[(\d+)\](?:,\s*q\[(\d+)\])?(?:,\s*q\[(\d+)\])?', line
            )
            if simple_match:
                gate_name = simple_match.group(1).lower()
                q0 = int(simple_match.group(2))
                q1 = int(simple_match.group(3)) if simple_match.group(3) else None
                q2 = int(simple_match.group(4)) if simple_match.group(4) else None

                if gate_name == 'h':
                    circuit.h(q0)
                elif gate_name == 'x':
                    circuit.x(q0)
                elif gate_name == 'y':
                    circuit.y(q0)
                elif gate_name == 'z':
                    circuit.z(q0)
                elif gate_name == 't':
                    circuit.t(q0)
                elif gate_name == 's':
                    circuit.s(q0)
                elif gate_name == 'tdg':
                    circuit.tdg(q0)
                elif gate_name == 'sdg':
                    circuit.sdg(q0)
                elif gate_name == 'cx':
                    circuit.cx(q0, q1)
                elif gate_name == 'cy':
                    circuit.cy(q0, q1)
                elif gate_name == 'cz':
                    circuit.cz(q0, q1)
                elif gate_name == 'swap':
                    circuit.swap(q0, q1)
                elif gate_name == 'ccx':
                    circuit.ccx(q0, q1, q2)

        return circuit

    def _eval_angle(self, expr: str) -> float:
        """Safely evaluate a QASM angle expression like pi/2 or 3*pi/4."""
        import math
        expr = expr.strip()
        expr = expr.replace('pi', str(math.pi))
        try:
            return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi}))
        except Exception:
            return 0.0


class AerConnector(IBMQuantumConnector):
    """Shortcut connector for local Qiskit Aer simulation (no token needed)."""

    def __init__(self) -> None:
        super().__init__(token="", use_simulator=True)