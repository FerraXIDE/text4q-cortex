"""
cortex.connectors.ibm
=====================
IBM Quantum connector using qiskit-ibm-runtime.
Supports both real QPU execution and local Aer simulation.
"""

from __future__ import annotations
import time
import os

from cortex.models import CortexResult, CircuitIntent


class IBMQuantumConnector:

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
        # Use qiskit.qasm3.loads() — native QASM 3.0 support
        from qiskit import qasm3
        circuit = qasm3.loads(qasm)

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


class AerConnector(IBMQuantumConnector):
    """Shortcut connector for local Qiskit Aer simulation (no token needed)."""

    def __init__(self) -> None:
        super().__init__(token="", use_simulator=True)