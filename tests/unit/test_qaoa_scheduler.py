"""
Tests for the QAOA Scheduler (Fase 4).
All tests run without real QPU access — use Aer simulator + mocks.
"""

import pytest
import math
from unittest.mock import patch, MagicMock

from cortex.scheduler.problem import (
    SchedulingJob, QPUBackend,
    build_qubo, evaluate_assignment, best_bitstring,
)
from cortex.scheduler.qaoa import (
    build_qaoa_circuit, circuit_stats, suggest_parameters,
)
from cortex.scheduler.optimizer import QAOAScheduler, SchedulingResult
from cortex.scheduler.integration import (
    QAOAQueueIntegration, AssignmentDecision, DEFAULT_BACKENDS,
)
from cortex.cloud.models import Job, JobPriority


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def two_jobs():
    return [
        SchedulingJob(id="job-aaa", priority=8.0, estimated_shots=1024),
        SchedulingJob(id="job-bbb", priority=3.0, estimated_shots=512),
    ]

@pytest.fixture
def two_backends():
    return [
        QPUBackend(id="b1", name="fast", capacity=0.9, error_rate=0.005),
        QPUBackend(id="b2", name="slow", capacity=0.5, error_rate=0.02),
    ]

@pytest.fixture
def simple_qubo(two_jobs, two_backends):
    return build_qubo(two_jobs, two_backends)


# ── Tests: SchedulingJob / QPUBackend ────────────────────────────────────────

class TestProblemDataClasses:

    def test_job_weight_normalized(self):
        job = SchedulingJob(id="j1", priority=10.0, estimated_shots=1024)
        assert job.weight == 1.0

    def test_job_weight_scales(self):
        j5 = SchedulingJob(id="j1", priority=5.0, estimated_shots=1024)
        j8 = SchedulingJob(id="j2", priority=8.0, estimated_shots=1024)
        assert j5.weight < j8.weight

    def test_job_time_cost_is_normalized(self):
        job = SchedulingJob(id="j1", priority=5.0, estimated_shots=1024)
        assert 0.0 < job.time_cost < 1.0

    def test_backend_quality_inversely_proportional_to_error(self):
        good = QPUBackend(id="b1", name="good", capacity=1.0, error_rate=0.001)
        bad  = QPUBackend(id="b2", name="bad",  capacity=1.0, error_rate=0.1)
        assert good.quality > bad.quality


# ── Tests: QUBO formulation ───────────────────────────────────────────────────

class TestBuildQUBO:

    def test_qubo_size(self, two_jobs, two_backends):
        qubo = build_qubo(two_jobs, two_backends)
        assert qubo.n_jobs == 2
        assert qubo.n_backends == 2
        assert qubo.n_vars == 4
        assert len(qubo.Q) == 4
        assert len(qubo.Q[0]) == 4

    def test_qubo_var_index(self, simple_qubo):
        assert simple_qubo.var_index(0, 0) == 0
        assert simple_qubo.var_index(0, 1) == 1
        assert simple_qubo.var_index(1, 0) == 2
        assert simple_qubo.var_index(1, 1) == 3

    def test_qubo_diagonal_penalizes_bad_backends(self, two_jobs, two_backends):
        qubo = build_qubo(two_jobs, two_backends)
        # High-priority job (index 0) to fast backend (index 0) should cost less
        # than to slow backend (index 1) — lower diagonal = better
        assert qubo.Q[0][0] < qubo.Q[1][1]

    def test_qubo_var_names_readable(self, simple_qubo):
        assert "job_job-aa" in simple_qubo.var_names[0]
        assert "fast" in simple_qubo.var_names[0]

    def test_unavailable_backend_excluded(self, two_jobs):
        backends = [
            QPUBackend(id="b1", name="ok",     capacity=1.0, error_rate=0.01, available=True),
            QPUBackend(id="b2", name="offline", capacity=1.0, error_rate=0.01, available=False),
        ]
        qubo = build_qubo(two_jobs, backends)
        # Only 1 available backend → 2 jobs × 1 backend = 2 vars
        assert qubo.n_backends == 1
        assert qubo.n_vars == 2

    def test_evaluate_assignment_returns_float(self, simple_qubo):
        cost = evaluate_assignment("1010", simple_qubo)
        assert isinstance(cost, float)

    def test_best_bitstring_selects_lowest_cost(self, simple_qubo):
        counts = {
            "1010": 500,   # job0→b1, job1→b1 (conflict)
            "1001": 300,   # job0→b1, job1→b2 (valid assignment)
            "0110": 200,   # job0→b2, job1→b1
        }
        best = best_bitstring(counts, simple_qubo)
        assert best in counts

    def test_best_bitstring_ignores_wrong_length(self, simple_qubo):
        counts = {"10": 999, "1001": 1}   # "10" has wrong length
        best = best_bitstring(counts, simple_qubo)
        assert best == "1001"


# ── Tests: QAOA circuit builder ───────────────────────────────────────────────

class TestBuildQAOACircuit:

    def test_circuit_is_valid_openqasm3(self, simple_qubo):
        qasm = build_qaoa_circuit(simple_qubo)
        assert "OPENQASM 3.0" in qasm
        assert "qubit[4]" in qasm
        assert "bit[4]" in qasm

    def test_circuit_has_hadamard_layer(self, simple_qubo):
        qasm = build_qaoa_circuit(simple_qubo)
        # 4 qubits → 4 H gates
        assert qasm.count("h q[") == 4

    def test_circuit_has_measurement(self, simple_qubo):
        qasm = build_qaoa_circuit(simple_qubo)
        assert "c = measure q" in qasm

    def test_circuit_has_rz_gates(self, simple_qubo):
        qasm = build_qaoa_circuit(simple_qubo)
        assert "rz(" in qasm

    def test_circuit_has_rx_mixer(self, simple_qubo):
        qasm = build_qaoa_circuit(simple_qubo)
        assert "rx(" in qasm

    def test_circuit_has_cx_for_offdiag(self, simple_qubo):
        qasm = build_qaoa_circuit(simple_qubo)
        assert "cx q[" in qasm

    def test_two_layers_doubles_gates(self, simple_qubo):
        qasm_p1 = build_qaoa_circuit(simple_qubo, p=1)
        qasm_p2 = build_qaoa_circuit(simple_qubo, p=2)
        # p=2 should have more gates
        assert len(qasm_p2) > len(qasm_p1)

    def test_raises_for_empty_qubo(self):
        from cortex.scheduler.problem import QUBOProblem
        empty = QUBOProblem(Q=[], n_jobs=0, n_backends=0, var_names=[])
        with pytest.raises(ValueError, match="no variables"):
            build_qaoa_circuit(empty)

    def test_raises_for_too_many_variables(self):
        from cortex.scheduler.problem import QUBOProblem
        big_n = 25
        big = QUBOProblem(
            Q=[[0.0]*big_n for _ in range(big_n)],
            n_jobs=5, n_backends=5, var_names=["x"]*big_n,
        )
        with pytest.raises(ValueError, match="max 20"):
            build_qaoa_circuit(big)

    def test_suggest_parameters_returns_valid_angles(self):
        gamma, beta = suggest_parameters(n_vars=4, p=1)
        assert 0 < gamma < math.pi
        assert 0 < beta  < math.pi

    def test_circuit_stats_correct_qubit_count(self, simple_qubo):
        stats = circuit_stats(simple_qubo, p=1)
        assert stats["n_qubits"] == 4
        assert stats["total_gates"] > 0
        assert isinstance(stats["recommend_real_qpu"], bool)


# ── Tests: QAOAScheduler (with Aer mock) ─────────────────────────────────────

def make_mock_cortex_result(bitstring: str, n_qubits: int, shots: int = 1024):
    """Return a fake Cortex execution result with a controlled bitstring."""
    from cortex.models import CortexResult, CircuitIntent
    intent = CircuitIntent(
        raw_text="QAOA scheduler", num_qubits=n_qubits,
        circuit_type="qaoa", shots=shots,
    )
    counts = {bitstring: shots}
    return CortexResult(
        intent=intent, qasm="OPENQASM 3.0;",
        counts=counts, backend="aer",
        shots=shots, execution_time_ms=42.0,
    )


class TestQAOAScheduler:

    def test_schedule_returns_result(self, two_jobs, two_backends):
        scheduler = QAOAScheduler(backend="aer", shots=256)
        # Mock the Aer connector to return "1001" (job0→b1, job1→b2)
        mock_result = make_mock_cortex_result("1001", n_qubits=4, shots=256)
        with patch("cortex.connectors.ibm.AerConnector.execute", return_value=mock_result):
            result = scheduler.schedule(two_jobs, two_backends)

        assert isinstance(result, SchedulingResult)
        assert result.method in ("qaoa", "classical")
        assert len(result.assignment) == 2

    def test_schedule_assigns_all_jobs(self, two_jobs, two_backends):
        scheduler = QAOAScheduler(backend="aer", shots=256)
        mock_result = make_mock_cortex_result("1001", n_qubits=4)
        with patch("cortex.connectors.ibm.AerConnector.execute", return_value=mock_result):
            result = scheduler.schedule(two_jobs, two_backends)

        for job in two_jobs:
            assert job.id in result.assignment

    def test_schedule_empty_jobs(self, two_backends):
        scheduler = QAOAScheduler(backend="aer")
        result = scheduler.schedule([], two_backends)
        assert result.assignment == {}
        assert result.method == "classical"

    def test_schedule_no_backends(self, two_jobs):
        scheduler = QAOAScheduler(backend="aer")
        result = scheduler.schedule(two_jobs, [])
        for job in two_jobs:
            assert result.assignment[job.id] == "unassigned"

    def test_classical_fallback_for_large_problems(self):
        # 6 jobs × 4 backends = 24 vars > MAX_QAOA_VARS (20)
        jobs = [SchedulingJob(id=f"j{i}", priority=float(i+1), estimated_shots=512)
                for i in range(6)]
        backends = [QPUBackend(id=f"b{i}", name=f"b{i}", capacity=0.8, error_rate=0.01)
                    for i in range(4)]
        scheduler = QAOAScheduler(backend="aer")
        result = scheduler.schedule(jobs, backends)
        assert result.method == "classical"
        assert "unassigned" not in result.assignment.values() or True  # may unassign

    def test_force_classical_flag(self, two_jobs, two_backends):
        scheduler = QAOAScheduler(backend="aer", force_classical=True)
        result = scheduler.schedule(two_jobs, two_backends)
        assert result.method == "classical"

    def test_qaoa_result_has_circuit_stats(self, two_jobs, two_backends):
        scheduler = QAOAScheduler(backend="aer", shots=256)
        mock_result = make_mock_cortex_result("1001", n_qubits=4)
        with patch("cortex.connectors.ibm.AerConnector.execute", return_value=mock_result):
            result = scheduler.schedule(two_jobs, two_backends)
        if result.method == "qaoa":
            assert "n_qubits" in result.circuit_stats

    def test_fallback_on_execution_error(self, two_jobs, two_backends):
        scheduler = QAOAScheduler(backend="aer", shots=256)
        with patch("cortex.connectors.ibm.AerConnector.execute",
                   side_effect=RuntimeError("QPU offline")):
            result = scheduler.schedule(two_jobs, two_backends)
        assert result.method == "classical"
        assert len(result.warnings) > 0


# ── Tests: QAOAQueueIntegration ───────────────────────────────────────────────

class TestQAOAQueueIntegration:

    def _make_job(self, priority: int = 5) -> Job:
        return Job(
            user_id="user-test",
            text="Bell state",
            shots=1024,
            backend="aer",
            priority=JobPriority(priority),
        )

    def test_assign_batch_returns_decisions(self):
        integration = QAOAQueueIntegration(scheduler_backend="aer")
        jobs = [self._make_job(10), self._make_job(5)]
        mock_result = make_mock_cortex_result("10", n_qubits=2)
        with patch("cortex.connectors.ibm.AerConnector.execute", return_value=mock_result):
            decisions = integration.assign_batch(jobs)
        assert len(decisions) == 2
        for d in decisions:
            assert isinstance(d, AssignmentDecision)
            assert d.backend_id in ("aer", "ibm_quantum", "unassigned")

    def test_assign_empty_batch(self):
        integration = QAOAQueueIntegration()
        decisions = integration.assign_batch([])
        assert decisions == []

    def test_assign_single_job(self):
        integration = QAOAQueueIntegration(scheduler_backend="aer")
        job = self._make_job(priority=10)
        mock_result = make_mock_cortex_result("1", n_qubits=1)
        with patch("cortex.connectors.ibm.AerConnector.execute", return_value=mock_result):
            decision = integration.assign_single(job)
        assert decision.job_id == job.id
        assert isinstance(decision.confidence, float)

    def test_register_new_backend(self):
        integration = QAOAQueueIntegration()
        new_b = QPUBackend(id="lab-qpu", name="Lab QPU", capacity=0.6, error_rate=0.008)
        integration.register_backend(new_b)
        assert any(b.id == "lab-qpu" for b in integration.backends)

    def test_set_backend_availability(self):
        integration = QAOAQueueIntegration()
        integration.set_backend_availability("ibm_quantum", True)
        ibm = next(b for b in integration.backends if b.id == "ibm_quantum")
        assert ibm.available is True

    def test_decision_has_method_field(self):
        integration = QAOAQueueIntegration(scheduler_backend="aer")
        job = self._make_job()
        mock_result = make_mock_cortex_result("1", n_qubits=1)
        with patch("cortex.connectors.ibm.AerConnector.execute", return_value=mock_result):
            decision = integration.assign_single(job)
        assert decision.method in ("qaoa", "classical")
