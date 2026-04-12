"""
Integration tests for the Cortex Cloud API.
Uses FastAPI's TestClient — no real server or QPU needed.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from cortex.cloud.server import app
from cortex.cloud.models import JobStatus
from cortex.models import CortexResult, CircuitIntent


# ── Fixtures ──────────────────────────────────────────────────────────────────

DEV_KEY  = "dev-key-0000"
DEMO_KEY = "demo-key-1111"

def make_mock_result(text: str = "Bell state") -> CortexResult:
    intent = CircuitIntent(
        raw_text=text,
        num_qubits=2,
        circuit_type="bell_state",
        shots=1024,
    )
    return CortexResult(
        intent=intent,
        qasm='OPENQASM 3.0;\nqubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c=measure q;',
        counts={"00": 512, "11": 512},
        backend="aer",
        shots=1024,
        execution_time_ms=42.0,
    )


@pytest.fixture(scope="module")
def client():
    """Spin up a TestClient with the app lifespan (starts/stops the queue)."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Helper ────────────────────────────────────────────────────────────────────

def submit(client, text="Bell state", api_key=DEV_KEY, **kwargs):
    return client.post(
        "/jobs",
        json={"text": text, "backend": "aer", **kwargs},
        headers={"x-api-key": api_key},
    )


def wait_for_done(client, job_id: str, timeout: int = 10) -> dict:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}", headers={"x-api-key": DEV_KEY})
        data = r.json()
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout}s")


# ── Auth tests ────────────────────────────────────────────────────────────────

class TestAuth:

    def test_no_key_returns_401(self, client):
        r = client.post("/jobs", json={"text": "Bell state", "backend": "aer"})
        assert r.status_code == 401

    def test_invalid_key_returns_403(self, client):
        r = client.post(
            "/jobs",
            json={"text": "Bell state", "backend": "aer"},
            headers={"x-api-key": "bad-key"},
        )
        assert r.status_code == 403

    def test_valid_dev_key_accepted(self, client):
        with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=make_mock_result()):
            r = submit(client)
        assert r.status_code == 202

    def test_me_endpoint(self, client):
        r = client.get("/users/me", headers={"x-api-key": DEV_KEY})
        assert r.status_code == 200
        data = r.json()
        assert data["username"] == "dev"
        assert data["is_admin"] is True


# ── Job submission tests ───────────────────────────────────────────────────────

class TestJobSubmit:

    def test_submit_returns_job_id(self, client):
        with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=make_mock_result()):
            r = submit(client, "Create a GHZ state with 3 qubits")
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"

    def test_submit_invalid_backend_returns_400(self, client):
        r = client.post(
            "/jobs",
            json={"text": "Bell state", "backend": "unknown_backend"},
            headers={"x-api-key": DEV_KEY},
        )
        assert r.status_code == 400

    def test_submit_with_priority(self, client):
        with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=make_mock_result()):
            r = submit(client, priority=10)
        assert r.status_code == 202

    def test_submit_with_tags(self, client):
        with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=make_mock_result()):
            r = submit(client, tags=["research", "phase1"])
        assert r.status_code == 202


# ── Job retrieval tests ───────────────────────────────────────────────────────

class TestJobRetrieval:

    def test_get_job_by_id(self, client):
        with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=make_mock_result()):
            r = submit(client)
        job_id = r.json()["job_id"]

        result = client.get(f"/jobs/{job_id}", headers={"x-api-key": DEV_KEY})
        assert result.status_code == 200
        assert result.json()["id"] == job_id

    def test_get_nonexistent_job_returns_404(self, client):
        r = client.get("/jobs/00000000-fake-fake-fake-000000000000",
                       headers={"x-api-key": DEV_KEY})
        assert r.status_code == 404

    def test_done_job_has_counts(self, client):
        mock_result = make_mock_result("Bell state")
        with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=mock_result):
            r = submit(client, "Bell state")
            job_id = r.json()["job_id"]
            data = wait_for_done(client, job_id)

        assert data["status"] == "done"
        assert data["counts"] == {"00": 512, "11": 512}
        assert data["circuit_type"] == "bell_state"

    def test_list_jobs_returns_array(self, client):
        r = client.get("/jobs", headers={"x-api-key": DEV_KEY})
        assert r.status_code == 200
        assert "jobs" in r.json()

    def test_list_filter_by_status(self, client):
        r = client.get("/jobs?status=done", headers={"x-api-key": DEV_KEY})
        assert r.status_code == 200
        for job in r.json()["jobs"]:
            assert job["status"] == "done"


# ── Cancellation tests ────────────────────────────────────────────────────────

class TestJobCancellation:

    def test_cancel_queued_job(self, client):
        # Pause workers by submitting to a fresh queue with 0 workers
        # — instead, we test via the API directly after submit
        with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=make_mock_result()):
            r = submit(client)
        job_id = r.json()["job_id"]

        # May be queued or already running — cancel attempt
        cancel_r = client.delete(f"/jobs/{job_id}", headers={"x-api-key": DEV_KEY})
        # 200 = cancelled, 409 = already running/done — both are valid outcomes
        assert cancel_r.status_code in (200, 409)

    def test_cancel_nonexistent_job_returns_404(self, client):
        r = client.delete("/jobs/nonexistent-id", headers={"x-api-key": DEV_KEY})
        assert r.status_code == 404


# ── Stats + health ────────────────────────────────────────────────────────────

class TestAdminEndpoints:

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "queue" in data

    def test_queue_stats_admin_only(self, client):
        r = client.get("/queue/stats", headers={"x-api-key": DEV_KEY})
        assert r.status_code == 200
        data = r.json()
        assert "queued" in data
        assert "workers" in data

    def test_queue_stats_forbidden_for_demo(self, client):
        r = client.get("/queue/stats", headers={"x-api-key": DEMO_KEY})
        assert r.status_code == 403


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:

    def test_rate_limit_enforced(self, client):
        from cortex.cloud.queue import JobQueue
        # Temporarily lower quota to 2 for the demo user
        from cortex.cloud import server as srv
        demo_user = srv._USERS[DEMO_KEY]
        original_quota = demo_user.quota_jobs_per_hour
        demo_user.quota_jobs_per_hour = 2

        try:
            with patch("cortex.cloud.queue.JobQueue._run_cortex", return_value=make_mock_result()):
                for _ in range(2):
                    r = submit(client, api_key=DEMO_KEY)
                    assert r.status_code == 202
                # 3rd should be blocked
                r = submit(client, api_key=DEMO_KEY)
                assert r.status_code == 429
        finally:
            demo_user.quota_jobs_per_hour = original_quota
