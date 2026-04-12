"""
Tests for the Cortex CLI and dashboard endpoint.
"""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from cortex.cli import app as cli_app
from cortex.cloud.server import app as server_app
from cortex.models import CortexResult, CircuitIntent

runner = CliRunner()
DEV_KEY = "dev-key-0000"


def make_mock_result(circuit="bell_state", n=2):
    intent = CircuitIntent(raw_text="Bell state", num_qubits=n,
                           circuit_type=circuit, shots=1024)
    return CortexResult(
        intent=intent,
        qasm="OPENQASM 3.0;\nqubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c=measure q;",
        counts={"00": 512, "11": 512},
        backend="aer", shots=1024, execution_time_ms=18.5,
    )


# ── CLI tests ─────────────────────────────────────────────────────────────────

class TestCLIRun:

    def test_run_bell_state(self):
        with patch("cortex.connectors.ibm.AerConnector.execute",
                   return_value=make_mock_result()):
            result = runner.invoke(cli_app, ["run", "Bell state with 2 qubits"])
        assert result.exit_code == 0
        assert "bell_state" in result.output

    def test_run_shows_counts(self):
        with patch("cortex.connectors.ibm.AerConnector.execute",
                   return_value=make_mock_result()):
            result = runner.invoke(cli_app, ["run", "Bell state"])
        assert "00" in result.output
        assert "11" in result.output

    def test_run_with_qasm_flag(self):
        with patch("cortex.connectors.ibm.AerConnector.execute",
                   return_value=make_mock_result()):
            result = runner.invoke(cli_app, ["run", "Bell state", "--qasm"])
        assert "OPENQASM" in result.output

    def test_run_shows_execution_time(self):
        with patch("cortex.connectors.ibm.AerConnector.execute",
                   return_value=make_mock_result()):
            result = runner.invoke(cli_app, ["run", "Bell state"])
        assert "ms" in result.output


class TestCLICompile:

    def test_compile_outputs_qasm(self):
        result = runner.invoke(cli_app, ["compile", "Bell state"])
        assert result.exit_code == 0
        assert "OPENQASM" in result.output
        assert "bell_state" in result.output

    def test_compile_ghz(self):
        result = runner.invoke(cli_app, ["compile", "GHZ state 3 qubits"])
        assert result.exit_code == 0
        assert "ghz" in result.output

    def test_compile_save_to_file(self, tmp_path):
        out = str(tmp_path / "circuit.qasm")
        result = runner.invoke(cli_app, ["compile", "Bell state", "--output", out])
        assert result.exit_code == 0
        import os
        assert os.path.exists(out)
        content = open(out).read()
        assert "OPENQASM" in content


class TestCLIInfo:

    def test_info_shows_version(self):
        result = runner.invoke(cli_app, ["info"])
        assert result.exit_code == 0
        assert "Cortex" in result.output

    def test_info_shows_backend(self):
        result = runner.invoke(cli_app, ["info"])
        assert "aer" in result.output or "ibm" in result.output


class TestCLIHelp:

    def test_help_lists_commands(self):
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        for cmd in ["run", "compile", "submit", "jobs", "status", "server", "info"]:
            assert cmd in result.output

    def test_run_help(self):
        import re
        result = runner.invoke(cli_app, ["run", "--help"])
        assert result.exit_code == 0
        # Strip ANSI codes before checking (CI may use color output)
        clean = re.sub(r"\[[0-9;]*[mGKHF]", "", result.output)
        assert "backend" in clean
        assert "shots" in clean
        assert "qasm" in clean

    def test_compile_help(self):
        import re
        result = runner.invoke(cli_app, ["compile", "--help"])
        assert result.exit_code == 0
        clean = re.sub(r"\[[0-9;]*[mGKHF]", "", result.output)
        # Check option exists regardless of locale translation
        assert "nlp" in clean.lower()
        assert "-o" in clean or "output" in clean.lower() or "salida" in clean.lower()


# ── Dashboard endpoint test ───────────────────────────────────────────────────

class TestDashboard:

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(server_app) as c:
            yield c

    def test_dashboard_returns_html(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_dashboard_contains_cortex_title(self, client):
        r = client.get("/dashboard")
        assert "text4q Cortex" in r.text

    def test_dashboard_no_auth_required(self, client):
        """Dashboard should be publicly accessible — no API key needed."""
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_has_submit_form(self, client):
        r = client.get("/dashboard")
        assert "submit-text" in r.text
        assert "submitJob" in r.text

    def test_dashboard_has_histogram_logic(self, client):
        r = client.get("/dashboard")
        assert "hist-bar" in r.text