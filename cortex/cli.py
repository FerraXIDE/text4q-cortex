"""
cortex.cli
==========
Command-line interface for text4q Cortex.

Commands:
    cortex run      "Bell state 2 qubits"       — compile + execute locally
    cortex compile  "GHZ 3 qubits"              — show QASM without executing
    cortex submit   "..." --server URL          — submit to cloud server
    cortex status   <job-id>                    — check job status
    cortex jobs                                 — list recent jobs
    cortex server                               — start the cloud API server
    cortex info                                 — show config + versions

Run with:
    python -m cortex.cli --help
    cortex --help   (after pip install)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint

app  = typer.Typer(
    name="cortex",
    help="[bold purple]text4q Cortex[/] — natural language quantum computing",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# ── Config helpers ────────────────────────────────────────────────────────────

def _get_server() -> str:
    return os.environ.get("CORTEX_SERVER", "http://localhost:8000")

def _get_api_key() -> str:
    key = os.environ.get("CORTEX_API_KEY", "dev-key-0000")
    return key

def _get_backend() -> str:
    return os.environ.get("CORTEX_BACKEND", "aer")

def _headers() -> dict:
    return {"x-api-key": _get_api_key(), "Content-Type": "application/json"}


# ── cortex run ────────────────────────────────────────────────────────────────

@app.command()
def run(
    text: str = typer.Argument(..., help="Natural language circuit description"),
    backend: str = typer.Option(None, "--backend", "-b", help="aer | ibm_quantum"),
    shots: int = typer.Option(1024, "--shots", "-s", help="Number of measurement shots"),
    nlp: str = typer.Option("pattern", "--nlp", help="NLP engine: pattern | llm"),
    llm_backend: str = typer.Option("anthropic", "--llm-backend", help="anthropic | openai"),
    show_qasm: bool = typer.Option(False, "--qasm", help="Print the generated QASM circuit"),
):
    """
    [bold]Compile and execute[/] a quantum circuit locally.

    Examples:
        cortex run "Bell state with 2 qubits"
        cortex run "GHZ 5 qubits 2048 shots" --backend aer --qasm
        cortex run "VQE para H2" --nlp llm
    """
    from cortex.core import Cortex

    backend = backend or _get_backend()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Parsing intent…", total=None)
        cx = Cortex(backend=backend, nlp=nlp, llm_backend=llm_backend)

        progress.update(task, description="Compiling to OpenQASM…")
        intent = cx.parse(text)
        qasm   = cx.compile(intent)

        progress.update(task, description=f"Executing on [bold]{backend}[/]…")
        result = cx._connector.execute(intent, qasm)

    _print_result(result, text, show_qasm=show_qasm)


# ── cortex compile ────────────────────────────────────────────────────────────

@app.command()
def compile(
    text: str = typer.Argument(..., help="Natural language circuit description"),
    nlp: str = typer.Option("pattern", "--nlp", help="NLP engine: pattern | llm"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save QASM to file"),
):
    """
    [bold]Translate[/] natural language to OpenQASM 3.0 (no execution).

    Examples:
        cortex compile "Bell state"
        cortex compile "QFT 4 qubits" --output circuit.qasm
    """
    from cortex.core import Cortex

    cx = Cortex(backend="aer", nlp=nlp)

    with console.status("Compiling…"):
        intent = cx.parse(text)
        qasm   = cx.compile(intent)

    console.print(f"\n[dim]Circuit:[/] [bold]{intent.circuit_type}[/]  "
                  f"[dim]Qubits:[/] {intent.num_qubits}  "
                  f"[dim]Shots:[/] {intent.shots}\n")

    syntax = Syntax(qasm, "qasm", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="[bold]OpenQASM 3.0[/]", border_style="purple"))

    if output:
        with open(output, "w") as f:
            f.write(qasm)
        console.print(f"\n[green]✓[/] Saved to [bold]{output}[/]")


# ── cortex submit ─────────────────────────────────────────────────────────────

@app.command()
def submit(
    text: str = typer.Argument(..., help="Natural language circuit description"),
    backend: str = typer.Option("aer", "--backend", "-b"),
    shots: int = typer.Option(1024, "--shots", "-s"),
    priority: int = typer.Option(5, "--priority", "-p", help="1=low 5=normal 10=high"),
    nlp: str = typer.Option("pattern", "--nlp"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion"),
    server: str = typer.Option(None, "--server", help="Server URL"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
):
    """
    [bold]Submit[/] a job to the Cortex cloud server.

    Examples:
        cortex submit "GHZ 5 qubits"
        cortex submit "Bell state" --wait --priority 10
        cortex submit "VQE" --server http://my-lab-server:8000
    """
    import httpx

    server_url = server or _get_server()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    payload = {
        "text": text,
        "backend": backend,
        "shots": shots,
        "priority": priority,
        "nlp_mode": nlp,
        "tags": tag_list,
    }

    with console.status(f"Submitting to [bold]{server_url}[/]…"):
        try:
            r = httpx.post(
                f"{server_url}/jobs",
                json=payload,
                headers=_headers(),
                timeout=10,
            )
            r.raise_for_status()
        except httpx.ConnectError:
            console.print(f"[red]✗[/] Cannot reach server at [bold]{server_url}[/]")
            console.print("  Start with: [dim]cortex server[/]")
            raise typer.Exit(1)
        except httpx.HTTPStatusError as e:
            console.print(f"[red]✗[/] Server error {e.response.status_code}: {e.response.text}")
            raise typer.Exit(1)

    data = r.json()
    job_id = data["job_id"]
    console.print(f"\n[green]✓[/] Job submitted: [bold cyan]{job_id}[/]")

    if not wait:
        console.print(f"  Track with: [dim]cortex status {job_id}[/]")
        return

    # Poll until done
    _poll_job(job_id, server_url)


# ── cortex status ─────────────────────────────────────────────────────────────

@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID to check"),
    server: str = typer.Option(None, "--server"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Poll until done"),
):
    """
    [bold]Check status[/] of a submitted job.

    Examples:
        cortex status abc123
        cortex status abc123 --watch
    """
    server_url = server or _get_server()

    if watch:
        _poll_job(job_id, server_url)
    else:
        _print_job_status(job_id, server_url)


# ── cortex jobs ───────────────────────────────────────────────────────────────

@app.command()
def jobs(
    status_filter: Optional[str] = typer.Option(None, "--status", help="queued|running|done|failed"),
    limit: int = typer.Option(15, "--limit", "-n"),
    server: str = typer.Option(None, "--server"),
):
    """
    [bold]List[/] recent jobs from the cloud server.

    Examples:
        cortex jobs
        cortex jobs --status done --limit 5
    """
    import httpx
    server_url = server or _get_server()

    params = {"limit": limit}
    if status_filter:
        params["status"] = status_filter

    try:
        r = httpx.get(
            f"{server_url}/jobs",
            params=params,
            headers=_headers(),
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        console.print(f"[red]✗[/] {e}")
        raise typer.Exit(1)

    data = r.json()
    job_list = data["jobs"]

    if not job_list:
        console.print("[dim]No jobs found.[/]")
        return

    table = Table(title="Recent Jobs", border_style="dim", show_lines=False)
    table.add_column("ID",         style="cyan dim",  no_wrap=True, max_width=10)
    table.add_column("Status",     no_wrap=True)
    table.add_column("Circuit",    style="dim")
    table.add_column("Qubits",     justify="right")
    table.add_column("Shots",      justify="right", style="dim")
    table.add_column("Time (ms)",  justify="right", style="dim")
    table.add_column("Input",      style="dim",     max_width=35)

    status_colors = {
        "queued": "yellow", "running": "blue",
        "done": "green",    "failed": "red", "cancelled": "dim",
    }

    for j in job_list:
        s = j["status"]
        color = status_colors.get(s, "white")
        dur = f"{j['duration_ms']:.0f}" if j.get("duration_ms") else "—"
        table.add_row(
            j["id"][:8],
            f"[{color}]{s}[/]",
            j.get("circuit_type") or "—",
            str(j.get("num_qubits") or "—"),
            str(j["shots"]),
            dur,
            j["text"],
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Total: {data['total']} jobs[/]")


# ── cortex server ─────────────────────────────────────────────────────────────

@app.command()
def server(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    workers: int = typer.Option(2, "--workers", help="Number of QPU worker threads"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """
    [bold]Start[/] the Cortex cloud API server.

    Examples:
        cortex server
        cortex server --port 9000 --workers 4
    """
    import uvicorn

    os.environ["CORTEX_WORKERS"] = str(workers)
    console.print(
        Panel(
            f"[bold purple]text4q Cortex Cloud[/]\n\n"
            f"  URL:      [cyan]http://{host}:{port}[/]\n"
            f"  Workers:  {workers}\n"
            f"  Docs:     [cyan]http://{host}:{port}/docs[/]\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            border_style="purple",
        )
    )

    uvicorn.run(
        "cortex.cloud.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )


# ── cortex info ───────────────────────────────────────────────────────────────

@app.command()
def info(
    server: str = typer.Option(None, "--server"),
):
    """Show local config and server health."""
    import httpx
    from cortex import __version__

    server_url = server or _get_server()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key",   style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Version",    __version__)
    table.add_row("Backend",    _get_backend())
    table.add_row("Server",     server_url)
    table.add_row("API Key",    _get_api_key()[:8] + "…")

    console.print(Panel(table, title="[bold]text4q Cortex[/]", border_style="purple"))

    # Try to reach the server
    try:
        r = httpx.get(f"{server_url}/health", headers=_headers(), timeout=3)
        health = r.json()
        q = health["queue"]
        console.print(
            f"\n[green]✓[/] Server reachable — "
            f"queued={q['queued']} running={q['running']} "
            f"done={q['done']} workers={q['workers']}"
        )
    except Exception:
        console.print(f"\n[yellow]⚠[/]  Server at [bold]{server_url}[/] not reachable")
        console.print("  Start with: [dim]cortex server[/]")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_result(result, text: str, show_qasm: bool = False):
    """Pretty-print a CortexResult."""
    status_color = "green" if result.success else "red"
    status_icon  = "✓" if result.success else "✗"

    console.print()
    console.print(Panel(
        f"[dim]Input:[/]   {text}\n"
        f"[dim]Circuit:[/] [bold]{result.intent.circuit_type}[/]  "
        f"({result.intent.num_qubits} qubits)\n"
        f"[dim]Backend:[/] {result.backend}\n"
        f"[dim]Time:[/]    {result.execution_time_ms:.1f} ms",
        title=f"[{status_color}]{status_icon} Result[/]",
        border_style=status_color,
    ))

    if result.success and result.counts:
        _print_histogram(result.counts, result.shots)

    if show_qasm and result.qasm:
        console.print()
        syntax = Syntax(result.qasm, "qasm", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="OpenQASM 3.0", border_style="dim"))

    if result.error:
        console.print(f"\n[red]Error:[/] {result.error}")


def _print_histogram(counts: dict, shots: int, max_bars: int = 10):
    """ASCII histogram of measurement results."""
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:max_bars]
    max_count = sorted_counts[0][1] if sorted_counts else 1
    bar_width = 30

    console.print("\n[dim]Measurement results:[/]")
    for state, count in sorted_counts:
        pct = count / shots * 100
        filled = int(bar_width * count / max_count)
        bar = "█" * filled + "░" * (bar_width - filled)
        console.print(f"  |{state}⟩  {bar}  {count:>5} ({pct:>5.1f}%)")


def _print_job_status(job_id: str, server_url: str):
    import httpx
    try:
        r = httpx.get(f"{server_url}/jobs/{job_id}", headers=_headers(), timeout=10)
        r.raise_for_status()
    except Exception as e:
        console.print(f"[red]✗[/] {e}")
        raise typer.Exit(1)

    j = r.json()
    status_colors = {
        "queued": "yellow", "running": "blue",
        "done": "green",    "failed": "red", "cancelled": "dim",
    }
    s = j["status"]
    color = status_colors.get(s, "white")

    console.print(f"\nJob [cyan]{job_id[:8]}[/]  [{color}]{s}[/]")

    if j.get("circuit_type"):
        console.print(f"  Circuit: {j['circuit_type']} ({j.get('num_qubits')}q)")

    if j.get("counts"):
        _print_histogram(j["counts"], j["shots"])

    if j.get("error"):
        console.print(f"  [red]Error:[/] {j['error']}")


def _poll_job(job_id: str, server_url: str, poll_interval: float = 0.5):
    import httpx
    terminal = {"done", "failed", "cancelled"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Waiting for job [cyan]{job_id[:8]}[/]…", total=None)

        while True:
            try:
                r = httpx.get(f"{server_url}/jobs/{job_id}", headers=_headers(), timeout=10)
                r.raise_for_status()
                j = r.json()
                s = j["status"]
                progress.update(task, description=f"Job [cyan]{job_id[:8]}[/] — {s}…")
                if s in terminal:
                    break
            except Exception as e:
                console.print(f"[red]✗[/] Poll error: {e}")
                raise typer.Exit(1)
            time.sleep(poll_interval)

    _print_job_status(job_id, server_url)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app()

if __name__ == "__main__":
    main()
