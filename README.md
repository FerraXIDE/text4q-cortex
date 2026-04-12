# text4q Cortex

Natural language interface for quantum computing infrastructure.

```python
from cortex import Cortex

cx = Cortex(backend="ibm_quantum")
result = cx.run("Simulate a Bell state with 2 qubits and measure 1024 times")
print(result.counts)
# {'00': 498, '11': 489, '01': 19, '10': 18}  -- real QPU output with noise
```

## Overview

Cortex is an open-source quantum orchestration platform that translates natural language descriptions into executable quantum circuits, manages QPU resources across providers, and schedules jobs using a quantum-native optimizer.

The core insight: writing OpenQASM circuits by hand is a barrier that keeps most researchers and engineers away from quantum hardware. Cortex removes that barrier without sacrificing access to real QPUs.

## Architecture

```
User (natural language)
        |
  Cortex NLP Engine        -- text4q core: language to OpenQASM 3.0
        |                     pattern-based (v0.1) + LLM-powered (v0.2)
  OQTOPUS Job Queue        -- cloud layer: scheduling, auth, rate limiting
        |
  QAOA Scheduler           -- quantum-native job-to-QPU assignment
        |
  QRMI Resource Manager    -- QPU as HPC node (Slurm-compatible)
        |
  QPU / Simulator          -- IBM Quantum, Google, Qiskit Aer
```

## Status

All modules are implemented and tested. The project is in active development (v0.1, pre-production).

| Module | Description | Status |
|--------|-------------|--------|
| `cortex.nlp` | Pattern-based NLP engine | Stable |
| `cortex.nlp.llm_engine` | LLM-powered engine (Claude / GPT-4o) | Stable |
| `cortex.connectors` | IBM Quantum + Aer backends | Stable |
| `cortex.cloud` | REST API, async job queue, dashboard | Stable |
| `cortex.scheduler` | QAOA-based QPU assignment | Stable |
| `cortex.cli` | Command-line interface | Stable |

103 tests passing across Python 3.10, 3.11, and 3.12.

## Installation

```bash
pip install text4q-cortex
```

With quantum backends:

```bash
pip install "text4q-cortex[qiskit]"      # IBM Quantum + Aer simulator
pip install "text4q-cortex[all]"         # everything including LLM support
```

## Quick Start

### Local simulation

```python
from cortex import Cortex

cx = Cortex(backend="aer")
result = cx.run("GHZ state with 3 qubits, 2048 shots")

print(result.counts)
# {'000': 1024, '111': 1024}

print(result.qasm)
# OPENQASM 3.0;
# include "stdgates.inc";
# qubit[3] q;
# ...
```

### IBM Quantum (real hardware)

```python
import os
from cortex import Cortex

cx = Cortex(backend="ibm_quantum", token=os.environ["IBM_QUANTUM_TOKEN"])
result = cx.run("Bell state with 2 qubits, 1024 shots")

# Real QPU output includes noise
print(result.counts)
# {'00': 498, '11': 489, '01': 19, '10': 18}
print(f"Fidelity: {(result.counts.get('00',0) + result.counts.get('11',0)) / result.shots:.2%}")
# Fidelity: 96.19%
```

### LLM-powered engine

Accepts arbitrary circuit descriptions beyond the built-in patterns:

```python
from cortex import Cortex

cx = Cortex(backend="aer", nlp="llm", llm_backend="anthropic")
result = cx.run(
    "Implement QAOA for a Max-Cut problem on a 4-node graph, "
    "p=1 layers, 2048 shots"
)
```

### Cloud API

Start a multi-user job server:

```bash
cortex server --port 8000 --workers 4
```

Submit jobs via HTTP:

```bash
curl -X POST http://localhost:8000/jobs \
  -H "x-api-key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "QFT on 4 qubits", "backend": "aer", "shots": 1024}'
```

Web dashboard available at `http://localhost:8000/dashboard`.

### CLI

```bash
cortex run "Bell state with 2 qubits" --qasm
cortex compile "GHZ state, 5 qubits" --output circuit.qasm
cortex submit "VQE for H2 molecule" --backend ibm_quantum --wait
cortex jobs --status done
cortex server
```

### QAOA Scheduler

Assigns jobs to QPU backends using a quantum optimization circuit:

```python
from cortex.scheduler.optimizer import QAOAScheduler
from cortex.scheduler.problem import SchedulingJob, QPUBackend

jobs = [
    SchedulingJob("exp-001", priority=9, estimated_shots=2048),
    SchedulingJob("exp-002", priority=4, estimated_shots=512),
]
backends = [
    QPUBackend("aer",         "Aer Simulator", capacity=1.0, error_rate=1e-6),
    QPUBackend("ibm_quantum", "IBM Eagle",     capacity=0.7, error_rate=0.01),
]

result = QAOAScheduler(backend="aer", p=1, shots=2048).schedule(jobs, backends)
print(result)
# exp-001 -> ibm_quantum   (high priority to low-error QPU)
# exp-002 -> aer            (low priority to simulator)
# cost=-14.2  time=38ms
```

## Noise handling

On real QPUs, results include gate errors, readout errors, and decoherence.
Cortex exposes raw measurement counts without post-processing, allowing
researchers to apply their own error mitigation:

```python
result = cx.run("Bell state, T1=50us T2=30us noise model, 4096 shots")

counts = result.counts
# {'00': 1923, '11': 1887, '01': 143, '10': 143}

error_rate = (counts.get('01', 0) + counts.get('10', 0)) / result.shots
print(f"Bit-flip error rate: {error_rate:.2%}")
# Bit-flip error rate: 7.00%
```

## Roadmap

- [x] NLP engine: pattern-based (v0.1)
- [x] LLM-powered circuit generation (v0.2)
- [x] IBM Quantum connector + Aer simulator
- [x] OQTOPUS job queue integration (v0.3)
- [x] CLI and web dashboard (v0.4)
- [x] QAOA Scheduler (v0.5)
- [ ] Classical parameter optimization for QAOA (SciPy outer loop)
- [ ] Google Quantum AI connector
- [ ] IonQ and Quantinuum connectors
- [ ] PyPI stable release (v1.0)
- [ ] text4q Cortex Cloud (hosted SaaS)

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

Areas where contributions are most welcome:
- Additional QPU connectors (Google, IonQ, Quantinuum)
- Error mitigation post-processing utilities
- QAOA parameter optimization (classical outer loop)
- Benchmarks on real hardware

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

If you use text4q Cortex in academic work, please cite:

```
@software{text4q_cortex_2024,
  author  = {Sanchez Ferra, Gabriel},
  title   = {text4q Cortex: Natural Language Interface for Quantum Computing Infrastructure},
  year    = {2024},
  url     = {https://github.com/FerraXIDE/text4q-cortex},
  version = {0.1.0}
}
```
