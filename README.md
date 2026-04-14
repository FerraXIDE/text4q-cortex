# text4q Cortex

Natural language interface for quantum computing infrastructure.

```python
from cortex import Cortex

cx = Cortex(backend="ibm_quantum", token="YOUR_TOKEN", backend_name="ibm_fez")
result = cx.run("Bell state with 2 qubits, 1024 shots")

print(result.counts)
# {'00': 504, '11': 459, '01': 27, '10': 34}  -- real QPU output, ibm_fez

print(f"Fidelity: {result.fidelity():.2%}")
# Fidelity: 94.04%
```

## Overview

Cortex is an open-source quantum orchestration platform that translates natural language into executable quantum circuits and runs them on real QPUs.

Tested on **ibm_fez** (156 qubits, IBM Eagle r3) — April 2026.

## Architecture

```
User (natural language)
        |
  Cortex NLP Engine        -- pattern-based (v0.1) + LLM-powered (v0.2)
        |                     + sequential command parser (v0.7)
  Validator                -- 4-layer error validation before QPU execution
        |
  Circuit Optimizer        -- removes redundant gates, improves fidelity
        |
  OQTOPUS Job Queue        -- cloud layer: multi-user, rate limiting, auth
        |
  QAOA Scheduler           -- quantum-native job-to-QPU assignment
        |
  QPU / Simulator          -- IBM Quantum (ibm_fez, ibm_marrakesh, ibm_kingston)
                              or local Aer simulator
```

## Hardware benchmark

Executed on **ibm_fez** (156-qubit IBM Eagle r3 processor) — April 13, 2026:

| Circuit | Shots | Expected states | Fidelity | Noise |
|---------|-------|----------------|----------|-------|
| Bell state | 1024 | \|00⟩, \|11⟩ | 94.34% | 5.66% |
| Bell state | 1024 | \|00⟩, \|11⟩ | 94.04% | 5.96% |

Raw counts from hardware:
```
{'00': 504, '11': 459, '01': 27, '10': 34}
```

The `01` and `10` counts represent real quantum noise — gate errors,
decoherence, and readout errors from the physical QPU. This is expected
behavior on real hardware and cannot be reproduced in simulation.

## Status

All modules implemented and tested. 192 tests passing across Python 3.10, 3.11, 3.12.

| Module | Description | Status |
|--------|-------------|--------|
| `cortex.nlp` | Pattern-based NLP engine | Stable |
| `cortex.nlp.llm_engine` | LLM-powered engine (Claude / GPT-4o) | Stable |
| `cortex.nlp.sequential` | Free-form sequential command parser | Stable |
| `cortex.nlp.validator` | 4-layer intelligent error validation | Stable |
| `cortex.nlp.optimizer` | Circuit optimization, gate cancellation | Stable |
| `cortex.connectors` | IBM Quantum (ibm_fez, ibm_marrakesh) + Aer | Stable |
| `cortex.cloud` | REST API, async job queue, web dashboard | Stable |
| `cortex.scheduler` | QAOA-based QPU assignment | Stable |
| `cortex.session` | Conversational circuit building | Stable |
| `cortex.cli` | Command-line interface | Stable |

## Installation

```bash
pip install text4q-cortex
```

With quantum backends:

```bash
pip install "text4q-cortex[qiskit]"   # IBM Quantum + Aer simulator
pip install "text4q-cortex[all]"      # everything
```

## Quick start

### Local simulation (no credentials needed)

```python
from cortex import Cortex

cx = Cortex(backend="aer")
result = cx.run("GHZ state with 3 qubits, 2048 shots")

print(result.counts)
# {'000': 1024, '111': 1024}

print(result.diagram())
#      ┌───┐          ┌─┐
# q_0: ┤ H ├──■────■──┤M├
#      └───┘┌─┴─┐  │  └╥┘
# q_1: ─────┤ X ├──┼───╫─
#           └───┘┌─┴─┐ ║
# q_2: ──────────┤ X ├─╫─
#                └───┘ ║
```

### IBM Quantum (real hardware)

```python
import os
from cortex import Cortex

cx = Cortex(
    backend="ibm_quantum",
    token=os.environ["IBM_QUANTUM_TOKEN"],
    backend_name="ibm_fez",   # 156 qubits, IBM Eagle r3
)

result = cx.run("Bell state with 2 qubits, 1024 shots")

# Real QPU output includes noise
print(result.counts)
# {'00': 504, '11': 459, '01': 27, '10': 34}

print(f"Fidelity: {result.fidelity():.2%}")
# Fidelity: 94.04%

print(f"Job ID: {result.job_id}")
# Job ID: d7er7h15a5qc73dr9mtg
```

### Sequential command parser

```python
cx = Cortex(backend="aer")

result = cx.run("Apply H to qubit 0, CNOT from 0 to 1, CNOT from 1 to 2, measure all")
print(result.diagram())
print(result.counts)
# {'000': 484, '111': 540}
```

### Conversational session

```python
cx = Cortex(backend="aer")

with cx.session():
    cx.run("Create 3 qubits")
    cx.run("Apply H to qubit 0")
    cx.run("CNOT from 0 to 1")
    cx.run("CNOT from 0 to 2")
    cx.run("show circuit")
    result = cx.run("measure all")

print(result.counts)
# {'000': 512, '111': 512}
```

### LLM-powered engine

```python
cx = Cortex(backend="aer", nlp="llm", llm_backend="anthropic")
result = cx.run("Implement QAOA for Max-Cut on a 4-node graph, p=1 layers")
```

### Circuit visualization

```python
result = cx.run("Bell state with 2 qubits")

# ASCII diagram
print(result.diagram())

# PNG image
result.save_diagram("bell_circuit.png")

# Fidelity metric
print(f"Fidelity: {result.fidelity():.2%}")
```

### Error validation

```python
from cortex.nlp.validator import CortexValidationError

try:
    cx.run("")                          # empty input
    cx.run("Bell state with 200 qubits") # too many qubits
    cx.run("CNOT from 0 to 0")          # duplicate qubit
except CortexValidationError as e:
    print(e.first.user_message)
    # "Please provide a circuit description..."
    # "You requested 200 qubits, but aer supports max 30..."
    # "A gate cannot use the same qubit twice..."
```

### Cloud API

```bash
cortex server --port 8000 --workers 4
```

```bash
curl -X POST http://localhost:8000/jobs \
  -H "x-api-key: dev-key-0000" \
  -H "Content-Type: application/json" \
  -d '{"text": "GHZ state with 3 qubits", "backend": "aer"}'
```

Dashboard at `http://localhost:8000/dashboard`.

### CLI

```bash
cortex run "Bell state with 2 qubits" --qasm
cortex compile "GHZ state, 5 qubits" --output circuit.qasm
cortex submit "VQE for H2" --backend ibm_quantum --wait
cortex jobs --status done
cortex server
```

### QAOA Scheduler

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

result = QAOAScheduler(backend="aer", p=1).schedule(jobs, backends)
# exp-001 -> ibm_quantum  (high priority to low-error QPU)
# exp-002 -> aer           (low priority to simulator)
```

## Available QPUs (IBM Quantum Open Plan)

| Backend | Qubits | Tested |
|---------|--------|--------|
| ibm_fez | 156 | Yes — 94.34% Bell fidelity |
| ibm_marrakesh | 156 | Available |
| ibm_kingston | 156 | Available |

## Roadmap

- [x] NLP engine: pattern-based (v0.1)
- [x] LLM-powered circuit generation (v0.2)
- [x] IBM Quantum connector + Aer simulator (v0.3)
- [x] OQTOPUS job queue + cloud API (v0.3)
- [x] CLI and web dashboard (v0.4)
- [x] QAOA Scheduler (v0.5)
- [x] Circuit visualization + fidelity metrics (v0.6)
- [x] Sequential command parser (v0.7)
- [x] Intelligent error validation (v0.8)
- [x] Circuit optimizer — gate cancellation (v0.8)
- [x] Conversational session mode (v0.8)
- [x] Validated on real QPU — ibm_fez 156 qubits (v0.8)
- [ ] Classical QAOA parameter optimization (SciPy outer loop)
- [ ] Google Quantum AI connector
- [ ] IonQ and Quantinuum connectors
- [ ] arXiv paper — benchmark vs hand-written QASM
- [ ] text4q Cortex Cloud (hosted SaaS)
- [ ] PyPI stable release (v1.0)

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

Areas where contributions are welcome:
- Additional QPU connectors (Google, IonQ, Quantinuum)
- Error mitigation post-processing
- QAOA parameter optimization
- Benchmarks on real hardware

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```
@software{text4q_cortex_2024,
  author  = {Sanchez Ferra, Gabriel},
  title   = {text4q Cortex: Natural Language Interface for Quantum Computing Infrastructure},
  year    = {2024},
  url     = {https://github.com/FerraXIDE/text4q-cortex},
  version = {0.1.8}
}
```
