# text4q Cortex

> Natural language interface for quantum computing infrastructure.  
> Write quantum intent in plain language — Cortex compiles, schedules, and executes it on real QPUs.

```python
from cortex import Cortex

cx = Cortex(backend="ibm_quantum")
result = cx.run("Simulate a Bell state with 2 qubits and measure 1024 times")
print(result.counts)
# {'00': 512, '11': 512}
```

## What is text4q Cortex?

Cortex is an open-source quantum orchestration platform built on three pillars:

- **NLP → Circuit**: Translate natural language descriptions into OpenQASM 3.0 circuits
- **Orchestration**: Manage QPU resources via QRMI, integrating quantum and classical HPC
- **Execution**: Schedule and run jobs across IBM Quantum, Google, or custom lab QPUs

## Architecture

```
User (natural language)
        ↓
  Cortex NLP Engine          ← text4q core: language → OpenQASM
        ↓
  OQTOPUS Job Queue          ← cloud layer: scheduling + auth
        ↓
  QAOA Scheduler             ← quantum-native optimization (roadmap)
        ↓
  QRMI Resource Manager      ← QPU as HPC node
        ↓
  QPU / Simulator            ← IBM Quantum, Google, Qiskit Aer
```

## Installation

```bash
pip install text4q-cortex
```

Or from source:

```bash
git clone https://github.com/your-org/text4q-cortex
cd text4q-cortex
pip install -e ".[dev]"
```

## Quick Start

```python
from cortex import Cortex
from cortex.connectors import IBMQuantumConnector

# Connect to IBM Quantum
connector = IBMQuantumConnector(token="YOUR_IBM_TOKEN")
cx = Cortex(connector=connector)

# Run from natural language
result = cx.run(
    "Create a GHZ state with 3 qubits, apply noise model T1=50us, measure 2048 shots"
)

print(result.circuit)   # the generated OpenQASM circuit
print(result.counts)    # measurement results
print(result.metadata)  # backend, shots, execution time
```

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `cortex.nlp` | NLP → OpenQASM translation engine | 🚧 v0.1 |
| `cortex.connectors` | IBM Quantum, Aer, Google backends | 🚧 v0.1 |
| `cortex.scheduler` | Job queue and QPU resource management | 📋 planned |
| `cortex.cloud` | Multi-user cloud layer (OQTOPUS-based) | 📋 planned |

## Roadmap

- [x] Project structure and architecture
- [ ] v0.1 — NLP engine (pattern-based) + IBM Quantum connector
- [ ] v0.2 — LLM-powered circuit generation + multi-backend
- [ ] v0.3 — OQTOPUS job queue integration
- [ ] v0.4 — QAOA Scheduler (quantum-native scheduling)
- [ ] v1.0 — text4q Cortex Cloud (SaaS)

## Contributing

Contributions welcome. Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) first.

## License

Apache 2.0 — see [LICENSE](LICENSE).
