"""
text4q Cortex — Quick start example
====================================
Run this with: python examples/quickstart.py

For local simulation (no IBM account needed):
    python examples/quickstart.py --sim

For IBM Quantum:
    IBM_QUANTUM_TOKEN=your_token python examples/quickstart.py
"""

import os
import sys
import argparse
from cortex import Cortex

EXAMPLES = [
    "Create a Bell state with 2 qubits and measure 1024 times",
    "Build a GHZ state with 3 qubits, 2048 shots",
    "Apply a quantum Fourier transform to 4 qubits",
    "Quantum teleportation protocol",
]


def main():
    parser = argparse.ArgumentParser(description="text4q Cortex quickstart")
    parser.add_argument("--sim", action="store_true", help="Use local Aer simulator")
    parser.add_argument("--text", type=str, help="Custom circuit description")
    args = parser.parse_args()

    use_sim = args.sim or not os.environ.get("IBM_QUANTUM_TOKEN")

    if use_sim:
        print("▶  Using local Aer simulator (no IBM token required)")
        cx = Cortex(backend="aer")
    else:
        print(f"▶  Connecting to IBM Quantum...")
        cx = Cortex(backend="ibm_quantum")

    print(f"▶  {cx}\n")

    texts = [args.text] if args.text else EXAMPLES

    for text in texts:
        print(f"{'─'*60}")
        print(f"  Input:   {text}")

        # Step 1: Parse
        intent = cx.parse(text)
        print(f"  Type:    {intent.circuit_type}  ({intent.num_qubits}q, {intent.shots} shots)")
        if intent.noise_model:
            print(f"  Noise:   {intent.noise_model}")

        # Step 2: Show compiled QASM
        qasm = cx.compile(intent)
        lines = qasm.strip().split("\n")
        print(f"  QASM:    {lines[0]}")
        if len(lines) > 1:
            print(f"           ... ({len(lines)} lines total)")

        # Step 3: Execute (only in sim mode for the quickstart)
        if use_sim:
            result = cx._connector.execute(intent, qasm)
            if result.success:
                top = result.most_probable()
                print(f"  Result:  {result.counts}")
                print(f"  Top:     |{top}⟩ ({result.counts[top]} / {result.shots})")
                print(f"  Time:    {result.execution_time_ms:.1f} ms")
            else:
                print(f"  Error:   {result.error}")

        print()


if __name__ == "__main__":
    main()
