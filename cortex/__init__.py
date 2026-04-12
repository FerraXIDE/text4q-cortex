"""
text4q Cortex
=============
Natural language interface for quantum computing infrastructure.

Quick start:
    from cortex import Cortex
    cx = Cortex(backend="ibm_quantum")
    result = cx.run("Bell state with 2 qubits, 1024 shots")
"""

from cortex.core import Cortex
from cortex.models import CortexResult, CircuitIntent

__version__ = "0.1.0-alpha"
__all__ = ["Cortex", "CortexResult", "CircuitIntent"]
