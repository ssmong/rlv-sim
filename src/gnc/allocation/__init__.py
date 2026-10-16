"""Control allocation — maps a virtual Command to a physical ActuatorCommand.

Contract: allocate(command, x_hat) -> ActuatorCommand. Reference implementation: default.py.
"""

from gnc.allocation.default import allocate

__all__ = ["allocate"]
