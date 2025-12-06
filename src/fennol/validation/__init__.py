"""
GB Implicit Solvent Validation Framework

This module provides reference implementations and validation tools for
validating CUDA GB optimizations while preserving physics accuracy.

Key components:
- gb_reference: NumPy FP64 reference implementations (GOLD STANDARD)
- test_generator: Automated test case generation
- tolerances: Precision specifications for validation
- cross_validator: Cross-implementation comparison tools
"""

__all__ = [
    'GBReferenceOBC',
    'GBEnergyReference',
    'GBForcesNumerical',
    'ValidationTolerances',
]

from .gb_reference import GBReferenceOBC, GBEnergyReference, GBForcesNumerical
from .tolerances import ValidationTolerances
