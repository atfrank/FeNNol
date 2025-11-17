"""
Implicit Solvent Models for FeNNol

This module provides GPU-native implicit solvent models for molecular dynamics simulations.
Supported models include:
- Generalized Born (GB/SA, OBC, GBn)
- Poisson-Boltzmann (PB) [future]
- COSMO [future]
- SMD [future]
"""

from .base import ImplicitSolventModel
from .generalized_born import GeneralizedBorn, OBC
from .gnn_solvent import GNNImplicitSolvent
from .parameters import AtomicParameters

__all__ = [
    "ImplicitSolventModel",
    "GeneralizedBorn",
    "OBC",
    "GNNImplicitSolvent",
    "AtomicParameters",
]

# Model registry for easy instantiation
MODEL_REGISTRY = {
    "GB": GeneralizedBorn,
    "GBSA": GeneralizedBorn,
    "OBC": OBC,
    "GNN": GNNImplicitSolvent,
    # Future models:
    # "PB": PoissonBoltzmann,
    # "COSMO": COSMO,
    # "SMD": SMD,
}


def create_implicit_solvent_model(model_name: str, parameters: dict):
    """
    Factory function to create implicit solvent models.

    Args:
        model_name: Name of the model ("GB", "OBC", "PB", etc.)
        parameters: Dictionary of model parameters

    Returns:
        ImplicitSolventModel instance

    Example:
        >>> model = create_implicit_solvent_model("OBC", {
        ...     "dielectric": 80.0,
        ...     "cutoff": 12.0,
        ...     "surface_tension": 0.005
        ... })
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown implicit solvent model '{model_name}'. "
            f"Available models: {available}"
        )

    return MODEL_REGISTRY[model_name](parameters)
