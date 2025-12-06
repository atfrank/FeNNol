"""
Base class for implicit solvent models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import jax.numpy as jnp


class ImplicitSolventModel(ABC):
    """
    Abstract base class for all implicit solvent models.

    All implicit solvent models should inherit from this class and implement
    the compute_energy_forces method.
    """

    def __init__(self, parameters: Dict):
        """
        Initialize the implicit solvent model.

        Args:
            parameters: Dictionary containing model parameters such as:
                - dielectric: Solvent dielectric constant
                - cutoff: Cutoff distance for interactions
                - surface_tension: Surface tension coefficient
                - etc. (model-specific)
        """
        self.parameters = parameters
        self.dielectric = parameters.get("dielectric", 80.0)
        self.cutoff = parameters.get("cutoff", 12.0)

        # Flag to indicate if CUDA kernels are available
        self.has_cuda = self._check_cuda_available()

    def _check_cuda_available(self) -> bool:
        """Check if CUDA kernels for this model are available and a CUDA device exists."""
        try:
            from fennol import cuda
            # Check for GB CUDA functions
            has_functions = (hasattr(cuda, "gb_compute_born_radii") and
                           hasattr(cuda, "gb_compute_energy_forces"))

            if not has_functions:
                return False

            # Also check if a CUDA device is actually available
            # Try to call a simple CUDA function to see if it works
            try:
                import numpy as np
                # Test with minimal inputs
                test_coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                test_radii = np.array([1.5], dtype=np.float32)
                test_b = np.array([0.8], dtype=np.float32)
                test_c = np.array([0.0], dtype=np.float32)
                # Try to compute Born radii - this will fail if no CUDA device
                cuda.gb_compute_born_radii(test_coords, test_radii, test_b, test_c, 8.0)
                return True
            except RuntimeError as e:
                # CUDA device not available
                if "CUDA" in str(e) or "no CUDA-capable device" in str(e):
                    return False
                raise
        except (ImportError, AttributeError, RuntimeError):
            return False

    @abstractmethod
    def compute_energy_forces(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        box: Optional[jnp.ndarray] = None,
        neighborlist: Optional[Tuple] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute solvation energy and forces.

        This is the main method that should be implemented by all subclasses.

        Args:
            coords: Atomic coordinates [natoms, 3] in Angstroms
            charges: Partial atomic charges [natoms] in electron units
            atomic_numbers: Atomic numbers [natoms]
            box: Simulation box vectors [3, 3] (optional, for PBC)
            neighborlist: Precomputed neighborlist (optional, for efficiency)

        Returns:
            energy: Solvation free energy in kcal/mol
            forces: Forces on atoms [natoms, 3] in kcal/mol/Å

        Note:
            Forces are computed as dE/dr, so they should be negated when
            added to MD forces (F = -dE/dr).
        """
        pass

    def __call__(self, coords, charges, atomic_numbers, box=None, neighborlist=None):
        """Allow calling the model as a function."""
        return self.compute_energy_forces(coords, charges, atomic_numbers, box, neighborlist)

    def get_info(self) -> Dict:
        """
        Get information about the model configuration.

        Returns:
            Dictionary containing model name, parameters, and status
        """
        return {
            "model": self.__class__.__name__,
            "parameters": self.parameters,
            "cuda_available": self.has_cuda,
            "dielectric": self.dielectric,
            "cutoff": self.cutoff,
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        cuda_status = "CUDA" if self.has_cuda else "JAX"
        return (
            f"{self.__class__.__name__}("
            f"dielectric={self.dielectric}, "
            f"cutoff={self.cutoff} Å, "
            f"backend={cuda_status})"
        )
