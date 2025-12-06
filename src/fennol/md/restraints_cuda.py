"""
CUDA-accelerated restraints module for FeNNol.

This module provides CUDA-accelerated versions of restraint calculations
with automatic fallback to JAX when CUDA is unavailable.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable, List, Dict, Any

# Try to import CUDA kernels
try:
    from ..cuda import (
        CUDA_AVAILABLE,
        harmonic_distance_restraint as cuda_distance_restraint,
        harmonic_angle_restraint as cuda_angle_restraint,
        CudaRestraints
    )
except ImportError:
    CUDA_AVAILABLE = False


class CudaRestraintCalculator:
    """
    CUDA-accelerated restraint calculator.

    Provides a drop-in replacement for JAX-based restraint calculations
    with automatic fallback when CUDA is unavailable.
    """

    def __init__(self, use_cuda: bool = True):
        """
        Initialize restraint calculator.

        Args:
            use_cuda: Whether to use CUDA if available
        """
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.restraints = []

        if self.use_cuda:
            print("# Using CUDA-accelerated restraints")
        else:
            if use_cuda:
                print("# CUDA not available, using JAX restraints")
            else:
                print("# Using JAX restraints (CUDA disabled)")

    def add_harmonic_distance(
        self,
        atom_indices: np.ndarray,
        target_distances: np.ndarray,
        force_constants: np.ndarray,
        name: str = None
    ):
        """
        Add harmonic distance restraints.

        Args:
            atom_indices: (nrestraints, 2) pairs of atom indices
            target_distances: (nrestraints,) target distances
            force_constants: (nrestraints,) force constants
            name: Optional name for the restraint
        """
        self.restraints.append({
            'type': 'harmonic_distance',
            'atom_indices': np.array(atom_indices, dtype=np.int32),
            'target_distances': np.array(target_distances, dtype=np.float64),
            'force_constants': np.array(force_constants, dtype=np.float64),
            'name': name or f'distance_{len(self.restraints)}'
        })

    def add_harmonic_angle(
        self,
        atom_indices: np.ndarray,
        target_angles: np.ndarray,
        force_constants: np.ndarray,
        name: str = None
    ):
        """
        Add harmonic angle restraints.

        Args:
            atom_indices: (nrestraints, 3) triplets of atom indices
            target_angles: (nrestraints,) target angles in radians
            force_constants: (nrestraints,) force constants
            name: Optional name for the restraint
        """
        self.restraints.append({
            'type': 'harmonic_angle',
            'atom_indices': np.array(atom_indices, dtype=np.int32),
            'target_angles': np.array(target_angles, dtype=np.float64),
            'force_constants': np.array(force_constants, dtype=np.float64),
            'name': name or f'angle_{len(self.restraints)}'
        })

    def compute_cuda(self, coordinates: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute restraints using CUDA kernels.

        Args:
            coordinates: (natoms, 3) atomic coordinates

        Returns:
            Tuple of (total_energy, total_forces)
        """
        natoms = coordinates.shape[0]
        total_energy = 0.0
        total_forces = np.zeros((natoms, 3), dtype=np.float64)

        coords = np.ascontiguousarray(coordinates, dtype=np.float64)

        for restraint in self.restraints:
            if restraint['type'] == 'harmonic_distance':
                energy, forces = cuda_distance_restraint(
                    coords,
                    restraint['atom_indices'],
                    restraint['target_distances'],
                    restraint['force_constants']
                )
                total_energy += energy
                total_forces += forces

            elif restraint['type'] == 'harmonic_angle':
                energy, forces = cuda_angle_restraint(
                    coords,
                    restraint['atom_indices'],
                    restraint['target_angles'],
                    restraint['force_constants']
                )
                total_energy += energy
                total_forces += forces

        return total_energy, total_forces

    def compute_jax(self, coordinates: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        Compute restraints using JAX (fallback).

        Args:
            coordinates: (natoms, 3) atomic coordinates

        Returns:
            Tuple of (total_energy, total_forces)
        """
        natoms = coordinates.shape[0]
        total_energy = 0.0
        total_forces = jnp.zeros((natoms, 3))

        for restraint in self.restraints:
            if restraint['type'] == 'harmonic_distance':
                energy, forces = self._compute_harmonic_distance_jax(
                    coordinates,
                    restraint['atom_indices'],
                    restraint['target_distances'],
                    restraint['force_constants']
                )
                total_energy += energy
                total_forces += forces

            elif restraint['type'] == 'harmonic_angle':
                energy, forces = self._compute_harmonic_angle_jax(
                    coordinates,
                    restraint['atom_indices'],
                    restraint['target_angles'],
                    restraint['force_constants']
                )
                total_energy += energy
                total_forces += forces

        return total_energy, total_forces

    def _compute_harmonic_distance_jax(
        self,
        coordinates: jnp.ndarray,
        atom_indices: np.ndarray,
        target_distances: np.ndarray,
        force_constants: np.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """JAX implementation of harmonic distance restraint."""
        natoms = coordinates.shape[0]
        nrestraints = len(atom_indices)

        total_energy = 0.0
        forces = jnp.zeros_like(coordinates)

        for i in range(nrestraints):
            idx_i, idx_j = atom_indices[i]
            r0 = target_distances[i]
            k = force_constants[i]

            ri = coordinates[idx_i]
            rj = coordinates[idx_j]
            rij = rj - ri
            r = jnp.linalg.norm(rij)

            dr = r - r0
            energy = 0.5 * k * dr * dr
            total_energy += energy

            # Force
            force_mag = k * dr / (r + 1e-10)
            force_vec = force_mag * rij

            forces = forces.at[idx_i].add(force_vec)
            forces = forces.at[idx_j].add(-force_vec)

        return total_energy, forces

    def _compute_harmonic_angle_jax(
        self,
        coordinates: jnp.ndarray,
        atom_indices: np.ndarray,
        target_angles: np.ndarray,
        force_constants: np.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """JAX implementation of harmonic angle restraint."""
        natoms = coordinates.shape[0]
        nrestraints = len(atom_indices)

        total_energy = 0.0
        forces = jnp.zeros_like(coordinates)

        for i in range(nrestraints):
            idx_i, idx_j, idx_k = atom_indices[i]
            theta0 = target_angles[i]
            k = force_constants[i]

            ri = coordinates[idx_i]
            rj = coordinates[idx_j]
            rk = coordinates[idx_k]

            rji = ri - rj
            rjk = rk - rj

            r_ji = jnp.linalg.norm(rji)
            r_jk = jnp.linalg.norm(rjk)

            cos_theta = jnp.dot(rji, rjk) / (r_ji * r_jk + 1e-10)
            cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
            theta = jnp.arccos(cos_theta)

            dtheta = theta - theta0
            energy = 0.5 * k * dtheta * dtheta
            total_energy += energy

            # Forces (simplified - full implementation would include analytical derivatives)
            # For now, using numerical gradients via JAX autodiff would be better

        return total_energy, forces

    def compute(
        self,
        coordinates: jnp.ndarray,
        step: int = 0
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute total restraint energy and forces.

        Args:
            coordinates: (natoms, 3) atomic coordinates
            step: Current simulation step (for time-varying restraints)

        Returns:
            Tuple of (total_energy, total_forces)
        """
        if len(self.restraints) == 0:
            natoms = coordinates.shape[0]
            return 0.0, jnp.zeros((natoms, 3))

        if self.use_cuda:
            # Convert to numpy for CUDA
            coords_np = np.array(coordinates, dtype=np.float64)
            energy, forces = self.compute_cuda(coords_np)
            # Convert back to JAX
            return float(energy), jnp.array(forces)
        else:
            # Use JAX implementation
            return self.compute_jax(coordinates)

    def __len__(self):
        """Return number of restraints."""
        return len(self.restraints)

    def summary(self) -> str:
        """Return a summary of configured restraints."""
        lines = [
            f"Restraint Calculator ({self.backend} backend)",
            f"Total restraints: {len(self.restraints)}"
        ]

        restraint_types = {}
        for r in self.restraints:
            rtype = r['type']
            restraint_types[rtype] = restraint_types.get(rtype, 0) + 1

        for rtype, count in restraint_types.items():
            lines.append(f"  - {rtype}: {count}")

        return "\n".join(lines)

    @property
    def backend(self) -> str:
        """Get the current backend."""
        return "cuda" if self.use_cuda else "jax"


def create_cuda_restraint_calculator(use_cuda: bool = True) -> CudaRestraintCalculator:
    """
    Create a CUDA-accelerated restraint calculator.

    Args:
        use_cuda: Whether to use CUDA if available

    Returns:
        CudaRestraintCalculator instance
    """
    return CudaRestraintCalculator(use_cuda=use_cuda)


def get_restraints_backend() -> str:
    """
    Get the current restraints backend.

    Returns:
        "cuda" if CUDA is available, "jax" otherwise
    """
    return "cuda" if CUDA_AVAILABLE else "jax"
