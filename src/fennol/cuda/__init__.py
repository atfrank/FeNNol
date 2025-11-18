"""
CUDA-accelerated molecular dynamics kernels for FeNNol.

This module provides CUDA native implementations of performance-critical
MD operations, with automatic fallback to JAX when CUDA is unavailable.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

# Try to import CUDA module
try:
    from . import fennol_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "CUDA kernels not available. Falling back to JAX implementation. "
        "To enable CUDA acceleration, build the CUDA extension with: "
        "python setup.py build_ext --inplace"
    )

__all__ = [
    'CUDA_AVAILABLE',
    'velocity_verlet_step_a',
    'velocity_verlet_step_b',
    'harmonic_distance_restraint',
    'harmonic_angle_restraint',
    'CudaIntegrator',
    'CudaRestraints',
    'gb_compute_born_radii',
    'gb_compute_energy_forces',
    'gb_compute_nonpolar'
]


def velocity_verlet_step_a(
    coordinates: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity Verlet integration step A (first half).

    Updates positions and half-step velocities:
    v' = v + (dt/2) * f/m
    x' = x + dt * v'

    Args:
        coordinates: (natoms, 3) atomic coordinates
        velocities: (natoms, 3) atomic velocities
        forces: (natoms, 3) atomic forces
        masses: (natoms,) atomic masses
        dt: timestep

    Returns:
        Tuple of (updated_coordinates, updated_velocities)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    # Ensure contiguous arrays
    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    vels = np.ascontiguousarray(velocities, dtype=np.float64)
    forces_arr = np.ascontiguousarray(forces, dtype=np.float64)
    masses_arr = np.ascontiguousarray(masses, dtype=np.float64)

    return fennol_cuda.velocity_verlet_step_a(coords, vels, forces_arr, masses_arr, dt)


def velocity_verlet_step_b(
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Velocity Verlet integration step B (second half).

    Completes velocity update and computes kinetic energy:
    v_new = v + (dt/2) * f/m
    Ek = 0.5 * sum(m * v^2)

    Args:
        velocities: (natoms, 3) atomic velocities
        forces: (natoms, 3) atomic forces
        masses: (natoms,) atomic masses
        dt: timestep

    Returns:
        Tuple of (updated_velocities, kinetic_energy, kinetic_tensor)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    vels = np.ascontiguousarray(velocities, dtype=np.float64)
    forces_arr = np.ascontiguousarray(forces, dtype=np.float64)
    masses_arr = np.ascontiguousarray(masses, dtype=np.float64)

    return fennol_cuda.velocity_verlet_step_b(vels, forces_arr, masses_arr, dt)


def harmonic_distance_restraint(
    coordinates: np.ndarray,
    atom_indices: np.ndarray,
    target_distances: np.ndarray,
    force_constants: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Harmonic distance restraint.

    E = 0.5 * k * (r - r0)^2

    Args:
        coordinates: (natoms, 3) atomic coordinates
        atom_indices: (nrestraints, 2) pairs of atom indices
        target_distances: (nrestraints,) target distances
        force_constants: (nrestraints,) force constants

    Returns:
        Tuple of (energy, forces)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    indices = np.ascontiguousarray(atom_indices, dtype=np.int32)
    targets = np.ascontiguousarray(target_distances, dtype=np.float64)
    fcs = np.ascontiguousarray(force_constants, dtype=np.float64)

    return fennol_cuda.harmonic_distance_restraint(coords, indices, targets, fcs)


def harmonic_angle_restraint(
    coordinates: np.ndarray,
    atom_indices: np.ndarray,
    target_angles: np.ndarray,
    force_constants: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Harmonic angle restraint.

    E = 0.5 * k * (theta - theta0)^2

    Args:
        coordinates: (natoms, 3) atomic coordinates
        atom_indices: (nrestraints, 3) triplets of atom indices
        target_angles: (nrestraints,) target angles in radians
        force_constants: (nrestraints,) force constants

    Returns:
        Tuple of (energy, forces)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    indices = np.ascontiguousarray(atom_indices, dtype=np.int32)
    targets = np.ascontiguousarray(target_angles, dtype=np.float64)
    fcs = np.ascontiguousarray(force_constants, dtype=np.float64)

    return fennol_cuda.harmonic_angle_restraint(coords, indices, targets, fcs)


def gb_compute_born_radii(
    coordinates: np.ndarray,
    intrinsic_radii: np.ndarray,
    b_params: np.ndarray,
    c_params: np.ndarray,
    cutoff: float
) -> np.ndarray:
    """
    Compute Born radii using OBC method.

    Args:
        coordinates: (natoms, 3) atomic coordinates
        intrinsic_radii: (natoms,) intrinsic atomic radii
        b_params: (natoms,) OBC b parameters
        c_params: (natoms,) OBC c parameters
        cutoff: cutoff distance

    Returns:
        born_radii: (natoms,) effective Born radii
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    radii = np.ascontiguousarray(intrinsic_radii, dtype=np.float64)
    b = np.ascontiguousarray(b_params, dtype=np.float64)
    c = np.ascontiguousarray(c_params, dtype=np.float64)

    return fennol_cuda.gb_compute_born_radii(coords, radii, b, c, cutoff)


def gb_compute_energy_forces(
    coordinates: np.ndarray,
    charges: np.ndarray,
    born_radii: np.ndarray,
    dielectric: float,
    cutoff: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GB electrostatic energy and forces.

    Args:
        coordinates: (natoms, 3) atomic coordinates
        charges: (natoms,) partial charges
        born_radii: (natoms,) Born radii
        dielectric: solvent dielectric constant
        cutoff: cutoff distance

    Returns:
        Tuple of (energy, forces)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    q = np.ascontiguousarray(charges, dtype=np.float64)
    radii = np.ascontiguousarray(born_radii, dtype=np.float64)

    return fennol_cuda.gb_compute_energy_forces(coords, q, radii, dielectric, cutoff)


def gb_compute_born_radii_with_psi(
    coordinates: np.ndarray,
    intrinsic_radii: np.ndarray,
    b_params: np.ndarray,
    c_params: np.ndarray,
    cutoff: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Born radii AND return descreening sum (needed for accurate forces).

    Args:
        coordinates: (natoms, 3) atomic coordinates
        intrinsic_radii: (natoms,) intrinsic atomic radii
        b_params: (natoms,) OBC b parameters
        c_params: (natoms,) OBC c parameters
        cutoff: cutoff distance

    Returns:
        Tuple of (born_radii, psi_sum)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    radii = np.ascontiguousarray(intrinsic_radii, dtype=np.float64)
    b = np.ascontiguousarray(b_params, dtype=np.float64)
    c = np.ascontiguousarray(c_params, dtype=np.float64)

    return fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b, c, cutoff)


def gb_compute_forces_complete(
    coordinates: np.ndarray,
    charges: np.ndarray,
    born_radii: np.ndarray,
    intrinsic_radii: np.ndarray,
    b_params: np.ndarray,
    c_params: np.ndarray,
    psi_sum: np.ndarray,
    dielectric: float,
    cutoff: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute COMPLETE GB forces including Born radii derivatives (CORRECT VERSION!).

    This includes the full chain rule: dE/dx = ∂E/∂x + ∂E/∂R · dR/dx

    Args:
        coordinates: (natoms, 3) atomic coordinates
        charges: (natoms,) partial charges
        born_radii: (natoms,) Born radii
        intrinsic_radii: (natoms,) intrinsic atomic radii
        b_params: (natoms,) OBC b parameters
        c_params: (natoms,) OBC c parameters
        psi_sum: (natoms,) descreening sum (from gb_compute_born_radii_with_psi)
        dielectric: solvent dielectric constant
        cutoff: cutoff distance

    Returns:
        Tuple of (energy, forces)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    q = np.ascontiguousarray(charges, dtype=np.float64)
    radii = np.ascontiguousarray(born_radii, dtype=np.float64)
    intrinsic = np.ascontiguousarray(intrinsic_radii, dtype=np.float64)
    b = np.ascontiguousarray(b_params, dtype=np.float64)
    c = np.ascontiguousarray(c_params, dtype=np.float64)
    psi = np.ascontiguousarray(psi_sum, dtype=np.float64)

    return fennol_cuda.gb_compute_forces_complete(
        coords, q, radii, intrinsic, b, c, psi, dielectric, cutoff
    )


def gb_compute_nonpolar(
    coordinates: np.ndarray,
    born_radii: np.ndarray,
    gamma_params: np.ndarray,
    probe_radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute non-polar (surface area) energy and forces.

    Args:
        coordinates: (natoms, 3) atomic coordinates
        born_radii: (natoms,) Born radii
        gamma_params: (natoms,) surface tension parameters
        probe_radius: solvent probe radius

    Returns:
        Tuple of (energy, forces)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")

    coords = np.ascontiguousarray(coordinates, dtype=np.float64)
    radii = np.ascontiguousarray(born_radii, dtype=np.float64)
    gamma = np.ascontiguousarray(gamma_params, dtype=np.float64)

    return fennol_cuda.gb_compute_nonpolar(coords, radii, gamma, probe_radius)


class CudaIntegrator:
    """
    CUDA-accelerated Velocity Verlet integrator.

    Provides a drop-in replacement for JAX-based integration.
    """

    def __init__(self, dt: float, masses: np.ndarray):
        """
        Initialize integrator.

        Args:
            dt: Timestep
            masses: (natoms,) atomic masses
        """
        self.dt = dt
        self.masses = masses

    def step_a(
        self,
        coordinates: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute integration step A."""
        return velocity_verlet_step_a(
            coordinates, velocities, forces, self.masses, self.dt
        )

    def step_b(
        self,
        velocities: np.ndarray,
        forces: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Execute integration step B."""
        return velocity_verlet_step_b(
            velocities, forces, self.masses, self.dt
        )


class CudaRestraints:
    """
    CUDA-accelerated restraint force calculator.

    Provides a drop-in replacement for JAX-based restraints.
    """

    def __init__(self):
        """Initialize restraint calculator."""
        self.restraints = []

    def add_harmonic_distance(
        self,
        atom_indices: np.ndarray,
        target_distances: np.ndarray,
        force_constants: np.ndarray
    ):
        """Add harmonic distance restraints."""
        self.restraints.append({
            'type': 'harmonic_distance',
            'atom_indices': atom_indices,
            'target_distances': target_distances,
            'force_constants': force_constants
        })

    def add_harmonic_angle(
        self,
        atom_indices: np.ndarray,
        target_angles: np.ndarray,
        force_constants: np.ndarray
    ):
        """Add harmonic angle restraints."""
        self.restraints.append({
            'type': 'harmonic_angle',
            'atom_indices': atom_indices,
            'target_angles': target_angles,
            'force_constants': force_constants
        })

    def compute(self, coordinates: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute total restraint energy and forces.

        Args:
            coordinates: (natoms, 3) atomic coordinates

        Returns:
            Tuple of (total_energy, total_forces)
        """
        natoms = coordinates.shape[0]
        total_energy = 0.0
        total_forces = np.zeros((natoms, 3), dtype=np.float64)

        for restraint in self.restraints:
            if restraint['type'] == 'harmonic_distance':
                energy, forces = harmonic_distance_restraint(
                    coordinates,
                    restraint['atom_indices'],
                    restraint['target_distances'],
                    restraint['force_constants']
                )
                total_energy += energy
                total_forces += forces
            elif restraint['type'] == 'harmonic_angle':
                energy, forces = harmonic_angle_restraint(
                    coordinates,
                    restraint['atom_indices'],
                    restraint['target_angles'],
                    restraint['force_constants']
                )
                total_energy += energy
                total_forces += forces

        return total_energy, total_forces
