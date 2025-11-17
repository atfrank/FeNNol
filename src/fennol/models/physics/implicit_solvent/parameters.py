"""
Atomic parameters for implicit solvent models.

Contains radii, OBC parameters, surface tension coefficients, etc.
"""

import jax.numpy as jnp
from typing import Dict, Tuple


class AtomicParameters:
    """
    Atomic parameters for implicit solvent calculations.

    Includes:
    - Bondi radii (intrinsic atomic radii)
    - OBC parameters (b, c coefficients)
    - Surface tension coefficients
    - van der Waals radii
    """

    # Bondi radii (Å) - intrinsic atomic radii
    # Reference: Bondi, A. (1964). J. Phys. Chem. 68, 441-451
    BONDI_RADII = {
        1: 1.20,   # H
        5: 1.92,   # B
        6: 1.70,   # C
        7: 1.55,   # N
        8: 1.52,   # O
        9: 1.47,   # F
        14: 2.10,  # Si
        15: 1.80,  # P
        16: 1.80,  # S
        17: 1.75,  # Cl
        35: 1.85,  # Br
        53: 1.98,  # I
    }

    # MBONDI radii (modified Bondi for AMBER)
    # Reference: Onufriev et al. (2004) Proteins 55, 383-394
    MBONDI_RADII = {
        1: 1.30,   # H (increased from 1.20)
        6: 1.70,   # C
        7: 1.55,   # N
        8: 1.50,   # O (reduced from 1.52)
        9: 1.47,   # F
        14: 2.10,  # Si
        15: 1.85,  # P
        16: 1.80,  # S
        17: 1.70,  # Cl
    }

    # OBC parameters (b, c) for each element
    # Reference: Onufriev et al. (2004)
    # 1/R_i = 1/ρ_i - tanh(ψ - b*ψ² + c*ψ³) / ρ_i
    OBC_PARAMS = {
        #  Z: (b, c)
        1: (0.85, 0.72),   # H
        6: (0.72, -0.01),  # C
        7: (0.79, 0.28),   # N
        8: (0.85, 0.10),   # O
        9: (0.88, 0.00),   # F
        14: (0.80, 0.00),  # Si
        15: (0.86, 0.00),  # P
        16: (0.96, -0.02), # S
        17: (0.80, 0.00),  # Cl
        35: (0.80, 0.00),  # Br
        53: (0.80, 0.00),  # I
    }

    # Surface tension coefficients (kcal/mol/Ų)
    # Reference: Schaefer & Karplus (1996)
    SURFACE_TENSION = {
        1: 0.005,   # H
        6: 0.005,   # C
        7: 0.005,   # N
        8: 0.005,   # O
        9: 0.005,   # F
        14: 0.005,  # Si
        15: 0.005,  # P
        16: 0.005,  # S
        17: 0.005,  # Cl
        35: 0.005,  # Br
        53: 0.005,  # I
    }

    # Alternative surface tension for non-polar atoms (carbon-based)
    # Used in some GB/SA variants
    SURFACE_TENSION_NONPOLAR = {
        6: 0.0072,  # C (non-polar)
        1: 0.0,     # H (usually bonded, no contribution)
    }

    def __init__(self, radii_set: str = "mbondi"):
        """
        Initialize atomic parameters.

        Args:
            radii_set: Which radii set to use ("bondi" or "mbondi")
        """
        self.radii_set = radii_set

        if radii_set == "bondi":
            self.radii = self.BONDI_RADII
        elif radii_set == "mbondi":
            self.radii = self.MBONDI_RADII
        else:
            raise ValueError(f"Unknown radii set: {radii_set}. Use 'bondi' or 'mbondi'")

    def get_radius(self, atomic_number: int) -> float:
        """
        Get the intrinsic radius for an atomic number.

        Args:
            atomic_number: Atomic number (Z)

        Returns:
            Radius in Angstroms

        Raises:
            KeyError: If atomic number not in parameter set
        """
        if atomic_number not in self.radii:
            # Default to carbon radius for unknown elements
            print(f"Warning: No radius for Z={atomic_number}, using carbon radius")
            return self.radii[6]
        return self.radii[atomic_number]

    def get_obc_params(self, atomic_number: int) -> Tuple[float, float]:
        """
        Get OBC b, c parameters for an atomic number.

        Args:
            atomic_number: Atomic number (Z)

        Returns:
            Tuple of (b, c) parameters
        """
        if atomic_number not in self.OBC_PARAMS:
            # Default to carbon parameters
            return self.OBC_PARAMS[6]
        return self.OBC_PARAMS[atomic_number]

    def get_surface_tension(self, atomic_number: int) -> float:
        """
        Get surface tension coefficient for an atomic number.

        Args:
            atomic_number: Atomic number (Z)

        Returns:
            Surface tension in kcal/mol/Ų
        """
        if atomic_number not in self.SURFACE_TENSION:
            return self.SURFACE_TENSION[6]
        return self.SURFACE_TENSION[atomic_number]

    def get_radii_array(self, atomic_numbers: jnp.ndarray) -> jnp.ndarray:
        """
        Get array of radii for a list of atomic numbers.

        Args:
            atomic_numbers: Array of atomic numbers [natoms]

        Returns:
            Array of radii [natoms]
        """
        radii = jnp.array([self.get_radius(int(z)) for z in atomic_numbers])
        return radii

    def get_obc_params_arrays(
        self, atomic_numbers: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get arrays of OBC parameters for a list of atomic numbers.

        Args:
            atomic_numbers: Array of atomic numbers [natoms]

        Returns:
            Tuple of (b_params, c_params) arrays [natoms]
        """
        b_params = []
        c_params = []
        for z in atomic_numbers:
            b, c = self.get_obc_params(int(z))
            b_params.append(b)
            c_params.append(c)

        return jnp.array(b_params), jnp.array(c_params)

    def get_surface_tension_array(self, atomic_numbers: jnp.ndarray) -> jnp.ndarray:
        """
        Get array of surface tension coefficients.

        Args:
            atomic_numbers: Array of atomic numbers [natoms]

        Returns:
            Array of surface tensions [natoms]
        """
        gammas = jnp.array([self.get_surface_tension(int(z)) for z in atomic_numbers])
        return gammas


# Predefined parameter sets
DEFAULT_PARAMS = AtomicParameters("mbondi")
BONDI_PARAMS = AtomicParameters("bondi")

# Physical constants
COULOMB_CONSTANT = 332.0636  # kcal·Å·mol⁻¹·e⁻²
ANGSTROM_TO_NM = 0.1
KCAL_TO_KJ = 4.184
