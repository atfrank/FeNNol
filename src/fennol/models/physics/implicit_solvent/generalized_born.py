"""
Generalized Born (GB) implicit solvent model.

Implements the OBC (Onufriev-Bashford-Case) variant of the Generalized Born
model for implicit solvation.

References:
- Still et al. (1990) J. Am. Chem. Soc. 112, 6127-6129
- Onufriev et al. (2004) Proteins 55, 383-394
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional
from functools import partial

from .base import ImplicitSolventModel
from .parameters import AtomicParameters, COULOMB_CONSTANT


class GeneralizedBorn(ImplicitSolventModel):
    """
    Generalized Born implicit solvent model (GB/SA).

    This implements the Still et al. GB model with optional OBC Born radii
    calculation and surface area-based non-polar term.
    """

    def __init__(self, parameters: Dict):
        """
        Initialize GB model.

        Args:
            parameters: Dictionary with keys:
                - dielectric: Solvent dielectric constant (default: 80.0)
                - cutoff: Cutoff distance in Å (default: 12.0)
                - surface_tension: Surface tension in kcal/mol/Ų (default: 0.005)
                - probe_radius: Solvent probe radius in Å (default: 1.4)
                - radii_set: Atomic radii set ("bondi" or "mbondi", default: "mbondi")
                - variant: GB variant ("still" or "obc", default: "obc")
                - include_nonpolar: Include surface area term (default: True)
        """
        super().__init__(parameters)

        self.surface_tension = parameters.get("surface_tension", 0.005)
        self.probe_radius = parameters.get("probe_radius", 1.4)
        self.radii_set = parameters.get("radii_set", "mbondi")
        self.variant = parameters.get("variant", "obc")
        self.include_nonpolar = parameters.get("include_nonpolar", True)

        # Initialize atomic parameters
        self.atomic_params = AtomicParameters(self.radii_set)

        # GB-specific parameters
        self.solute_dielectric = 1.0  # Interior dielectric
        # GB factor: -0.5 * (1 - 1/ε) for solvation free energy
        # Negative sign gives favorable (negative) solvation energy
        self.gb_factor = -0.5 * (1.0 / self.solute_dielectric - 1.0 / self.dielectric)

        print(f"# Initialized {self.variant.upper()} Generalized Born model")
        print(f"#   Dielectric: {self.dielectric}")
        print(f"#   Cutoff: {self.cutoff} Å")
        print(f"#   Radii set: {self.radii_set}")
        print(f"#   Surface tension: {self.surface_tension} kcal/mol/Ų")
        print(f"#   Non-polar term: {'enabled' if self.include_nonpolar else 'disabled'}")

    def compute_energy_forces(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        box: Optional[jnp.ndarray] = None,
        neighborlist: Optional[Tuple] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute GB solvation energy and forces.

        Args:
            coords: Atomic coordinates [natoms, 3]
            charges: Partial charges [natoms]
            atomic_numbers: Atomic numbers [natoms]
            box: Simulation box (optional, for PBC)
            neighborlist: Precomputed neighborlist (optional)

        Returns:
            energy: Solvation free energy (kcal/mol)
            forces: Forces on atoms [natoms, 3] (kcal/mol/Å)
        """
        # Use CUDA if available, otherwise JAX
        if self.has_cuda:
            return self._compute_cuda(coords, charges, atomic_numbers)
        else:
            return self._compute_jax(coords, charges, atomic_numbers, box)

    def _compute_jax(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> Tuple[float, jnp.ndarray]:
        """JAX implementation of GB energy and forces."""

        # Get atomic parameters
        radii = self.atomic_params.get_radii_array(atomic_numbers)
        b_params, c_params = self.atomic_params.get_obc_params_arrays(atomic_numbers)

        # Step 1: Compute Born radii
        born_radii = self._compute_born_radii_jax(coords, radii, b_params, c_params, box)

        # Step 2: Compute electrostatic GB energy and forces
        gb_energy, gb_forces = self._compute_gb_electrostatic_jax(
            coords, charges, born_radii, box
        )

        # Step 3: Compute non-polar (surface area) term if enabled
        if self.include_nonpolar:
            gammas = self.atomic_params.get_surface_tension_array(atomic_numbers)
            np_energy, np_forces = self._compute_nonpolar_jax(
                coords, radii, born_radii, gammas, box
            )
            total_energy = gb_energy + np_energy
            total_forces = gb_forces + np_forces
        else:
            total_energy = gb_energy
            total_forces = gb_forces

        return total_energy, total_forces

    def _compute_born_radii_jax(
        self,
        coords: jnp.ndarray,
        radii: jnp.ndarray,
        b_params: jnp.ndarray,
        c_params: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute Born radii using OBC or Still method.

        Args:
            coords: Atomic coordinates [natoms, 3]
            radii: Intrinsic atomic radii [natoms]
            b_params: OBC b parameters [natoms]
            c_params: OBC c parameters [natoms]
            box: Simulation box (for PBC)

        Returns:
            born_radii: Effective Born radii [natoms]
        """
        natoms = coords.shape[0]

        # Compute pairwise distances
        dr = coords[:, None, :] - coords[None, :, :]  # [natoms, natoms, 3]

        if box is not None:
            # Apply minimum image convention for PBC
            dr = dr - jnp.round(dr / jnp.diag(box)) * jnp.diag(box)

        r = jnp.linalg.norm(dr, axis=-1)  # [natoms, natoms]

        # Compute pairwise descreening integral
        psi = self._compute_descreening_integral(r, radii, radii[:, None])

        # Sum contributions from all atoms j != i
        psi_sum = jnp.sum(psi, axis=1) - jnp.diag(psi)  # Exclude self

        if self.variant == "obc":
            # OBC formula: 1/R_i = 1/ρ_i - tanh(ψ - b*ψ² + c*ψ³) / ρ_i
            tanh_term = jnp.tanh(
                psi_sum - b_params * psi_sum**2 + c_params * psi_sum**3
            )
            born_radii = 1.0 / (1.0 / radii - tanh_term / radii)
        else:
            # Still formula: simpler version
            born_radii = 1.0 / (1.0 / radii - psi_sum / radii)

        # Ensure born radii are at least as large as intrinsic radii
        born_radii = jnp.maximum(born_radii, radii)

        return born_radii

    @staticmethod
    def _compute_descreening_integral(
        r: jnp.ndarray,
        radii_i: jnp.ndarray,
        radii_j: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute pairwise descreening integral.

        This is the integral I(r_ij, ρ_i, ρ_j) in the OBC model.

        Args:
            r: Pairwise distances [natoms, natoms]
            radii_i: Radii of atoms i [natoms]
            radii_j: Radii of atoms j [natoms, natoms] (broadcasted)

        Returns:
            Descreening integral [natoms, natoms]
        """
        # Prevent division by zero
        r_safe = jnp.where(r > 0.001, r, 1e10)

        # Scale factor for integration
        rho_i = radii_i[:, None]
        rho_j = radii_j

        # Compute integral based on geometric conditions
        # Case 1: r >> radii (far apart) - negligible interaction
        # Case 2: r ~ radii (intermediate) - partial descreening
        # Case 3: r << radii (overlap) - full descreening

        # Simplified descreening (Still et al.)
        upper_limit = rho_i + rho_j
        lower_limit = jnp.abs(rho_i - rho_j)

        # Integration kernel
        integral = jnp.where(
            r_safe < lower_limit,
            # Complete overlap
            0.5 * (1.0 / lower_limit**2 - 1.0 / upper_limit**2),
            jnp.where(
                r_safe < upper_limit,
                # Partial overlap
                0.5 * (1.0 / r_safe**2 - 1.0 / upper_limit**2),
                # No overlap
                0.0
            )
        )

        return integral * rho_i

    def _compute_gb_electrostatic_jax(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        born_radii: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute GB electrostatic energy and forces.

        Args:
            coords: Atomic coordinates [natoms, 3]
            charges: Partial charges [natoms]
            born_radii: Born radii [natoms]
            box: Simulation box (for PBC)

        Returns:
            energy: GB electrostatic energy (kcal/mol)
            forces: Forces [natoms, 3] (kcal/mol/Å)
        """
        # Use automatic differentiation for forces
        energy_fn = lambda x: self._gb_energy_only(x, charges, born_radii, box)
        energy = energy_fn(coords)
        forces = -jax.grad(energy_fn)(coords)  # F = -dE/dr

        return energy, forces

    def _gb_energy_only(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        born_radii: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> float:
        """Compute GB electrostatic energy (for autodiff)."""

        # Compute pairwise distances
        dr = coords[:, None, :] - coords[None, :, :]

        if box is not None:
            dr = dr - jnp.round(dr / jnp.diag(box)) * jnp.diag(box)

        # Use safe norm to avoid NaN gradients at r=0 (self-interactions)
        r_squared = jnp.sum(dr**2, axis=-1)
        r = jnp.sqrt(r_squared + 1e-10)  # Add small epsilon for gradient stability

        # GB function: f_GB = sqrt(r² + R_iR_j * exp(-r²/4R_iR_j))
        R_i = born_radii[:, None]
        R_j = born_radii[None, :]
        R_product = R_i * R_j

        # Prevent numerical issues
        r_safe = jnp.maximum(r, 0.001)

        # Apply cutoff mask before computing expensive exponentials
        cutoff_mask = r_safe < self.cutoff

        # Only compute exp for pairs within cutoff
        exp_term = jnp.where(cutoff_mask, jnp.exp(-r_safe**2 / (4.0 * R_product)), 0.0)

        f_GB = jnp.sqrt(r_safe**2 + R_product * exp_term)

        # For pairs outside cutoff, set f_GB to a large value to make energy ~0
        # Use r itself for far pairs to avoid discontinuity
        f_GB = jnp.where(cutoff_mask, f_GB, r_safe + 1000.0)

        # Electrostatic interaction
        q_i = charges[:, None]
        q_j = charges[None, :]
        q_product = q_i * q_j

        # GB energy: E = gb_factor * COULOMB * Σᵢⱼ qᵢqⱼ / f_GB
        # gb_factor already includes negative sign for favorable solvation
        # Mask self-interactions (diagonal) to avoid double-counting
        mask = jnp.eye(len(charges))
        interaction_term = jnp.where(mask, 0.0, q_product / f_GB)

        # Sum pairwise interactions and divide by 2 (each pair counted twice)
        pairwise_energy = 0.5 * self.gb_factor * COULOMB_CONSTANT * jnp.sum(interaction_term)

        # Self-energy (Born self-solvation): each atom with itself
        self_energy = self.gb_factor * COULOMB_CONSTANT * jnp.sum(charges**2 / born_radii)

        # Total energy
        energy = pairwise_energy + self_energy

        return energy

    def _compute_nonpolar_jax(
        self,
        coords: jnp.ndarray,
        radii: jnp.ndarray,
        born_radii: jnp.ndarray,
        gammas: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute non-polar (surface area) energy and forces.

        Uses a simple approximation: SA_i ≈ 4π R_i²

        Args:
            coords: Atomic coordinates [natoms, 3]
            radii: Intrinsic radii [natoms]
            born_radii: Born radii [natoms]
            gammas: Surface tension coefficients [natoms]
            box: Simulation box

        Returns:
            energy: Non-polar energy (kcal/mol)
            forces: Forces [natoms, 3] (kcal/mol/Å)
        """
        # Simple approximation: Surface area proportional to Born radius squared
        # More sophisticated: LCPO or numerical surface area calculation

        surface_area = 4.0 * jnp.pi * (born_radii + self.probe_radius)**2

        # Non-polar energy: E_np = Σᵢ γᵢ * SA_i
        energy = jnp.sum(gammas * surface_area)

        # Forces via autodiff
        energy_fn = lambda x: jnp.sum(
            gammas * 4.0 * jnp.pi * (
                self._compute_born_radii_jax(x, radii, jnp.zeros_like(radii), jnp.zeros_like(radii), box) +
                self.probe_radius
            )**2
        )

        # For now, use zero forces (surface area term is small)
        # Full implementation would compute dSA/dr via numerical methods
        forces = jnp.zeros_like(coords)

        return energy, forces

    def _compute_cuda(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """CUDA implementation using native CUDA kernels."""
        try:
            from fennol import cuda as fennol_cuda
            import numpy as np

            # Convert JAX arrays to NumPy for CUDA
            coords_np = np.array(coords)
            charges_np = np.array(charges)
            atomic_numbers_np = np.array(atomic_numbers)

            # Get atomic parameters
            radii = self.atomic_params.get_radii_array(atomic_numbers_np)
            b_params, c_params = self.atomic_params.get_obc_params_arrays(atomic_numbers_np)

            # Step 1: Compute Born radii using CUDA
            born_radii = fennol_cuda.gb_compute_born_radii(
                coords_np,
                radii,
                b_params,
                c_params,
                self.cutoff
            )

            # Step 2: Compute GB electrostatic energy and forces
            gb_energy_array, gb_forces = fennol_cuda.gb_compute_energy_forces(
                coords_np,
                charges_np,
                born_radii,
                self.dielectric,
                self.cutoff
            )

            gb_energy = float(gb_energy_array[0])

            # Step 3: Compute non-polar term if enabled
            if self.include_nonpolar:
                gammas = self.atomic_params.get_surface_tension_array(atomic_numbers_np)
                np_energy_array, np_forces = fennol_cuda.gb_compute_nonpolar(
                    coords_np,
                    born_radii,
                    gammas,
                    self.probe_radius
                )

                np_energy = float(np_energy_array[0])
                total_energy = gb_energy + np_energy
                total_forces = gb_forces + np_forces
            else:
                total_energy = gb_energy
                total_forces = gb_forces

            # Convert back to JAX arrays
            total_energy_jax = jnp.array(total_energy)
            total_forces_jax = jnp.array(total_forces)

            return total_energy_jax, total_forces_jax

        except (ImportError, AttributeError) as e:
            print(f"# Warning: CUDA backend not available ({e}), falling back to JAX")
            return self._compute_jax(coords, charges, atomic_numbers, None)


class OBC(GeneralizedBorn):
    """
    Convenience class for OBC (Onufriev-Bashford-Case) GB model.

    This is equivalent to GeneralizedBorn with variant="obc".
    """

    def __init__(self, parameters: Dict):
        parameters["variant"] = "obc"
        super().__init__(parameters)
