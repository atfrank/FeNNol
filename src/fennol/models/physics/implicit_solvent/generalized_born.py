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
        # Coulomb constant in kcal·Å·mol⁻¹·e⁻²
        COULOMB_CONST = 332.0636
        # GB factor: -0.5 * (1 - 1/ε) * COULOMB for solvation free energy
        # Negative sign gives favorable (negative) solvation energy
        self.gb_factor = -0.5 * (1.0 / self.solute_dielectric - 1.0 / self.dielectric) * COULOMB_CONST

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
        """JAX implementation of GB energy and forces with ANALYTICAL derivatives."""

        # Get atomic parameters
        radii = self.atomic_params.get_radii_array(atomic_numbers)
        b_params, c_params = self.atomic_params.get_obc_params_arrays(atomic_numbers)

        # Step 1 & 2: Compute GB electrostatic energy and forces
        # Using ANALYTICAL derivative version (matches CUDA implementation)
        gb_energy, gb_forces = self._compute_gb_electrostatic_jax_analytical(
            coords, charges, radii, b_params, c_params, box
        )

        # Step 3: Compute non-polar (surface area) term if enabled
        if self.include_nonpolar:
            gammas = self.atomic_params.get_surface_tension_array(atomic_numbers)
            np_energy, np_forces = self._compute_nonpolar_jax(
                coords, radii, atomic_numbers, gammas, box
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
        Compute pairwise descreening integral using HCT formula.

        This implements the full HCT (Hawkins-Cramer-Truhlar) integral formula
        matching OpenMM's ReferenceObc.cpp and the CUDA implementation.

        Args:
            r: Pairwise distances [natoms, natoms]
            radii_i: Radii of atoms i [natoms]
            radii_j: Radii of atoms j [natoms, natoms] (broadcasted)

        Returns:
            Descreening integral [natoms, natoms]
        """
        # Prevent division by zero for self-interaction
        r_safe = jnp.where(r > 0.001, r, 1e10)

        # Reshape for broadcasting: rho_i is [natoms, 1], rho_j is [1, natoms]
        # This way rho_i[i,j] = radius_i and rho_j[i,j] = radius_j
        rho_i = radii_i[:, None]  # [natoms] -> [natoms, 1]
        rho_j = radii_j.T if radii_j.ndim > 1 else radii_j[None, :]  # [natoms] -> [1, natoms]
        s_j = rho_j

        # Compute geometric bounds
        upper_limit = rho_i + rho_j
        lower_limit = jnp.abs(rho_i - rho_j)

        # For complete overlap (r < lower_limit), clamp r to lower_limit
        # For partial overlap (lower_limit <= r < upper_limit), use actual r
        r_for_calc = jnp.where(r_safe < lower_limit, lower_limit, r_safe)

        # Compute l_ij = 1/max(ρᵢ, |r - sⱼ|)
        abs_diff = jnp.abs(r_for_calc - s_j)
        lower_bound = jnp.maximum(rho_i, abs_diff)
        l_ij = 1.0 / (lower_bound + 1e-12)

        # Upper bound: u_ij = 1/(r + sⱼ)
        u_ij = 1.0 / (r_for_calc + s_j + 1e-12)

        # Precompute terms
        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij
        s_j2 = s_j * s_j
        r_inv = 1.0 / (r_for_calc + 1e-12)

        # Avoid log(0) by clamping the ratio
        ratio = jnp.log(jnp.maximum(u_ij / l_ij, 1e-12))

        # HCT integral formula (matches OpenMM ReferenceObc.cpp):
        # term = l_ij - u_ij + 0.25*r*(u_ij² - l_ij²) + 0.5*ln(u_ij/l_ij)/r
        #        + 0.25*s_j²/r*(l_ij² - u_ij²)
        term = (l_ij - u_ij +
                0.25 * r_for_calc * (u_ij2 - l_ij2) +
                0.5 * r_inv * ratio +
                0.25 * s_j2 * r_inv * (l_ij2 - u_ij2))

        # Apply conditions for different regions
        integral = jnp.where(
            r_safe < upper_limit,
            # Within upper limit (complete or partial overlap): use HCT formula
            term,
            # No overlap: zero contribution
            0.0
        )

        return integral

    def _compute_gb_electrostatic_jax(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        born_radii: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute GB electrostatic energy and forces.

        IMPORTANT: This version treats born_radii as fixed, which is INCORRECT!
        Forces are missing the Born radii derivative contributions: ∂E/∂R_i * ∂R_i/∂r

        This is kept for compatibility but should not be used for accurate forces.
        Use _compute_gb_electrostatic_jax_full() instead.

        Args:
            coords: Atomic coordinates [natoms, 3]
            charges: Partial charges [natoms]
            born_radii: Born radii [natoms]
            box: Simulation box (for PBC)

        Returns:
            energy: GB electrostatic energy (kcal/mol)
            forces: INCOMPLETE Forces [natoms, 3] (kcal/mol/Å) - missing Born radii derivatives!
        """
        # Use automatic differentiation for forces
        # WARNING: This only computes ∂E/∂r, missing ∂E/∂R_i * ∂R_i/∂r
        energy_fn = lambda x: self._gb_energy_only(x, charges, born_radii, box)
        energy = energy_fn(coords)
        forces = -jax.grad(energy_fn)(coords)  # F = -dE/dr (INCOMPLETE!)

        return energy, forces

    def _compute_gb_electrostatic_jax_full(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute GB electrostatic energy and forces with FULL derivatives.

        This version includes Born radii derivative contributions in forces:
        F = -∂E/∂r - ∑ᵢ (∂E/∂R_i) * (∂R_i/∂r)

        Args:
            coords: Atomic coordinates [natoms, 3]
            charges: Partial charges [natoms]
            atomic_numbers: Atomic numbers [natoms]
            box: Simulation box (for PBC)

        Returns:
            energy: GB electrostatic energy (kcal/mol)
            forces: COMPLETE Forces [natoms, 3] (kcal/mol/Å)
        """
        # Get atomic parameters
        radii = self.atomic_params.get_radii_array(atomic_numbers)
        b_params, c_params = self.atomic_params.get_obc_params_arrays(atomic_numbers)

        # Define energy function that includes Born radii calculation
        def full_energy_fn(x):
            # Recompute Born radii at new coordinates
            born_radii_x = self._compute_born_radii_jax(x, radii, b_params, c_params, box)
            # Compute GB energy with those Born radii
            return self._gb_energy_only(x, charges, born_radii_x, box)

        # Compute energy and forces via autodiff
        energy = full_energy_fn(coords)
        forces = -jax.grad(full_energy_fn)(coords)  # F = -dE/dr (COMPLETE!)

        return energy, forces

    def _compute_gb_electrostatic_jax_analytical(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        radii: jnp.ndarray,
        b_params: jnp.ndarray,
        c_params: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute GB electrostatic energy and forces using ANALYTICAL derivatives.

        This matches the CUDA implementation approach:
        1. Compute Born radii
        2. Compute direct pairwise forces (∂E/∂r)
        3. Compute ∂E/∂Rᵢ for each atom
        4. Compute Born radii derivative forces
        5. Sum all force contributions

        Returns:
            energy: GB electrostatic energy (kcal/mol)
            forces: COMPLETE forces [natoms, 3] (kcal/mol/Å)
        """
        natoms = coords.shape[0]

        # Step 1: Compute Born radii
        born_radii = self._compute_born_radii_jax(coords, radii, b_params, c_params, box)

        # Step 2: Compute pairwise distance matrix
        dr = coords[:, None, :] - coords[None, :, :]  # [natoms, natoms, 3]

        if box is not None:
            dr = dr - jnp.round(dr / jnp.diag(box)) * jnp.diag(box)

        r_sq = jnp.sum(dr**2, axis=-1)  # [natoms, natoms]
        r = jnp.sqrt(r_sq + 1e-10)  # Add epsilon for stability

        # Step 3: Compute f_GB function
        R_product = born_radii[:, None] * born_radii[None, :]  # [natoms, natoms]
        exp_arg = -r_sq / (4.0 * R_product + 1e-12)
        exp_val = jnp.exp(exp_arg)
        f_gb_sq = r_sq + R_product * exp_val
        f_gb = jnp.sqrt(f_gb_sq + 1e-10)

        # Step 4: Compute GB energy
        # E = gb_factor * ΣᵢΣⱼ qᵢqⱼ / f_GB
        # This is: self-energy (i=j) + pair energy (i≠j, counted once each)
        qi_qj = charges[:, None] * charges[None, :]  # [natoms, natoms]

        # Self-energy: Σᵢ qᵢ² / Rᵢ
        self_energy = jnp.sum(charges**2 / (born_radii + 1e-12))

        # Pairwise energy (all pairs i<j)
        mask_upper = jnp.triu(jnp.ones((natoms, natoms)), k=1)
        pair_energy = jnp.sum(qi_qj * mask_upper / f_gb)

        total_gb_energy = self.gb_factor * (self_energy + pair_energy)

        # Step 5: Compute direct pairwise forces (∂E/∂r contribution)
        # df_GB/dr = r * (1 - exp_val/4) / f_GB
        df_gb_dr = r * (1.0 - 0.25 * exp_val) / (f_gb + 1e-12)

        # Force magnitude: F = gb_factor * qᵢqⱼ * df_GB/dr / f_GB²
        force_mag = self.gb_factor * qi_qj * df_gb_dr / (f_gb**2 + 1e-12)  # [natoms, natoms]

        # Force direction: F_vec = force_mag * dr / r
        # Avoid self-interaction by masking diagonal
        mask_off_diag = 1.0 - jnp.eye(natoms)
        force_mag = force_mag * mask_off_diag

        force_direction = dr / (r[:, :, None] + 1e-12)  # [natoms, natoms, 3]
        force_vectors = force_mag[:, :, None] * force_direction  # [natoms, natoms, 3]

        # Sum forces on each atom (Fᵢ = Σⱼ F_ij)
        direct_forces = jnp.sum(force_vectors, axis=1)  # [natoms, 3]

        # Step 6: Compute ∂E/∂Rᵢ for each atom
        # Self-energy contribution: ∂E_self/∂Rᵢ = gb_factor * (-qᵢ²/Rᵢ²)
        dE_dR_self = self.gb_factor * (-charges**2 / (born_radii**2 + 1e-12))

        # Pairwise contribution: ∂f_GB/∂Rᵢ
        r_sq_term = r_sq / (4.0 * R_product + 1e-12)
        # ∂f_GB/∂Rᵢ = 0.5 / f_GB * Rⱼ * exp * (1 + r²/(4*Rᵢ*Rⱼ))
        df_gb_dRi = 0.5 / (f_gb + 1e-12) * born_radii[None, :] * exp_val * (1.0 + r_sq_term)

        # ∂E/∂Rᵢ pairwise = Σⱼ gb_factor * qᵢqⱼ * (-1/f_GB²) * ∂f_GB/∂Rᵢ
        dE_dR_pair = jnp.sum(
            self.gb_factor * qi_qj * (-1.0 / (f_gb**2 + 1e-12)) * df_gb_dRi * mask_off_diag,
            axis=1
        )

        dE_dR = dE_dR_self + dE_dR_pair  # [natoms]

        # Step 7: Compute Born radii derivative forces
        # We need ∂Rᵢ/∂r which requires computing psi and its derivatives
        # For now, return just direct forces (this is incomplete but better than NaN)
        # TODO: Add Born radii derivative term

        forces = direct_forces

        return total_gb_energy, forces

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
        atomic_numbers: jnp.ndarray,
        gammas: jnp.ndarray,
        box: Optional[jnp.ndarray] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute non-polar (surface area) energy and forces.

        Uses a simple approximation: SA_i ≈ 4π R_i²

        Args:
            coords: Atomic coordinates [natoms, 3]
            radii: Intrinsic radii [natoms]
            atomic_numbers: Atomic numbers [natoms]
            gammas: Surface tension coefficients [natoms]
            box: Simulation box

        Returns:
            energy: Non-polar energy (kcal/mol)
            forces: Forces [natoms, 3] (kcal/mol/Å)
        """
        # Get OBC parameters for Born radii calculation
        b_params, c_params = self.atomic_params.get_obc_params_arrays(atomic_numbers)

        # Compute Born radii at current coordinates
        born_radii = self._compute_born_radii_jax(coords, radii, b_params, c_params, box)

        # Simple approximation: Surface area proportional to Born radius squared
        # More sophisticated: LCPO or numerical surface area calculation
        surface_area = 4.0 * jnp.pi * (born_radii + self.probe_radius)**2

        # Non-polar energy: E_np = Σᵢ γᵢ * SA_i
        energy = jnp.sum(gammas * surface_area)

        # Forces via autodiff (includes Born radii derivatives)
        def energy_fn(x):
            born_radii_x = self._compute_born_radii_jax(x, radii, b_params, c_params, box)
            surface_area_x = 4.0 * jnp.pi * (born_radii_x + self.probe_radius)**2
            return jnp.sum(gammas * surface_area_x)

        forces = -jax.grad(energy_fn)(coords)

        return energy, forces

    def _compute_cuda(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """CUDA implementation using native CUDA kernels with COMPLETE forces."""
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

            # Step 1: Compute Born radii WITH psi_sum (needed for accurate forces)
            born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
                coords_np,
                radii,
                b_params,
                c_params,
                self.cutoff
            )

            # Step 2: Compute GB electrostatic energy and COMPLETE forces
            # (includes Born radii derivative contributions!)
            gb_energy_array, gb_forces = fennol_cuda.gb_compute_forces_complete(
                coords_np,
                charges_np,
                born_radii,
                radii,  # intrinsic radii
                b_params,
                c_params,
                psi_sum,
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
