"""
NumPy FP64 Reference Implementations for GB Implicit Solvent

This module provides SLOW but CORRECT reference implementations used to
validate CUDA optimizations. These are intentionally unoptimized - they
prioritize clarity and correctness over performance.

**DO NOT OPTIMIZE THIS CODE!** It's the gold standard reference.

Key classes:
- GBReferenceOBC: Born radii calculation (OBC model)
- GBEnergyReference: GB electrostatic energy
- GBForcesNumerical: Forces via finite differences (ultimate reference)

Reference: OpenMM ReferenceObc.cpp, Hawkins et al. (1996)
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class GBReferenceOBC:
    """
    Reference implementation of OBC Born radii calculation.

    This is intentionally SLOW but CORRECT. Uses:
    - Double precision (FP64) everywhere
    - Explicit loops (no vectorization)
    - Direct implementation of HCT integral formula
    - Extensive validation checks

    DO NOT optimize this code! It's the reference.

    Reference:
        Hawkins, Cramer, Truhlar, Chem. Phys. Lett. 246, 122 (1996)
        OpenMM: ReferenceObc.cpp
    """

    def __init__(self, cutoff: float = 12.0):
        """
        Initialize OBC reference calculator.

        Args:
            cutoff: Cutoff distance for pairwise interactions (Angstroms)
        """
        self.cutoff = cutoff

    def compute_descreening_integral_HCT(
        self,
        r: float,
        rho_i: float,
        rho_j: float
    ) -> float:
        """
        Compute HCT descreening integral I(r, ρ_i, ρ_j).

        This implements the pairwise integral used in computing Born radii.
        The formula matches OpenMM's ReferenceObc.cpp exactly.

        Args:
            r: Distance between atoms i and j (Angstroms)
            rho_i: Intrinsic radius of atom i (Angstroms)
            rho_j: Intrinsic radius of atom j (Angstroms), called s_j in papers

        Returns:
            Descreening integral value (dimensionless)

        Notes:
            - Returns 0.0 for self-interaction (r < 1e-6)
            - Returns 0.0 when r >= rho_i + rho_j (no overlap)
            - Uses clamping for complete overlap region (r < |rho_i - rho_j|)
        """
        # Handle self-interaction and nearly overlapping atoms
        if r < 1e-6:
            return 0.0

        # Use OpenMM notation: s_j = rho_j
        s_j = rho_j

        # Geometric bounds for overlap regions
        upper_limit = rho_i + s_j
        lower_limit = abs(rho_i - s_j)

        # Region 1: No overlap (r >= upper_limit)
        if r >= upper_limit:
            return 0.0

        # Region 2 & 3: Complete or partial overlap (r < upper_limit)
        # Clamp r to lower_limit for numerical stability in complete overlap region
        r_calc = max(r, lower_limit)

        # Integration bounds (HCT formula)
        # l_ij = 1/max(ρ_i, |r - s_j|)
        abs_diff = abs(r_calc - s_j)
        lower_bound = max(rho_i, abs_diff)
        l_ij = 1.0 / lower_bound

        # u_ij = 1/(r + s_j)
        u_ij = 1.0 / (r_calc + s_j)

        # Precompute powers for efficiency (still slow, but clearer)
        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij
        s_j2 = s_j * s_j
        r_inv = 1.0 / r_calc

        # Avoid log(0) - should never happen but be safe
        if u_ij <= 0 or l_ij <= 0:
            warnings.warn(f"Invalid integration bounds: u_ij={u_ij}, l_ij={l_ij}")
            return 0.0

        ratio = np.log(u_ij / l_ij)

        # HCT integral formula (OpenMM ReferenceObc.cpp lines 145-147):
        # I = l_ij - u_ij + 0.25*r*(u_ij² - l_ij²)
        #     + 0.5*ln(u_ij/l_ij)/r + 0.25*s_j²/r*(l_ij² - u_ij²)
        integral = (
            l_ij - u_ij
            + 0.25 * r_calc * (u_ij2 - l_ij2)
            + 0.5 * r_inv * ratio
            + 0.25 * s_j2 * r_inv * (l_ij2 - u_ij2)
        )

        return integral

    def compute_born_radii(
        self,
        coords: np.ndarray,      # [N, 3] in Angstroms
        radii: np.ndarray,       # [N] intrinsic radii in Angstroms
        b_params: np.ndarray,    # [N] OBC b parameters
        c_params: np.ndarray     # [N] OBC c parameters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Born radii using OBC model.

        Args:
            coords: Atomic coordinates [N, 3] in Angstroms
            radii: Intrinsic atomic radii [N] (ρ) in Angstroms
            b_params: OBC b parameters [N] (dimensionless)
            c_params: OBC c parameters [N] (dimensionless)

        Returns:
            born_radii: Effective Born radii [N] in Angstroms
            psi_sum: Descreening sums [N] (for force calculation)

        Notes:
            - Uses explicit O(N²) double loop (SLOW!)
            - This is the reference - don't optimize
            - CRITICAL: Applies 0.5*rho scaling before OBC formula (OpenMM convention)
        """
        N = len(coords)

        # Input validation
        assert coords.shape == (N, 3), f"coords shape {coords.shape} != ({N}, 3)"
        assert len(radii) == N, f"radii length {len(radii)} != {N}"
        assert len(b_params) == N, f"b_params length {len(b_params)} != {N}"
        assert len(c_params) == N, f"c_params length {len(c_params)} != {N}"

        # Use FP64 explicitly
        psi_sum = np.zeros(N, dtype=np.float64)

        # Compute pairwise descreening (explicit O(N²) loop)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Compute distance
                dr = coords[i] - coords[j]
                r = np.linalg.norm(dr)

                # Apply cutoff
                if r > self.cutoff:
                    continue

                # Accumulate descreening integral
                I_ij = self.compute_descreening_integral_HCT(
                    r, radii[i], radii[j]
                )
                psi_sum[i] += I_ij

        # Apply OBC formula to get Born radii
        born_radii = np.zeros(N, dtype=np.float64)

        for i in range(N):
            rho_i = radii[i]
            psi_i = psi_sum[i]

            # CRITICAL: Scale psi by 0.5*rho_i (OpenMM convention)
            # This matches ReferenceObc.cpp line 207: sum *= 0.5*offsetRadiusI
            psi_scaled = 0.5 * rho_i * psi_i

            # OBC tanh polynomial: tanh(ψ - b*ψ² + c*ψ³)
            psi_2 = psi_scaled * psi_scaled
            psi_3 = psi_2 * psi_scaled
            tanh_arg = psi_scaled - b_params[i] * psi_2 + c_params[i] * psi_3
            tanh_val = np.tanh(tanh_arg)

            # Born radius formula: 1/R = 1/rho - tanh(...)/rho
            # Rearranged: R = rho / (1 - tanh(...))
            R_inv = 1.0 / rho_i - tanh_val / rho_i
            R = 1.0 / R_inv

            # Clamp Born radius to be at least as large as intrinsic radius
            # (physical constraint - atom can't be more buried than its own size)
            born_radii[i] = max(R, rho_i)

        return born_radii, psi_sum


class GBEnergyReference:
    """
    Reference GB electrostatic energy calculation.

    Implements the GB solvation energy formula with explicit loops.
    Slow but obviously correct.
    """

    # Coulomb constant in kcal·Å·mol⁻¹·e⁻²
    COULOMB_CONST = 332.0636

    def __init__(self, dielectric: float = 80.0, cutoff: float = 12.0):
        """
        Initialize GB energy calculator.

        Args:
            dielectric: Solvent dielectric constant (dimensionless)
            cutoff: Cutoff distance for pairwise interactions (Angstroms)
        """
        self.dielectric = dielectric
        self.cutoff = cutoff
        # GB factor: -0.5 * (1 - 1/ε) * COULOMB
        self.gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * self.COULOMB_CONST

    def compute_f_GB(
        self,
        r: float,
        R_i: float,
        R_j: float
    ) -> float:
        """
        Compute GB screening function.

        f_GB(r, R_i, R_j) = sqrt(r² + R_i*R_j * exp(-r²/(4*R_i*R_j)))

        Args:
            r: Distance between atoms (Angstroms)
            R_i: Born radius of atom i (Angstroms)
            R_j: Born radius of atom j (Angstroms)

        Returns:
            f_GB value (Angstroms)

        Reference:
            Still et al., J. Am. Chem. Soc. 112, 6127 (1990)
        """
        R_product = R_i * R_j
        r_sq = r * r

        # Exponential screening term
        exp_arg = -r_sq / (4.0 * R_product)
        exp_term = np.exp(exp_arg)

        # f_GB formula
        f_gb = np.sqrt(r_sq + R_product * exp_term)

        return f_gb

    def compute_energy(
        self,
        coords: np.ndarray,       # [N, 3] in Angstroms
        charges: np.ndarray,      # [N] in electron charges
        born_radii: np.ndarray    # [N] in Angstroms
    ) -> float:
        """
        Compute GB electrostatic solvation energy.

        E_GB = gb_factor * (E_self + E_pair)

        where:
        - E_self = Σ_i q_i²/R_i (self-solvation)
        - E_pair = Σ_i<j q_i*q_j/f_GB(r_ij, R_i, R_j) (pairwise screening)

        Args:
            coords: Atomic coordinates [N, 3] in Angstroms
            charges: Partial atomic charges [N] in electron charges
            born_radii: Effective Born radii [N] in Angstroms

        Returns:
            Total GB solvation energy in kcal/mol

        Notes:
            - Uses explicit loop over unique pairs (i < j)
            - Self-energy computed separately
        """
        N = len(coords)

        # Input validation
        assert coords.shape == (N, 3)
        assert len(charges) == N
        assert len(born_radii) == N

        # Self-energy term: Σ_i q_i²/R_i
        E_self = np.sum(charges**2 / born_radii)

        # Pairwise energy term: Σ_i<j q_i*q_j/f_GB(r_ij, R_i, R_j)
        # Explicit sum over upper triangle (i < j) to avoid double counting
        E_pair = 0.0
        for i in range(N):
            for j in range(i + 1, N):  # Only upper triangle
                # Compute distance
                dr = coords[i] - coords[j]
                r = np.linalg.norm(dr)

                # Apply cutoff (same as Born radii calculation)
                if r > self.cutoff:
                    continue

                # Compute f_GB
                f_gb = self.compute_f_GB(r, born_radii[i], born_radii[j])

                # Accumulate pairwise energy
                E_pair += charges[i] * charges[j] / f_gb

        # Total GB energy
        E_total = self.gb_factor * (E_self + E_pair)

        return E_total


class GBForcesNumerical:
    """
    Numerical gradient computation for GB forces.

    This is the ULTIMATE reference - if analytical forces disagree with this,
    the analytical forces are wrong!

    Uses central finite differences with careful step size selection.
    VERY SLOW - only for small systems (N < 20) in validation.
    """

    def __init__(
        self,
        radii: np.ndarray,
        b_params: np.ndarray,
        c_params: np.ndarray,
        dielectric: float = 80.0,
        cutoff: float = 12.0
    ):
        """
        Initialize numerical force calculator.

        Args:
            radii: Intrinsic radii [N]
            b_params: OBC b parameters [N]
            c_params: OBC c parameters [N]
            dielectric: Solvent dielectric
            cutoff: Interaction cutoff (Angstroms)
        """
        self.born_calculator = GBReferenceOBC(cutoff)
        self.energy_calculator = GBEnergyReference(dielectric)
        self.radii = radii
        self.b_params = b_params
        self.c_params = c_params

    def compute_energy_at_coords(
        self,
        coords: np.ndarray,
        charges: np.ndarray
    ) -> float:
        """
        Compute total GB energy at given coordinates.

        This recomputes Born radii at the given coordinates, then
        computes the energy. This is the function we differentiate.

        Args:
            coords: Coordinates [N, 3]
            charges: Charges [N]

        Returns:
            Total GB solvation energy (kcal/mol)
        """
        # Recompute Born radii at these coordinates
        born_radii, _ = self.born_calculator.compute_born_radii(
            coords, self.radii, self.b_params, self.c_params
        )

        # Compute energy
        energy = self.energy_calculator.compute_energy(
            coords, charges, born_radii
        )

        return energy

    def compute_forces_numerical(
        self,
        coords: np.ndarray,
        charges: np.ndarray,
        step: float = 1e-5
    ) -> np.ndarray:
        """
        Compute forces using central finite differences.

        F_i = -∂E/∂x_i ≈ -(E(x+h) - E(x-h))/(2h)

        This is THE reference for force validation. If your analytical
        forces don't match this (within tolerance), they are WRONG.

        Args:
            coords: Coordinates [N, 3] in Angstroms
            charges: Charges [N] in electron charges
            step: Finite difference step size (Angstroms)
                  Default 1e-5 is usually good for FP64

        Returns:
            forces: Forces [N, 3] in kcal/(mol·Å)

        Notes:
            - Uses central differences for O(h²) accuracy
            - VERY SLOW: 6*N energy evaluations
            - Only use for small systems (N < 20)
        """
        N = len(coords)
        forces = np.zeros((N, 3), dtype=np.float64)

        # Loop over all atoms and all dimensions
        for i in range(N):
            for d in range(3):  # x, y, z
                # Forward step
                coords_plus = coords.copy()
                coords_plus[i, d] += step
                E_plus = self.compute_energy_at_coords(coords_plus, charges)

                # Backward step
                coords_minus = coords.copy()
                coords_minus[i, d] -= step
                E_minus = self.compute_energy_at_coords(coords_minus, charges)

                # Central difference: F = -dE/dx
                forces[i, d] = -(E_plus - E_minus) / (2.0 * step)

        return forces

    def estimate_optimal_step_size(
        self,
        coords: np.ndarray,
        charges: np.ndarray,
        test_atom: int = 0,
        test_dim: int = 0
    ) -> float:
        """
        Estimate optimal finite difference step size using Richardson extrapolation.

        Tests different step sizes and finds the one that minimizes
        truncation + roundoff error.

        Args:
            coords: Coordinates [N, 3]
            charges: Charges [N]
            test_atom: Which atom to test on (default: 0)
            test_dim: Which dimension to test on (default: 0 = x)

        Returns:
            Optimal step size in Angstroms

        Notes:
            - Tests step sizes from 1e-8 to 1e-3
            - Uses Richardson extrapolation to estimate error
            - Run this once to find good step size for your system
        """
        steps = np.logspace(-8, -3, 20)
        errors = []

        for h in steps:
            # Compute derivative with step h
            coords_plus = coords.copy()
            coords_plus[test_atom, test_dim] += h
            E_plus = self.compute_energy_at_coords(coords_plus, charges)

            coords_minus = coords.copy()
            coords_minus[test_atom, test_dim] -= h
            E_minus = self.compute_energy_at_coords(coords_minus, charges)

            deriv_h = (E_plus - E_minus) / (2 * h)

            # Compute derivative with step h/2 (higher accuracy)
            coords_plus2 = coords.copy()
            coords_plus2[test_atom, test_dim] += h / 2
            E_plus2 = self.compute_energy_at_coords(coords_plus2, charges)

            coords_minus2 = coords.copy()
            coords_minus2[test_atom, test_dim] -= h / 2
            E_minus2 = self.compute_energy_at_coords(coords_minus2, charges)

            deriv_h2 = (E_plus2 - E_minus2) / h

            # Richardson extrapolation error estimate
            error = abs(deriv_h - deriv_h2)
            errors.append(error)

        # Find step size with minimum error
        optimal_idx = np.argmin(errors)
        optimal_step = steps[optimal_idx]

        print(f"Optimal step size: {optimal_step:.2e}")
        print(f"Minimum error: {errors[optimal_idx]:.2e}")

        return optimal_step
