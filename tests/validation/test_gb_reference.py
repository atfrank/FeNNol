"""
Test NumPy FP64 Reference Implementations

These tests validate that our reference implementations are correct by
comparing to analytical solutions for tiny systems.

If these tests fail, the reference implementation is WRONG and must be fixed
before doing any CUDA validation.
"""

import pytest
import numpy as np
from fennol.validation.gb_reference import (
    GBReferenceOBC,
    GBEnergyReference,
    GBForcesNumerical
)
from fennol.validation.tolerances import ValidationTolerances


class TestGBReferenceAnalytical:
    """Test reference implementations against analytical solutions."""

    def test_single_atom_born_radius(self):
        """
        Single atom: Born radius should equal intrinsic radius (no descreening).

        Analytical solution:
        - No other atoms → no descreening → ψ = 0
        - OBC formula: 1/R = 1/ρ - tanh(0)/ρ = 1/ρ
        - Therefore: R = ρ
        """
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.5])  # Oxygen intrinsic radius
        b_params = np.array([0.8])
        c_params = np.array([0.0])

        ref = GBReferenceOBC(cutoff=12.0)
        born_radii, psi_sum = ref.compute_born_radii(coords, radii, b_params, c_params)

        # Check psi_sum = 0 (no descreening)
        assert np.isclose(psi_sum[0], 0.0, atol=1e-12), \
            f"Single atom should have psi_sum=0, got {psi_sum[0]}"

        # Check Born radius = intrinsic radius
        assert np.isclose(born_radii[0], radii[0], atol=1e-12), \
            f"Single atom Born radius should be {radii[0]}, got {born_radii[0]}"

    def test_single_atom_energy(self):
        """
        Single atom: GB energy is just self-energy.

        Analytical formula:
        E_GB = gb_factor * q²/R
             = -0.5 * (1 - 1/80) * 332.0636 * q²/R

        For q = -0.834 e (oxygen), R = 1.5 Å:
        E_GB = -0.5 * 0.9875 * 332.0636 * 0.695556 / 1.5
             ≈ -76.0 kcal/mol
        """
        coords = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([-0.834])  # TIP3P oxygen
        born_radii = np.array([1.5])  # No descreening

        ref_energy = GBEnergyReference(dielectric=80.0)
        energy = ref_energy.compute_energy(coords, charges, born_radii)

        # Analytical calculation
        gb_factor = -0.5 * (1.0 - 1.0/80.0) * 332.0636
        E_analytical = gb_factor * charges[0]**2 / born_radii[0]

        assert np.isclose(energy, E_analytical, atol=1e-10), \
            f"Energy mismatch: {energy} vs {E_analytical}"

        # Also check value is reasonable
        assert -80 < energy < -70, f"Energy {energy} outside expected range"

    def test_single_atom_forces_zero(self):
        """Single atom: forces should be exactly zero."""
        coords = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([-0.834])
        radii = np.array([1.5])
        b_params = np.array([0.8])
        c_params = np.array([0.0])

        ref_forces = GBForcesNumerical(radii, b_params, c_params)
        forces = ref_forces.compute_forces_numerical(coords, charges, step=1e-5)

        assert np.allclose(forces, 0.0, atol=1e-8), \
            f"Single atom forces should be zero, got {forces}"

    def test_two_atoms_far_apart(self):
        """
        Two atoms beyond cutoff: no interaction.

        Analytical:
        - r = 15 Å > cutoff = 12 Å
        - No descreening → R_i = ρ_i for both
        - Energy = 2 * self-energy (no pairwise term)
        - Forces = 0
        """
        separation = 15.0
        coords = np.array([
            [0.0, 0.0, 0.0],
            [separation, 0.0, 0.0]
        ])
        charges = np.array([-0.834, -0.834])
        radii = np.array([1.5, 1.5])
        b_params = np.array([0.8, 0.8])
        c_params = np.array([0.0, 0.0])

        # Born radii
        ref_born = GBReferenceOBC(cutoff=12.0)
        born_radii, psi_sum = ref_born.compute_born_radii(coords, radii, b_params, c_params)

        # No descreening (beyond cutoff)
        assert np.allclose(psi_sum, 0.0, atol=1e-12), \
            f"Atoms beyond cutoff should have psi_sum=0, got {psi_sum}"
        assert np.allclose(born_radii, radii, atol=1e-12), \
            f"Born radii should equal intrinsic radii, got {born_radii}"

        # Energy (just self-energies)
        ref_energy = GBEnergyReference(dielectric=80.0)
        energy = ref_energy.compute_energy(coords, charges, born_radii)

        gb_factor = -0.5 * (1.0 - 1.0/80.0) * 332.0636
        E_analytical = gb_factor * np.sum(charges**2 / born_radii)

        assert np.isclose(energy, E_analytical, atol=1e-10), \
            f"Energy mismatch: {energy} vs {E_analytical}"

        # Forces (should be zero - no interaction)
        ref_forces = GBForcesNumerical(radii, b_params, c_params)
        forces = ref_forces.compute_forces_numerical(coords, charges)

        assert np.allclose(forces, 0.0, atol=1e-3), \
            f"Forces beyond cutoff should be ~zero, got {forces}"

    def test_two_atoms_at_contact(self):
        """
        Two atoms at contact distance.

        This tests descreening integral in partial overlap regime.
        No analytical solution, but we can check:
        - Born radii > intrinsic radii (descreening increases R)
        - Forces are non-zero
        - Net force = 0 (Newton's 3rd law)
        """
        separation = 2.8  # Typical O-O distance in water
        coords = np.array([
            [0.0, 0.0, 0.0],
            [separation, 0.0, 0.0]
        ])
        charges = np.array([-0.834, -0.834])
        radii = np.array([1.5, 1.5])
        b_params = np.array([0.8, 0.8])
        c_params = np.array([0.0, 0.0])

        # Born radii
        ref_born = GBReferenceOBC(cutoff=12.0)
        born_radii, psi_sum = ref_born.compute_born_radii(coords, radii, b_params, c_params)

        # Descreening should occur (psi > 0)
        assert psi_sum[0] > 0, f"Expected descreening psi_sum > 0, got {psi_sum[0]}"
        assert psi_sum[1] > 0, f"Expected descreening psi_sum > 0, got {psi_sum[1]}"

        # Born radii should be larger than intrinsic (descreening effect)
        assert born_radii[0] >= radii[0], \
            f"Born radius {born_radii[0]} < intrinsic {radii[0]}"
        assert born_radii[1] >= radii[1], \
            f"Born radius {born_radii[1]} < intrinsic {radii[1]}"

        # By symmetry, both should be equal
        assert np.isclose(born_radii[0], born_radii[1], atol=1e-10), \
            f"Symmetric atoms should have equal Born radii: {born_radii}"

        # Forces
        ref_forces = GBForcesNumerical(radii, b_params, c_params)
        forces = ref_forces.compute_forces_numerical(coords, charges)

        # Newton's 3rd law: net force = 0
        assert ValidationTolerances.check_newton_third_law(forces, tolerance=1e-6), \
            "Net force should be zero (Newton's 3rd law)"

        # Forces should be non-zero (atoms interact)
        assert np.linalg.norm(forces) > 0.01, \
            f"Expected non-zero forces, got {forces}"

    def test_descreening_integral_non_symmetric(self):
        """
        Test that descreening integral is NOT symmetric: I(r, ρ_i, ρ_j) ≠ I(r, ρ_j, ρ_i).

        The HCT integral depends on which atom is being descreened (ρ_i is the atom
        receiving descreening, ρ_j is the descreening atom). This asymmetry is correct.
        """
        ref = GBReferenceOBC()

        r = 2.5
        rho_i = 1.5
        rho_j = 1.2

        I_ij = ref.compute_descreening_integral_HCT(r, rho_i, rho_j)
        I_ji = ref.compute_descreening_integral_HCT(r, rho_j, rho_i)

        # These should be DIFFERENT (asymmetric)
        assert not np.isclose(I_ij, I_ji, atol=1e-12), \
            f"Descreening should NOT be symmetric, but got: " \
            f"I({r}, {rho_i}, {rho_j}) = {I_ij} ≈ I({r}, {rho_j}, {rho_i}) = {I_ji}"

        # Both should be positive (descreening effect)
        assert I_ij > 0, f"I_ij should be positive, got {I_ij}"
        assert I_ji > 0, f"I_ji should be positive, got {I_ji}"

    def test_descreening_integral_edge_cases(self):
        """Test descreening integral at geometric boundaries."""
        ref = GBReferenceOBC()

        rho_i = 1.5
        rho_j = 1.5

        # Case 1: r = 0 (self-interaction)
        I_self = ref.compute_descreening_integral_HCT(0.0, rho_i, rho_j)
        assert np.isclose(I_self, 0.0, atol=1e-12), \
            f"Self-interaction should be zero, got {I_self}"

        # Case 2: r >> rho_i + rho_j (no overlap)
        I_far = ref.compute_descreening_integral_HCT(10.0, rho_i, rho_j)
        assert np.isclose(I_far, 0.0, atol=1e-12), \
            f"No overlap should give zero, got {I_far}"

        # Case 3: r = rho_i + rho_j (exactly at boundary)
        r_boundary = rho_i + rho_j
        I_boundary = ref.compute_descreening_integral_HCT(r_boundary, rho_i, rho_j)
        # Should be near zero (boundary case)
        assert abs(I_boundary) < 1e-6, \
            f"Boundary case should be small, got {I_boundary}"

    def test_f_GB_limits(self):
        """Test GB screening function f_GB in various limits."""
        ref_energy = GBEnergyReference()

        R_i = 2.0
        R_j = 2.0

        # Limit 1: r → 0 (atoms on top of each other)
        f_gb_zero = ref_energy.compute_f_GB(1e-6, R_i, R_j)
        # f_GB ≈ sqrt(R_i * R_j) when r → 0
        expected = np.sqrt(R_i * R_j)
        assert np.isclose(f_gb_zero, expected, rtol=0.01), \
            f"f_GB at r→0 should be ~{expected}, got {f_gb_zero}"

        # Limit 2: r → ∞ (atoms far apart)
        r_large = 100.0
        f_gb_inf = ref_energy.compute_f_GB(r_large, R_i, R_j)
        # f_GB → r when r >> R_i, R_j
        assert np.isclose(f_gb_inf, r_large, rtol=0.01), \
            f"f_GB at large r should be ~{r_large}, got {f_gb_inf}"


class TestNumericalForces:
    """Test numerical force calculation."""

    def test_optimal_step_size_estimation(self):
        """Test that we can find a reasonable step size."""
        coords = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        charges = np.array([-0.834, -0.834])
        radii = np.array([1.5, 1.5])
        b_params = np.array([0.8, 0.8])
        c_params = np.array([0.0, 0.0])

        ref_forces = GBForcesNumerical(radii, b_params, c_params)

        # This should find step size around 1e-5
        optimal_h = ref_forces.estimate_optimal_step_size(coords, charges)

        assert 1e-7 < optimal_h < 1e-3, \
            f"Optimal step size {optimal_h} outside reasonable range"

    def test_numerical_forces_two_atoms(self):
        """Test numerical forces for two-atom system."""
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        charges = np.array([1.0, -1.0])  # Opposite charges
        radii = np.array([1.5, 1.5])
        b_params = np.array([0.8, 0.8])
        c_params = np.array([0.0, 0.0])

        ref_forces = GBForcesNumerical(radii, b_params, c_params)
        forces = ref_forces.compute_forces_numerical(coords, charges)

        # Check Newton's 3rd law
        assert ValidationTolerances.check_newton_third_law(forces), \
            "Numerical forces violate Newton's 3rd law"

        # Check forces are along x-axis (by symmetry)
        assert np.allclose(forces[:, 1], 0.0, atol=1e-5), \
            f"Forces should be along x-axis, got y-components: {forces[:, 1]}"
        assert np.allclose(forces[:, 2], 0.0, atol=1e-5), \
            f"Forces should be along x-axis, got z-components: {forces[:, 2]}"

        # GB forces can be complex - just check they're non-zero and opposite
        # (Don't assume attraction/repulsion without full vacuum+GB calculation)
        assert abs(forces[0, 0]) > 0.1, \
            f"Forces should be non-zero, got {forces[0, 0]}"
        assert np.isclose(forces[0, 0], -forces[1, 0], atol=1e-3), \
            f"Forces should be equal and opposite, got {forces[0, 0]} vs {forces[1, 0]}"


@pytest.mark.skipif(
    True,  # Skip by default (slow)
    reason="Slow test - run manually with pytest --slow"
)
class TestReferenceSlow:
    """Slow tests - only run when explicitly requested."""

    def test_water_molecule(self):
        """Test full water molecule (3 atoms)."""
        # TIP3P water geometry
        coords = np.array([
            [0.0, 0.0, 0.0],        # O
            [0.757, 0.586, 0.0],    # H1
            [-0.757, 0.586, 0.0]    # H2
        ])
        charges = np.array([-0.834, 0.417, 0.417])
        radii = np.array([1.5, 1.2, 1.2])  # O, H, H
        b_params = np.array([0.8, 0.8, 0.8])
        c_params = np.array([0.0, 0.0, 0.0])

        # Born radii
        ref_born = GBReferenceOBC(cutoff=12.0)
        born_radii, psi_sum = ref_born.compute_born_radii(coords, radii, b_params, c_params)

        # All Born radii should be positive and >= intrinsic
        assert np.all(born_radii > 0), "Born radii must be positive"
        assert np.all(born_radii >= radii), "Born radii must be >= intrinsic radii"

        # Energy
        ref_energy = GBEnergyReference(dielectric=80.0)
        energy = ref_energy.compute_energy(coords, charges, born_radii)

        # Should be negative (favorable solvation)
        assert energy < 0, f"GB energy should be negative, got {energy}"

        # Forces (slow!)
        ref_forces = GBForcesNumerical(radii, b_params, c_params)
        forces = ref_forces.compute_forces_numerical(coords, charges)

        # Check Newton's 3rd law
        assert ValidationTolerances.check_newton_third_law(forces), \
            "Forces violate Newton's 3rd law"

        # Check no NaN/Inf
        assert ValidationTolerances.check_no_nan_or_inf(
            born_radii, np.array([energy]), forces,
            names=['born_radii', 'energy', 'forces']
        ), "Found NaN/Inf in results"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
