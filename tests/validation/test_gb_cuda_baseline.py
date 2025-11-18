"""
Test CUDA GB Implementation Against NumPy Reference

This validates that the current CUDA GB implementation produces correct results
by comparing against the NumPy FP64 reference implementations.

If these tests fail, the CUDA baseline is WRONG and must be fixed before
attempting any optimizations!
"""

import pytest
import numpy as np

# Import reference implementations
from fennol.validation import (
    GBReferenceOBC,
    GBEnergyReference,
    ValidationTolerances
)

# Check if CUDA is available
try:
    from fennol import cuda as fennol_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAvsReference:
    """Compare CUDA implementation to NumPy reference."""

    def test_single_atom_born_radius(self):
        """Single atom: CUDA should match reference exactly."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.5])
        b_params = np.array([0.8])
        c_params = np.array([0.0])
        cutoff = 12.0

        # NumPy reference
        ref = GBReferenceOBC(cutoff=cutoff)
        born_radii_ref, psi_sum_ref = ref.compute_born_radii(coords, radii, b_params, c_params)

        # CUDA implementation
        born_radii_cuda, psi_sum_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        # Should match within tight tolerance (both FP64)
        tol = ValidationTolerances.REFERENCE_TOLERANCE['born_radii']
        assert ValidationTolerances.check_array_tolerance(
            born_radii_cuda, born_radii_ref, tol, 'born_radii'
        ), "Single atom Born radius mismatch"

        assert ValidationTolerances.check_array_tolerance(
            psi_sum_cuda, psi_sum_ref, tol, 'psi_sum'
        ), "Single atom psi_sum mismatch"

    def test_two_atoms_born_radii(self):
        """Two atoms at contact: validate descreening."""
        separation = 2.8  # Typical O-O distance
        coords = np.array([
            [0.0, 0.0, 0.0],
            [separation, 0.0, 0.0]
        ])
        radii = np.array([1.5, 1.5])
        b_params = np.array([0.8, 0.8])
        c_params = np.array([0.0, 0.0])
        cutoff = 12.0

        # NumPy reference
        ref = GBReferenceOBC(cutoff=cutoff)
        born_radii_ref, psi_sum_ref = ref.compute_born_radii(coords, radii, b_params, c_params)

        # CUDA implementation
        born_radii_cuda, psi_sum_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        # Validate Born radii
        tol = ValidationTolerances.REFERENCE_TOLERANCE['born_radii']
        assert ValidationTolerances.check_array_tolerance(
            born_radii_cuda, born_radii_ref, tol, 'born_radii'
        ), "Two-atom Born radii mismatch"

        # Validate psi_sum
        assert ValidationTolerances.check_array_tolerance(
            psi_sum_cuda, psi_sum_ref, tol, 'psi_sum'
        ), "Two-atom psi_sum mismatch"

        # Check descreening occurred
        assert np.all(born_radii_cuda > radii), "Descreening should increase Born radii"
        assert np.all(psi_sum_cuda > 0), "psi_sum should be positive"

    def test_single_atom_energy(self):
        """Single atom: validate GB self-energy."""
        coords = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([-0.834])  # TIP3P oxygen
        radii = np.array([1.5])
        b_params = np.array([0.8])
        c_params = np.array([0.0])
        dielectric = 80.0
        cutoff = 12.0

        # NumPy reference
        ref_born = GBReferenceOBC(cutoff=cutoff)
        born_radii_ref, _ = ref_born.compute_born_radii(coords, radii, b_params, c_params)

        ref_energy = GBEnergyReference(dielectric=dielectric, cutoff=cutoff)
        energy_ref = ref_energy.compute_energy(coords, charges, born_radii_ref)

        # CUDA implementation
        born_radii_cuda, psi_sum_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        energy_cuda_array, _ = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_radii_cuda, radii, b_params, c_params,
            psi_sum_cuda, dielectric, cutoff
        )
        energy_cuda = float(energy_cuda_array[0])

        # Validate energy
        tol = ValidationTolerances.REFERENCE_TOLERANCE['energy']
        assert ValidationTolerances.check_tolerance(
            energy_cuda, energy_ref, tol, 'energy'
        ), "Single atom energy mismatch"

    def test_water_molecule_born_radii(self):
        """Water molecule (3 atoms): validate Born radii calculation."""
        # TIP3P water geometry
        coords = np.array([
            [0.0, 0.0, 0.0],        # O
            [0.757, 0.586, 0.0],    # H1
            [-0.757, 0.586, 0.0]    # H2
        ])
        radii = np.array([1.5, 1.2, 1.2])  # O, H, H
        b_params = np.array([0.8, 0.8, 0.8])
        c_params = np.array([0.0, 0.0, 0.0])
        cutoff = 12.0

        # NumPy reference
        ref = GBReferenceOBC(cutoff=cutoff)
        born_radii_ref, psi_sum_ref = ref.compute_born_radii(coords, radii, b_params, c_params)

        # CUDA implementation
        born_radii_cuda, psi_sum_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        # Validate
        tol = ValidationTolerances.REFERENCE_TOLERANCE['born_radii']
        assert ValidationTolerances.check_array_tolerance(
            born_radii_cuda, born_radii_ref, tol, 'born_radii'
        ), "Water molecule Born radii mismatch"

        # Check no NaN/Inf
        assert ValidationTolerances.check_no_nan_or_inf(
            born_radii_cuda, psi_sum_cuda,
            names=['born_radii', 'psi_sum']
        ), "CUDA produced NaN/Inf"

    def test_water_molecule_energy(self):
        """Water molecule: validate GB energy."""
        # TIP3P water geometry
        coords = np.array([
            [0.0, 0.0, 0.0],        # O
            [0.757, 0.586, 0.0],    # H1
            [-0.757, 0.586, 0.0]    # H2
        ])
        charges = np.array([-0.834, 0.417, 0.417])
        radii = np.array([1.5, 1.2, 1.2])
        b_params = np.array([0.8, 0.8, 0.8])
        c_params = np.array([0.0, 0.0, 0.0])
        dielectric = 80.0
        cutoff = 12.0

        # NumPy reference
        ref_born = GBReferenceOBC(cutoff=cutoff)
        born_radii_ref, _ = ref_born.compute_born_radii(coords, radii, b_params, c_params)

        ref_energy = GBEnergyReference(dielectric=dielectric, cutoff=cutoff)
        energy_ref = ref_energy.compute_energy(coords, charges, born_radii_ref)

        # CUDA implementation
        born_radii_cuda, psi_sum_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        energy_cuda_array, _ = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_radii_cuda, radii, b_params, c_params,
            psi_sum_cuda, dielectric, cutoff
        )
        energy_cuda = float(energy_cuda_array[0])

        # Validate energy
        tol = ValidationTolerances.REFERENCE_TOLERANCE['energy']
        assert ValidationTolerances.check_tolerance(
            energy_cuda, energy_ref, tol, 'energy'
        ), f"Water molecule energy mismatch: CUDA={energy_cuda:.6f}, Ref={energy_ref:.6f}"

        # Should be negative (favorable solvation)
        assert energy_cuda < 0, f"GB energy should be negative, got {energy_cuda}"

    def test_water_molecule_forces_newton_third_law(self):
        """Water molecule: forces must obey Newton's 3rd law."""
        # TIP3P water geometry
        coords = np.array([
            [0.0, 0.0, 0.0],        # O
            [0.757, 0.586, 0.0],    # H1
            [-0.757, 0.586, 0.0]    # H2
        ])
        charges = np.array([-0.834, 0.417, 0.417])
        radii = np.array([1.5, 1.2, 1.2])
        b_params = np.array([0.8, 0.8, 0.8])
        c_params = np.array([0.0, 0.0, 0.0])
        dielectric = 80.0
        cutoff = 12.0

        # CUDA implementation
        born_radii_cuda, psi_sum_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        _, forces_cuda = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_radii_cuda, radii, b_params, c_params,
            psi_sum_cuda, dielectric, cutoff
        )

        # Check Newton's 3rd law (net force = 0)
        assert ValidationTolerances.check_newton_third_law(forces_cuda, tolerance=1e-6), \
            "CUDA forces violate Newton's 3rd law"

        # Check no NaN/Inf
        assert ValidationTolerances.check_no_nan_or_inf(
            forces_cuda, names=['forces']
        ), "CUDA forces contain NaN/Inf"

    def test_two_water_molecules_energy(self):
        """Two water molecules: larger system validation."""
        # Two TIP3P waters separated by 3.5 Å
        coords = np.array([
            # Water 1
            [0.0, 0.0, 0.0],        # O
            [0.757, 0.586, 0.0],    # H1
            [-0.757, 0.586, 0.0],   # H2
            # Water 2 (translated)
            [3.5, 0.0, 0.0],        # O
            [4.257, 0.586, 0.0],    # H1
            [2.743, 0.586, 0.0]     # H2
        ])
        charges = np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417])
        radii = np.array([1.5, 1.2, 1.2, 1.5, 1.2, 1.2])
        b_params = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
        c_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dielectric = 80.0
        cutoff = 12.0

        # NumPy reference
        ref_born = GBReferenceOBC(cutoff=cutoff)
        born_radii_ref, _ = ref_born.compute_born_radii(coords, radii, b_params, c_params)

        ref_energy = GBEnergyReference(dielectric=dielectric, cutoff=cutoff)
        energy_ref = ref_energy.compute_energy(coords, charges, born_radii_ref)

        # CUDA implementation
        born_radii_cuda, psi_sum_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        energy_cuda_array, forces_cuda = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_radii_cuda, radii, b_params, c_params,
            psi_sum_cuda, dielectric, cutoff
        )
        energy_cuda = float(energy_cuda_array[0])

        # Validate Born radii
        tol_born = ValidationTolerances.REFERENCE_TOLERANCE['born_radii']
        assert ValidationTolerances.check_array_tolerance(
            born_radii_cuda, born_radii_ref, tol_born, 'born_radii'
        ), "Two-water Born radii mismatch"

        # Validate energy
        tol_energy = ValidationTolerances.REFERENCE_TOLERANCE['energy']
        assert ValidationTolerances.check_tolerance(
            energy_cuda, energy_ref, tol_energy, 'energy'
        ), f"Two-water energy mismatch: CUDA={energy_cuda:.6f}, Ref={energy_ref:.6f}"

        # Check Newton's 3rd law
        assert ValidationTolerances.check_newton_third_law(forces_cuda), \
            "Two-water forces violate Newton's 3rd law"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDANumericalStability:
    """Test CUDA implementation for numerical stability issues."""

    def test_no_nan_in_born_radii(self):
        """Ensure Born radii calculation doesn't produce NaN."""
        # Create challenging case: atoms very close
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Very close to first atom
            [0.0, 1.0, 0.0]
        ])
        radii = np.array([1.5, 1.2, 1.3])
        b_params = np.array([0.8, 0.8, 0.8])
        c_params = np.array([0.0, 0.0, 0.0])
        cutoff = 12.0

        born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        assert not np.any(np.isnan(born_radii)), "Born radii contains NaN"
        assert not np.any(np.isinf(born_radii)), "Born radii contains Inf"
        assert not np.any(np.isnan(psi_sum)), "psi_sum contains NaN"
        assert not np.any(np.isinf(psi_sum)), "psi_sum contains Inf"

    def test_no_nan_in_forces(self):
        """Ensure force calculation doesn't produce NaN."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0]
        ])
        charges = np.array([-0.5, 0.25, 0.25])
        radii = np.array([1.5, 1.2, 1.3])
        b_params = np.array([0.8, 0.8, 0.8])
        c_params = np.array([0.0, 0.0, 0.0])
        dielectric = 80.0
        cutoff = 12.0

        born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff
        )

        _, forces = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_radii, radii, b_params, c_params,
            psi_sum, dielectric, cutoff
        )

        assert not np.any(np.isnan(forces)), "Forces contain NaN"
        assert not np.any(np.isinf(forces)), "Forces contain Inf"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
