"""
Unit tests for CUDA-accelerated restraint kernels.

Tests verify numerical correctness against JAX implementation and
check for proper error handling.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose

# Try to import CUDA module
try:
    from fennol.cuda import CUDA_AVAILABLE
    if CUDA_AVAILABLE:
        from fennol.cuda import (
            harmonic_distance_restraint,
            harmonic_angle_restraint,
        )
except ImportError:
    CUDA_AVAILABLE = False

# Import restraints calculator
from fennol.md.restraints_cuda import CudaRestraintCalculator


class TestCudaDistanceRestraints:
    """Test suite for CUDA distance restraint kernels."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        np.random.seed(42)
        return {
            'coordinates': np.random.randn(10, 3).astype(np.float64),
            'atom_pairs': np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            'target_distances': np.array([1.5, 2.0, 1.8], dtype=np.float64),
            'force_constants': np.array([100.0, 100.0, 100.0], dtype=np.float64),
        }

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_harmonic_distance_shape(self, simple_system):
        """Test that harmonic distance restraint returns correct shapes."""
        energy, forces = harmonic_distance_restraint(
            simple_system['coordinates'],
            simple_system['atom_pairs'],
            simple_system['target_distances'],
            simple_system['force_constants']
        )

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == simple_system['coordinates'].shape

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_harmonic_distance_energy_positive(self, simple_system):
        """Test that energy is positive for violated restraints."""
        energy, forces = harmonic_distance_restraint(
            simple_system['coordinates'],
            simple_system['atom_pairs'],
            simple_system['target_distances'],
            simple_system['force_constants']
        )

        # Energy should be non-negative for harmonic restraints
        assert energy >= 0.0

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_harmonic_distance_zero_at_target(self):
        """Test that energy is zero when distance equals target."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],  # Distance = 1.5
        ], dtype=np.float64)

        atom_pairs = np.array([[0, 1]], dtype=np.int32)
        target_distances = np.array([1.5], dtype=np.float64)
        force_constants = np.array([100.0], dtype=np.float64)

        energy, forces = harmonic_distance_restraint(
            coords, atom_pairs, target_distances, force_constants
        )

        # Energy should be ~zero when distance equals target
        assert abs(energy) < 1e-10

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_harmonic_distance_forces_opposite(self):
        """Test that forces on atom pair are opposite."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Distance = 2.0, target = 1.5
        ], dtype=np.float64)

        atom_pairs = np.array([[0, 1]], dtype=np.int32)
        target_distances = np.array([1.5], dtype=np.float64)
        force_constants = np.array([100.0], dtype=np.float64)

        energy, forces = harmonic_distance_restraint(
            coords, atom_pairs, target_distances, force_constants
        )

        # Forces should be opposite (Newton's third law)
        assert_allclose(forces[0], -forces[1], rtol=1e-10)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_harmonic_distance_force_magnitude(self):
        """Test force magnitude for simple case."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Distance = 2.0, target = 1.5
        ], dtype=np.float64)

        atom_pairs = np.array([[0, 1]], dtype=np.int32)
        target_distances = np.array([1.5], dtype=np.float64)
        force_constants = np.array([100.0], dtype=np.float64)

        energy, forces = harmonic_distance_restraint(
            coords, atom_pairs, target_distances, force_constants
        )

        # Calculate expected values
        r = 2.0
        r0 = 1.5
        k = 100.0
        dr = r - r0  # 0.5

        expected_energy = 0.5 * k * dr**2  # 12.5
        expected_force_mag = k * dr  # 50.0

        assert_allclose(energy, expected_energy, rtol=1e-10)
        # Force should be in x-direction
        assert_allclose(abs(forces[0, 0]), expected_force_mag, rtol=1e-10)

    def test_cuda_vs_jax_distance_restraints(self, simple_system):
        """Compare CUDA and JAX implementations."""
        coords = simple_system['coordinates']
        pairs = simple_system['atom_pairs']
        targets = simple_system['target_distances']
        fcs = simple_system['force_constants']

        # JAX implementation (simplified)
        energy_jax = 0.0
        forces_jax = np.zeros_like(coords)

        for i, (idx_i, idx_j) in enumerate(pairs):
            ri = coords[idx_i]
            rj = coords[idx_j]
            rij = rj - ri
            r = np.linalg.norm(rij)

            dr = r - targets[i]
            k = fcs[i]

            energy_jax += 0.5 * k * dr**2

            force_mag = k * dr / (r + 1e-10)
            force_vec = force_mag * rij

            forces_jax[idx_i] += force_vec
            forces_jax[idx_j] -= force_vec

        if CUDA_AVAILABLE:
            # CUDA implementation
            energy_cuda, forces_cuda = harmonic_distance_restraint(
                coords, pairs, targets, fcs
            )

            # Compare
            assert_allclose(energy_cuda, energy_jax, rtol=1e-12, atol=1e-12)
            assert_allclose(forces_cuda, forces_jax, rtol=1e-12, atol=1e-12)
        else:
            pytest.skip("CUDA not available for comparison")


class TestCudaAngleRestraints:
    """Test suite for CUDA angle restraint kernels."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        return {
            'coordinates': np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ], dtype=np.float64),
            'atom_triplets': np.array([[0, 1, 2]], dtype=np.int32),
            'target_angles': np.array([np.pi/2], dtype=np.float64),  # 90 degrees
            'force_constants': np.array([100.0], dtype=np.float64),
        }

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_harmonic_angle_shape(self, simple_system):
        """Test that harmonic angle restraint returns correct shapes."""
        energy, forces = harmonic_angle_restraint(
            simple_system['coordinates'],
            simple_system['atom_triplets'],
            simple_system['target_angles'],
            simple_system['force_constants']
        )

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == simple_system['coordinates'].shape

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_harmonic_angle_zero_at_target(self, simple_system):
        """Test that energy is zero when angle equals target."""
        # This system already has a 90-degree angle
        energy, forces = harmonic_angle_restraint(
            simple_system['coordinates'],
            simple_system['atom_triplets'],
            simple_system['target_angles'],
            simple_system['force_constants']
        )

        # Energy should be ~zero
        assert abs(energy) < 1e-10


class TestCudaRestraintCalculator:
    """Test suite for CudaRestraintCalculator."""

    def test_calculator_creation(self):
        """Test creation of restraint calculator."""
        calc = CudaRestraintCalculator(use_cuda=CUDA_AVAILABLE)

        assert calc.backend in ['cuda', 'jax']
        assert len(calc) == 0

    def test_add_distance_restraints(self):
        """Test adding distance restraints."""
        calc = CudaRestraintCalculator(use_cuda=CUDA_AVAILABLE)

        calc.add_harmonic_distance(
            atom_indices=np.array([[0, 1], [2, 3]], dtype=np.int32),
            target_distances=np.array([1.5, 2.0], dtype=np.float64),
            force_constants=np.array([100.0, 100.0], dtype=np.float64),
            name='test_distance'
        )

        assert len(calc) == 1

    def test_add_angle_restraints(self):
        """Test adding angle restraints."""
        calc = CudaRestraintCalculator(use_cuda=CUDA_AVAILABLE)

        calc.add_harmonic_angle(
            atom_indices=np.array([[0, 1, 2]], dtype=np.int32),
            target_angles=np.array([np.pi/2], dtype=np.float64),
            force_constants=np.array([100.0], dtype=np.float64),
            name='test_angle'
        )

        assert len(calc) == 1

    def test_compute_empty(self):
        """Test computing with no restraints."""
        calc = CudaRestraintCalculator(use_cuda=CUDA_AVAILABLE)

        coords = jnp.array(np.random.randn(10, 3), dtype=jnp.float64)
        energy, forces = calc.compute(coords)

        assert energy == 0.0
        assert forces.shape == coords.shape
        assert jnp.all(forces == 0.0)

    def test_compute_with_restraints(self):
        """Test computing with restraints."""
        calc = CudaRestraintCalculator(use_cuda=CUDA_AVAILABLE)

        calc.add_harmonic_distance(
            atom_indices=np.array([[0, 1]], dtype=np.int32),
            target_distances=np.array([1.5], dtype=np.float64),
            force_constants=np.array([100.0], dtype=np.float64)
        )

        coords = jnp.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=jnp.float64)

        energy, forces = calc.compute(coords)

        assert energy > 0.0  # Should have non-zero energy
        assert forces.shape == coords.shape
        assert not jnp.all(forces == 0.0)  # Should have non-zero forces

    def test_summary(self):
        """Test summary generation."""
        calc = CudaRestraintCalculator(use_cuda=CUDA_AVAILABLE)

        calc.add_harmonic_distance(
            atom_indices=np.array([[0, 1]], dtype=np.int32),
            target_distances=np.array([1.5], dtype=np.float64),
            force_constants=np.array([100.0], dtype=np.float64)
        )

        summary = calc.summary()
        assert 'Restraint Calculator' in summary
        assert 'harmonic_distance' in summary


class TestNumericalStability:
    """Test numerical stability of CUDA restraint kernels."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_very_small_distances(self):
        """Test with very small distances."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0],
        ], dtype=np.float64)

        atom_pairs = np.array([[0, 1]], dtype=np.int32)
        target_distances = np.array([1.5], dtype=np.float64)
        force_constants = np.array([100.0], dtype=np.float64)

        # Should not crash or produce NaN
        energy, forces = harmonic_distance_restraint(
            coords, atom_pairs, target_distances, force_constants
        )

        assert not np.isnan(energy)
        assert not np.any(np.isnan(forces))

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_collinear_angle(self):
        """Test angle restraint with collinear atoms (180 degrees)."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Collinear
        ], dtype=np.float64)

        atom_triplets = np.array([[0, 1, 2]], dtype=np.int32)
        target_angles = np.array([np.pi], dtype=np.float64)  # 180 degrees
        force_constants = np.array([100.0], dtype=np.float64)

        # Should not crash
        energy, forces = harmonic_angle_restraint(
            coords, atom_triplets, target_angles, force_constants
        )

        assert not np.isnan(energy)
        assert not np.any(np.isnan(forces))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
