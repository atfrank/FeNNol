"""
Unit tests for CUDA-accelerated integration kernels.

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
            velocity_verlet_step_a,
            velocity_verlet_step_b,
        )
except ImportError:
    CUDA_AVAILABLE = False

# Import JAX-based integration for comparison
from fennol.md.integrate_cuda import create_cuda_integrator


class TestCudaIntegration:
    """Test suite for CUDA integration kernels."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        natoms = 100
        np.random.seed(42)

        return {
            'coordinates': np.random.randn(natoms, 3).astype(np.float64),
            'velocities': np.random.randn(natoms, 3).astype(np.float64),
            'forces': np.random.randn(natoms, 3).astype(np.float64),
            'masses': np.random.uniform(1.0, 20.0, natoms).astype(np.float64),
            'dt': 0.001,
        }

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_step_a_shape(self, simple_system):
        """Test that step A returns correct shapes."""
        coords_new, vels_new = velocity_verlet_step_a(
            simple_system['coordinates'],
            simple_system['velocities'],
            simple_system['forces'],
            simple_system['masses'],
            simple_system['dt']
        )

        assert coords_new.shape == simple_system['coordinates'].shape
        assert vels_new.shape == simple_system['velocities'].shape

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_step_b_shape(self, simple_system):
        """Test that step B returns correct shapes."""
        vels_new, ke, ke_tensor = velocity_verlet_step_b(
            simple_system['velocities'],
            simple_system['forces'],
            simple_system['masses'],
            simple_system['dt']
        )

        assert vels_new.shape == simple_system['velocities'].shape
        assert isinstance(ke, (float, np.floating))
        assert ke_tensor.shape == (3, 3)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_energy_conservation_nve(self, simple_system):
        """Test energy conservation in NVE simulation."""
        # Run short NVE simulation
        coords = simple_system['coordinates'].copy()
        vels = simple_system['velocities'].copy()
        forces = simple_system['forces'].copy()
        masses = simple_system['masses']
        dt = simple_system['dt']

        energies = []

        for _ in range(100):
            # Step A
            coords, vels = velocity_verlet_step_a(coords, vels, forces, masses, dt)

            # Step B (forces would be recomputed here in real MD)
            vels, ke, _ = velocity_verlet_step_b(vels, forces, masses, dt)

            # Total energy (potential is constant since forces don't change)
            pe = 0.0  # Constant in this test
            total_energy = ke + pe
            energies.append(total_energy)

        # Energy should be conserved (within numerical precision)
        energy_drift = (energies[-1] - energies[0]) / energies[0]
        assert abs(energy_drift) < 1e-10, f"Energy drift: {energy_drift}"

    def test_cuda_vs_jax_step_a(self, simple_system):
        """Compare CUDA and JAX implementations of step A."""
        coords = simple_system['coordinates'].copy()
        vels = simple_system['velocities'].copy()
        forces = simple_system['forces']
        masses = simple_system['masses']
        dt = simple_system['dt']

        # JAX implementation
        dt2 = 0.5 * dt
        dt2m = dt2 / masses[:, None]

        coords_jax = coords.copy()
        vels_jax = vels.copy()

        vels_jax = vels_jax + forces * dt2m
        coords_jax = coords_jax + dt2 * vels_jax
        coords_jax = coords_jax + dt2 * vels_jax

        if CUDA_AVAILABLE:
            # CUDA implementation
            coords_cuda, vels_cuda = velocity_verlet_step_a(
                coords, vels, forces, masses, dt
            )

            # Compare results
            assert_allclose(coords_cuda, coords_jax, rtol=1e-14, atol=1e-14)
            assert_allclose(vels_cuda, vels_jax, rtol=1e-14, atol=1e-14)
        else:
            pytest.skip("CUDA not available for comparison")

    def test_cuda_vs_jax_step_b(self, simple_system):
        """Compare CUDA and JAX implementations of step B."""
        vels = simple_system['velocities'].copy()
        forces = simple_system['forces']
        masses = simple_system['masses']
        dt = simple_system['dt']

        # JAX implementation
        dt2 = 0.5 * dt
        dt2m = dt2 / masses[:, None]

        vels_jax = vels.copy()
        vels_jax = vels_jax + forces * dt2m

        # Kinetic energy
        ke_jax = 0.5 * np.sum(masses[:, None] * vels_jax * vels_jax)

        # Kinetic tensor
        ke_tensor_jax = 0.5 * np.sum(
            masses[:, None, None] * vels_jax[:, :, None] * vels_jax[:, None, :],
            axis=0
        )

        if CUDA_AVAILABLE:
            # CUDA implementation
            vels_cuda, ke_cuda, ke_tensor_cuda = velocity_verlet_step_b(
                vels, forces, masses, dt
            )

            # Compare results
            assert_allclose(vels_cuda, vels_jax, rtol=1e-14, atol=1e-14)
            assert_allclose(ke_cuda, ke_jax, rtol=1e-13, atol=1e-13)
            assert_allclose(ke_tensor_cuda, ke_tensor_jax, rtol=1e-13, atol=1e-13)
        else:
            pytest.skip("CUDA not available for comparison")

    def test_hybrid_integrator_creation(self, simple_system):
        """Test creation of hybrid integrator."""
        integrator = create_cuda_integrator(
            simple_system['dt'],
            simple_system['masses'],
            use_cuda=CUDA_AVAILABLE
        )

        assert 'stepA' in integrator
        assert 'stepB' in integrator
        assert 'backend' in integrator
        assert integrator['backend'] in ['cuda', 'jax']

    def test_hybrid_integrator_execution(self, simple_system):
        """Test execution of hybrid integrator."""
        integrator = create_cuda_integrator(
            simple_system['dt'],
            jnp.array(simple_system['masses']),
            use_cuda=CUDA_AVAILABLE
        )

        system = {
            'coordinates': jnp.array(simple_system['coordinates']),
            'vel': jnp.array(simple_system['velocities']),
            'forces': jnp.array(simple_system['forces']),
        }

        # Execute step A
        system = integrator['stepA'](system)
        assert 'coordinates' in system
        assert 'vel' in system

        # Execute step B
        system = integrator['stepB'](system)
        assert 'ek' in system
        assert 'ek_tensor' in system

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_large_system_performance(self):
        """Test performance on larger system."""
        import time

        natoms = 10000
        coords = np.random.randn(natoms, 3).astype(np.float64)
        vels = np.random.randn(natoms, 3).astype(np.float64)
        forces = np.random.randn(natoms, 3).astype(np.float64)
        masses = np.ones(natoms, dtype=np.float64)
        dt = 0.001

        # Warmup
        for _ in range(5):
            velocity_verlet_step_a(coords, vels, forces, masses, dt)

        # Benchmark
        nsteps = 100
        start = time.time()
        for _ in range(nsteps):
            coords, vels = velocity_verlet_step_a(coords, vels, forces, masses, dt)
        elapsed = time.time() - start

        time_per_step = elapsed / nsteps
        print(f"\nCUDA performance ({natoms} atoms):")
        print(f"  Time per step: {time_per_step*1000:.3f} ms")
        print(f"  Steps per second: {nsteps/elapsed:.1f}")

        # Should be reasonably fast
        assert time_per_step < 0.01, f"Too slow: {time_per_step} s/step"


class TestNumericalStability:
    """Test numerical stability of CUDA kernels."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_zero_forces(self):
        """Test with zero forces."""
        natoms = 100
        coords = np.random.randn(natoms, 3).astype(np.float64)
        vels = np.random.randn(natoms, 3).astype(np.float64)
        forces = np.zeros((natoms, 3), dtype=np.float64)
        masses = np.ones(natoms, dtype=np.float64)
        dt = 0.001

        coords_new, vels_new = velocity_verlet_step_a(coords, vels, forces, masses, dt)

        # With zero forces, velocities should be unchanged
        assert_allclose(vels_new, vels, rtol=1e-14)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_very_small_masses(self):
        """Test with very small masses."""
        natoms = 10
        coords = np.random.randn(natoms, 3).astype(np.float64)
        vels = np.zeros((natoms, 3), dtype=np.float64)
        forces = np.random.randn(natoms, 3).astype(np.float64) * 1e-10
        masses = np.ones(natoms, dtype=np.float64) * 1e-10
        dt = 0.001

        # Should not crash or produce NaN
        coords_new, vels_new = velocity_verlet_step_a(coords, vels, forces, masses, dt)

        assert not np.any(np.isnan(coords_new))
        assert not np.any(np.isnan(vels_new))

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_single_atom(self):
        """Test with single atom."""
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        vels = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        forces = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        masses = np.array([1.0], dtype=np.float64)
        dt = 0.001

        coords_new, vels_new = velocity_verlet_step_a(coords, vels, forces, masses, dt)

        # Should move in x direction
        assert coords_new[0, 0] > coords[0, 0]
        assert_allclose(coords_new[0, 1:], coords[0, 1:], atol=1e-15)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
