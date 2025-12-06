"""
CUDA-accelerated integration module for FeNNol.

This module provides CUDA-accelerated versions of MD integration functions
with automatic fallback to JAX when CUDA is unavailable.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable, Dict, Any

# Try to import CUDA kernels
try:
    from ..cuda import (
        CUDA_AVAILABLE,
        velocity_verlet_step_a as cuda_step_a,
        velocity_verlet_step_b as cuda_step_b,
        CudaIntegrator
    )
except ImportError:
    CUDA_AVAILABLE = False


def create_cuda_integrator(dt: float, masses: jnp.ndarray, use_cuda: bool = True) -> Dict[str, Any]:
    """
    Create integration functions using CUDA if available.

    Args:
        dt: Timestep
        masses: (natoms,) atomic masses
        use_cuda: Whether to use CUDA if available

    Returns:
        Dictionary containing stepA, stepB, and update_forces functions
    """
    dt2 = 0.5 * dt

    if CUDA_AVAILABLE and use_cuda:
        print("# Using CUDA-accelerated integration")

        # Convert masses to numpy for CUDA
        masses_np = np.array(masses, dtype=np.float64)

        def stepA_cuda(system):
            """CUDA-accelerated step A."""
            # Convert JAX arrays to numpy
            coords = np.array(system["coordinates"], dtype=np.float64)
            vels = np.array(system["vel"], dtype=np.float64)
            forces = np.array(system["forces"], dtype=np.float64)

            # Execute CUDA kernel
            coords_new, vels_new = cuda_step_a(coords, vels, forces, masses_np, dt)

            # Convert back to JAX arrays
            return {
                **system,
                "coordinates": jnp.array(coords_new),
                "vel": jnp.array(vels_new),
            }

        def stepB_cuda(system):
            """CUDA-accelerated step B."""
            # Convert JAX arrays to numpy
            vels = np.array(system["vel"], dtype=np.float64)
            forces = np.array(system["forces"], dtype=np.float64)

            # Execute CUDA kernel
            vels_new, ek, ek_tensor = cuda_step_b(vels, forces, masses_np, dt)

            # Convert back to JAX arrays
            return {
                **system,
                "vel": jnp.array(vels_new),
                "ek": float(ek),
                "ek_tensor": jnp.array(ek_tensor),
            }

        return {
            "stepA": stepA_cuda,
            "stepB": stepB_cuda,
            "backend": "cuda",
        }

    else:
        if use_cuda:
            print("# CUDA not available, using JAX integration")
        else:
            print("# Using JAX integration (CUDA disabled)")

        # Fallback to JAX implementation
        dt2m = jnp.asarray(dt2 / masses[:, None])

        @jax.jit
        def stepA_jax(system):
            """JAX-based step A."""
            v = system["vel"]
            f = system["forces"]
            x = system["coordinates"]

            v = v + f * dt2m
            x = x + dt2 * v
            # Note: thermostat would be applied here in full implementation
            x = x + dt2 * v

            return {**system, "coordinates": x, "vel": v}

        @jax.jit
        def stepB_jax(system):
            """JAX-based step B."""
            v = system["vel"]
            f = system["forces"]

            v = v + f * dt2m

            # Compute kinetic energy
            ek_tensor = (
                0.5 * jnp.sum(
                    masses[:, None, None] * v[:, :, None] * v[:, None, :],
                    axis=0
                )
            )
            ek = jnp.trace(ek_tensor)

            return {
                **system,
                "vel": v,
                "ek": ek,
                "ek_tensor": ek_tensor,
            }

        return {
            "stepA": stepA_jax,
            "stepB": stepB_jax,
            "backend": "jax",
        }


def get_integration_backend() -> str:
    """
    Get the current integration backend.

    Returns:
        "cuda" if CUDA is available, "jax" otherwise
    """
    return "cuda" if CUDA_AVAILABLE else "jax"


def benchmark_integration(
    natoms: int = 1000,
    nsteps: int = 100,
    use_cuda: bool = True
) -> Dict[str, float]:
    """
    Benchmark integration performance.

    Args:
        natoms: Number of atoms
        nsteps: Number of steps to benchmark
        use_cuda: Whether to use CUDA

    Returns:
        Dictionary with timing results
    """
    import time

    # Create test system
    masses = jnp.ones(natoms)
    dt = 0.001

    integrator = create_cuda_integrator(dt, masses, use_cuda)

    # Initialize test data
    system = {
        "coordinates": jnp.zeros((natoms, 3)),
        "vel": jnp.ones((natoms, 3)),
        "forces": jnp.ones((natoms, 3)),
    }

    # Warmup
    for _ in range(10):
        system = integrator["stepA"](system)
        system = integrator["stepB"](system)

    # Benchmark
    start = time.time()
    for _ in range(nsteps):
        system = integrator["stepA"](system)
        system = integrator["stepB"](system)

    # Ensure completion
    if isinstance(system["coordinates"], jnp.ndarray):
        system["coordinates"].block_until_ready()

    elapsed = time.time() - start

    return {
        "backend": integrator["backend"],
        "total_time": elapsed,
        "time_per_step": elapsed / nsteps,
        "steps_per_second": nsteps / elapsed,
    }
