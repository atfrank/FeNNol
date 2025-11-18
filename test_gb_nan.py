#!/usr/bin/env python3
"""Debug NaN in GB forces."""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

# Force JAX to use CPU and enable 64-bit
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Water molecule coordinates from initialization
coords = np.array([
    [0.000, 0.000, 0.000],  # O
    [0.757, 0.586, 0.000],  # H
    [-0.757, 0.586, 0.000],  # H
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

gb_params = {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "radii_set": "mbondi",
    "include_nonpolar": False  # Disable non-polar for debugging
}

print("Creating GB model...")
gb_model = create_implicit_solvent_model("OBC", gb_params)

# Force JAX backend by disabling CUDA
gb_model.has_cuda = False
print(f"Using backend: {'CUDA' if gb_model.has_cuda else 'JAX'} (forced JAX for testing)")

print("\nTesting GB with initial coordinates:")
energy, forces = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
print(f"Energy: {energy}")
print(f"Forces:\n{forces}")
print(f"Forces contain NaN: {np.any(np.isnan(forces))}")

# Try with JAX arrays explicitly
print("\n\nTrying with explicit JAX arrays:")
coords_jax = jnp.array(coords, dtype=jnp.float64)
charges_jax = jnp.array(charges, dtype=jnp.float64)
atomic_numbers_jax = jnp.array(atomic_numbers, dtype=jnp.int32)

energy2, forces2 = gb_model.compute_energy_forces(coords_jax, charges_jax, atomic_numbers_jax)
print(f"Energy: {energy2}")
print(f"Forces:\n{forces2}")
print(f"Forces contain NaN: {np.any(np.isnan(forces2))}")
