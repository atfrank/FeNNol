#!/usr/bin/env python3
"""Detailed comparison of CUDA vs JAX intermediate values."""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

# Force JAX to use CPU and enable 64-bit
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Water molecule
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
    "include_nonpolar": False
}

print("Testing CUDA Born radii calculation:")
print("=" * 80)

gb_model = create_implicit_solvent_model("OBC", gb_params)

if gb_model.has_cuda:
    # Test CUDA
    try:
        from fennol import cuda

        # Get parameters
        radii = gb_model.atomic_params.get_radii_array(atomic_numbers)
        b_params, c_params = gb_model.atomic_params.get_obc_params_arrays(atomic_numbers)

        print(f"Intrinsic radii: {radii}")
        print(f"OBC b params: {b_params}")
        print(f"OBC c params: {c_params}")

        # Call CUDA directly (needs 2D coords)
        born_radii_cuda = cuda.gb_compute_born_radii(
            coords,  # Keep as [natoms, 3]
            radii,
            b_params,
            c_params,
            8.0  # cutoff
        )
        print(f"\nCUDA Born radii: {born_radii_cuda}")

        # Compute energy with CUDA
        energy_cuda, forces_cuda = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
        print(f"CUDA Energy: {energy_cuda:.6f} kcal/mol")
        print(f"CUDA Forces:\n{forces_cuda}")

    except Exception as e:
        print(f"CUDA test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Testing JAX Born radii calculation:")
print("=" * 80)

# Force JAX backend
gb_model.has_cuda = False

# Get parameters
radii = gb_model.atomic_params.get_radii_array(atomic_numbers)
b_params, c_params = gb_model.atomic_params.get_obc_params_arrays(atomic_numbers)

print(f"Intrinsic radii: {radii}")
print(f"OBC b params: {b_params}")
print(f"OBC c params: {c_params}")

# Call internal method
coords_jax = jnp.array(coords)
radii_jax = jnp.array(radii)
b_params_jax = jnp.array(b_params)
c_params_jax = jnp.array(c_params)

born_radii_jax = gb_model._compute_born_radii_jax(coords_jax, radii_jax, b_params_jax, c_params_jax)
print(f"\nJAX Born radii: {born_radii_jax}")

# Compute energy with JAX
energy_jax, forces_jax = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
print(f"JAX Energy: {energy_jax:.6f} kcal/mol")
print(f"JAX Forces:\n{forces_jax}")

if gb_model.has_cuda:
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print(f"Born radii difference: {np.array(born_radii_jax) - np.array(born_radii_cuda)}")
    print(f"Energy difference: {energy_jax - energy_cuda:.6f} kcal/mol ({100*(energy_jax - energy_cuda)/energy_cuda:.2f}%)")
    print(f"Forces RMS difference: {np.sqrt(np.mean((forces_jax - forces_cuda)**2)):.6f} kcal/mol/Å")
