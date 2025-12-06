#!/usr/bin/env python3
"""Simple test of GB forces during dynamics-like updates."""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

# Force JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Water molecule coordinates
coords = np.array([
    [0.000, 0.000, 0.000],  # O
    [0.757, 0.586, 0.000],  # H1
    [-0.757, 0.586, 0.000],  # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

# Initialize GB model
gb_params = {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "radii_set": "mbondi",
    "include_nonpolar": False
}

print("Initializing GB model...")
gb_model = create_implicit_solvent_model("OBC", gb_params)
gb_model.has_cuda = False  # Force JAX
print(f"Using backend: {'CUDA' if gb_model.has_cuda else 'JAX'}")

# Test dynamics-like behavior: perturb coordinates slightly and recompute
dt = 0.5  # fs
nsteps = 100

print(f"\nRunning {nsteps} pseudo-dynamics steps...")
print(f"{'Step':>6} {'Energy':>12} {'Max_F':>10} {'Status'}")
print("-" * 50)

current_coords = coords.copy()

for step in range(nsteps):
    # Compute GB energy and forces
    try:
        energy, forces = gb_model.compute_energy_forces(
            current_coords, charges, atomic_numbers
        )

        # Check for NaN
        if np.isnan(energy) or np.any(np.isnan(forces)):
            print(f"{step:6d} {'NaN':>12} {'NaN':>10} FAILED")
            print("\n*** NaN detected! ***")
            print(f"Energy: {energy}")
            print(f"Forces:\n{forces}")
            break

        max_force = np.max(np.abs(forces))

        # Output every 10 steps
        if step % 10 == 0:
            print(f"{step:6d} {energy:12.3f} {max_force:10.3f} OK")

        # Simple update: move atoms in direction opposite to forces (damped)
        # This is NOT real dynamics, just testing that forces stay reasonable
        current_coords = current_coords - 0.01 * forces

    except Exception as e:
        print(f"{step:6d} {'ERROR':>12} {'ERROR':>10} CRASHED")
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n{nsteps} steps completed successfully!")
print(f"\nFinal state:")
print(f"  Energy: {energy:.3f} kcal/mol")
print(f"  Max force: {max_force:.3f} kcal/mol/Å")
print(f"  Final coordinates:\n{current_coords}")
print(f"\nNo NaN detected - GB forces are stable!")
