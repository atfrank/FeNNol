#!/usr/bin/env python3
"""Test GB JAX implementation on water molecule."""

import numpy as np
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

# Water molecule coordinates (Angstroms)
coords = np.array([
    [0.000, 0.000, 0.000],  # O
    [0.757, 0.586, 0.000],  # H
    [-0.757, 0.586, 0.000],  # H
], dtype=np.float64)

# Partial charges (electron units)
charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)

# Atomic numbers
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

# GB parameters
gb_params = {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "radii_set": "mbondi",
    "include_nonpolar": True
}

print("Creating GB model...")
gb_model = create_implicit_solvent_model("OBC", gb_params)
print(f"GB model created: {gb_model}")
print(f"Using backend: {'CUDA' if gb_model.has_cuda else 'JAX'}")

print("\nComputing GB energy and forces...")
energy, forces = gb_model.compute_energy_forces(coords, charges, atomic_numbers)

print(f"\nGB Energy: {energy:.6f} kcal/mol")
print(f"GB Forces:\n{forces}")
print(f"\nForce magnitudes: {np.linalg.norm(forces, axis=1)}")
print(f"Max force component: {np.max(np.abs(forces)):.6f} kcal/mol/Å")
print(f"Force contains NaN: {np.any(np.isnan(forces))}")
print(f"Force contains Inf: {np.any(np.isinf(forces))}")
