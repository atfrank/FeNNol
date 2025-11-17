#!/usr/bin/env python
"""
Quick test of implicit solvent CUDA implementation
"""

import jax.numpy as jnp
import numpy as np

# Test 1: Import and create model
print("=" * 60)
print("Test 1: Import and create OBC model")
print("=" * 60)

from fennol.models.physics.implicit_solvent import OBC

model = OBC({
    "dielectric": 80.0,
    "cutoff": 12.0,
    "surface_tension": 0.005,
    "include_nonpolar": True
})

print(model)
print(f"CUDA available: {model.has_cuda}")
print()

# Test 2: Small water molecule test
print("=" * 60)
print("Test 2: Small water molecule")
print("=" * 60)

# Water molecule: O-H-H
coords = jnp.array([
    [0.0, 0.0, 0.0],      # O
    [0.96, 0.0, 0.0],     # H1
    [-0.24, 0.93, 0.0]    # H2
])

charges = jnp.array([-0.834, 0.417, 0.417])  # TIP3P charges
atomic_numbers = jnp.array([8, 1, 1])  # O, H, H

print(f"Coordinates:\n{coords}")
print(f"Charges: {charges}")
print(f"Atomic numbers: {atomic_numbers}")
print()

energy, forces = model(coords, charges, atomic_numbers)

print(f"Solvation energy: {energy:.6f} kcal/mol")
print(f"Forces:\n{forces}")
print(f"Force magnitude: {np.linalg.norm(forces):.6f} kcal/mol/Å")
print()

# Test 3: Multiple molecules
print("=" * 60)
print("Test 3: Two water molecules (6 atoms)")
print("=" * 60)

coords2 = jnp.array([
    [0.0, 0.0, 0.0],      # O
    [0.96, 0.0, 0.0],     # H1
    [-0.24, 0.93, 0.0],   # H2
    [3.0, 0.0, 0.0],      # O
    [3.96, 0.0, 0.0],     # H1
    [2.76, 0.93, 0.0]     # H2
])

charges2 = jnp.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417])
atomic_numbers2 = jnp.array([8, 1, 1, 8, 1, 1])

energy2, forces2 = model(coords2, charges2, atomic_numbers2)

print(f"Solvation energy: {energy2:.6f} kcal/mol")
print(f"Energy per molecule: {energy2/2:.6f} kcal/mol")
print(f"Forces magnitude: {np.linalg.norm(forces2):.6f} kcal/mol/Å")
print()

print("=" * 60)
print("All tests passed! ✓")
print("=" * 60)
