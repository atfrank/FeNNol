#!/usr/bin/env python
"""Debug NaN forces - test without non-polar term"""

import numpy as np
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import OBC

coords = jnp.array([
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0]
])

charges = jnp.array([-0.834, 0.417, 0.417])
atomic_numbers = jnp.array([8, 1, 1])

print("Testing WITHOUT non-polar term...")
model_no_np = OBC({
    "dielectric": 80.0,
    "cutoff": 12.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "include_nonpolar": False  # Disable non-polar
})
model_no_np.has_cuda = False

energy, forces = model_no_np(coords, charges, atomic_numbers)
print(f"Energy: {energy}")
print(f"Forces:\n{forces}")
print(f"Any NaN: {jnp.any(jnp.isnan(forces))}")
print()

print("Testing WITH non-polar term...")
model_with_np = OBC({
    "dielectric": 80.0,
    "cutoff": 12.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "include_nonpolar": True  # Enable non-polar
})
model_with_np.has_cuda = False

energy2, forces2 = model_with_np(coords, charges, atomic_numbers)
print(f"Energy: {energy2}")
print(f"Forces:\n{forces2}")
print(f"Any NaN: {jnp.any(jnp.isnan(forces2))}")
