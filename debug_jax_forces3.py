#!/usr/bin/env python
"""Debug NaN - test autodiff with fixed Born radii"""

import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import OBC

coords = jnp.array([
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0]
])

charges = jnp.array([-0.834, 0.417, 0.417])
atomic_numbers = jnp.array([8, 1, 1])

model = OBC({
    "dielectric": 80.0,
    "cutoff": 12.0,
    "include_nonpolar": False
})
model.has_cuda = False

# Pre-compute Born radii (no autodiff)
import numpy as np
radii = model.atomic_params.get_radii_array(np.array(atomic_numbers))
b_params, c_params = model.atomic_params.get_obc_params_arrays(np.array(atomic_numbers))
born_radii = model._compute_born_radii_jax(coords, jnp.array(radii), jnp.array(b_params), jnp.array(c_params), None)

print("Born radii:", born_radii)
print()

# Test 1: Differentiate GB energy with FIXED Born radii
print("Test 1: Autodiff GB energy with FIXED Born radii...")
energy_fn = lambda x: model._gb_energy_only(x, charges, born_radii, None)
energy = energy_fn(coords)
forces = -jax.grad(energy_fn)(coords)

print(f"Energy: {energy}")
print(f"Forces:\n{forces}")
print(f"Any NaN: {jnp.any(jnp.isnan(forces))}")
print()

# Test 2: Full calculation (autodiff through Born radii)
print("Test 2: Autodiff through entire calculation...")
try:
    energy2, forces2 = model._compute_gb_electrostatic_jax(coords, charges, born_radii, None)
    print(f"Energy: {energy2}")
    print(f"Forces:\n{forces2}")
    print(f"Any NaN: {jnp.any(jnp.isnan(forces2))}")
except Exception as e:
    print(f"ERROR: {e}")
