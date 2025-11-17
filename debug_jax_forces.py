#!/usr/bin/env python
"""Debug NaN forces in JAX implementation"""

import numpy as np
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import OBC

# Simple water molecule
coords = jnp.array([
    [0.0, 0.0, 0.0],      # O
    [0.96, 0.0, 0.0],     # H1
    [-0.24, 0.93, 0.0]    # H2
])

charges = jnp.array([-0.834, 0.417, 0.417])  # TIP3P
atomic_numbers = jnp.array([8, 1, 1])

# Force JAX backend
model = OBC({
    "dielectric": 80.0,
    "cutoff": 12.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "include_nonpolar": True
})
model.has_cuda = False

print("Testing JAX implementation on single water molecule...")
print(f"Coordinates: {coords}")
print(f"Charges: {charges}")
print()

# Get atomic parameters
radii = model.atomic_params.get_radii_array(np.array(atomic_numbers))
b_params, c_params = model.atomic_params.get_obc_params_arrays(np.array(atomic_numbers))

print(f"Radii: {radii}")
print(f"B params: {b_params}")
print(f"C params: {c_params}")
print()

# Test Born radii calculation
print("Computing Born radii...")
born_radii = model._compute_born_radii_jax(coords, jnp.array(radii), jnp.array(b_params), jnp.array(c_params), None)
print(f"Born radii: {born_radii}")
print(f"Any NaN in Born radii: {jnp.any(jnp.isnan(born_radii))}")
print(f"Any Inf in Born radii: {jnp.any(jnp.isinf(born_radii))}")
print()

# Test GB energy only
print("Computing GB energy...")
gb_energy = model._gb_energy_only(coords, charges, born_radii, None)
print(f"GB energy: {gb_energy}")
print(f"Is NaN: {jnp.isnan(gb_energy)}")
print(f"Is Inf: {jnp.isinf(gb_energy)}")
print()

# Test full energy and forces
print("Computing energy and forces...")
try:
    energy, forces = model(coords, charges, atomic_numbers)
    print(f"Energy: {energy}")
    print(f"Forces:\n{forces}")
    print(f"Any NaN in forces: {jnp.any(jnp.isnan(forces))}")
    print(f"Any Inf in forces: {jnp.any(jnp.isinf(forces))}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
