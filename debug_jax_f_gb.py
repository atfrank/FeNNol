#!/usr/bin/env python
"""Debug f_GB calculation"""

import jax.numpy as jnp

coords = jnp.array([
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0]
])

born_radii = jnp.array([1.5, 2.89, 2.89])

# Compute pairwise distances
dr = coords[:, None, :] - coords[None, :, :]
r = jnp.linalg.norm(dr, axis=-1)

print("Pairwise distances:")
print(r)
print()

# Compute f_GB
R_i = born_radii[:, None]
R_j = born_radii[None, :]
R_product = R_i * R_j

r_safe = jnp.maximum(r, 0.001)

cutoff = 12.0
cutoff_mask = r_safe < cutoff

exp_term = jnp.where(cutoff_mask, jnp.exp(-r_safe**2 / (4.0 * R_product)), 0.0)

f_GB = jnp.sqrt(r_safe**2 + R_product * exp_term)

print("f_GB:")
print(f_GB)
print()

print("Diagonal f_GB (self-interactions):")
print(jnp.diag(f_GB))
print()

# Check what happens when we divide
charges = jnp.array([-0.834, 0.417, 0.417])
q_i = charges[:, None]
q_j = charges[None, :]
q_product = q_i * q_j

print("q_product / f_GB (includes self-interactions):")
print(q_product / f_GB)
print()

# With mask
mask = jnp.eye(len(charges))
interaction_term = jnp.where(mask, 0.0, q_product / f_GB)

print("With mask (self-interactions set to 0):")
print(interaction_term)
print()

# Check if there are any inf/nan
print(f"Any NaN in f_GB: {jnp.any(jnp.isnan(f_GB))}")
print(f"Any Inf in f_GB: {jnp.any(jnp.isinf(f_GB))}")
print(f"Any NaN in interaction_term: {jnp.any(jnp.isnan(interaction_term))}")
print(f"Any Inf in interaction_term: {jnp.any(jnp.isinf(interaction_term))}")
