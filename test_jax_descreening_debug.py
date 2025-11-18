#!/usr/bin/env python3
"""Debug JAX descreening integral calculation."""

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
    [0.757, 0.586, 0.000],  # H1
    [-0.757, 0.586, 0.000],  # H2
], dtype=np.float64)

atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

gb_params = {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "radii_set": "mbondi",
}

gb_model = create_implicit_solvent_model("OBC", gb_params)
gb_model.has_cuda = False

# Get parameters
radii = gb_model.atomic_params.get_radii_array(atomic_numbers)
print(f"Radii: {radii}")

# Compute distance matrix
coords_jax = jnp.array(coords)
dr = coords_jax[:, None, :] - coords_jax[None, :, :]
r = jnp.linalg.norm(dr, axis=-1)
print(f"\nDistance matrix:\n{r}")

# Call the descreening integral function
radii_i = radii
radii_j = radii[:, None]

print(f"\nInput shapes to descreening_integral:")
print(f"  r shape: {r.shape}")
print(f"  radii_i shape: {radii_i.shape}")
print(f"  radii_j shape: {radii_j.shape}")

psi = gb_model._compute_descreening_integral(r, radii_i, radii_j)

print(f"\nDescreening integral matrix (psi):\n{psi}")

# Sum contributions (exclude self)
psi_sum = jnp.sum(psi, axis=1) - jnp.diag(psi)
print(f"\nψ_sum (excluding self):\n{psi_sum}")

# Get OBC parameters
b_params, c_params = gb_model.atomic_params.get_obc_params_arrays(atomic_numbers)
print(f"\nOBC b parameters: {b_params}")
print(f"\nOBC c parameters: {c_params}")

# Compute Born radii
tanh_arg = psi_sum - b_params * psi_sum**2 + c_params * psi_sum**3
tanh_val = jnp.tanh(tanh_arg)
born_radii_inv = 1.0 / radii - tanh_val / radii
born_radii = 1.0 / born_radii_inv
born_radii = jnp.maximum(born_radii, radii)

print(f"\nBorn radii calculation:")
print(f"  tanh_arg: {tanh_arg}")
print(f"  tanh_val: {tanh_val}")
print(f"  1/R: {born_radii_inv}")
print(f"  R (before clamping): {1.0/born_radii_inv}")
print(f"  R (after clamping): {born_radii}")
