#!/usr/bin/env python3
"""Compare CUDA and JAX GB implementations in detail."""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

# Force JAX to use CPU and enable 64-bit
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Water molecule coordinates
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

print("=" * 80)
print("COMPARING CUDA vs JAX GB IMPLEMENTATIONS")
print("=" * 80)

gb_model = create_implicit_solvent_model("OBC", gb_params)

# Test with CUDA if available
if gb_model.has_cuda:
    print("\n### CUDA IMPLEMENTATION ###")
    gb_model_cuda = create_implicit_solvent_model("OBC", gb_params)
    energy_cuda, forces_cuda = gb_model_cuda.compute_energy_forces(coords, charges, atomic_numbers)
    print(f"CUDA Energy: {energy_cuda:.6f} kcal/mol")
    print(f"CUDA Forces:\n{forces_cuda}")
    print(f"CUDA Forces magnitude: {np.linalg.norm(forces_cuda, axis=1)}")
else:
    print("\n### CUDA NOT AVAILABLE ###")
    energy_cuda = None
    forces_cuda = None

# Test with JAX
print("\n### JAX IMPLEMENTATION ###")
gb_model.has_cuda = False  # Force JAX
energy_jax, forces_jax = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
print(f"JAX Energy: {energy_jax:.6f} kcal/mol")
print(f"JAX Forces:\n{forces_jax}")
print(f"JAX Forces magnitude: {np.linalg.norm(forces_jax, axis=1)}")

# Compare if both available
if energy_cuda is not None:
    print("\n### COMPARISON ###")
    print(f"Energy difference: {energy_jax - energy_cuda:.6f} kcal/mol")
    print(f"Energy ratio: {energy_jax / energy_cuda:.6f}")
    print(f"Forces difference (RMS): {np.sqrt(np.mean((forces_jax - forces_cuda)**2)):.6f} kcal/mol/Å")

# Now let's debug the JAX internals
print("\n" + "=" * 80)
print("DEBUGGING JAX INTERNALS")
print("=" * 80)

# Get atomic parameters
radii = gb_model.atomic_params.get_radii_array(atomic_numbers)
b_params, c_params = gb_model.atomic_params.get_obc_params_arrays(atomic_numbers)

print(f"\nAtomic radii: {radii}")
print(f"OBC b parameters: {b_params}")
print(f"OBC c parameters: {c_params}")

# Compute Born radii manually
coords_jax = jnp.array(coords)
radii_jax = jnp.array(radii)
b_params_jax = jnp.array(b_params)
c_params_jax = jnp.array(c_params)

# Call internal method to get Born radii
born_radii = gb_model._compute_born_radii_jax(coords_jax, radii_jax, b_params_jax, c_params_jax)
print(f"\nBorn radii: {born_radii}")

# Compute distance matrix
dr = coords_jax[:, None, :] - coords_jax[None, :, :]
r_sq = jnp.sum(dr**2, axis=-1)
r = jnp.sqrt(r_sq + 1e-10)
print(f"\nDistance matrix:\n{r}")

# Compute f_GB
R_product = born_radii[:, None] * born_radii[None, :]
exp_arg = -r_sq / (4.0 * R_product + 1e-12)
exp_val = jnp.exp(exp_arg)
f_gb = jnp.sqrt(r_sq + R_product * exp_val + 1e-10)
print(f"\nf_GB matrix:\n{f_gb}")

# Compute energy components
qi_qj = charges[:, None] * charges[None, :]
print(f"\nCharge products qi*qj:\n{qi_qj}")

# Self energy
self_energy = jnp.sum(charges**2 / (born_radii + 1e-12))
print(f"\nSelf energy term: {self_energy:.6f}")
print(f"  q_O^2/R_O = {charges[0]**2 / born_radii[0]:.6f}")
print(f"  q_H^2/R_H = {charges[1]**2 / born_radii[1]:.6f}")
print(f"  q_H^2/R_H = {charges[2]**2 / born_radii[2]:.6f}")

# Pair energy
natoms = 3
mask_upper = jnp.triu(jnp.ones((natoms, natoms)), k=1)
pair_energy_terms = qi_qj * mask_upper / f_gb
print(f"\nPair energy terms (i<j):\n{pair_energy_terms}")
pair_energy = jnp.sum(pair_energy_terms)
print(f"Total pair energy: {pair_energy:.6f}")

# GB factor
eps_in = 1.0
eps_out = gb_model.dielectric
gb_factor = 332.0636 * (1.0/eps_in - 1.0/eps_out)
print(f"\nGB factor: {gb_factor:.6f}")
print(f"  (eps_in={eps_in}, eps_out={eps_out})")

# Total energy
total_gb_energy = gb_factor * (self_energy + pair_energy)
print(f"\nTotal GB energy: {total_gb_energy:.6f} kcal/mol")
print(f"  = {gb_factor:.4f} * ({self_energy:.6f} + {pair_energy:.6f})")

# Compare to what we got from compute_energy_forces
print(f"\nDirect calculation: {total_gb_energy:.6f}")
print(f"From compute_energy_forces: {energy_jax:.6f}")
print(f"Match: {np.isclose(total_gb_energy, energy_jax)}")
