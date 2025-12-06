#!/usr/bin/env python3
"""
Test COMPLETE GB forces (direct + Born radii derivatives) on water molecule
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("COMPLETE GB FORCE TEST - Water Molecule")
print("="*80)
print()

# Single water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H1
    [-0.757, 0.586, 0.0], # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Compute complete forces
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born_deriv = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

forces_total = forces_direct + forces_born_deriv

print(f"Energy: {float(energy):.10f}")
print()

print("Direct GB forces:")
for i, f in enumerate(forces_direct):
    print(f"  F[{i}] = {f}")
print()

print("Born radii derivative forces:")
for i, f in enumerate(forces_born_deriv):
    print(f"  F[{i}] = {f}")
print()

print("TOTAL forces:")
for i, f in enumerate(forces_total):
    print(f"  F[{i}] = {f}")
print()

# Check Newton's 3rd law
total_force = np.sum(forces_total, axis=0)
print(f"Sum of all forces (should be ~0): {total_force}")
print(f"Newton's 3rd law satisfied: {np.linalg.norm(total_force) < 1e-6}")
print()

# Numerical gradient
delta = 0.0001

def compute_energy(coords_temp):
    born_radii_temp, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_temp, radii, b_params, c_params, cutoff)
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii_temp, dielectric, cutoff)
    return float(E_temp)

forces_numerical = np.zeros_like(coords)

for atom in range(3):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta

        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        E_plus = compute_energy(coords_plus)
        E_minus = compute_energy(coords_minus)

        dE_dx = (E_plus - E_minus) / (2*delta)
        forces_numerical[atom, dim] = -dE_dx

print("Numerical gradient:")
for i, f in enumerate(forces_numerical):
    print(f"  F[{i}] = {f}")
print()

print("="*80)
print("COMPARISON")
print("="*80)

errors = forces_total - forces_numerical
print()
print("Errors:")
for i, (err, f_num) in enumerate(zip(errors, forces_numerical)):
    rel_err = np.linalg.norm(err) / max(np.linalg.norm(f_num), 1e-10) * 100
    print(f"  Atom {i}: {err} (relative: {rel_err:.2f}%)")
print()

max_rel_error = max([
    np.linalg.norm(err) / max(np.linalg.norm(f_num), 1e-10) * 100
    for err, f_num in zip(errors, forces_numerical)
])

print(f"Max relative error: {max_rel_error:.2f}%")
print()

if max_rel_error < 1.0:
    print("✓ Forces are CORRECT!")
else:
    print("✗ Forces have errors")
