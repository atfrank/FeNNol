#!/usr/bin/env python3
"""
Test forces on simple 2-atom system
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("2-ATOM FORCE TEST")
print("="*80)
print()

# Simple 2-atom system (O and H)
coords = np.array([
    [0.0, 0.0, 0.0],      # O at origin
    [0.757, 0.586, 0.0],  # H
], dtype=np.float64)

charges = np.array([-0.834, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Compute forces analytically
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy_analytical, forces_analytical = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

print(f"Energy: {float(energy_analytical):.10f} kcal/mol")
print(f"Born radii: {born_radii}")
print(f"Psi sum: {psi_sum}")
print()

print("Analytical forces:")
print(f"  F_O = {forces_analytical[0]}")
print(f"  F_H = {forces_analytical[1]}")
print()

# Compute numerical gradient
delta = 0.0001

def compute_energy(coords_temp):
    born_radii_temp, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_temp, radii, b_params, c_params, cutoff)
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii_temp, dielectric, cutoff)
    return float(E_temp)

forces_numerical = np.zeros_like(coords)

for atom in range(2):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta

        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        E_plus = compute_energy(coords_plus)
        E_minus = compute_energy(coords_minus)

        dE_dx = (E_plus - E_minus) / (2*delta)
        forces_numerical[atom, dim] = -dE_dx

print("Numerical forces:")
print(f"  F_O = {forces_numerical[0]}")
print(f"  F_H = {forces_numerical[1]}")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print()

error_O = forces_analytical[0] - forces_numerical[0]
error_H = forces_analytical[1] - forces_numerical[1]

print(f"Error on O: {error_O}")
print(f"Error on H: {error_H}")
print()

rel_error_O = np.linalg.norm(error_O) / np.linalg.norm(forces_numerical[0]) * 100
rel_error_H = np.linalg.norm(error_H) / np.linalg.norm(forces_numerical[1]) * 100

print(f"Relative error on O: {rel_error_O:.2f}%")
print(f"Relative error on H: {rel_error_H:.2f}%")
print()

if rel_error_O < 1.0 and rel_error_H < 1.0:
    print("✓ Forces are CORRECT!")
else:
    print("✗ Forces have errors!")
