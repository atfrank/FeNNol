#!/usr/bin/env python3
"""
Test TOTAL forces = direct + Born radii derivative
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("TOTAL GB FORCE TEST")
print("="*80)
print()

# 2-atom system
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H
], dtype=np.float64)

charges = np.array([-0.834, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Compute forces using multi-pass approach
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born_deriv = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

print(f"Energy: {float(energy):.10f}")
print()

print("Direct GB forces (∂E/∂r with R fixed):")
print(f"  F_O = {forces_direct[0]}")
print(f"  F_H = {forces_direct[1]}")
print()

print("Born radii derivative forces (∂E/∂R × ∂R/∂r):")
print(f"  F_O = {forces_born_deriv[0]}")
print(f"  F_H = {forces_born_deriv[1]}")
print()

# Total forces
forces_total = forces_direct + forces_born_deriv

print("TOTAL forces (direct + Born derivative):")
print(f"  F_O = {forces_total[0]}")
print(f"  F_H = {forces_total[1]}")
print()

# Numerical gradient
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

print("Numerical gradient:")
print(f"  F_O = {forces_numerical[0]}")
print(f"  F_H = {forces_numerical[1]}")
print()

print("="*80)
print("COMPARISON")
print("="*80)

error_O = forces_total[0] - forces_numerical[0]
error_H = forces_total[1] - forces_numerical[1]

print(f"Error on O: {error_O}")
print(f"Error on H: {error_H}")
print()

rel_error_O = np.linalg.norm(error_O) / np.linalg.norm(forces_numerical[0]) * 100
rel_error_H = np.linalg.norm(error_H) / np.linalg.norm(forces_numerical[1]) * 100

print(f"Relative error on O: {rel_error_O:.2f}%")
print(f"Relative error on H: {rel_error_H:.2f}%")
print()

if rel_error_O < 1.0 and rel_error_H < 1.0:
    print("✓ TOTAL forces are CORRECT!")
else:
    print("✗ TOTAL forces still have errors")
