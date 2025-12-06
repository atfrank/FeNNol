#!/usr/bin/env python3
"""
Diagnostic test to understand energy vs force calculation
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("ENERGY AND FORCE DIAGNOSTIC")
print("="*80)
print()

# 2-atom system (O-H)
coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0]], dtype=np.float64)
charges = np.array([-0.834, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Compute Born radii
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
print(f"Born radii: {born_radii}")
print()

# Compute analytical energy and forces
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
print(f"Analytical energy: {float(energy):.10f}")
print(f"Analytical direct forces: F_O = {forces_direct[0]}, F_H = {forces_direct[1]}")
print()

# Numerical energy gradient (should match direct forces)
delta = 0.0001

def compute_energy_fixed_R(coords_temp):
    """Compute energy with FIXED Born radii"""
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii, dielectric, cutoff)
    return float(E_temp)

forces_numerical = np.zeros_like(coords)
for atom in range(2):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta
        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        E_plus = compute_energy_fixed_R(coords_plus)
        E_minus = compute_energy_fixed_R(coords_minus)

        forces_numerical[atom, dim] = -(E_plus - E_minus) / (2*delta)

print(f"Numerical gradient (Born radii FIXED): F_O = {forces_numerical[0]}, F_H = {forces_numerical[1]}")
print()

# Check if forces sum to zero (conservation of momentum)
print("FORCE BALANCE CHECK:")
print(f"Sum of analytical direct forces: {forces_direct[0] + forces_direct[1]}")
print(f"Sum of numerical forces: {forces_numerical[0] + forces_numerical[1]}")
print()

# Compute error
error = forces_direct - forces_numerical
rel_error = np.linalg.norm(error) / np.linalg.norm(forces_numerical) * 100
print(f"Relative error: {rel_error:.2f}%")
print(f"Error vector: {error[0]}")
print()

# The analytical force should be EXACTLY 2x what we're getting if the 0.5 factor is wrong
ratio = forces_numerical[0] / forces_direct[0]
print(f"Ratio (numerical/analytical): {ratio}")
