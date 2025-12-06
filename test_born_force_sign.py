#!/usr/bin/env python3
"""
Test the sign of Born forces
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("BORN FORCE SIGN DIAGNOSTIC")
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

# Compute Born radii and forces
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

print(f"Analytical Born forces: F_O = {forces_born[0]}, F_H = {forces_born[1]}")
print()

# Numerical gradient for Born contribution
delta = 0.0001

def compute_total_energy(coords_temp):
    """Compute total GB energy"""
    born_radii_temp, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_temp, radii, b_params, c_params, cutoff)
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii_temp, dielectric, cutoff)
    return float(E_temp)

def compute_direct_energy(coords_temp):
    """Compute direct GB energy with FIXED Born radii"""
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii, dielectric, cutoff)
    return float(E_temp)

forces_total_numerical = np.zeros_like(coords)
forces_direct_numerical = np.zeros_like(coords)

for atom in range(2):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta
        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        E_plus_total = compute_total_energy(coords_plus)
        E_minus_total = compute_total_energy(coords_minus)
        forces_total_numerical[atom, dim] = -(E_plus_total - E_minus_total) / (2*delta)

        E_plus_direct = compute_direct_energy(coords_plus)
        E_minus_direct = compute_direct_energy(coords_minus)
        forces_direct_numerical[atom, dim] = -(E_plus_direct - E_minus_direct) / (2*delta)

forces_born_numerical = forces_total_numerical - forces_direct_numerical

print(f"Numerical Born forces: F_O = {forces_born_numerical[0]}, F_H = {forces_born_numerical[1]}")
print()

print("SIGN CHECK:")
print(f"Analytical F_O[0]: {forces_born[0, 0]:.8f}")
print(f"Numerical F_O[0]:  {forces_born_numerical[0, 0]:.8f}")
print(f"Ratio (analytical/numerical): {forces_born[0, 0] / forces_born_numerical[0, 0]:.8f}")
print()

# Check if they're opposite signs
if forces_born[0, 0] * forces_born_numerical[0, 0] < 0:
    print("WARNING: Signs are OPPOSITE! The analytical and numerical forces have opposite signs.")
    print(f"The correct sign is: {np.sign(forces_born_numerical[0, 0])}")
else:
    print("Signs match!")
