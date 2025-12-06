#!/usr/bin/env python3
"""
Test if forces are being double-counted.

The apply_born_forces kernel processes each pair (i,j) from BOTH directions:
- Thread i processes (i,j) and modifies both force[i] and force[j]
- Thread j processes (j,i) and modifies both force[j] and force[i]

This could lead to 2× overcounting!
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("DOUBLE-COUNTING TEST")
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

# Compute Born radii forces
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

print("Current CUDA forces:")
print(f"  F_O = {forces_born[0]}")
print(f"  F_H = {forces_born[1]}")
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

# Get direct forces for comparison
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

# Total analytical forces
forces_total = forces_direct + forces_born

print("Full force comparison:")
print(f"  Direct:    F_O = {forces_direct[0]}")
print(f"  Born:      F_O = {forces_born[0]}")
print(f"  Total:     F_O = {forces_total[0]}")
print(f"  Numerical: F_O = {forces_numerical[0]}")
print()

error = forces_total[0] - forces_numerical[0]
print(f"Error: {error}")
print(f"Error magnitude: {np.linalg.norm(error):.6f}")
print()

# Test hypothesis: If we divide Born forces by 2, does it match?
forces_born_half = forces_born / 2.0
forces_total_half = forces_direct + forces_born_half

print("TEST: What if Born forces are 2× too large?")
print(f"  Born/2:    F_O = {forces_born_half[0]}")
print(f"  Total:     F_O = {forces_total_half[0]}")
print(f"  Numerical: F_O = {forces_numerical[0]}")
print()

error_half = forces_total_half[0] - forces_numerical[0]
print(f"Error with Born/2: {error_half}")
print(f"Error magnitude: {np.linalg.norm(error_half):.6f}")
print()

# Calculate improvement
orig_error_mag = np.linalg.norm(forces_total - forces_numerical)
half_error_mag = np.linalg.norm(forces_total_half - forces_numerical)

print(f"Original error: {orig_error_mag:.6f}")
print(f"Error with Born/2: {half_error_mag:.6f}")
print(f"Improvement: {(1 - half_error_mag/orig_error_mag)*100:.1f}%")
print()

if half_error_mag < orig_error_mag * 0.1:
    print("✓ HYPOTHESIS CONFIRMED: Born forces are being DOUBLE-COUNTED!")
    print()
    print("Root cause: apply_born_forces kernel processes each pair (i,j) from")
    print("both thread i and thread j, and modifies forces for BOTH atoms in")
    print("each iteration. This leads to 2× overcounting.")
    print()
    print("Fix: The kernel should either:")
    print("  1. Only process each pair once (i < j), OR")
    print("  2. Only modify force[i] from thread i (use Newton's 3rd law implicitly), OR")
    print("  3. Include a 0.5 factor in the force calculation")
else:
    print("✗ Hypothesis rejected: Error not improved by dividing by 2")
    print(f"  Ratio of errors: {half_error_mag / orig_error_mag:.3f}")
