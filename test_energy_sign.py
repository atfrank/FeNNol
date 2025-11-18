#!/usr/bin/env python3
"""
Test to check if energy has correct sign by perturbing a single atom.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

# Single water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H1
    [-0.757, 0.586, 0.0], # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)

# OBC parameters
radii = np.array([1.5, 1.2, 1.2])
b_params = np.array([0.8, 0.85, 0.85])
c_params = np.array([0.0, 0.0, 0.0])
dielectric = 80.0
cutoff = 12.0

print("Testing energy sign by moving O atom in +y direction")
print()

# Compute energy at equilibrium
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
E0, _ = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
print(f"E(y=0.0):   {float(E0):.6f} kcal/mol")

# Move O in +y by small amount
coords_plus = coords.copy()
coords_plus[0, 1] += 0.001  # Move O in +y

born_radii_plus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_plus, radii, b_params, c_params, cutoff)
E_plus, _ = fennol_cuda.gb_compute_energy_forces(coords_plus, charges, born_radii_plus, dielectric, cutoff)
print(f"E(y=+0.001): {float(E_plus):.6f} kcal/mol")

# Move O in -y by small amount
coords_minus = coords.copy()
coords_minus[0, 1] -= 0.001  # Move O in -y

born_radii_minus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_minus, radii, b_params, c_params, cutoff)
E_minus, _ = fennol_cuda.gb_compute_energy_forces(coords_minus, charges, born_radii_minus, dielectric, cutoff)
print(f"E(y=-0.001): {float(E_minus):.6f} kcal/mol")

print()
print(f"ΔE (moving +y): {float(E_plus - E0):.6f} kcal/mol")
print(f"ΔE (moving -y): {float(E_minus - E0):.6f} kcal/mol")
print()

if E_plus > E0:
    print("Moving O in +y INCREASES energy (repulsive)")
    print("Therefore, force on O should be in -y direction (negative)")
elif E_plus < E0:
    print("Moving O in +y DECREASES energy (attractive)")
    print("Therefore, force on O should be in +y direction (positive)")

print()
numerical_force_y = -(float(E_plus) - float(E_minus)) / 0.002
print(f"Numerical force (from finite diff): F_y = {numerical_force_y:.6f} kcal/(mol·Å)")

# Compute analytical force
_,  forces = fennol_cuda.gb_compute_forces_complete(
    coords, charges, born_radii, radii, b_params, c_params, psi_sum, dielectric, cutoff
)
forces = forces.reshape(-1, 3)
print(f"Analytical force:                    F_y = {forces[0, 1]:.6f} kcal/(mol·Å)")
print()

if numerical_force_y * forces[0, 1] > 0:
    print("✅ Signs match!")
else:
    print("❌ SIGN ERROR: Forces have opposite signs!")
