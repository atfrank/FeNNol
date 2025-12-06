#!/usr/bin/env python3
"""
Simple test: just compute energy at different geometries to validate numerical gradient
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
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

print("Testing energy calculation with small perturbations")
print()

# Reference
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
E0, _ = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
print(f"E(reference):  {float(E0):.10f} kcal/mol")
print(f"Born radii: {born_radii}")
print()

# Move O in +y by 0.001
coords_plus = coords.copy()
coords_plus[0, 1] += 0.001
born_radii_plus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_plus, radii, b_params, c_params, cutoff)
E_plus, _ = fennol_cuda.gb_compute_energy_forces(coords_plus, charges, born_radii_plus, dielectric, cutoff)
print(f"E(y+0.001):    {float(E_plus):.10f} kcal/mol")
print(f"Born radii: {born_radii_plus}")
print(f"ΔE: {float(E_plus - E0):.10f}")
print()

# Move O in -y by 0.001
coords_minus = coords.copy()
coords_minus[0, 1] -= 0.001
born_radii_minus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_minus, radii, b_params, c_params, cutoff)
E_minus, _ = fennol_cuda.gb_compute_energy_forces(coords_minus, charges, born_radii_minus, dielectric, cutoff)
print(f"E(y-0.001):    {float(E_minus):.10f} kcal/mol")
print(f"Born radii: {born_radii_minus}")
print(f"ΔE: {float(E_minus - E0):.10f}")
print()

# Numerical derivative
dE_dy = (float(E_plus) - float(E_minus)) / 0.002
print(f"Numerical ∂E/∂y_O: {dE_dy:.10f} kcal/(mol·Å)")
print(f"Force F_y = -∂E/∂y: {-dE_dy:.10f} kcal/(mol·Å)")
