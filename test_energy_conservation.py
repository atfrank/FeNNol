#!/usr/bin/env python3
"""
Test energy conservation - are forces truly the gradient of energy?
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("ENERGY CONSERVATION TEST")
print("="*80)
print()

# Water molecule
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

# Compute forces at current position
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
forces_analytical = forces_direct + forces_born

print(f"Initial energy: {float(energy):.10f} kcal/mol")
print()

# Compute numerical gradient
delta = 0.00001
forces_numerical = np.zeros_like(coords)

def compute_total_energy(coords_temp):
    """Compute total GB energy"""
    born_radii_temp, psi_sum_temp = fennol_cuda.gb_compute_born_radii_with_psi(coords_temp, radii, b_params, c_params, cutoff)
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii_temp, dielectric, cutoff)
    return float(E_temp)

for atom in range(3):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta
        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        E_plus = compute_total_energy(coords_plus)
        E_minus = compute_total_energy(coords_minus)

        forces_numerical[atom, dim] = -(E_plus - E_minus) / (2*delta)

print("Force comparison:")
for i in range(3):
    print(f"Atom {i}:")
    print(f"  Analytical: {forces_analytical[i]}")
    print(f"  Numerical:  {forces_numerical[i]}")
    error = forces_analytical[i] - forces_numerical[i]
    print(f"  Error:      {error}")
    print()

total_error = np.linalg.norm(forces_analytical - forces_numerical) / np.linalg.norm(forces_numerical) * 100
print(f"Relative force error: {total_error:.6f}%")
print()

# Test energy conservation along force direction
# Move in the direction of force - energy should DECREASE
print("Testing energy change along force direction:")
force_direction = forces_analytical / np.linalg.norm(forces_analytical)
step_sizes = [0.0001, 0.001, 0.01]

for step_size in step_sizes:
    coords_test = coords + step_size * force_direction
    E_test = compute_total_energy(coords_test)
    dE = E_test - float(energy)
    predicted_dE = -np.sum(forces_analytical * force_direction) * step_size

    print(f"Step size {step_size:.5f}: dE = {dE:.8f}, predicted = {predicted_dE:.8f}, ratio = {dE/predicted_dE:.6f}")

print()
if total_error < 0.01:
    print("✓ Forces are correct gradient of energy (error < 0.01%)")
else:
    print(f"❌ Forces are NOT correct gradient of energy (error = {total_error:.6f}%)")
