#!/usr/bin/env python3
"""
Test force accuracy when Born radii are near clamping
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("FORCE ACCURACY TEST NEAR BORN RADIUS CLAMPING")
print("="*80)
print()

# Water molecule with atoms far apart (near clamping)
coords = np.array([
    [0.0, 0.0, 0.0],        # O
    [2.70, 2.10, 0.0],      # H1 - far away
    [-2.70, 2.10, 0.0],     # H2 - far away
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

print(f"Initial coordinates:\\n{coords}")
print()

# Compute analytical forces
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
forces_total = forces_direct + forces_born

print(f"Energy: {float(energy):.6f} kcal/mol")
print(f"Born radii: {born_radii}")
print(f"Clamped? O={abs(born_radii[0]-radii[0])<1e-6}, H1={abs(born_radii[1]-radii[1])<1e-6}, H2={abs(born_radii[2]-radii[2])<1e-6}")
print()

print("Analytical Forces:")
for i in range(3):
    print(f"  Atom {i}: F = {forces_total[i]}")
print()

# Numerical gradient
delta = 0.0001
forces_numerical = np.zeros_like(coords)

def compute_total_energy(coords_temp):
    """Compute total GB energy"""
    born_radii_temp, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_temp, radii, b_params, c_params, cutoff)
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

print("Numerical gradient:")
for i in range(3):
    print(f"  Atom {i}: F = {forces_numerical[i]}")
print()

# Compare
error = forces_total - forces_numerical
rel_error = np.linalg.norm(error) / (np.linalg.norm(forces_numerical) + 1e-10) * 100

print("="*80)
print("ERROR ANALYSIS")
print("="*80)
print()

for i in range(3):
    print(f"Atom {i} error: {error[i]}")
    atom_rel_error = np.linalg.norm(error[i]) / (np.linalg.norm(forces_numerical[i]) + 1e-10) * 100
    print(f"Atom {i} relative error: {atom_rel_error:.2f}%")
    print()

print(f"Overall relative error: {rel_error:.2f}%")
print()

if rel_error < 1.0:
    print("✓ Forces are correct even near clamping!")
else:
    print(f"✗ Forces have {rel_error:.1f}% error near clamping - THIS IS THE PROBLEM!")
