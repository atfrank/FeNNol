#!/usr/bin/env python3
"""
Isolate the source of force errors by testing direct and Born forces separately
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("FORCE COMPONENT ANALYSIS - 2-Atom System")
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

# Compute analytical forces
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

forces_total = forces_direct + forces_born

print(f"Born radii: {born_radii}")
print(f"Energy: {float(energy):.10f}")
print()

print("Analytical Forces:")
print(f"  Direct:  F_O = {forces_direct[0]}, F_H = {forces_direct[1]}")
print(f"  Born:    F_O = {forces_born[0]}, F_H = {forces_born[1]}")
print(f"  TOTAL:   F_O = {forces_total[0]}, F_H = {forces_total[1]}")
print()

# Numerical gradient for TOTAL energy (direct + Born combined)
delta = 0.0001

def compute_total_energy(coords_temp):
    """Compute total GB energy including Born radii recalculation"""
    born_radii_temp, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_temp, radii, b_params, c_params, cutoff)
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii_temp, dielectric, cutoff)
    return float(E_temp)

def compute_direct_energy(coords_temp, born_radii_fixed):
    """Compute direct GB energy with FIXED Born radii (no Born recalculation)"""
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii_fixed, dielectric, cutoff)
    return float(E_temp)

# Numerical gradient for total energy
forces_total_numerical = np.zeros_like(coords)
for atom in range(2):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta
        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        E_plus = compute_total_energy(coords_plus)
        E_minus = compute_total_energy(coords_minus)

        forces_total_numerical[atom, dim] = -(E_plus - E_minus) / (2*delta)

print("Numerical gradient (total energy):")
print(f"  F_O = {forces_total_numerical[0]}")
print(f"  F_H = {forces_total_numerical[1]}")
print()

# Numerical gradient for DIRECT energy only (Born radii held fixed)
forces_direct_numerical = np.zeros_like(coords)
for atom in range(2):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta
        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        # Use ORIGINAL Born radii (no recalculation!)
        E_plus = compute_direct_energy(coords_plus, born_radii)
        E_minus = compute_direct_energy(coords_minus, born_radii)

        forces_direct_numerical[atom, dim] = -(E_plus - E_minus) / (2*delta)

print("Numerical gradient (direct energy, Born radii FIXED):")
print(f"  F_O = {forces_direct_numerical[0]}")
print(f"  F_H = {forces_direct_numerical[1]}")
print()

# Numerical gradient for Born contribution (total - direct with fixed R)
forces_born_numerical = forces_total_numerical - forces_direct_numerical

print("Numerical gradient (Born contribution = total - direct_fixed):")
print(f"  F_O = {forces_born_numerical[0]}")
print(f"  F_H = {forces_born_numerical[1]}")
print()

print("="*80)
print("ERROR ANALYSIS")
print("="*80)
print()

# Compare direct forces
error_direct = forces_direct - forces_direct_numerical
rel_error_direct = np.linalg.norm(error_direct) / np.linalg.norm(forces_direct_numerical) * 100
print(f"Direct forces error: {error_direct[0]}")
print(f"Direct forces relative error: {rel_error_direct:.2f}%")
print()

# Compare Born forces
error_born = forces_born - forces_born_numerical
rel_error_born = np.linalg.norm(error_born) / np.linalg.norm(forces_born_numerical) * 100
print(f"Born forces error: {error_born[0]}")
print(f"Born forces relative error: {rel_error_born:.2f}%")
print()

# Compare total forces
error_total = forces_total - forces_total_numerical
rel_error_total = np.linalg.norm(error_total) / np.linalg.norm(forces_total_numerical) * 100
print(f"Total forces error: {error_total[0]}")
print(f"Total forces relative error: {rel_error_total:.2f}%")
print()

if rel_error_direct < 1.0 and rel_error_born < 1.0:
    print("✓ Both force components are correct!")
elif rel_error_direct < 1.0:
    print("✓ Direct forces are correct, ✗ Born forces have errors")
elif rel_error_born < 1.0:
    print("✓ Born forces are correct, ✗ Direct forces have errors")
else:
    print("✗ Both force components have errors")
