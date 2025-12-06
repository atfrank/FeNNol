#!/usr/bin/env python3
"""
Simple test: just check forces and see if they're reasonable
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("FORCE CHECK - Water Molecule")
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

print(f"Initial coordinates:\n{coords}")
print()

# Compute forces
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

forces_total = forces_direct + forces_born

print(f"Energy: {float(energy):.6f} kcal/mol")
print(f"Born radii: {born_radii}")
print()

print("Forces:")
print(f"  Direct forces:  F[0] = {forces_direct[0]}")
print(f"  Born forces:    F[0] = {forces_born[0]}")
print(f"  TOTAL forces:   F[0] = {forces_total[0]}")
print()

# Check if forces sum to zero (Newton's 3rd law)
total_force = np.sum(forces_total, axis=0)
print(f"Sum of all forces: {total_force}")
print(f"Newton's 3rd law satisfied: {np.linalg.norm(total_force) < 1e-6}")
print()

# Compute force magnitudes
force_mags = np.linalg.norm(forces_total, axis=1)
print(f"Force magnitudes: {force_mags} kcal/mol/Angstrom")
print()

# Check if forces are reasonable
# For a molecule in solution, forces should be ~ 1-100 kcal/mol/A
max_force = np.max(force_mags)
if max_force > 1000:
    print(f"❌ UNREASONABLY LARGE FORCES (max = {max_force:.1f} kcal/mol/A)")
elif max_force > 100:
    print(f"⚠️  WARNING: Large forces (max = {max_force:.1f} kcal/mol/A)")
else:
    print(f"✓ Forces seem reasonable (max = {max_force:.1f} kcal/mol/A)")
print()

# Try a tiny displacement and see what happens
print("Testing single step with dt = 0.0001 ps:")
dt = 0.0001
masses = np.array([15.999, 1.008, 1.008], dtype=np.float64)
conversion = 418.4
acceleration = forces_total * conversion / masses[:, np.newaxis]

velocities = np.zeros_like(coords)
coords_new = coords + velocities * dt + 0.5 * acceleration * dt**2

print(f"Displacement: {coords_new - coords}")
print(f"New coords:\n{coords_new}")
print()

# Compute new energy
born_radii_new, psi_sum_new = fennol_cuda.gb_compute_born_radii_with_psi(coords_new, radii, b_params, c_params, cutoff)
energy_new, _ = fennol_cuda.gb_compute_energy_forces(coords_new, charges, born_radii_new, dielectric, cutoff)

print(f"Old energy: {float(energy):.6f} kcal/mol")
print(f"New energy: {float(energy_new):.6f} kcal/mol")
print(f"Change:     {float(energy_new) - float(energy):.6f} kcal/mol")

if float(energy_new) > float(energy):
    print("⚠️  Energy INCREASED (forces are pulling system to higher energy!)")
else:
    print("✓ Energy decreased (forces are correct direction)")
