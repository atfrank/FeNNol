#!/usr/bin/env python3
"""
Test GB forces on a SINGLE water molecule (3 atoms: O-H-H)

This is simpler than water dimer but more complex than 2 identical atoms.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

def compute_numerical_gradient(coords, charges, atomic_numbers, radii, b_params, c_params, step=1e-5):
    """Compute numerical gradient using finite differences"""
    natoms = len(coords)
    grad = np.zeros_like(coords)

    dielectric = 80.0
    cutoff = 12.0

    for i in range(natoms):
        for d in range(3):
            # Forward step
            coords_plus = coords.copy()
            coords_plus[i, d] += step

            born_radii_plus, psi_plus = fennol_cuda.gb_compute_born_radii_with_psi(
                coords_plus, radii, b_params, c_params, cutoff
            )

            E_plus, _ = fennol_cuda.gb_compute_energy_forces(
                coords_plus, charges, born_radii_plus, dielectric, cutoff
            )

            # Backward step
            coords_minus = coords.copy()
            coords_minus[i, d] -= step

            born_radii_minus, psi_minus = fennol_cuda.gb_compute_born_radii_with_psi(
                coords_minus, radii, b_params, c_params, cutoff
            )

            E_minus, _ = fennol_cuda.gb_compute_energy_forces(
                coords_minus, charges, born_radii_minus, dielectric, cutoff
            )

            # Numerical gradient
            grad[i, d] = -(float(E_plus) - float(E_minus)) / (2.0 * step)

    return grad

# Single water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H1
    [-0.757, 0.586, 0.0], # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

# OBC parameters (mbondi radii)
radii = np.array([1.5, 1.2, 1.2])  # O, H, H
b_params = np.array([0.8, 0.85, 0.85])
c_params = np.array([0.0, 0.0, 0.0])
dielectric = 80.0
cutoff = 12.0

print("=" * 70)
print("SINGLE WATER MOLECULE GB FORCE TEST")
print("=" * 70)
print()
print("System:")
print(f"  Atoms: {len(coords)}")
print(f"  Total charge: {charges.sum():.6f}")
print()

# Compute Born radii
print("Computing Born radii...")
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
    coords, radii, b_params, c_params, cutoff
)
print(f"  Born radii: {born_radii}")
print(f"  Psi sum: {psi_sum}")
print()

# Compute energy
energy, _ = fennol_cuda.gb_compute_energy_forces(
    coords, charges, born_radii, dielectric, cutoff
)
print(f"Energy: {float(energy):.6f} kcal/mol")
print()

# Compute analytical forces
print("Computing analytical forces...")
_, forces_analytical = fennol_cuda.gb_compute_forces_complete(
    coords, charges, born_radii, radii, b_params, c_params, psi_sum,
    dielectric, cutoff
)
forces_analytical = forces_analytical.reshape(-1, 3)
print(f"Analytical forces:")
for i in range(len(forces_analytical)):
    atom_type = ['O', 'H1', 'H2'][i]
    print(f"  {atom_type}: [{forces_analytical[i, 0]:12.6f}, {forces_analytical[i, 1]:12.6f}, {forces_analytical[i, 2]:12.6f}]")
print(f"  Max force: {np.max(np.abs(forces_analytical)):.6f} kcal/(mol·Å)")
print()

# Compute numerical gradient
print("Computing numerical gradient...")
forces_numerical = compute_numerical_gradient(coords, charges, atomic_numbers, radii, b_params, c_params, step=1e-5)
print(f"Numerical forces:")
for i in range(len(forces_numerical)):
    atom_type = ['O', 'H1', 'H2'][i]
    print(f"  {atom_type}: [{forces_numerical[i, 0]:12.6f}, {forces_numerical[i, 1]:12.6f}, {forces_numerical[i, 2]:12.6f}]")
print(f"  Max force: {np.max(np.abs(forces_numerical)):.6f} kcal/(mol·Å)")
print()

# Compare
error = forces_analytical - forces_numerical
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Error:")
for i in range(len(error)):
    atom_type = ['O', 'H1', 'H2'][i]
    print(f"  {atom_type}: [{error[i, 0]:12.6f}, {error[i, 1]:12.6f}, {error[i, 2]:12.6f}]")
max_error = np.max(np.abs(error))
print(f"  Max error: {max_error:.6f} kcal/(mol·Å)")
ratio = np.max(np.abs(forces_analytical)) / max(np.max(np.abs(forces_numerical)), 1e-10)
print(f"  Ratio (analytical/numerical): {ratio:.2f}×")
print()

if max_error < 0.01:
    print("✅ PASS: Forces match within tolerance!")
else:
    print(f"❌ FAIL: Error {max_error:.6f} exceeds tolerance 0.01")
