#!/usr/bin/env python3
"""
Simple 2-atom test to debug GB forces

Uses two oxygen atoms at 2.8 Å separation (same as in debug session summary)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

def compute_numerical_gradient(coords, charges, atomic_numbers, step=1e-5):
    """Compute numerical gradient using finite differences"""
    natoms = len(coords)
    grad = np.zeros_like(coords)

    # OBC parameters for oxygen
    radii = np.array([1.5, 1.5])
    b_params = np.array([0.8, 0.8])
    c_params = np.array([0.0, 0.0])
    dielectric = 80.0
    cutoff = 12.0

    for i in range(natoms):
        for d in range(3):
            # Forward step
            coords_plus = coords.copy()
            coords_plus[i, d] += step

            # Compute Born radii
            born_radii_plus, psi_plus = fennol_cuda.gb_compute_born_radii_with_psi(
                coords_plus, radii, b_params, c_params, cutoff
            )

            # Compute energy and forces (we only need energy)
            E_plus, _ = fennol_cuda.gb_compute_energy_forces(
                coords_plus, charges, born_radii_plus, dielectric, cutoff
            )

            # Backward step
            coords_minus = coords.copy()
            coords_minus[i, d] -= step

            # Compute Born radii
            born_radii_minus, psi_minus = fennol_cuda.gb_compute_born_radii_with_psi(
                coords_minus, radii, b_params, c_params, cutoff
            )

            # Compute energy and forces (we only need energy)
            E_minus, _ = fennol_cuda.gb_compute_energy_forces(
                coords_minus, charges, born_radii_minus, dielectric, cutoff
            )

            # Numerical gradient
            grad[i, d] = -(E_plus - E_minus) / (2.0 * step)

    return grad

# Simple 2-atom system (same as debug session)
coords = np.array([
    [0.0, 0.0, 0.0],  # Oxygen 1
    [2.8, 0.0, 0.0],  # Oxygen 2
], dtype=np.float64)

charges = np.array([-0.834, -0.834], dtype=np.float64)
atomic_numbers = np.array([8, 8], dtype=np.int32)

# OBC parameters for oxygen
radii = np.array([1.5, 1.5])
b_params = np.array([0.8, 0.8])
c_params = np.array([0.0, 0.0])
dielectric = 80.0
cutoff = 12.0

print("=" * 70)
print("SIMPLE 2-ATOM GB FORCE DEBUG TEST")
print("=" * 70)
print()
print("System:")
print(f"  Coords: {coords}")
print(f"  Charges: {charges}")
print(f"  Dielectric: {dielectric}")
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

# Compute analytical forces (COMPLETE with Born radii derivatives)
print("Computing COMPLETE analytical forces...")
_, forces_complete = fennol_cuda.gb_compute_forces_complete(
    coords, charges, born_radii, radii, b_params, c_params, psi_sum,
    dielectric, cutoff
)
forces_complete = forces_complete.reshape(-1, 3)
print(f"Complete forces:")
for i in range(len(forces_complete)):
    print(f"  Atom {i}: [{forces_complete[i, 0]:12.6f}, {forces_complete[i, 1]:12.6f}, {forces_complete[i, 2]:12.6f}]")
print(f"  Max force: {np.max(np.abs(forces_complete)):.6f} kcal/(mol·Å)")
print()

# Compute numerical gradient
print("Computing numerical gradient...")
grad_numerical = compute_numerical_gradient(coords, charges, atomic_numbers, step=1e-5)
print(f"Numerical forces:")
for i in range(len(grad_numerical)):
    print(f"  Atom {i}: [{grad_numerical[i, 0]:12.6f}, {grad_numerical[i, 1]:12.6f}, {grad_numerical[i, 2]:12.6f}]")
print(f"  Max force: {np.max(np.abs(grad_numerical)):.6f} kcal/(mol·Å)")
print()

# Compare
error = forces_complete - grad_numerical
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Error:")
for i in range(len(error)):
    print(f"  Atom {i}: [{error[i, 0]:12.6f}, {error[i, 1]:12.6f}, {error[i, 2]:12.6f}]")
print(f"  Max error: {np.max(np.abs(error)):.6f} kcal/(mol·Å)")
print(f"  Ratio (analytical/numerical): {np.max(np.abs(forces_complete)) / np.max(np.abs(grad_numerical)):.2f}×")
print()

# From GPU debug output, we know:
print("=" * 70)
print("DEBUG INFO FROM GPU")
print("=" * 70)
print("From CUDA printf output above:")
print("  dE_dR_i (atom 0) = 86.48 (with 2× self-energy factor)")
print("  force_mag_i (pair 0,1) = -197.03")
print("  born_forces (atom 0) = [25.18, 246.17, 0.00]")
print()
print("Expected (normal 1× self-energy):")
print("  dE_dR_i should be ~43 (half of 86.48)")
print("  Born forces should be ~123 (half of 246)")
print()
print("But numerical gradient expects only ~0.17 kcal/(mol·Å)!")
print("This is a 700× discrepancy!")
print("=" * 70)
