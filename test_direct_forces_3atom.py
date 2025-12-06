#!/usr/bin/env python3
"""
Manually compute direct GB forces for 3-atom water to verify kernel
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("MANUAL DIRECT FORCE CALCULATION - 3 Atoms")
print("="*80)
print()

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

# Get Born radii
born_radii, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

# Get kernel forces
energy_kernel, forces_kernel = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

print(f"Kernel forces:")
for i, f in enumerate(forces_kernel):
    print(f"  F[{i}] = {f}")
print()

# Manual calculation
COULOMB = 332.0636
gb_factor = -0.5 * (1.0 - 1.0/dielectric) * COULOMB

def compute_f_gb_and_deriv(r, R_i, R_j):
    """Compute f_GB and df_GB/dr"""
    R_product = R_i * R_j
    r_sq = r**2
    exp_arg = -r_sq / (4.0 * R_product)
    exp_term = np.exp(exp_arg)

    f_gb = np.sqrt(r_sq + R_product * exp_term)
    df_gb_dr = r * (1.0 - 0.25 * exp_term) / f_gb

    return f_gb, df_gb_dr

# Compute forces manually
forces_manual = np.zeros((3, 3))

# For each atom i, loop over all j
for i in range(3):
    for j in range(3):
        if i == j:
            continue

        # Displacement vector (i - j)
        dr = coords[i] - coords[j]
        r = np.linalg.norm(dr)
        r_hat = dr / r

        # Compute f_GB and derivative
        f_gb, df_gb_dr = compute_f_gb_and_deriv(r, born_radii[i], born_radii[j])

        # Force magnitude (with 0.5 factor because each pair is processed twice)
        force_mag = 0.5 * gb_factor * charges[i] * charges[j] * df_gb_dr / (f_gb**2)

        # Force vector
        force_vec = force_mag * r_hat

        # Accumulate
        forces_manual[i] += force_vec

        print(f"Pair ({i},{j}): r={r:.6f}, f_GB={f_gb:.6f}, df/dr={df_gb_dr:.6f}, force_mag={force_mag:.6f}")
        print(f"  r_hat={r_hat}, force_vec={force_vec}")

print()
print(f"Manual forces:")
for i, f in enumerate(forces_manual):
    print(f"  F[{i}] = {f}")
print()

print("Comparison:")
for i in range(3):
    error = forces_manual[i] - forces_kernel[i]
    print(f"  Atom {i}: error={error}, norm={np.linalg.norm(error):.6f}")
print()

# Check if the 0.5 factor is the issue
print("="*80)
print("TESTING WITHOUT 0.5 FACTOR")
print("="*80)
print()

forces_manual_no_half = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        if i == j:
            continue

        dr = coords[i] - coords[j]
        r = np.linalg.norm(dr)
        r_hat = dr / r

        f_gb, df_gb_dr = compute_f_gb_and_deriv(r, born_radii[i], born_radii[j])

        # Force magnitude WITHOUT 0.5 factor
        force_mag = gb_factor * charges[i] * charges[j] * df_gb_dr / (f_gb**2)

        force_vec = force_mag * r_hat
        forces_manual_no_half[i] += force_vec

print(f"Manual forces (no 0.5 factor):")
for i, f in enumerate(forces_manual_no_half):
    print(f"  F[{i}] = {f}")
print()

print("Comparison (no 0.5):")
for i in range(3):
    error = forces_manual_no_half[i] - forces_kernel[i]
    print(f"  Atom {i}: error={error}, norm={np.linalg.norm(error):.6f}")
