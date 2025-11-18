#!/usr/bin/env python3
"""
Test that Born radii calculation matches the expected values
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("BORN RADII TEST")
print("="*80)
print()

# Single water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H1
    [-0.757, 0.586, 0.0], # H2
], dtype=np.float64)

radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
cutoff = 12.0

# Compute Born radii
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

print("Intrinsic radii:", radii)
print("Descreening sum (ψ):", psi_sum)
print("Scaled sum (0.5*ρ*ψ):")
for i in range(3):
    print(f"  Atom {i}: 0.5 * {radii[i]:.3f} * {psi_sum[i]:.6f} = {0.5 * radii[i] * psi_sum[i]:.6f}")
print()
print("Born radii:", born_radii)
print()

# Verify that Born radii >= intrinsic radii
for i in range(3):
    if born_radii[i] < radii[i]:
        print(f"ERROR: Born radius {born_radii[i]:.6f} < intrinsic radius {radii[i]:.6f} for atom {i}")
    else:
        print(f"✓ Atom {i}: R = {born_radii[i]:.6f} >= ρ = {radii[i]:.6f}")
