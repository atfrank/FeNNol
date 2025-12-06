#!/usr/bin/env python3
"""
Test if the self-energy derivative is computed correctly.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("SELF-ENERGY DERIVATIVE TEST")
print("="*80)
print()

# Simple 1-atom system to isolate self-energy
coords = np.array([
    [0.0, 0.0, 0.0],
], dtype=np.float64)

charges = np.array([-0.834], dtype=np.float64)
radii = np.array([1.5], dtype=np.float64)
b_params = np.array([0.8], dtype=np.float64)
c_params = np.array([0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Compute Born radius (should equal intrinsic radius for isolated atom)
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

print(f"Isolated atom:")
print(f"  q = {charges[0]}")
print(f"  ρ = {radii[0]}")
print(f"  R = {born_radii[0]:.6f} (should equal ρ)")
print(f"  ψ = {psi_sum[0]:.6f} (should be ~0)")
print()

# Compute energy
energy, _ = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

COULOMB = 332.0636
gb_factor = -0.5 * (1.0 - 1.0/dielectric) * COULOMB

print(f"Self-energy:")
print(f"  gb_factor = {gb_factor:.6f}")
print(f"  E_self = gb_factor * q² / R")
print(f"         = {gb_factor:.4f} * {charges[0]**2:.6f} / {born_radii[0]:.6f}")
print(f"         = {float(energy):.10f} kcal/mol")
print()

# Expected self-energy
E_self_expected = gb_factor * charges[0]**2 / born_radii[0]
print(f"Expected E_self = {E_self_expected:.10f}")
print(f"CUDA E_self     = {float(energy):.10f}")
print(f"Match: {abs(float(energy) - E_self_expected) < 1e-6}")
print()

# Derivative of self-energy w.r.t. R
print("Derivative ∂E_self/∂R:")
print(f"  ∂E_self/∂R = ∂/∂R (gb_factor * q² / R)")
print(f"             = gb_factor * q² * (-1/R²)")

dE_dR_expected = gb_factor * charges[0]**2 * (-1.0 / (born_radii[0]**2))
print(f"             = {gb_factor:.4f} * {charges[0]**2:.6f} * (-1/{born_radii[0]:.6f}²)")
print(f"             = {dE_dR_expected:.8f}")
print()

# Compute what CUDA gives
dE_dR_cuda = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
print(f"CUDA dE_dR[0] = {dE_dR_cuda[0]:.8f}")
print()

if abs(dE_dR_cuda[0] - dE_dR_expected) < 1e-6:
    print("✓ Self-energy derivative is CORRECT")
else:
    print("✗ Self-energy derivative is WRONG")
    print(f"  Error: {dE_dR_cuda[0] - dE_dR_expected:.6e}")
    ratio = dE_dR_cuda[0] / dE_dR_expected if abs(dE_dR_expected) > 1e-10 else 0
    print(f"  Ratio: {ratio:.6f}")

print()
print("="*80)
print("CHECK SIGN OF gb_factor")
print("="*80)
print()

print(f"gb_factor = -0.5 * (1 - 1/ε) * COULOMB")
print(f"          = -0.5 * (1 - 1/{dielectric}) * {COULOMB:.4f}")
print(f"          = -0.5 * {1 - 1/dielectric:.6f} * {COULOMB:.4f}")
print(f"          = {gb_factor:.6f}")
print()

if gb_factor < 0:
    print("gb_factor is NEGATIVE")
    print()
    print("Therefore:")
    print("  E_self = (negative) * q² / R < 0  ✓ (stabilizing)")
    print("  ∂E_self/∂R = (negative) * q² * (-1/R²)")
    print("             = (negative) * (negative)")
    print("             = POSITIVE")
    print()
    print("This means: Increasing R makes energy less negative (less stable)")
    print("           → System wants to minimize R")
    print("           → Forces should push atoms together (correct!)")
else:
    print("ERROR: gb_factor should be negative!")
