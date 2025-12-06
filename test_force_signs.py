#!/usr/bin/env python3
"""
Test if the force signs and magnitudes are correct by comparing
Born radii forces with numerical gradient.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("FORCE SIGN TEST")
print("="*80)
print()

# 2-atom system
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H
], dtype=np.float64)

charges = np.array([-0.834, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Compute current values
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

print("Current state:")
print(f"  R_O = {born_radii[0]:.8f}, psi_O = {psi_sum[0]:.8f}")
print(f"  R_H = {born_radii[1]:.8f}, psi_H = {psi_sum[1]:.8f}")
print(f"  Energy = {float(energy):.10f}")
print()

# Compute Born radii forces
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

print("Born radii forces (CUDA):")
print(f"  F_O = {forces_born[0]}")
print(f"  F_H = {forces_born[1]}")
print()

# Numerical gradient: Perturb coordinates and see how Born radii change
# Then compute energy change
delta = 0.0001

# Test: Move oxygen in +x direction
coords_test = coords.copy()
coords_test[0, 0] += delta

# Compute new Born radii
born_radii_plus, psi_sum_plus = fennol_cuda.gb_compute_born_radii_with_psi(coords_test, radii, b_params, c_params, cutoff)

print(f"After moving O by +{delta} in x:")
print(f"  R_O: {born_radii[0]:.8f} → {born_radii_plus[0]:.8f} (ΔR = {born_radii_plus[0] - born_radii[0]:.8e})")
print(f"  R_H: {born_radii[1]:.8f} → {born_radii_plus[1]:.8f} (ΔR = {born_radii_plus[1] - born_radii[1]:.8e})")
print(f"  psi_O: {psi_sum[0]:.8f} → {psi_sum_plus[0]:.8f} (Δpsi = {psi_sum_plus[0] - psi_sum[0]:.8e})")
print(f"  psi_H: {psi_sum[1]:.8f} → {psi_sum_plus[1]:.8f} (Δpsi = {psi_sum_plus[1] - psi_sum[1]:.8e})")
print()

# Energy change
energy_plus, _ = fennol_cuda.gb_compute_energy_forces(coords_test, charges, born_radii_plus, dielectric, cutoff)
dE_dx = (float(energy_plus) - float(energy)) / delta

print(f"Energy change:")
print(f"  E: {float(energy):.10f} → {float(energy_plus):.10f}")
print(f"  dE/dx = {dE_dx:.8f}")
print(f"  F_x (numerical) = {-dE_dx:.8f}")
print()

# Expected force from chain rule:
# F_x = -dE/dx = -(∂E/∂R_O × ∂R_O/∂x + ∂E/∂R_H × ∂R_H/∂x)
#              = -(∂E/∂R_O × ∂R_O/∂psi_O × ∂psi_O/∂x + ∂E/∂R_H × ∂R_H/∂psi_H × ∂psi_H/∂x)

# Numerical derivatives of Born radii w.r.t. x
dR_O_dx_numerical = (born_radii_plus[0] - born_radii[0]) / delta
dR_H_dx_numerical = (born_radii_plus[1] - born_radii[1]) / delta

print("Numerical derivatives of R w.r.t. x:")
print(f"  ∂R_O/∂x = {dR_O_dx_numerical:.8e}")
print(f"  ∂R_H/∂x = {dR_H_dx_numerical:.8e}")
print()

# Expected force from chain rule
F_x_expected = -(dE_dR[0] * dR_O_dx_numerical + dE_dR[1] * dR_H_dx_numerical)

print("Expected force from chain rule:")
print(f"  F_x = -(dE_dR[0] × ∂R_O/∂x + dE_dR[1] × ∂R_H/∂x)")
print(f"  F_x = -({dE_dR[0]:.4f} × {dR_O_dx_numerical:.6e} + {dE_dR[1]:.4f} × {dR_H_dx_numerical:.6e})")
print(f"  F_x = {F_x_expected:.8f}")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print()

print(f"CUDA F_O[x] = {forces_born[0, 0]:.8f}")
print(f"Numerical    = {-dE_dx:.8f}")
print(f"Chain rule   = {F_x_expected:.8f}")
print()

error_cuda = abs(forces_born[0, 0] - (-dE_dx))
error_chain = abs(F_x_expected - (-dE_dx))

print(f"CUDA error: {error_cuda:.6e}")
print(f"Chain rule error: {error_chain:.6e}")
print()

if error_cuda < error_chain:
    print("CUDA is more accurate than simple chain rule")
else:
    print("Chain rule is more accurate than CUDA")
    ratio = forces_born[0, 0] / F_x_expected if abs(F_x_expected) > 1e-10 else 0
    print(f"CUDA/Chain ratio: {ratio:.6f}")

    # Check if there's a consistent scaling factor
    if abs(ratio - 2.0) < 0.1:
        print("→ CUDA force is ~2× too large!")
    elif abs(ratio - 0.5) < 0.1:
        print("→ CUDA force is ~0.5× too small!")
    elif abs(ratio + 1.0) < 0.1:
        print("→ CUDA force has WRONG SIGN!")

print()
print("="*80)
print("DETAILED BREAKDOWN")
print("="*80)
print()

# Let's trace through the entire calculation manually
print("Step 1: Compute ∂E/∂R (done by compute_dE_dR)")
print(f"  dE_dR[0] = {dE_dR[0]:.8f}")
print(f"  dE_dR[1] = {dE_dR[1]:.8f}")
print()

print("Step 2: Convert to ∂E/∂ψ (done by reduce_born_force)")
print(f"  dE_dpsi[0] = {dE_dpsi[0]:.8f}")
print(f"  dE_dpsi[1] = {dE_dpsi[1]:.8f}")
print()

# Numerical ∂psi/∂x
dpsi_O_dx_numerical = (psi_sum_plus[0] - psi_sum[0]) / delta
dpsi_H_dx_numerical = (psi_sum_plus[1] - psi_sum[1]) / delta

print("Step 3: Numerical ∂ψ/∂x")
print(f"  ∂psi_O/∂x = {dpsi_O_dx_numerical:.8e}")
print(f"  ∂psi_H/∂x = {dpsi_H_dx_numerical:.8e}")
print()

print("Step 4: Force from ψ chain (what apply_born_forces should compute)")
print(f"  F_x = -(dE_dpsi[0] × ∂psi_O/∂x + dE_dpsi[1] × ∂psi_H/∂x)")
F_x_psi_chain = -(dE_dpsi[0] * dpsi_O_dx_numerical + dE_dpsi[1] * dpsi_H_dx_numerical)
print(f"  F_x = -({dE_dpsi[0]:.4f} × {dpsi_O_dx_numerical:.6e} + {dE_dpsi[1]:.4f} × {dpsi_H_dx_numerical:.6e})")
print(f"  F_x = {F_x_psi_chain:.8f}")
print()

print("FINAL COMPARISON:")
print(f"  CUDA:          {forces_born[0, 0]:.8f}")
print(f"  ψ chain rule:  {F_x_psi_chain:.8f}")
print(f"  Numerical:     {-dE_dx:.8f}")
print()

if abs(forces_born[0, 0] - F_x_psi_chain) < 1e-5:
    print("✓ CUDA matches ψ chain rule")
else:
    ratio = forces_born[0, 0] / F_x_psi_chain if abs(F_x_psi_chain) > 1e-10 else 0
    print(f"✗ CUDA differs from ψ chain rule (ratio: {ratio:.6f})")

if abs(F_x_psi_chain - (-dE_dx)) < error_cuda * 0.1:
    print("✓ ψ chain rule matches numerical gradient")
else:
    print("✗ ψ chain rule differs from numerical gradient")
    print(f"  Error: {abs(F_x_psi_chain - (-dE_dx)):.6e}")
