#!/usr/bin/env python3
"""
Directly test the descreening derivative ∂ψ/∂r.

The apply_born_forces kernel uses descreening_integral_derivative()
to compute ∂ψ_i/∂r. Let's verify this is correct.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("PSI DERIVATIVE TEST")
print("="*80)
print()

# 2-atom system
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H
], dtype=np.float64)

radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
cutoff = 12.0

# Compute psi at current positions
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

r = np.linalg.norm(coords[1] - coords[0])

print(f"Initial state:")
print(f"  r = {r:.6f} Å")
print(f"  psi_O = {psi_sum[0]:.8f}")
print(f"  psi_H = {psi_sum[1]:.8f}")
print()

# Numerical derivative: perturb distance along the line connecting atoms
delta_r = 0.0001

# Move H away from O by delta_r
direction = (coords[1] - coords[0]) / r
coords_plus = coords.copy()
coords_plus[1] = coords[0] + (r + delta_r) * direction

# Move H toward O by delta_r
coords_minus = coords.copy()
coords_minus[1] = coords[0] + (r - delta_r) * direction

# Compute psi at perturbed positions
_, psi_plus = fennol_cuda.gb_compute_born_radii_with_psi(coords_plus, radii, b_params, c_params, cutoff)
_, psi_minus = fennol_cuda.gb_compute_born_radii_with_psi(coords_minus, radii, b_params, c_params, cutoff)

# Numerical derivatives
dpsi_O_dr_numerical = (psi_plus[0] - psi_minus[0]) / (2 * delta_r)
dpsi_H_dr_numerical = (psi_plus[1] - psi_minus[1]) / (2 * delta_r)

print(f"Numerical derivatives:")
print(f"  ∂psi_O/∂r = {dpsi_O_dr_numerical:.8e}")
print(f"  ∂psi_H/∂r = {dpsi_H_dr_numerical:.8e}")
print()

# Now let's check what the CUDA kernel computes
# The kernel should compute the same values

# From the GPU debug output in test_force_signs.py, we saw:
# GPU[apply_born_forces]: pair (0,1): dpsi_i_dr=-0.02647456
# This should match our numerical dpsi_O_dr

print("Expected CUDA values (from descreening_integral_derivative):")
print(f"  For atom 0 (O): dpsi_i_dr should be ≈ {dpsi_O_dr_numerical:.8f}")
print(f"  For atom 1 (H): dpsi_i_dr should be ≈ {dpsi_H_dr_numerical:.8f}")
print()

# The OpenMM formula involves complex HCT integral derivatives
# Let's manually compute what it should be using the HCT formula

rho_O = radii[0]
rho_H = radii[1]

# HCT integral derivative (from OpenMM)
def hct_derivative(r, rho_i, rho_j):
    """Compute ∂I/∂r for HCT descreening integral."""
    if r < 0.001:
        return 0.0

    upper_limit = rho_i + rho_j
    lower_limit = abs(rho_i - rho_j)

    if r < lower_limit:
        return 0.0
    elif r < upper_limit:
        # Partial overlap - use HCT derivative formula
        s_j = rho_j

        # Lower bound: l_ij = 1/max(ρᵢ, |r - sⱼ|)
        abs_diff = abs(r - s_j)
        lower_bound = max(rho_i, abs_diff)
        l_ij = 1.0 / lower_bound

        # Upper bound: u_ij = 1/(r + sⱼ)
        u_ij = 1.0 / (r + s_j)

        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij
        s_j2 = s_j * s_j
        r_inv = 1.0 / r
        r2_inv = r_inv * r_inv

        # HCT derivative formula from OpenMM
        t3 = 0.125 * (1.0 + s_j2 * r2_inv) * (l_ij2 - u_ij2) + 0.25 * np.log(u_ij / l_ij) * r2_inv
        deriv = t3 * r_inv

        return deriv
    else:
        return 0.0

dpsi_O_dr_analytical = hct_derivative(r, rho_O, rho_H)
dpsi_H_dr_analytical = hct_derivative(r, rho_H, rho_O)

print("Analytical HCT derivative:")
print(f"  ∂psi_O/∂r = {dpsi_O_dr_analytical:.8e}")
print(f"  ∂psi_H/∂r = {dpsi_H_dr_analytical:.8e}")
print()

print("Comparison:")
print(f"  O: Numerical={dpsi_O_dr_numerical:.8e}, Analytical={dpsi_O_dr_analytical:.8e}")
print(f"     Error: {abs(dpsi_O_dr_numerical - dpsi_O_dr_analytical):.2e}")
print(f"  H: Numerical={dpsi_H_dr_numerical:.8e}, Analytical={dpsi_H_dr_analytical:.8e}")
print(f"     Error: {abs(dpsi_H_dr_numerical - dpsi_H_dr_analytical):.2e}")
print()

if abs(dpsi_O_dr_numerical - dpsi_O_dr_analytical) < 1e-6:
    print("✓ HCT derivative formula is correct for O")
else:
    print("✗ HCT derivative formula has error for O")

if abs(dpsi_H_dr_numerical - dpsi_H_dr_analytical) < 1e-6:
    print("✓ HCT derivative formula is correct for H")
else:
    print("✗ HCT derivative formula has error for H")

print()
print("="*80)
print("TOTAL FORCE CHAIN VERIFICATION")
print("="*80)
print()

# Now let's trace through the entire force calculation manually
charges = np.array([-0.834, 0.417], dtype=np.float64)
dielectric = 80.0

# Compute full energy derivatives
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)

print("Energy derivatives:")
print(f"  dE_dR[0] = {dE_dR[0]:.6f}")
print(f"  dE_dR[1] = {dE_dR[1]:.6f}")
print(f"  dE_dpsi[0] = {dE_dpsi[0]:.6f}")
print(f"  dE_dpsi[1] = {dE_dpsi[1]:.6f}")
print()

# Manual force calculation along the O-H bond direction
# F_x = -(dE_dpsi[0] * dpsi_O_dr + dE_dpsi[1] * dpsi_H_dr) * (dx/r)

dx = coords[1, 0] - coords[0, 0]
F_O_x_manual = -(dE_dpsi[0] * dpsi_O_dr_numerical + dE_dpsi[1] * dpsi_H_dr_numerical) * (dx / r)

print("Manual force calculation (using numerical ∂ψ/∂r):")
print(f"  F_O[x] = -(dE_dpsi[0] × ∂psi_O/∂r + dE_dpsi[1] × ∂psi_H/∂r) × (dx/r)")
print(f"        = -({dE_dpsi[0]:.4f} × {dpsi_O_dr_numerical:.6e} + {dE_dpsi[1]:.4f} × {dpsi_H_dr_numerical:.6e}) × {dx/r:.6f}")
print(f"        = {F_O_x_manual:.8f}")
print()

# Get CUDA forces
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
print(f"CUDA F_O[x] = {forces_born[0, 0]:.8f}")
print()

if abs(forces_born[0, 0] - F_O_x_manual) < 1e-5:
    print("✓ CUDA apply_born_forces matches manual calculation")
else:
    print("✗ CUDA apply_born_forces differs from manual")
    ratio = forces_born[0, 0] / F_O_x_manual if abs(F_O_x_manual) > 1e-10 else 0
    print(f"  Ratio: {ratio:.6f}")
