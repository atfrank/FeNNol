#!/usr/bin/env python3
"""
Check: When atoms move apart, does psi increase or decrease?
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("PSI DIRECTION TEST")
print("="*80)
print()

#2-atom system
radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
cutoff = 12.0

# Test at different distances
distances = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0]

print("Distance (Å)  psi_O      psi_H      Comments")
print("-" * 60)

for r in distances:
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
    ], dtype=np.float64)

    _, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

    comment = ""
    if r < abs(radii[0] - radii[1]):
        comment = "complete overlap"
    elif r < radii[0] + radii[1]:
        comment = "partial overlap"
    else:
        comment = "no overlap"

    print(f"{r:7.2f}      {psi_sum[0]:.6f}  {psi_sum[1]:.6f}  {comment}")

print()
print("Observation:")
print("  As r increases (atoms move apart):")
print("  - psi should DECREASE (less overlap → less descreening)")
print("  - Therefore ∂psi/∂r should be NEGATIVE")
print()

# Compute numerical derivative at r = 0.957
r_test = 0.957
delta = 0.0001

coords_center = np.array([[0.0, 0.0, 0.0], [r_test, 0.0, 0.0]], dtype=np.float64)
coords_plus = np.array([[0.0, 0.0, 0.0], [r_test + delta, 0.0, 0.0]], dtype=np.float64)
coords_minus = np.array([[0.0, 0.0, 0.0], [r_test - delta, 0.0, 0.0]], dtype=np.float64)

_, psi_center = fennol_cuda.gb_compute_born_radii_with_psi(coords_center, radii, b_params, c_params, cutoff)
_, psi_plus = fennol_cuda.gb_compute_born_radii_with_psi(coords_plus, radii, b_params, c_params, cutoff)
_, psi_minus = fennol_cuda.gb_compute_born_radii_with_psi(coords_minus, radii, b_params, c_params, cutoff)

dpsi_O_dr = (psi_plus[0] - psi_minus[0]) / (2 * delta)
dpsi_H_dr = (psi_plus[1] - psi_minus[1]) / (2 * delta)

print(f"At r = {r_test:.3f} Å:")
print(f"  psi_O(r - δ) = {psi_minus[0]:.8f}")
print(f"  psi_O(r    ) = {psi_center[0]:.8f}")
print(f"  psi_O(r + δ) = {psi_plus[0]:.8f}")
print()
print(f"  Change: {psi_plus[0] - psi_minus[0]:.8e}")
print(f"  ∂psi_O/∂r = {dpsi_O_dr:.8e}")
print()

if dpsi_O_dr < 0:
    print("✓ ∂psi_O/∂r is NEGATIVE (correct: psi decreases as r increases)")
else:
    print("✗ ∂psi_O/∂r is POSITIVE (wrong: psi INCREASES as r increases?!)")
    print()
    print("This is backwards! As atoms move apart, psi should DECREASE.")
    print("There must be a sign error in the descreening integral.")

print()
print("="*80)
print("CHECK DESCREENING INTEGRAL FORMULA")
print("="*80)
print()

# The descreening integral should increase when atoms are closer
# Let's check the HCT formula directly

def hct_integral(r, rho_i, rho_j):
    """HCT descreening integral from OpenMM."""
    if r < 0.001:
        return 0.0

    upper_limit = rho_i + rho_j
    lower_limit = abs(rho_i - rho_j)

    if r < lower_limit:
        # Use clamped formula
        r_clamped = lower_limit
        s_j = rho_j
        abs_diff = abs(r_clamped - s_j)
        lower_bound = max(rho_i, abs_diff)
        l_ij = 1.0 / lower_bound
        u_ij = 1.0 / (r_clamped + s_j)

        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij
        s_j2 = s_j * s_j
        r_inv = 1.0 / r_clamped
        ratio = np.log(u_ij / l_ij)

        term = l_ij - u_ij + 0.25 * r_clamped * (u_ij2 - l_ij2) + \
               0.5 * r_inv * ratio + \
               0.25 * s_j2 * r_inv * (l_ij2 - u_ij2)

        return term

    elif r < upper_limit:
        # Partial overlap
        s_j = rho_j
        abs_diff = abs(r - s_j)
        lower_bound = max(rho_i, abs_diff)
        l_ij = 1.0 / lower_bound
        u_ij = 1.0 / (r + s_j)

        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij
        s_j2 = s_j * s_j
        r_inv = 1.0 / r
        ratio = np.log(u_ij / l_ij)

        term = l_ij - u_ij + 0.25 * r * (u_ij2 - l_ij2) + \
               0.5 * r_inv * ratio + \
               0.25 * s_j2 * r_inv * (l_ij2 - u_ij2)

        return term
    else:
        return 0.0

# Test integral at different r
print("HCT integral values:")
for r in [0.8, 0.9, 0.95, 0.957, 1.0, 1.1, 1.2]:
    I_val = hct_integral(r, radii[0], radii[1])
    print(f"  I(r={r:.3f}) = {I_val:.8f}")

print()
print("If I increases with r, then ∂I/∂r > 0")
print("If I decreases with r, then ∂I/∂r < 0")
