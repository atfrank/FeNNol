#!/usr/bin/env python3
"""
Test the HCT descreening derivative formula by comparing with numerical derivative
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("TESTING HCT DESCREENING DERIVATIVE FORMULA")
print("="*80)
print()

# Simple 2-atom system
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H
], dtype=np.float64)

radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
cutoff = 12.0

# Compute psi at current position
_, psi_0 = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
print(f"Current psi[0] = {psi_0[0]:.10f}")
print()

# Perturb H atom slightly in x direction and recompute
delta = 0.0001
coords_plus = coords.copy()
coords_plus[1, 0] += delta

_, psi_plus = fennol_cuda.gb_compute_born_radii_with_psi(coords_plus, radii, b_params, c_params, cutoff)

coords_minus = coords.copy()
coords_minus[1, 0] -= delta

_, psi_minus = fennol_cuda.gb_compute_born_radii_with_psi(coords_minus, radii, b_params, c_params, cutoff)

# Numerical derivative
dpsi_dr_numerical = (psi_plus[0] - psi_minus[0]) / (2*delta)

print(f"Numerical ∂ψ[0]/∂r = {dpsi_dr_numerical:.10f}")
print()

# Now compute analytical derivative using our kernel
# The kernel computes force = -dE_dpsi * dpsi_dr
# We can extract dpsi_dr by setting dE_dpsi = 1.0

# Create dummy dE_dpsi array
dE_dpsi = np.array([1.0, 0.0], dtype=np.float64)

# Call apply_born_forces
forces = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

# Force on atom 0 from atom 1 is: -dE_dpsi[0] * dpsi_0_dr * (r_0 - r_1)/r
# We set dE_dpsi[0] = 1.0, so force = -dpsi_0_dr * (r_0 - r_1)/r

r_vec = coords[0] - coords[1]
r = np.linalg.norm(r_vec)
r_hat = r_vec / r

print(f"Distance r = {r:.10f}")
print(f"Direction (O - H) = {r_vec}")
print(f"Unit vector = {r_hat}")
print()

# Force = -dpsi_0_dr * r_hat
# So dpsi_0_dr = -Force / r_hat

# The x-component: F_x = -dpsi_0_dr * r_hat_x
# dpsi_0_dr = -F_x / r_hat_x

dpsi_dr_analytical = -forces[0, 0] / r_hat[0]

print(f"Force on O: {forces[0]}")
print(f"Analytical ∂ψ[0]/∂r = -F_x / r_hat_x = {dpsi_dr_analytical:.10f}")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print(f"Numerical:  {dpsi_dr_numerical:.10f}")
print(f"Analytical: {dpsi_dr_analytical:.10f}")
print(f"Ratio: {dpsi_dr_analytical / dpsi_dr_numerical:.6f}")
print()

if abs(dpsi_dr_analytical / dpsi_dr_numerical - 1.0) < 0.01:
    print("✓ HCT derivative formula is CORRECT!")
else:
    print(f"✗ HCT derivative formula has error: {abs(dpsi_dr_analytical / dpsi_dr_numerical - 1.0) * 100:.1f}%")
