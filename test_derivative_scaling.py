#!/usr/bin/env python3
"""
Test if the ∂R/∂ψ derivative needs to account for the 0.5*rho scaling.

The Born radii calculation now uses:
  psi_scaled = 0.5 * rho * psi

We need to verify that the derivative ∂R/∂ψ includes this scaling factor.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("DERIVATIVE SCALING TEST")
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

# Compute Born radii and psi
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

print("Born radii and descreening:")
print(f"  R_O = {born_radii[0]:.6f} Å (rho={radii[0]:.2f})")
print(f"  R_H = {born_radii[1]:.6f} Å (rho={radii[1]:.2f})")
print(f"  psi_O = {psi_sum[0]:.6f}")
print(f"  psi_H = {psi_sum[1]:.6f}")
print()

# Compute energy
energy, _ = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
print(f"Energy = {float(energy):.10f} kcal/mol")
print()

# Numerical derivative: ∂R/∂ψ by finite difference on psi
delta_psi = 0.0001

# For atom 0 (oxygen)
rho = radii[0]
psi = psi_sum[0]
b = b_params[0]
c = c_params[0]

# Function to compute R from psi
def compute_R_from_psi(psi_val, rho, b, c):
    """Compute Born radius from psi using OBC formula."""
    psi_scaled = 0.5 * rho * psi_val
    psi_2 = psi_scaled * psi_scaled
    psi_3 = psi_2 * psi_scaled
    tanh_arg = psi_scaled - b * psi_2 + c * psi_3
    tanh_val = np.tanh(tanh_arg)
    R_inv = 1.0 / rho - tanh_val / rho
    R = 1.0 / R_inv
    if R < rho:
        R = rho
    return R

# Numerical derivative
R_plus = compute_R_from_psi(psi + delta_psi, rho, b, c)
R_minus = compute_R_from_psi(psi - delta_psi, rho, b, c)
dR_dpsi_numerical = (R_plus - R_minus) / (2 * delta_psi)

print("DERIVATIVE TEST (oxygen):")
print(f"  Current R = {born_radii[0]:.8f}")
print(f"  R(psi + δ) = {R_plus:.8f}")
print(f"  R(psi - δ) = {R_minus:.8f}")
print(f"  ∂R/∂ψ (numerical) = {dR_dpsi_numerical:.8f}")
print()

# Analytical derivative (what CUDA code should compute)
# Given: R = 1 / (1/rho - tanh(psi_scaled - b*psi_scaled² + c*psi_scaled³) / rho)
# where psi_scaled = 0.5 * rho * psi
#
# ∂R/∂ψ = ∂R/∂psi_scaled × ∂psi_scaled/∂ψ
#       = ∂R/∂psi_scaled × (0.5 * rho)

psi_scaled = 0.5 * rho * psi
psi_2 = psi_scaled * psi_scaled
psi_3 = psi_2 * psi_scaled
tanh_arg = psi_scaled - b * psi_2 + c * psi_3

# Derivative of tanh argument w.r.t. psi_scaled
dtanh_arg_dpsi_scaled = 1.0 - 2.0 * b * psi_scaled + 3.0 * c * psi_2

# sech²(x) = 1 - tanh²(x)
tanh_val = np.tanh(tanh_arg)
sech_squared = 1.0 - tanh_val * tanh_val

# ∂(1/R)/∂psi_scaled = -sech²(...) * d(...)/dpsi_scaled / rho
d_invR_dpsi_scaled = -sech_squared * dtanh_arg_dpsi_scaled / rho

# ∂(1/R)/∂ψ = ∂(1/R)/∂psi_scaled × ∂psi_scaled/∂ψ
#           = d_invR_dpsi_scaled × (0.5 * rho)
d_invR_dpsi = d_invR_dpsi_scaled * (0.5 * rho)

# ∂R/∂ψ = -R² × ∂(1/R)/∂ψ
R = born_radii[0]
dR_dpsi_analytical = -R * R * d_invR_dpsi

print("ANALYTICAL DERIVATIVE:")
print(f"  psi_scaled = 0.5 * {rho:.2f} * {psi:.6f} = {psi_scaled:.6f}")
print(f"  dtanh_arg/dpsi_scaled = {dtanh_arg_dpsi_scaled:.8f}")
print(f"  sech²(tanh_arg) = {sech_squared:.8f}")
print(f"  ∂(1/R)/∂psi_scaled = {d_invR_dpsi_scaled:.8f}")
print(f"  ∂(1/R)/∂ψ = {d_invR_dpsi:.8f}  (includes 0.5*rho factor)")
print(f"  ∂R/∂ψ = {dR_dpsi_analytical:.8f}")
print()

# Compare
error = abs(dR_dpsi_analytical - dR_dpsi_numerical)
rel_error = error / abs(dR_dpsi_numerical) * 100

print("COMPARISON:")
print(f"  Numerical:  ∂R/∂ψ = {dR_dpsi_numerical:.8f}")
print(f"  Analytical: ∂R/∂ψ = {dR_dpsi_analytical:.8f}")
print(f"  Error: {error:.2e} ({rel_error:.4f}%)")
print()

if rel_error < 0.01:
    print("✓ Derivative calculation is CORRECT")
else:
    print("✗ Derivative calculation has error")
    print()
    print("The CUDA code born_radius_derivative_wrt_psi() returns:")
    print(f"  ∂(1/R)/∂ψ = {d_invR_dpsi:.8f}")
    print()
    print("This should match the numerical derivative of 1/R w.r.t. ψ")

print()
print("="*80)
print("CHECKING CUDA COMPUTATION")
print("="*80)
print()

# Get what CUDA actually computes
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)

print(f"CUDA dE_dpsi[0] = {dE_dpsi[0]:.8f}")
print()

# Expected: dE_dpsi = dE_dR × ∂R/∂ψ
# But CUDA computes: dE_dpsi = dE_dR × R² × obcChain
# where obcChain = ∂(1/R)/∂ψ
#
# So: dE_dpsi = dE_dR × R² × ∂(1/R)/∂ψ
#             = dE_dR × R² × (-∂R/∂ψ / R²)  [since ∂(1/R)/∂ψ = -∂R/∂ψ / R²]
#             = -dE_dR × ∂R/∂ψ

expected_dE_dpsi = dE_dR[0] * R * R * d_invR_dpsi
print(f"Expected dE_dpsi[0] = {expected_dE_dpsi:.8f}  (using analytical ∂(1/R)/∂ψ)")

cuda_obcChain = dE_dpsi[0] / (dE_dR[0] * R * R)
print(f"CUDA obcChain = {cuda_obcChain:.8f}")
print(f"Expected obcChain = {d_invR_dpsi:.8f}")
print()

if abs(cuda_obcChain - d_invR_dpsi) < 1e-6:
    print("✓ CUDA born_radius_derivative_wrt_psi() is CORRECT")
else:
    print("✗ CUDA born_radius_derivative_wrt_psi() has error")
    ratio = cuda_obcChain / d_invR_dpsi if abs(d_invR_dpsi) > 1e-10 else 0
    print(f"  Ratio: {ratio:.8f}")
    if abs(ratio - 2.0) < 0.01:
        print("  → Missing 0.5 factor in derivative!")
    elif abs(ratio - 0.5) < 0.01:
        print("  → Extra 0.5 factor in derivative!")
