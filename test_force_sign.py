#!/usr/bin/env python3
"""
Carefully trace through force calculation to find sign error
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("TRACING FORCE SIGN CALCULATION")
print("="*80)
print()

coords = np.array([
    [0.0, 0.0, 0.0],      # O at origin
    [0.757, 0.586, 0.0],  # H1 in +x, +y direction
], dtype=np.float64)

charges = np.array([-0.834, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

print("Geometry:")
print(f"  O at: {coords[0]}")
print(f"  H at: {coords[1]}")
print(f"  Direction O→H: {coords[1] - coords[0]}")
r_vec_O_to_H = coords[1] - coords[0]
r = np.linalg.norm(r_vec_O_to_H)
print(f"  Distance: {r:.6f}")
print()

print("Energy test:")
print("  Moving O in +y moves it TOWARD H")
print("  If E increases → force should push O in -y (away from H)")
print()

# Test energy change
delta = 0.001
coords_plus = coords.copy()
coords_plus[0, 1] += delta
born_radii_plus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_plus, radii, b_params, c_params, cutoff)
E_plus, _ = fennol_cuda.gb_compute_energy_forces(coords_plus, charges, born_radii_plus, dielectric, cutoff)

E_0, _ = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

coords_minus = coords.copy()
coords_minus[0, 1] -= delta
born_radii_minus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_minus, radii, b_params, c_params, cutoff)
E_minus, _ = fennol_cuda.gb_compute_energy_forces(coords_minus, charges, born_radii_minus, dielectric, cutoff)

print(f"  E(y=0):     {float(E_0):.6f}")
print(f"  E(y=+{delta}): {float(E_plus):.6f}")
print(f"  E(y=-{delta}): {float(E_minus):.6f}")
print()

dE_dy = (float(E_plus) - float(E_minus)) / (2*delta)
F_y_numerical = -dE_dy

print(f"  ∂E/∂y = {dE_dy:.6f}")
print(f"  F_y = -∂E/∂y = {F_y_numerical:.6f}")
print()

if F_y_numerical < 0:
    print("  ✓ Force is negative (pushes O in -y, away from H)")
else:
    print("  ✗ Force is positive (would push O toward H - wrong!)")
print()

print("="*80)
print("ANALYTICAL FORCE CALCULATION")
print("="*80)
print()

# Compute derivatives
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)

print(f"∂E/∂R_O = {dE_dR[0]:.6f}")
print(f"∂E/∂ψ_O = {dE_dpsi[0]:.6f}")
print()

# Descreening derivative
def compute_dpsi_dr(r, rho_i, rho_j):
    upper = rho_i + rho_j
    lower = abs(rho_i - rho_j)
    if r < lower:
        return 0.0
    elif r < upper:
        return -rho_i / r**3
    else:
        return 0.0

dpsi_O_dr = compute_dpsi_dr(r, radii[0], radii[1])
print(f"∂ψ_O/∂r = {dpsi_O_dr:.6f}")
print()

# Force magnitude
force_mag = -dE_dpsi[0] * dpsi_O_dr
print(f"Force magnitude = -(∂E/∂ψ_O) × (∂ψ_O/∂r)")
print(f"                = -({dE_dpsi[0]:.6f}) × ({dpsi_O_dr:.6f})")
print(f"                = {force_mag:.6f}")
print()

# Direction vector
# When we compute force on O from its interaction with H:
# The direction should be along (O - H) for repulsion
# But our code uses (r_i - r_j) where i=O, j=H
# So direction = coords[O] - coords[H] = -r_vec_O_to_H

direction_in_code = coords[0] - coords[1]  # This is what our kernel uses
print(f"Direction vector in code: r_O - r_H = {direction_in_code}")
print(f"Unit vector: {direction_in_code / r}")
print()

# Force vector
F_vec = force_mag * (direction_in_code / r)
print(f"Force vector = {force_mag:.6f} × {direction_in_code / r}")
print(f"             = [{F_vec[0]:.6f}, {F_vec[1]:.6f}, {F_vec[2]:.6f}]")
print()

print(f"Force F_y (analytical) = {F_vec[1]:.6f}")
print(f"Force F_y (numerical)  = {F_y_numerical:.6f}")
print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

print("The issue is:")
print()
print("1. When ψ_O increases, R_O should INCREASE (less screening)")
print(f"   Check: ∂R_O/∂ψ_O = -R_O² × obcChain = ?")
print()

# Manually compute ∂R/∂ψ
R_O = born_radii[0]
rho_O = radii[0]
psi_O = psi_sum[0]
b_O = b_params[0]
c_O = c_params[0]

psi_2 = psi_O**2
tanh_arg = psi_O - b_O * psi_2 + c_O * psi_2 * psi_O
dtanh_arg_dpsi = 1.0 - 2.0 * b_O * psi_O + 3.0 * c_O * psi_2
tanh_val = np.tanh(tanh_arg)
sech_squared = 1.0 - tanh_val**2

# Our obcChain = ∂(1/R)/∂ψ
obcChain = -sech_squared * dtanh_arg_dpsi / rho_O
print(f"   obcChain = ∂(1/R)/∂ψ = {obcChain:.6f}")

# ∂R/∂ψ = -R² × ∂(1/R)/∂ψ
dR_dpsi = -R_O**2 * obcChain
print(f"   ∂R/∂ψ = -R² × obcChain = {dR_dpsi:.6f}")
print()

if dR_dpsi > 0:
    print("   ✓ R increases with ψ (correct)")
else:
    print("   ✗ R decreases with ψ (wrong!)")
print()

print("2. When r increases (O moves away from H), ψ_O DECREASES (less descreening)")
print(f"   ∂ψ_O/∂r = {dpsi_O_dr:.6f}")

if dpsi_O_dr < 0:
    print("   ✓ ψ decreases with r (correct)")
else:
    print("   ✗ ψ increases with r (wrong!)")
print()

print("3. Chain rule: ∂R_O/∂r = (∂R_O/∂ψ_O) × (∂ψ_O/∂r)")
dR_O_dr = dR_dpsi * dpsi_O_dr
print(f"   ∂R_O/∂r = {dR_dpsi:.6f} × {dpsi_O_dr:.6f} = {dR_O_dr:.6f}")
print()

if dR_O_dr < 0:
    print("   When r increases, R_O decreases")
    print("   This means Born radius gets SMALLER when atoms move apart")
    print("   This makes sense: less screening from neighbors")
else:
    print("   When r increases, R_O increases")
    print("   This is WRONG!")
print()

print("4. Force via chain rule: F = -∂E/∂r = -(∂E/∂R_O) × (∂R_O/∂r)")
print(f"   But we compute: F = -(∂E/∂ψ_O) × (∂ψ_O/∂r)")
print(f"   These should be equivalent since ∂E/∂ψ = (∂E/∂R) × (∂R/∂ψ)")
print()

F_via_R = -dE_dR[0] * dR_O_dr
F_via_psi = -dE_dpsi[0] * dpsi_O_dr

print(f"   F (via R)   = -{dE_dR[0]:.6f} × {dR_O_dr:.6f} = {F_via_R:.6f}")
print(f"   F (via ψ)   = -{dE_dpsi[0]:.6f} × {dpsi_O_dr:.6f} = {F_via_psi:.6f}")
print(f"   Match: {abs(F_via_R - F_via_psi) < 1e-6}")
print()

print("5. Now apply direction:")
print(f"   Force scalar along O→H direction: {F_via_psi:.6f}")
print(f"   Unit vector (O - H)/r = {direction_in_code / r}")
print(f"   Force component in y: {F_via_psi * (direction_in_code[1] / r):.6f}")
print()

print("Expected from numerical gradient: {:.6f}".format(F_y_numerical))
print()

print("="*80)
print("FINDING THE BUG")
print("="*80)
print()

print("Hypothesis: The direction vector might be wrong!")
print()
print("When computing force on O due to H:")
print("  - If force is repulsive, it should push O AWAY from H")
print("  - Direction should be (O - H) normalized")
print("  - But if force_mag is negative, then negative force_mag times (O-H) gives force toward H!")
print()

print(f"force_mag = {force_mag:.6f}")
if force_mag < 0:
    print("Force magnitude is NEGATIVE")
    print("With direction (O-H), this gives force toward H")
    print("But force should be repulsive (away from H)")
    print()
    print("SOLUTION: Either flip sign of force_mag, or flip direction vector!")
