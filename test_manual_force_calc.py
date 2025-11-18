#!/usr/bin/env python3
"""
Manually compute what the Born radii force should be
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("MANUAL FORCE CALCULATION")
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

# Get Born radii and psi
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, _ = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

print(f"Born radii: {born_radii}")
print(f"Psi sum: {psi_sum}")
print(f"Energy: {float(energy):.10f}")
print()

# Compute ∂E/∂R
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
print(f"∂E/∂R: {dE_dR}")
print()

# Compute ∂E/∂ψ
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
print(f"∂E/∂ψ: {dE_dpsi}")
print()

# Now manually compute ∂ψ/∂r using HCT formula
r_vec = coords[1] - coords[0]
r = np.linalg.norm(r_vec)

rho_i = radii[0]
rho_j = radii[1]
s_j = rho_j

# Compute l_ij and u_ij
abs_diff = abs(r - s_j)
lower_bound = max(rho_i, abs_diff)
l_ij = 1.0 / lower_bound
u_ij = 1.0 / (r + s_j)

l_ij2 = l_ij**2
u_ij2 = u_ij**2
s_j2 = s_j**2
r2_inv = 1.0 / r**2

# HCT derivative
t3 = 0.125 * (1.0 + s_j2 * r2_inv) * (l_ij2 - u_ij2) + 0.25 * np.log(u_ij / l_ij) * r2_inv
dpsi_dr = t3 / r

print(f"r = {r:.10f}")
print(f"rho_i = {rho_i}, rho_j = {rho_j}")
print(f"l_ij = {l_ij:.10f}, u_ij = {u_ij:.10f}")
print(f"t3 = {t3:.10f}")
print(f"∂ψ/∂r = {dpsi_dr:.10f}")
print()

# Force from ψ₀ changing
de = dE_dpsi[0] * dpsi_dr / r
force_vec = de * r_vec

print(f"de = (∂E/∂ψ₀) × (∂ψ₀/∂r) / r")
print(f"   = {dE_dpsi[0]:.6f} × {dpsi_dr:.6f} / {r:.6f}")
print(f"   = {de:.6f}")
print()

print(f"Force on O from ψ₀: -{de:.6f} × {r_vec} = -{force_vec}")
print(f"Force on H from ψ₀: +{de:.6f} × {r_vec} = +{force_vec}")
print()

# Now compute force from ψ₁ changing
# When ψ₁ changes, we need ∂ψ₁/∂r where ψ₁ is the descreening on atom H from atom O

# For atom 1 (H), descreening from atom 0 (O):
rho_i_1 = radii[1]  # H radius
rho_j_1 = radii[0]  # O radius (the "j" in the pair from H's perspective)
s_j_1 = rho_j_1

abs_diff_1 = abs(r - s_j_1)
lower_bound_1 = max(rho_i_1, abs_diff_1)
l_ij_1 = 1.0 / lower_bound_1
u_ij_1 = 1.0 / (r + s_j_1)

l_ij2_1 = l_ij_1**2
u_ij2_1 = u_ij_1**2
s_j2_1 = s_j_1**2

t3_1 = 0.125 * (1.0 + s_j2_1 * r2_inv) * (l_ij2_1 - u_ij2_1) + 0.25 * np.log(u_ij_1 / l_ij_1) * r2_inv
dpsi_dr_1 = t3_1 / r

print(f"For atom H (computing ∂ψ₁/∂r):")
print(f"rho_i = {rho_i_1}, rho_j = {rho_j_1}")
print(f"l_ij = {l_ij_1:.10f}, u_ij = {u_ij_1:.10f}")
print(f"t3 = {t3_1:.10f}")
print(f"∂ψ₁/∂r = {dpsi_dr_1:.10f}")
print()

# Force from ψ₁ changing
de_1 = dE_dpsi[1] * dpsi_dr_1 / r
force_vec_1 = de_1 * r_vec

print(f"de_1 = (∂E/∂ψ₁) × (∂ψ₁/∂r) / r")
print(f"     = {dE_dpsi[1]:.6f} × {dpsi_dr_1:.6f} / {r:.6f}")
print(f"     = {de_1:.6f}")
print()

print(f"Force on O from ψ₁: -{de_1:.6f} × {r_vec} = -{force_vec_1}")
print(f"Force on H from ψ₁: +{de_1:.6f} × {r_vec} = +{force_vec_1}")
print()

# Total force
total_force_O = -(force_vec + force_vec_1)
total_force_H = +(force_vec + force_vec_1)

print("="*80)
print("TOTAL FORCES (manual calculation)")
print("="*80)
print(f"F_O = {total_force_O}")
print(f"F_H = {total_force_H}")
print()

# Compare with multi-pass
forces_multipass = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
print("Multi-pass forces:")
print(f"F_O = {forces_multipass[0]}")
print(f"F_H = {forces_multipass[1]}")
print()

print("Ratio (multipass / manual):")
print(f"  O: {np.linalg.norm(forces_multipass[0]) / np.linalg.norm(total_force_O):.6f}")
