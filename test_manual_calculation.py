#!/usr/bin/env python3
"""
Phase 1: Manual calculation validation
Extract all intermediate values and calculate force by hand
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

# Single water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O (atom 0)
    [0.757, 0.586, 0.0],  # H1 (atom 1)
    [-0.757, 0.586, 0.0], # H2 (atom 2)
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

print("="*80)
print("PHASE 1: MANUAL CALCULATION VALIDATION")
print("="*80)
print()

# Step 1: Compute Born radii
print("Step 1: Computing Born radii and psi...")
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
print(f"  Born radii: {born_radii}")
print(f"  Psi sum:    {psi_sum}")
print()

# Step 2: Compute energy and direct forces
print("Step 2: Computing energy and direct forces...")
energy, direct_forces = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
direct_forces = direct_forces.reshape(-1, 3)
print(f"  Energy: {float(energy):.10f} kcal/mol")
print(f"  Direct forces:")
for i in range(3):
    print(f"    Atom {i}: [{direct_forces[i,0]:12.6f}, {direct_forces[i,1]:12.6f}, {direct_forces[i,2]:12.6f}]")
print()

# Step 3: Compute dE/dR
print("Step 3: Computing ∂E/∂R...")
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
print(f"  ∂E/∂R: {dE_dR}")
print()

# Step 4: Convert ∂E/∂R → ∂E/∂ψ
print("Step 4: Converting ∂E/∂R → ∂E/∂ψ...")
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
print(f"  ∂E/∂ψ: {dE_dpsi}")
print()

# Step 5: Apply Born forces
print("Step 5: Applying Born forces...")
born_forces = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
born_forces = born_forces.reshape(-1, 3)
print(f"  Born forces:")
for i in range(3):
    print(f"    Atom {i}: [{born_forces[i,0]:12.6f}, {born_forces[i,1]:12.6f}, {born_forces[i,2]:12.6f}]")
print()

# Step 6: Total forces
total_forces = direct_forces + born_forces
print("Step 6: Total forces (direct + Born):")
for i in range(3):
    print(f"    Atom {i}: [{total_forces[i,0]:12.6f}, {total_forces[i,1]:12.6f}, {total_forces[i,2]:12.6f}]")
print()

print("="*80)
print("MANUAL CALCULATION FOR PAIR (0, 1)")
print("="*80)
print()

# Extract values
R_O = born_radii[0]
R_H = born_radii[1]
rho_O = radii[0]
rho_H = radii[1]
psi_O = psi_sum[0]
psi_H = psi_sum[1]
q_O = charges[0]
q_H = charges[1]
b_O = b_params[0]
c_O = c_params[0]
b_H = b_params[1]
c_H = c_params[1]

print(f"Atom 0 (O): R={R_O:.6f}, ρ={rho_O:.6f}, ψ={psi_O:.6f}, q={q_O:.6f}")
print(f"Atom 1 (H): R={R_H:.6f}, ρ={rho_H:.6f}, ψ={psi_H:.6f}, q={q_H:.6f}")
print()

# Geometry
dx = coords[0,0] - coords[1,0]  # O - H
dy = coords[0,1] - coords[1,1]
dz = coords[0,2] - coords[1,2]
r = np.sqrt(dx**2 + dy**2 + dz**2)
r_inv = 1.0 / r

print(f"Distance vector (O→H): ({dx:.6f}, {dy:.6f}, {dz:.6f})")
print(f"Distance r = {r:.6f} Å")
print(f"Unit vector: ({dx*r_inv:.6f}, {dy*r_inv:.6f}, {dz*r_inv:.6f})")
print()

# Manual calculation of ∂E/∂R
print("--- Manual: ∂E/∂R calculation ---")
COULOMB = 332.0636
gb_factor = -0.5 * (1.0 - 1.0/dielectric) * COULOMB
print(f"gb_factor = {gb_factor:.6f}")

# Self-energy derivative for O
dE_dR_self_O = gb_factor * (-q_O**2 / R_O**2)
print(f"∂E_self/∂R_O = {dE_dR_self_O:.6f}")

# Pair contribution for O from pair (O,H)
# f_GB = sqrt(r² + R_O*R_H*exp(-r²/(4*R_O*R_H)))
RiRj = R_O * R_H
exp_term = np.exp(-r**2 / (4*RiRj))
f_gb = np.sqrt(r**2 + RiRj * exp_term)
print(f"f_GB = {f_gb:.6f}")

# ∂f_GB/∂R_O
df_gb_dRi = 0.5 / f_gb * R_H * exp_term * (1.0 - r**2 / (4 * R_O * RiRj))
print(f"∂f_GB/∂R_O = {df_gb_dRi:.6f}")

# Pairwise contribution
dE_dR_pair_O = gb_factor * q_O * q_H * (-1.0 / f_gb**2) * df_gb_dRi
print(f"∂E_pair/∂R_O = {dE_dR_pair_O:.6f}")

dE_dR_total_O_manual = dE_dR_self_O + dE_dR_pair_O
print(f"∂E/∂R_O (manual) = {dE_dR_total_O_manual:.6f}")
print(f"∂E/∂R_O (kernel) = {dE_dR[0]:.6f}")
print(f"Match: {np.abs(dE_dR_total_O_manual - dE_dR[0]) < 1e-6}")
print()

# Manual calculation of obcChain = ∂(1/R)/∂ψ
print("--- Manual: obcChain = ∂(1/R)/∂ψ calculation ---")
psi_2 = psi_O**2
psi_3 = psi_2 * psi_O
tanh_arg = psi_O - b_O * psi_2 + c_O * psi_3
print(f"tanh_arg = ψ - b*ψ² + c*ψ³ = {tanh_arg:.6f}")

dtanh_arg_dpsi = 1.0 - 2.0 * b_O * psi_O + 3.0 * c_O * psi_2
print(f"d(tanh_arg)/dψ = {dtanh_arg_dpsi:.6f}")

tanh_val = np.tanh(tanh_arg)
sech_squared = 1.0 - tanh_val**2
print(f"tanh(tanh_arg) = {tanh_val:.6f}")
print(f"sech²(tanh_arg) = {sech_squared:.6f}")

obcChain_manual = -sech_squared * dtanh_arg_dpsi / rho_O
print(f"obcChain (manual) = -sech² * d(tanh_arg)/dψ / ρ = {obcChain_manual:.6f}")
print()

# Manual calculation of ∂E/∂ψ
print("--- Manual: ∂E/∂ψ = (∂E/∂R) × (-R²) × obcChain ---")
dE_dpsi_manual_O = -dE_dR[0] * R_O**2 * obcChain_manual
print(f"∂E/∂ψ_O (manual) = -{dE_dR[0]:.6f} × {R_O**2:.6f} × {obcChain_manual:.6f}")
print(f"                  = {dE_dpsi_manual_O:.6f}")
print(f"∂E/∂ψ_O (kernel) = {dE_dpsi[0]:.6f}")
print(f"Match: {np.abs(dE_dpsi_manual_O - dE_dpsi[0]) < 1e-6}")
print()

# Manual calculation of ∂ψ/∂r (descreening integral derivative)
print("--- Manual: ∂ψ/∂r calculation ---")
upper_limit = rho_O + rho_H
lower_limit = np.abs(rho_O - rho_H)
print(f"Limits: lower={lower_limit:.6f}, upper={upper_limit:.6f}, r={r:.6f}")

if r < lower_limit:
    dpsi_O_dr_manual = 0.0
    print("Region: Complete overlap → ∂ψ/∂r = 0")
elif r < upper_limit:
    dpsi_O_dr_manual = -rho_O / r**3
    print(f"Region: Partial overlap → ∂ψ/∂r = -ρ_O/r³ = {dpsi_O_dr_manual:.6f}")
else:
    dpsi_O_dr_manual = 0.0
    print("Region: No overlap → ∂ψ/∂r = 0")

# Similarly for H
if r < lower_limit:
    dpsi_H_dr_manual = 0.0
elif r < upper_limit:
    dpsi_H_dr_manual = -rho_H / r**3
    print(f"           ∂ψ_H/∂r = -ρ_H/r³ = {dpsi_H_dr_manual:.6f}")
else:
    dpsi_H_dr_manual = 0.0
print()

# Manual calculation of force magnitude
print("--- Manual: Force magnitude calculation ---")
print(f"∂E/∂ψ_O = {dE_dpsi[0]:.6f}")
print(f"∂E/∂ψ_H = {dE_dpsi[1]:.6f}")
print(f"∂ψ_O/∂r = {dpsi_O_dr_manual:.6f}")
print(f"∂ψ_H/∂r = {dpsi_H_dr_manual:.6f}")
print()

force_mag_from_O = -dE_dpsi[0] * dpsi_O_dr_manual
force_mag_from_H = -dE_dpsi[1] * dpsi_H_dr_manual
force_mag_total = force_mag_from_O + force_mag_from_H

print(f"Force from ∂E/∂ψ_O: -{dE_dpsi[0]:.6f} × {dpsi_O_dr_manual:.6f} = {force_mag_from_O:.6f}")
print(f"Force from ∂E/∂ψ_H: -{dE_dpsi[1]:.6f} × {dpsi_H_dr_manual:.6f} = {force_mag_from_H:.6f}")
print(f"Total force magnitude: {force_mag_total:.6f}")
print()

# Force in y-direction on atom O
F_O_y_manual = force_mag_total * dy * r_inv
print(f"Force on O (y-component): {force_mag_total:.6f} × {dy:.6f} × {r_inv:.6f}")
print(f"                        = {F_O_y_manual:.6f} kcal/(mol·Å)")
print()

print(f"Born force on O (y) from kernel: {born_forces[0,1]:.6f} kcal/(mol·Å)")
print(f"Ratio (kernel / manual): {born_forces[0,1] / F_O_y_manual:.3f}")
print()

print("="*80)
print("COMPARISON WITH NUMERICAL GRADIENT")
print("="*80)
print()

# Numerical gradient
delta = 1e-5
coords_plus = coords.copy()
coords_plus[0, 1] += delta
born_radii_plus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_plus, radii, b_params, c_params, cutoff)
E_plus, _ = fennol_cuda.gb_compute_energy_forces(coords_plus, charges, born_radii_plus, dielectric, cutoff)

coords_minus = coords.copy()
coords_minus[0, 1] -= delta
born_radii_minus, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_minus, radii, b_params, c_params, cutoff)
E_minus, _ = fennol_cuda.gb_compute_energy_forces(coords_minus, charges, born_radii_minus, dielectric, cutoff)

F_numerical = -(float(E_plus) - float(E_minus)) / (2 * delta)

print(f"Numerical gradient F_y = {F_numerical:.6f} kcal/(mol·Å)")
print(f"Total force (kernel)   = {total_forces[0,1]:.6f} kcal/(mol·Å)")
print(f"Manual calculation     = {F_O_y_manual:.6f} kcal/(mol·Å)")
print()

print(f"Ratio (kernel / numerical): {total_forces[0,1] / F_numerical:.3f}×")
print(f"Ratio (manual / numerical): {F_O_y_manual / F_numerical:.3f}×")
print()

print("="*80)
print("KEY FINDINGS")
print("="*80)
print()
print(f"1. Manual calculation gives: F_y = {F_O_y_manual:.3f}")
print(f"2. Numerical gradient gives: F_y = {F_numerical:.3f}")
print(f"3. Error factor: {F_O_y_manual / F_numerical:.1f}×")
print()
print("Next: Identify where the ~15× factor comes from!")
