#!/usr/bin/env python3
"""
Detailed diagnostic of 3-atom water molecule to find the source of error
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("3-ATOM DIAGNOSTIC - Water Molecule")
print("="*80)
print()

# Single water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H1
    [-0.757, 0.586, 0.0], # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Compute Born radii and psi
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

print("Step 1: Born Radii and Psi")
print("-" * 80)
print(f"Born radii: {born_radii}")
print(f"Psi sum:    {psi_sum}")
print()

# Manually compute psi sums using HCT formula
def compute_hct_integral(r, rho_i, rho_j):
    """Compute HCT descreening integral"""
    upper = rho_i + rho_j
    lower = abs(rho_i - rho_j)

    if r < lower:
        r_use = lower
    elif r < upper:
        r_use = r
    else:
        return 0.0

    s_j = rho_j
    abs_diff = abs(r_use - s_j)
    lower_bound = max(rho_i, abs_diff)
    l_ij = 1.0 / lower_bound
    u_ij = 1.0 / (r_use + s_j)

    l_ij2 = l_ij**2
    u_ij2 = u_ij**2
    s_j2 = s_j**2
    r_inv = 1.0 / r_use
    ratio = np.log(u_ij / l_ij)

    term = l_ij - u_ij + 0.25 * r_use * (u_ij2 - l_ij2) + \
           0.5 * r_inv * ratio + \
           0.25 * s_j2 * r_inv * (l_ij2 - u_ij2)

    return term

# Compute distances
r01 = np.linalg.norm(coords[0] - coords[1])
r02 = np.linalg.norm(coords[0] - coords[2])
r12 = np.linalg.norm(coords[1] - coords[2])

print("Distances:")
print(f"  r(O-H1) = {r01:.6f}")
print(f"  r(O-H2) = {r02:.6f}")
print(f"  r(H1-H2) = {r12:.6f}")
print()

# Manually compute psi for each atom
psi_0_manual = compute_hct_integral(r01, radii[0], radii[1]) + \
               compute_hct_integral(r02, radii[0], radii[2])

psi_1_manual = compute_hct_integral(r01, radii[1], radii[0]) + \
               compute_hct_integral(r12, radii[1], radii[2])

psi_2_manual = compute_hct_integral(r02, radii[2], radii[0]) + \
               compute_hct_integral(r12, radii[2], radii[1])

print("Manual Psi Calculation:")
print(f"  psi[0] (O):  manual={psi_0_manual:.10f}, kernel={psi_sum[0]:.10f}, match={abs(psi_0_manual - psi_sum[0]) < 1e-8}")
print(f"  psi[1] (H1): manual={psi_1_manual:.10f}, kernel={psi_sum[1]:.10f}, match={abs(psi_1_manual - psi_sum[1]) < 1e-8}")
print(f"  psi[2] (H2): manual={psi_2_manual:.10f}, kernel={psi_sum[2]:.10f}, match={abs(psi_2_manual - psi_sum[2]) < 1e-8}")
print()

# Compute Born radii manually using OBC formula
def compute_born_radius(rho_i, psi_i, b, c):
    """Compute Born radius from psi using OBC formula"""
    psi_2 = psi_i**2
    psi_3 = psi_2 * psi_i
    tanh_arg = psi_i - b * psi_2 + c * psi_3
    tanh_val = np.tanh(tanh_arg)
    inv_R = (1.0 / rho_i) - (tanh_val / rho_i)
    R = 1.0 / inv_R
    return R

R_0_manual = compute_born_radius(radii[0], psi_sum[0], b_params[0], c_params[0])
R_1_manual = compute_born_radius(radii[1], psi_sum[1], b_params[1], c_params[1])
R_2_manual = compute_born_radius(radii[2], psi_sum[2], b_params[2], c_params[2])

print("Manual Born Radii Calculation:")
print(f"  R[0] (O):  manual={R_0_manual:.10f}, kernel={born_radii[0]:.10f}, match={abs(R_0_manual - born_radii[0]) < 1e-8}")
print(f"  R[1] (H1): manual={R_1_manual:.10f}, kernel={born_radii[1]:.10f}, match={abs(R_1_manual - born_radii[1]) < 1e-8}")
print(f"  R[2] (H2): manual={R_2_manual:.10f}, kernel={born_radii[2]:.10f}, match={abs(R_2_manual - born_radii[2]) < 1e-8}")
print()

print("="*80)
print("Step 2: Energy and Direct Forces")
print("-" * 80)

energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
print(f"Energy: {float(energy):.10f}")
print()

# Manually compute energy
COULOMB = 332.0636
gb_factor = -0.5 * (1.0 - 1.0/dielectric) * COULOMB

# Self energies
E_self = 0.0
for i in range(3):
    E_self_i = gb_factor * charges[i]**2 / born_radii[i]
    E_self += E_self_i
    print(f"E_self[{i}] = {E_self_i:.6f}")

print()

# Pair energies
E_pair_total = 0.0
for i in range(3):
    for j in range(i+1, 3):
        r_ij = np.linalg.norm(coords[i] - coords[j])
        R_i = born_radii[i]
        R_j = born_radii[j]

        RiRj = R_i * R_j
        exp_term = np.exp(-r_ij**2 / (4*RiRj))
        f_gb = np.sqrt(r_ij**2 + RiRj * exp_term)

        E_pair_ij = gb_factor * charges[i] * charges[j] / f_gb
        E_pair_total += E_pair_ij
        print(f"E_pair[{i},{j}] = {E_pair_ij:.6f} (r={r_ij:.6f}, f_GB={f_gb:.6f})")

print()
E_total_manual = E_self + E_pair_total
print(f"E_total (manual) = {E_total_manual:.10f}")
print(f"E_total (kernel) = {float(energy):.10f}")
print(f"Match: {abs(E_total_manual - float(energy)) < 1e-6}")
print()

print("="*80)
print("Step 3: Force Analysis (focus on O atom)")
print("-" * 80)

# Compute dE/dR
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
print(f"∂E/∂R: {dE_dR}")
print()

# Manually compute dE/dR for oxygen
# Self-energy contribution
dE_dR_self_0 = gb_factor * (-charges[0]**2 / born_radii[0]**2)
print(f"∂E_self/∂R[0] = {dE_dR_self_0:.6f}")

# Pair contributions
dE_dR_pair_0 = 0.0
for j in [1, 2]:
    r_ij = np.linalg.norm(coords[0] - coords[j])
    R_i = born_radii[0]
    R_j = born_radii[j]

    RiRj = R_i * R_j
    exp_arg = -r_ij**2 / (4*RiRj)
    exp_term = np.exp(exp_arg)
    f_gb = np.sqrt(r_ij**2 + RiRj * exp_term)

    # df_GB/dR_i (correct formula using chain rule through α² = R_i*R_j)
    df_gb_dRi = 0.5 / f_gb * R_j * exp_term * (1.0 + r_ij**2 / (4 * RiRj))

    # dE_pair/dR_i
    dE_pair_dRi = gb_factor * charges[0] * charges[j] * (-1.0 / f_gb**2) * df_gb_dRi
    dE_dR_pair_0 += dE_pair_dRi

    print(f"  Pair (0,{j}): df_GB/dR_0={df_gb_dRi:.6f}, dE/dR_0={dE_pair_dRi:.6f}")

print()
dE_dR_0_manual = dE_dR_self_0 + dE_dR_pair_0
print(f"∂E/∂R[0] (manual) = {dE_dR_0_manual:.6f}")
print(f"∂E/∂R[0] (kernel) = {dE_dR[0]:.6f}")
print(f"Match: {abs(dE_dR_0_manual - dE_dR[0]) < 0.01}")
print()

# Convert to dE/dpsi
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
print(f"∂E/∂ψ: {dE_dpsi}")
print()

# Apply Born forces
forces_born_deriv = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

print("Direct forces:")
print(f"  F[0] (O)  = {forces_direct[0]}")
print(f"  F[1] (H1) = {forces_direct[1]}")
print(f"  F[2] (H2) = {forces_direct[2]}")
print()

print("Born derivative forces:")
print(f"  F[0] (O)  = {forces_born_deriv[0]}")
print(f"  F[1] (H1) = {forces_born_deriv[1]}")
print(f"  F[2] (H2) = {forces_born_deriv[2]}")
print()

forces_total = forces_direct + forces_born_deriv
print("TOTAL forces:")
print(f"  F[0] (O)  = {forces_total[0]}")
print(f"  F[1] (H1) = {forces_total[1]}")
print(f"  F[2] (H2) = {forces_total[2]}")
print()

# Numerical gradient for comparison
delta = 0.0001

def compute_energy_full(coords_temp):
    born_radii_temp, _ = fennol_cuda.gb_compute_born_radii_with_psi(coords_temp, radii, b_params, c_params, cutoff)
    E_temp, _ = fennol_cuda.gb_compute_energy_forces(coords_temp, charges, born_radii_temp, dielectric, cutoff)
    return float(E_temp)

forces_numerical = np.zeros_like(coords)
for atom in range(3):
    for dim in range(3):
        coords_plus = coords.copy()
        coords_plus[atom, dim] += delta
        coords_minus = coords.copy()
        coords_minus[atom, dim] -= delta

        E_plus = compute_energy_full(coords_plus)
        E_minus = compute_energy_full(coords_minus)

        forces_numerical[atom, dim] = -(E_plus - E_minus) / (2*delta)

print("Numerical gradient:")
print(f"  F[0] (O)  = {forces_numerical[0]}")
print(f"  F[1] (H1) = {forces_numerical[1]}")
print(f"  F[2] (H2) = {forces_numerical[2]}")
print()

print("="*80)
print("ERROR ANALYSIS")
print("="*80)
print()

for i in range(3):
    error = forces_total[i] - forces_numerical[i]
    rel_error = np.linalg.norm(error) / max(np.linalg.norm(forces_numerical[i]), 1e-10) * 100
    print(f"Atom {i}: error={error}, relative={rel_error:.2f}%")

    # Break down by component
    error_direct = forces_direct[i] - forces_numerical[i]
    error_born = forces_born_deriv[i]

    print(f"  If direct forces were perfect: error would be {-error_born}")
    print(f"  Current error in direct forces: {error_direct}")
    print()
