#!/usr/bin/env python3
"""
Check which pairs are being processed and their contributions
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

born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)

print("Manual calculation of Born forces for each pair:")
print()

def compute_dpsi_dr(r, rho_i, rho_j):
    """Compute ∂ψ_i/∂r"""
    upper_limit = rho_i + rho_j
    lower_limit = abs(rho_i - rho_j)

    if r < lower_limit:
        return 0.0
    elif r < upper_limit:
        return -rho_i / r**3
    else:
        return 0.0

# Process all pairs
total_force_O = np.zeros(3)

print("Atom 0 (O) processing its neighbors:")
for j in [1, 2]:
    r_vec = coords[0] - coords[j]
    r = np.linalg.norm(r_vec)
    r_unit = r_vec / r

    dpsi_0_dr = compute_dpsi_dr(r, radii[0], radii[j])
    force_mag = -dE_dpsi[0] * dpsi_0_dr
    force_vec = force_mag * r_unit

    print(f"  Pair (0,{j}): r={r:.6f}, dpsi_O_dr={dpsi_0_dr:.6f}, force_mag={force_mag:.6f}")
    print(f"            Force vector: [{force_vec[0]:.6f}, {force_vec[1]:.6f}, {force_vec[2]:.6f}]")

    total_force_O += force_vec

print()
print(f"Total Born force on O (manual): [{total_force_O[0]:.6f}, {total_force_O[1]:.6f}, {total_force_O[2]:.6f}]")
print()

# Now check what kernel gives
born_forces = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
born_forces = born_forces.reshape(-1, 3)
print(f"Total Born force on O (kernel): [{born_forces[0,0]:.6f}, {born_forces[0,1]:.6f}, {born_forces[0,2]:.6f}]")
print()

print(f"Ratio (kernel/manual) for y-component: {born_forces[0,1] / total_force_O[1]:.3f}")
