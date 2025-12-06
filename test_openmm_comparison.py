#!/usr/bin/env python3
"""
Compare our GB force implementation with OpenMM's logic step-by-step
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("OpenMM vs FeNNol GB Force Comparison")
print("="*80)
print()

# 3-atom water
coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]], dtype=np.float64)
charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# Get Born radii
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

print("Step 1: Born Radii")
print(f"  R[0] = {born_radii[0]:.10f}")
print(f"  R[1] = {born_radii[1]:.10f}")
print(f"  R[2] = {born_radii[2]:.10f}")
print()

# Manually compute "bornForces" array (OpenMM's intermediate)
# This is dE/dpsi in our notation
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)

print("Step 2: Born Force Intermediate (dE/dpsi)")
print(f"  bornForces[0] = {dE_dpsi[0]:.10f}")
print(f"  bornForces[1] = {dE_dpsi[1]:.10f}")
print(f"  bornForces[2] = {dE_dpsi[2]:.10f}")
print()

# Now manually compute the Cartesian forces following OpenMM's logic
print("Step 3: Convert to Cartesian Forces (OpenMM's second loop)")
print("-"*80)

def compute_hct_derivative(r, rho_i, rho_j):
    """Compute HCT descreening derivative (OpenMM's t3 term)"""
    upper_limit = rho_i + rho_j
    lower_limit = abs(rho_i - rho_j)

    if r < lower_limit:
        return 0.0
    elif r < upper_limit:
        s_j = rho_j
        abs_diff = abs(r - s_j)
        lower_bound = max(rho_i, abs_diff)
        l_ij = 1.0 / lower_bound
        u_ij = 1.0 / (r + s_j)

        l_ij2 = l_ij**2
        u_ij2 = u_ij**2
        s_j2 = s_j**2
        r_inv = 1.0 / r
        r2_inv = r_inv**2

        # OpenMM's t3 formula
        t3 = 0.125 * (1.0 + s_j2 * r2_inv) * (l_ij2 - u_ij2) + 0.25 * np.log(u_ij / l_ij) * r2_inv
        return t3
    else:
        return 0.0

# Manual force calculation following OpenMM
forces_manual = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        if i == j:
            continue

        # Displacement vector (j - i) [OpenMM convention]
        dr = coords[j] - coords[i]
        r = np.linalg.norm(dr)
        r_inv = 1.0 / r

        # Compute HCT derivative
        t3 = compute_hct_derivative(r, radii[i], radii[j])

        # Force magnitude (OpenMM: de = bornForces[i] * t3 * r_inv)
        de = dE_dpsi[i] * t3 * r_inv

        # Force vector
        force_vec = de * dr

        # Apply forces (OpenMM: inputForces[i] -= deltaX * de, inputForces[j] += deltaX * de)
        forces_manual[i] -= force_vec
        forces_manual[j] += force_vec

        print(f"Pair ({i},{j}): r={r:.6f}, t3={t3:.10f}, de={de:.10f}")
        print(f"  dr={dr}, force_vec={force_vec}")

print()
print("Manual forces (OpenMM logic):")
for i, f in enumerate(forces_manual):
    print(f"  F[{i}] = {f}")
print()

# Get kernel forces
forces_kernel = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

print("Kernel forces:")
for i, f in enumerate(forces_kernel):
    print(f"  F[{i}] = {f}")
print()

print("Comparison:")
for i in range(3):
    error = forces_manual[i] - forces_kernel[i]
    print(f"  Atom {i}: error={error}, norm={np.linalg.norm(error):.10f}")
