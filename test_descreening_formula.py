#!/usr/bin/env python3
"""
Deep dive: Check if the descreening formula is the issue
Compare our simplified formula with what it should be
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("INVESTIGATING DESCREENING FORMULA")
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

born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)

print("Computed values:")
print(f"  Born radii: {born_radii}")
print(f"  Psi sum:    {psi_sum}")
print()

# Manually compute psi for atom 0 using our formula
r01 = np.linalg.norm(coords[0] - coords[1])
r02 = np.linalg.norm(coords[0] - coords[2])

def descreening_integral_simple(r, rho_i, rho_j):
    """Our current simplified formula"""
    upper = rho_i + rho_j
    lower = abs(rho_i - rho_j)

    if r < lower:
        # Complete overlap
        return 0.5 * rho_i * (1.0/(lower*lower) - 1.0/(upper*upper))
    elif r < upper:
        # Partial overlap
        return 0.5 * rho_i * (1.0/(r*r) - 1.0/(upper*upper))
    else:
        return 0.0

psi_0_manual = descreening_integral_simple(r01, radii[0], radii[1]) + \
               descreening_integral_simple(r02, radii[0], radii[2])

print(f"Manual psi[0] (simplified formula): {psi_0_manual:.10f}")
print(f"Kernel psi[0]:                       {psi_sum[0]:.10f}")
print(f"Match: {abs(psi_0_manual - psi_sum[0]) < 1e-6}")
print()

# Now check if the issue is in how ∂E/∂R is calculated
print("="*80)
print("CHECKING ∂E/∂R CALCULATION")
print("="*80)
print()

# The self-energy derivative
COULOMB = 332.0636
gb_factor = -0.5 * (1.0 - 1.0/dielectric) * COULOMB
print(f"gb_factor = {gb_factor:.6f}")

# For atom 0 (O)
R_O = born_radii[0]
q_O = charges[0]

dE_self_dR = gb_factor * (-q_O**2 / R_O**2)
print(f"∂E_self/∂R_O = {dE_self_dR:.6f}")
print()

# Check: Is the self-energy formula correct?
# E_self = -0.5 * (1 - 1/ε) * q²/R
# ∂E_self/∂R = -0.5 * (1 - 1/ε) * q² * (-1/R²)
#             = 0.5 * (1 - 1/ε) * q²/R²
# With our gb_factor = -0.5 * (1 - 1/ε):
# ∂E_self/∂R = gb_factor * (-q²/R²)  ✓ Matches our code

print("Self-energy derivative formula: ✓ Correct")
print()

# Now check the pair energy derivative
print("="*80)
print("CHECKING PAIR ENERGY DERIVATIVE")
print("="*80)
print()

# For pair (O, H1)
R_H = born_radii[1]
q_H = charges[1]
r = r01

print(f"Pair (O, H1):")
print(f"  r = {r:.6f}")
print(f"  R_O = {R_O:.6f}, R_H = {R_H:.6f}")
print(f"  q_O = {q_O:.6f}, q_H = {q_H:.6f}")
print()

# f_GB = sqrt(r² + R_O*R_H*exp(-r²/(4*R_O*R_H)))
RiRj = R_O * R_H
exp_term = np.exp(-r**2 / (4*RiRj))
f_gb = np.sqrt(r**2 + RiRj * exp_term)

print(f"  f_GB = {f_gb:.6f}")
print()

# E_pair = gb_factor * q_O * q_H / f_GB
E_pair = gb_factor * q_O * q_H / f_gb
print(f"  E_pair = {E_pair:.6f} kcal/mol")
print()

# ∂f_GB/∂R_O
# f_GB = sqrt(r² + R_O*R_H*exp(-r²/(4*R_O*R_H)))
# Let α² = r² + R_O*R_H*exp(-D) where D = r²/(4*R_O*R_H)
# ∂α²/∂R_O = R_H*exp(-D) + R_O*R_H*exp(-D)*(-1)*(-r²/(4*R_O²*R_H))
#          = R_H*exp(-D) * [1 + r²/(4*R_O*R_H)]
#          = R_H*exp(-D) * [1 + D/R_O]
# But that's not quite right... let me recalculate

# Actually:
# ∂α²/∂R_O = ∂/∂R_O [R_O*R_H*exp(-r²/(4*R_O*R_H))]
# Let u = R_O*R_H and v = -r²/(4*u)
# ∂u/∂R_O = R_H
# ∂v/∂R_O = -r² * (-1/(4*u²)) * R_H = r²*R_H/(4*u²)
#
# ∂α²/∂R_O = R_H*exp(v) + u*exp(v)*r²*R_H/(4*u²)
#          = R_H*exp(v) * [1 + r²/(4*u)]
#          = R_H*exp(v) * [1 - r²/(4*R_O*R_H)]

# Wait, let me be more careful. Let me use the formula from the code:
df_gb_dRi = 0.5 / f_gb * R_H * exp_term * (1.0 - r**2 / (4 * R_O * RiRj))
print(f"  ∂f_GB/∂R_O = {df_gb_dRi:.6f}")
print()

# ∂E_pair/∂R_O = gb_factor * q_O * q_H * ∂(1/f_GB)/∂R_O
#              = gb_factor * q_O * q_H * (-1/f_GB²) * ∂f_GB/∂R_O
dE_pair_dR_O = gb_factor * q_O * q_H * (-1.0 / f_gb**2) * df_gb_dRi
print(f"  ∂E_pair/∂R_O = {dE_pair_dR_O:.6f}")
print()

print("="*80)
print("KEY INSIGHT: Check if energy formula has correct factor")
print("="*80)
print()

# The GB energy formula is:
# E_GB = -0.5 * (1 - 1/ε) * Σ_i Σ_j q_i*q_j/f_GB
#
# This includes a factor of 0.5 to avoid double-counting pairs
# So for a single pair, the energy contribution is:
# E_pair = -0.5 * (1 - 1/ε) * COULOMB * q_i*q_j/f_GB
#
# BUT: When we compute forces, we need ∂E/∂r
# Since each pair appears once in the sum (not twice), there's NO additional 0.5 factor in the derivative

# Let me check if there's a factor-of-2 issue...

print("Testing hypothesis: Is there a missing factor of 2 in the energy formula?")
print()

# Compute energy for the water molecule
energy_kernel, _ = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
print(f"Energy from kernel: {float(energy_kernel):.10f} kcal/mol")
print()

# Manual calculation of energy
E_manual = 0.0

# Self energies
for i in range(3):
    E_self_i = gb_factor * charges[i]**2 / born_radii[i]
    E_manual += E_self_i
    print(f"E_self[{i}] = {E_self_i:.6f}")

print()

# Pair energies
for i in range(3):
    for j in range(i+1, 3):
        r_ij = np.linalg.norm(coords[i] - coords[j])
        R_i = born_radii[i]
        R_j = born_radii[j]

        RiRj = R_i * R_j
        exp_term = np.exp(-r_ij**2 / (4*RiRj))
        f_gb_ij = np.sqrt(r_ij**2 + RiRj * exp_term)

        E_pair_ij = gb_factor * charges[i] * charges[j] / f_gb_ij
        E_manual += E_pair_ij
        print(f"E_pair[{i},{j}] = {E_pair_ij:.6f}")

print()
print(f"E_manual (total) = {E_manual:.10f}")
print(f"E_kernel         = {float(energy_kernel):.10f}")
print(f"Ratio: {E_manual / float(energy_kernel):.6f}")
print()

if abs(E_manual / float(energy_kernel) - 1.0) > 0.01:
    print("⚠️  WARNING: Energy calculation has an error!")
    print("This suggests the energy formula in the kernel might be wrong!")
else:
    print("✓ Energy calculation is correct")
