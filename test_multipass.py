#!/usr/bin/env python3
"""
Test the multi-pass GB force implementation (OpenMM approach)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

# Single water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H1
    [-0.757, 0.586, 0.0], # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)

# OBC parameters
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

print("=" * 70)
print("Multi-Pass GB Force Implementation Test")
print("=" * 70)
print()

# Step 1: Compute Born radii with psi
print("Step 1: Computing Born radii and descreening sum...")
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
    coords, radii, b_params, c_params, cutoff
)
print(f"  Born radii: {born_radii}")
print(f"  Psi sum:    {psi_sum}")
print()

# Step 2: Compute energy and direct forces
print("Step 2: Computing energy and direct forces...")
energy, direct_forces = fennol_cuda.gb_compute_energy_forces(
    coords, charges, born_radii, dielectric, cutoff
)
direct_forces = direct_forces.reshape(-1, 3)
print(f"  Energy: {float(energy):.6f} kcal/mol")
print(f"  Direct forces:\n{direct_forces}")
print()

# Step 3: Compute dE/dR (using GPU implementation)
print("Step 3: Computing ∂E/∂R using compute_dE_dR...")
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
print(f"  ∂E/∂R: {dE_dR}")
print()

# Step 4: Convert ∂E/∂R → ∂E/∂ψ (NEW multi-pass function)
print("Step 4: Converting ∂E/∂R → ∂E/∂ψ using reduce_born_force...")
dE_dpsi = fennol_cuda.reduce_born_force(
    dE_dR, born_radii, radii, b_params, c_params, psi_sum
)
print(f"  ∂E/∂ψ: {dE_dpsi}")
print()

# Step 5: Apply Born forces (NEW multi-pass function)
print("Step 5: Applying Born forces using apply_born_forces...")
born_forces = fennol_cuda.apply_born_forces(
    coords, radii, dE_dpsi, cutoff
)
born_forces = born_forces.reshape(-1, 3)
print(f"  Born forces:\n{born_forces}")
print()

# Step 6: Combine forces
print("Step 6: Combining forces...")
total_forces_multipass = direct_forces + born_forces
print(f"  Total forces (multi-pass):\n{total_forces_multipass}")
print()

# Compare with existing gb_compute_forces_complete
print("Comparing with existing gb_compute_forces_complete...")
_, total_forces_complete = fennol_cuda.gb_compute_forces_complete(
    coords, charges, born_radii, radii, b_params, c_params, psi_sum, dielectric, cutoff
)
total_forces_complete = total_forces_complete.reshape(-1, 3)
print(f"  Total forces (complete):\n{total_forces_complete}")
print()

# Compute difference
diff = total_forces_multipass - total_forces_complete
max_diff = np.max(np.abs(diff))
print(f"Maximum difference: {max_diff:.6e}")
print()

if max_diff < 1e-6:
    print("✅ Multi-pass implementation matches existing implementation!")
else:
    print(f"⚠️  Difference detected: {max_diff:.6e}")
    print(f"   Difference:\n{diff}")
print()

# Now test with numerical gradient
print("=" * 70)
print("Numerical Gradient Validation")
print("=" * 70)
print()

delta = 1e-5
numerical_forces = np.zeros_like(coords)

for i in range(len(coords)):
    for d in range(3):
        # Perturb +delta
        coords_plus = coords.copy()
        coords_plus[i, d] += delta

        born_radii_plus, psi_plus = fennol_cuda.gb_compute_born_radii_with_psi(
            coords_plus, radii, b_params, c_params, cutoff
        )
        E_plus, _ = fennol_cuda.gb_compute_energy_forces(
            coords_plus, charges, born_radii_plus, dielectric, cutoff
        )

        # Perturb -delta
        coords_minus = coords.copy()
        coords_minus[i, d] -= delta

        born_radii_minus, psi_minus = fennol_cuda.gb_compute_born_radii_with_psi(
            coords_minus, radii, b_params, c_params, cutoff
        )
        E_minus, _ = fennol_cuda.gb_compute_energy_forces(
            coords_minus, charges, born_radii_minus, dielectric, cutoff
        )

        # Finite difference
        numerical_forces[i, d] = -(float(E_plus) - float(E_minus)) / (2 * delta)

print(f"Numerical forces:\n{numerical_forces}")
print()
print(f"Multi-pass forces:\n{total_forces_multipass}")
print()

force_diff = total_forces_multipass - numerical_forces
max_force_diff = np.max(np.abs(force_diff))
rel_error = max_force_diff / np.max(np.abs(numerical_forces))

print(f"Maximum absolute difference: {max_force_diff:.6e}")
print(f"Relative error: {rel_error:.6e}")
print()

if rel_error < 0.01:  # 1% error
    print("✅ Multi-pass forces match numerical gradient!")
elif rel_error < 0.1:
    print("⚠️  Multi-pass forces within 10% of numerical gradient")
else:
    print("❌ SIGNIFICANT ERROR: Multi-pass forces differ from numerical gradient")
    print(f"   Difference:\n{force_diff}")
