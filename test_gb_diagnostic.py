#!/usr/bin/env python3
"""
Diagnostic test to understand what's going wrong with GB forces.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fennol.models.physics.implicit_solvent.parameters import AtomicParameters

try:
    from fennol import cuda as fennol_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("ERROR: CUDA not available. This test requires CUDA.")
    sys.exit(1)


def main():
    print("=" * 70)
    print("GB DIAGNOSTIC TEST")
    print("=" * 70)

    # Create simple water dimer
    coords = np.array([
        # Water 1
        [0.0, 0.0, 0.0],      # O
        [0.757, 0.586, 0.0],  # H
        [-0.757, 0.586, 0.0], # H
        # Water 2 (2.8 Å away)
        [2.8, 0.0, 0.0],      # O
        [3.557, 0.586, 0.0],  # H
        [2.043, 0.586, 0.0],  # H
    ])
    charges = np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417])
    atomic_numbers = np.array([8, 1, 1, 8, 1, 1])

    # Parameters
    atomic_params = AtomicParameters("mbondi")
    radii = atomic_params.get_radii_array(atomic_numbers)
    b_params, c_params = atomic_params.get_obc_params_arrays(atomic_numbers)
    dielectric = 80.0
    cutoff = 12.0

    print(f"\nTest system: Water dimer")
    print(f"  Atoms: {len(coords)}")
    print(f"  Radii: {radii}")
    print(f"  b_params: {b_params}")
    print(f"  c_params: {c_params}")

    # Test 1: Compute Born radii (old way)
    print("\n" + "=" * 70)
    print("TEST 1: Born radii (OLD method)")
    print("=" * 70)
    born_radii_old = fennol_cuda.gb_compute_born_radii(
        coords, radii, b_params, c_params, cutoff
    )
    print(f"Born radii: {born_radii_old}")

    # Test 2: Compute Born radii (new way with psi)
    print("\n" + "=" * 70)
    print("TEST 2: Born radii (NEW method with psi)")
    print("=" * 70)
    born_radii_new, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
        coords, radii, b_params, c_params, cutoff
    )
    print(f"Born radii: {born_radii_new}")
    print(f"psi_sum: {psi_sum}")

    # Compare
    print(f"\nDifference in Born radii: {np.abs(born_radii_old - born_radii_new).max()}")

    # Test 3: Compute energy and forces (OLD)
    print("\n" + "=" * 70)
    print("TEST 3: Energy and forces (OLD)")
    print("=" * 70)
    energy_old, forces_old = fennol_cuda.gb_compute_energy_forces(
        coords, charges, born_radii_old, dielectric, cutoff
    )
    print(f"Energy: {energy_old[0]:.6f} kcal/mol")
    print(f"Forces shape: {forces_old.shape}")
    print(f"Max force: {np.abs(forces_old).max():.6f} kcal/(mol·Å)")
    print(f"Forces:\n{forces_old}")

    # Test 4: Compute energy and forces (NEW)
    print("\n" + "=" * 70)
    print("TEST 4: Energy and forces (NEW - COMPLETE)")
    print("=" * 70)
    energy_new, forces_new = fennol_cuda.gb_compute_forces_complete(
        coords, charges, born_radii_new, radii, b_params, c_params, psi_sum, dielectric, cutoff
    )
    print(f"Energy: {energy_new[0]:.6f} kcal/mol")
    print(f"Forces shape: {forces_new.shape}")
    print(f"Max force: {np.abs(forces_new).max():.6f} kcal/(mol·Å)")
    print(f"Forces:\n{forces_new}")

    # Compare energies
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Energy difference: {abs(energy_old[0] - energy_new[0]):.6f} kcal/mol")
    print(f"Force difference (RMS): {np.sqrt(((forces_old - forces_new)**2).mean()):.6f} kcal/(mol·Å)")
    print(f"Force difference (max): {np.abs(forces_old - forces_new).max():.6f} kcal/(mol·Å)")

    # Manually compute numerical gradient for one component
    print("\n" + "=" * 70)
    print("TEST 5: Manual numerical gradient (atom 0, x component)")
    print("=" * 70)

    delta = 1e-5
    i, j = 0, 0  # Atom 0, x component

    # Forward step
    coords_plus = coords.copy()
    coords_plus[i, j] += delta
    born_radii_plus, psi_plus = fennol_cuda.gb_compute_born_radii_with_psi(
        coords_plus, radii, b_params, c_params, cutoff
    )
    energy_plus, _ = fennol_cuda.gb_compute_forces_complete(
        coords_plus, charges, born_radii_plus, radii, b_params, c_params, psi_plus, dielectric, cutoff
    )

    # Backward step
    coords_minus = coords.copy()
    coords_minus[i, j] -= delta
    born_radii_minus, psi_minus = fennol_cuda.gb_compute_born_radii_with_psi(
        coords_minus, radii, b_params, c_params, cutoff
    )
    energy_minus, _ = fennol_cuda.gb_compute_forces_complete(
        coords_minus, charges, born_radii_minus, radii, b_params, c_params, psi_minus, dielectric, cutoff
    )

    numerical_force = -(energy_plus[0] - energy_minus[0]) / (2 * delta)
    analytical_force = forces_new[i, j]

    print(f"Delta: {delta} Å")
    print(f"E(x+δ): {energy_plus[0]:.6f} kcal/mol")
    print(f"E(x-δ): {energy_minus[0]:.6f} kcal/mol")
    print(f"Numerical force: {numerical_force:.6f} kcal/(mol·Å)")
    print(f"Analytical force: {analytical_force:.6f} kcal/(mol·Å)")
    print(f"Error: {abs(numerical_force - analytical_force):.6e} kcal/(mol·Å)")
    print(f"Relative error: {abs(numerical_force - analytical_force) / abs(numerical_force) * 100:.2f}%")


if __name__ == "__main__":
    main()
