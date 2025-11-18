#!/usr/bin/env python3
"""Debug GB energy behavior with small perturbations."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

os.environ['FENNOL_NO_CUDA'] = '1'

from fennol.models.physics.implicit_solvent import OBC
from fennol.utils.pdb import read_pdb, assign_charges


def debug_energy_sensitivity():
    """Check how energy changes with small position perturbations."""
    print("\n" + "="*70)
    print("GB Energy Sensitivity Debug")
    print("="*70)

    # Load system
    pdb_file = "/home/aaron/ATX/software/VelocityMD/examples/input_files/ternary_complex.pdb"
    structure = read_pdb(pdb_file)

    # Use just 10 atoms for clarity
    coords = structure.coordinates[:10].copy()
    charges = assign_charges(structure)[:10]
    atomic_numbers = structure.atomic_numbers[:10]

    print(f"\nTesting with {len(coords)} atoms")
    print(f"Total charge: {charges.sum():.6f} e")

    # Create OBC model
    config = {
        "dielectric": 80.0,
        "cutoff": 12.0,
        "surface_tension": 0.005,
        "use_cuda": False
    }
    model = OBC(config)

    # Verify which backend is being used
    print(f"\nBackend being used:")
    print(f"  has_cuda: {model.has_cuda}")
    print(f"  Using: {'CUDA' if model.has_cuda else 'JAX'}")

    # Compute energy and forces at original position
    E0, F0 = model(coords, charges, atomic_numbers)
    E0 = float(E0)

    print(f"\nOriginal energy: {E0:.6f} kcal/mol")
    print(f"Max force: {float(np.abs(F0).max()):.6f} kcal/(mol·Å)")

    # Perturb atom 0 in x direction by small amounts
    print("\n" + "="*70)
    print(f"Perturbing atom 0, coordinate x:")
    print(f"{'Delta (Å)':<12} {'Energy':<15} {'ΔE':<15} {'Numerical F':<15} {'Analytical F':<15}")
    print("-" * 70)

    deltas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    for delta in deltas:
        # Forward
        coords_plus = coords.copy()
        coords_plus[0, 0] += delta
        E_plus, _ = model(coords_plus, charges, atomic_numbers)
        E_plus = float(E_plus)

        # Backward
        coords_minus = coords.copy()
        coords_minus[0, 0] -= delta
        E_minus, _ = model(coords_minus, charges, atomic_numbers)
        E_minus = float(E_minus)

        # Energy difference
        dE = E_plus - E0
        dE_symmetric = (E_plus - E_minus) / 2

        # Numerical force
        F_numerical = -(E_plus - E_minus) / (2 * delta)

        # Analytical force
        F_analytical = float(F0[0, 0])

        print(f"{delta:<12.2e} {E_plus:<15.6f} {dE_symmetric:<15.6e} {F_numerical:<15.6f} {F_analytical:<15.6f}")

    # Check if energy is discontinuous or has numerical issues
    print("\n" + "="*70)
    print("Analysis")
    print("="*70)

    # Very small perturbation
    delta_tiny = 1e-6
    coords_tiny = coords.copy()
    coords_tiny[0, 0] += delta_tiny
    E_tiny, _ = model(coords_tiny, charges, atomic_numbers)
    E_tiny = float(E_tiny)

    F_expected_tiny = (E_tiny - E0) / delta_tiny
    F_analytical = float(F0[0, 0])

    print(f"\nWith δ = {delta_tiny:.2e} Å:")
    print(f"  ΔE = {E_tiny - E0:.2e} kcal/mol")
    print(f"  Numerical force estimate: {-F_expected_tiny:.6f} kcal/(mol·Å)")
    print(f"  Analytical force: {F_analytical:.6f} kcal/(mol·Å)")
    print(f"  Ratio: {-F_expected_tiny / F_analytical if F_analytical != 0 else np.inf:.1f}x")

    if abs(F_expected_tiny / F_analytical) > 100:
        print("\n⚠️  HUGE DISCREPANCY!")
        print("    Energy is changing much faster than analytical forces suggest.")
        print("    This indicates a bug in the force calculation.")


if __name__ == "__main__":
    debug_energy_sensitivity()
