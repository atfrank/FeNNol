#!/usr/bin/env python3
"""Test GB force accuracy with JAX (no CUDA) using numerical gradient."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Temporarily disable CUDA to test JAX implementation
os.environ['FENNOL_NO_CUDA'] = '1'

from fennol.models.physics.implicit_solvent import OBC
from fennol.utils.pdb import read_pdb, assign_charges


def numerical_gradient(coords, charges, atomic_numbers, model, delta=1e-5):
    """Compute numerical gradient of energy."""
    natoms = len(coords)
    numerical_forces = np.zeros_like(coords)

    for i in range(natoms):
        for j in range(3):
            coords_plus = coords.copy()
            coords_plus[i, j] += delta
            E_plus, _ = model(coords_plus, charges, atomic_numbers)

            coords_minus = coords.copy()
            coords_minus[i, j] -= delta
            E_minus, _ = model(coords_minus, charges, atomic_numbers)

            numerical_forces[i, j] = -(E_plus - E_minus) / (2 * delta)

    return numerical_forces


def test_force_accuracy_jax():
    """Test JAX forces against numerical gradient."""
    print("\n" + "="*70)
    print("GB Force Accuracy Test (JAX - Fixed Version)")
    print("="*70)

    # Load ternary complex
    pdb_file = "/home/aaron/ATX/software/VelocityMD/examples/input_files/ternary_complex.pdb"
    structure = read_pdb(pdb_file)

    # Use only first 20 atoms for speed (JAX autodiff is slower)
    coords = structure.coordinates[:20].copy()
    charges = assign_charges(structure)[:20]
    atomic_numbers = structure.atomic_numbers[:20]

    print(f"\nTesting with {len(coords)} atoms (JAX implementation)")
    print(f"Total charge: {charges.sum():.6f} e")

    # Create OBC model
    config = {
        "dielectric": 80.0,
        "cutoff": 12.0,
        "surface_tension": 0.005,
        "use_cuda": False  # Force JAX
    }
    model = OBC(config)

    # Compute analytical forces
    print("\nComputing analytical forces (with full derivatives)...")
    energy_analytical, forces_analytical = model(coords, charges, atomic_numbers)

    print(f"  Energy: {float(energy_analytical):.4f} kcal/mol")
    print(f"  Max force: {float(np.abs(forces_analytical).max()):.4f} kcal/(mol·Å)")
    print(f"  RMS force: {float(np.sqrt((forces_analytical**2).mean())):.4f} kcal/(mol·Å)")

    # Compute numerical forces
    print("\nComputing numerical gradient...")
    forces_numerical = numerical_gradient(
        np.array(coords), np.array(charges), np.array(atomic_numbers),
        model, delta=1e-4
    )

    print(f"  Max numerical force: {np.abs(forces_numerical).max():.4f} kcal/(mol·Å)")

    # Compare forces
    print("\n" + "="*70)
    print("Force Comparison")
    print("="*70)

    force_diff = np.array(forces_analytical) - forces_numerical
    force_error = np.abs(force_diff)

    max_error = force_error.max()
    mean_error = force_error.mean()
    rms_error = np.sqrt((force_diff**2).mean())

    # Relative error
    force_magnitude = np.abs(forces_analytical)
    mask = force_magnitude > 0.1
    if mask.any():
        rel_error = np.abs(force_diff[mask] / np.array(forces_analytical)[mask])
        mean_rel_error = rel_error.mean()
        max_rel_error = rel_error.max()
    else:
        mean_rel_error = 0
        max_rel_error = 0

    print(f"\nAbsolute errors:")
    print(f"  Max error:  {max_error:.4e} kcal/(mol·Å)")
    print(f"  Mean error: {mean_error:.4e} kcal/(mol·Å)")
    print(f"  RMS error:  {rms_error:.4e} kcal/(mol·Å)")

    print(f"\nRelative errors (for forces > 0.1 kcal/(mol·Å)):")
    print(f"  Max relative error:  {max_rel_error:.4e}")
    print(f"  Mean relative error: {mean_rel_error:.4e}")

    # Check if forces are reasonable
    print("\n" + "="*70)
    print("Assessment")
    print("="*70)

    MAX_ABS_ERROR = 1e-2  # 0.01 kcal/(mol·Å)
    MAX_REL_ERROR = 0.05   # 5%

    if max_error < MAX_ABS_ERROR and max_rel_error < MAX_REL_ERROR:
        print("\n✅ JAX Forces are ACCURATE!")
        print(f"   Max absolute error {max_error:.2e} < {MAX_ABS_ERROR} ✓")
        print(f"   Max relative error {max_rel_error:.2%} < {MAX_REL_ERROR:.0%} ✓")
        return True
    else:
        print("\n⚠️  JAX Forces still have DISCREPANCIES!")
        if max_error >= MAX_ABS_ERROR:
            print(f"   Max absolute error {max_error:.2e} >= {MAX_ABS_ERROR} ✗")
        if max_rel_error >= MAX_REL_ERROR:
            print(f"   Max relative error {max_rel_error:.2%} >= {MAX_REL_ERROR:.0%} ✗")
        return False


if __name__ == "__main__":
    success = test_force_accuracy_jax()

    if not success:
        sys.exit(1)
