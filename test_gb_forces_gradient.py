#!/usr/bin/env python3
"""
Comprehensive numerical gradient test for GB forces implementation.

Tests BOTH implementations:
1. OLD (incomplete): gb_compute_energy_forces() - missing Born radii derivatives
2. NEW (complete): gb_compute_forces_complete() - full chain rule implementation

Compares both against numerical gradients using finite differences.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fennol.models.physics.implicit_solvent.parameters import AtomicParameters
from fennol.utils.pdb import read_pdb, assign_charges

try:
    from fennol import cuda as fennol_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("ERROR: CUDA not available. This test requires CUDA.")
    sys.exit(1)


def compute_energy_old(coords, charges, atomic_numbers, atomic_params, dielectric=80.0, cutoff=12.0):
    """
    Compute energy using OLD (incomplete) force implementation.

    This uses gb_compute_energy_forces() which is missing Born radii derivatives.
    NOTE: We only compute the ELECTROSTATIC part, not nonpolar.
    """
    # Get atomic parameters
    radii = atomic_params.get_radii_array(atomic_numbers)
    b_params, c_params = atomic_params.get_obc_params_arrays(atomic_numbers)

    # Compute Born radii
    born_radii = fennol_cuda.gb_compute_born_radii(
        coords, radii, b_params, c_params, cutoff
    )

    # Compute energy (and incomplete forces, but we only use energy for numerical gradient)
    energy_array, forces = fennol_cuda.gb_compute_energy_forces(
        coords, charges, born_radii, dielectric, cutoff
    )

    # DEBUG: Print some values
    # print(f"  [OLD] Energy: {float(energy_array[0]):.4f}, max Born radii: {born_radii.max():.4f}")

    return float(energy_array[0])


def compute_energy_new(coords, charges, atomic_numbers, atomic_params, dielectric=80.0, cutoff=12.0):
    """
    Compute energy using NEW (complete) force implementation.

    This uses gb_compute_forces_complete() with full Born radii derivatives.
    NOTE: We only compute the ELECTROSTATIC part, not nonpolar.
    """
    # Get atomic parameters
    radii = atomic_params.get_radii_array(atomic_numbers)
    b_params, c_params = atomic_params.get_obc_params_arrays(atomic_numbers)

    # Compute Born radii WITH psi_sum
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
        coords, radii, b_params, c_params, cutoff
    )

    # Compute energy (and complete forces, but we only use energy for numerical gradient)
    energy_array, forces = fennol_cuda.gb_compute_forces_complete(
        coords, charges, born_radii, radii, b_params, c_params, psi_sum, dielectric, cutoff
    )

    # DEBUG: Print some values
    # print(f"  [NEW] Energy: {float(energy_array[0]):.4f}, max Born radii: {born_radii.max():.4f}")

    return float(energy_array[0])


def get_forces_old(coords, charges, atomic_numbers, atomic_params, dielectric=80.0, cutoff=12.0):
    """Get analytical forces using OLD implementation."""
    # Get atomic parameters
    radii = atomic_params.get_radii_array(atomic_numbers)
    b_params, c_params = atomic_params.get_obc_params_arrays(atomic_numbers)

    # Compute Born radii
    born_radii = fennol_cuda.gb_compute_born_radii(
        coords, radii, b_params, c_params, cutoff
    )

    # Compute forces (INCOMPLETE - missing dR/dx terms)
    energy_array, forces = fennol_cuda.gb_compute_energy_forces(
        coords, charges, born_radii, dielectric, cutoff
    )

    return forces


def get_forces_new(coords, charges, atomic_numbers, atomic_params, dielectric=80.0, cutoff=12.0):
    """Get analytical forces using NEW implementation."""
    # Get atomic parameters
    radii = atomic_params.get_radii_array(atomic_numbers)
    b_params, c_params = atomic_params.get_obc_params_arrays(atomic_numbers)

    # Compute Born radii WITH psi_sum
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(
        coords, radii, b_params, c_params, cutoff
    )

    # Compute COMPLETE forces (including dR/dx terms)
    energy_array, forces = fennol_cuda.gb_compute_forces_complete(
        coords, charges, born_radii, radii, b_params, c_params, psi_sum, dielectric, cutoff
    )

    return forces


def numerical_gradient(coords, charges, atomic_numbers, atomic_params, energy_func, delta=1e-5):
    """
    Compute numerical gradient of energy using finite differences.

    F_i = -dE/dx_i ≈ -(E(x+δ) - E(x-δ)) / (2δ)

    Args:
        coords: Atomic coordinates
        charges: Atomic charges
        atomic_numbers: Atomic numbers
        atomic_params: Atomic parameters object
        energy_func: Function to compute energy (either compute_energy_old or compute_energy_new)
        delta: Finite difference step size

    Returns:
        Numerical forces array
    """
    natoms = len(coords)
    numerical_forces = np.zeros_like(coords)

    print(f"  Computing numerical gradient for {natoms} atoms ({natoms * 3} force components)...")

    for i in range(natoms):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{natoms} atoms", end='\r')

        for j in range(3):
            # Forward step
            coords_plus = coords.copy()
            coords_plus[i, j] += delta
            E_plus = energy_func(coords_plus, charges, atomic_numbers, atomic_params)

            # Backward step
            coords_minus = coords.copy()
            coords_minus[i, j] -= delta
            E_minus = energy_func(coords_minus, charges, atomic_numbers, atomic_params)

            # Central difference
            numerical_forces[i, j] = -(E_plus - E_minus) / (2 * delta)

    print(f"    Progress: {natoms}/{natoms} atoms - DONE    ")

    return numerical_forces


def analyze_force_errors(forces_analytical, forces_numerical, label):
    """
    Analyze and report errors between analytical and numerical forces.

    Returns:
        Dictionary with error metrics
    """
    force_diff = forces_analytical - forces_numerical
    force_error = np.abs(force_diff)

    max_error = force_error.max()
    mean_error = force_error.mean()
    rms_error = np.sqrt((force_diff**2).mean())

    # Relative error (where forces are significant)
    force_magnitude = np.abs(forces_numerical)
    mask = force_magnitude > 0.1  # Only compare where forces are > 0.1

    if mask.any():
        rel_error = np.abs(force_diff[mask] / forces_numerical[mask])
        mean_rel_error = rel_error.mean()
        max_rel_error = rel_error.max()
    else:
        mean_rel_error = 0.0
        max_rel_error = 0.0

    print(f"\n{label}")
    print("=" * 70)

    print(f"\nAbsolute errors:")
    print(f"  Max error:  {max_error:.4e} kcal/(mol·Å)")
    print(f"  Mean error: {mean_error:.4e} kcal/(mol·Å)")
    print(f"  RMS error:  {rms_error:.4e} kcal/(mol·Å)")

    print(f"\nRelative errors (for forces > 0.1 kcal/(mol·Å)):")
    print(f"  Max relative error:  {max_rel_error:.2%}")
    print(f"  Mean relative error: {mean_rel_error:.2%}")

    # Show worst discrepancies
    print(f"\nWorst 5 force components:")
    print(f"{'Atom':<6} {'Coord':<6} {'Analytical':<15} {'Numerical':<15} {'Error':<15}")
    print("-" * 70)

    flat_idx = np.argsort(force_error.flatten())[::-1][:5]
    for idx in flat_idx:
        i = idx // 3
        j = idx % 3
        coord_name = ['x', 'y', 'z'][j]

        print(f"{i:<6} {coord_name:<6} {forces_analytical[i,j]:<15.6f} "
              f"{forces_numerical[i,j]:<15.6f} {force_diff[i,j]:<15.6e}")

    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'rms_error': rms_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error
    }


def test_gradient_accuracy():
    """Main test function."""
    print("\n" + "="*70)
    print("COMPREHENSIVE GB FORCES NUMERICAL GRADIENT TEST")
    print("="*70)
    print("\nThis test validates analytical forces against numerical gradients")
    print("for BOTH the old (incomplete) and new (complete) implementations.")

    # Load test system
    # Use a SMALL test system for easier debugging
    USE_WATER_DIMER = True  # Set to False to use ternary complex

    if USE_WATER_DIMER:
        print("\nCreating water dimer test system...")
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
    else:
        pdb_file = "/home/aaron/ATX/software/VelocityMD/examples/input_files/ternary_complex.pdb"
        print(f"\nLoading structure from: {pdb_file}")
        structure = read_pdb(pdb_file)

        # Use only first 30 atoms for reasonable speed
        # (50 atoms = 150 force components = 300 energy evaluations)
        natoms_test = 30
        coords = structure.coordinates[:natoms_test].copy()
        charges = assign_charges(structure)[:natoms_test]
        atomic_numbers = structure.atomic_numbers[:natoms_test]

    print(f"\nTest system:")
    print(f"  Atoms: {len(coords)}")
    print(f"  Total charge: {charges.sum():.6f} e")
    print(f"  Elements: {np.unique(atomic_numbers)}")

    # Create atomic parameters
    atomic_params = AtomicParameters("mbondi")

    # Test parameters
    dielectric = 80.0
    cutoff = 12.0
    delta = 1e-5  # Finite difference step size

    print(f"\nTest parameters:")
    print(f"  Dielectric: {dielectric}")
    print(f"  Cutoff: {cutoff} Å")
    print(f"  Finite difference step: {delta} Å")

    # =========================================================================
    # TEST 1: OLD (INCOMPLETE) IMPLEMENTATION
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: OLD (INCOMPLETE) IMPLEMENTATION")
    print("="*70)
    print("\nThis uses gb_compute_energy_forces() which is MISSING Born radii")
    print("derivatives. We expect this to FAIL the gradient test.")

    print("\nComputing analytical forces (OLD)...")
    forces_old = get_forces_old(coords, charges, atomic_numbers, atomic_params)
    print(f"  Max force: {np.abs(forces_old).max():.4f} kcal/(mol·Å)")
    print(f"  RMS force: {np.sqrt((forces_old**2).mean()):.4f} kcal/(mol·Å)")

    print("\nComputing numerical gradient...")
    forces_numerical_old = numerical_gradient(
        coords, charges, atomic_numbers, atomic_params, compute_energy_old, delta=delta
    )
    print(f"  Max numerical force: {np.abs(forces_numerical_old).max():.4f} kcal/(mol·Å)")
    print(f"  RMS numerical force: {np.sqrt((forces_numerical_old**2).mean()):.4f} kcal/(mol·Å)")

    metrics_old = analyze_force_errors(forces_old, forces_numerical_old, "OLD IMPLEMENTATION ERRORS")

    # =========================================================================
    # TEST 2: NEW (COMPLETE) IMPLEMENTATION
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: NEW (COMPLETE) IMPLEMENTATION")
    print("="*70)
    print("\nThis uses gb_compute_forces_complete() with FULL Born radii")
    print("derivatives. We expect this to PASS the gradient test.")

    print("\nComputing analytical forces (NEW)...")
    forces_new = get_forces_new(coords, charges, atomic_numbers, atomic_params)
    print(f"  Max force: {np.abs(forces_new).max():.4f} kcal/(mol·Å)")
    print(f"  RMS force: {np.sqrt((forces_new**2).mean()):.4f} kcal/(mol·Å)")

    print("\nComputing numerical gradient...")
    forces_numerical_new = numerical_gradient(
        coords, charges, atomic_numbers, atomic_params, compute_energy_new, delta=delta
    )
    print(f"  Max numerical force: {np.abs(forces_numerical_new).max():.4f} kcal/(mol·Å)")
    print(f"  RMS numerical force: {np.sqrt((forces_numerical_new**2).mean()):.4f} kcal/(mol·Å)")

    metrics_new = analyze_force_errors(forces_new, forces_numerical_new, "NEW IMPLEMENTATION ERRORS")

    # =========================================================================
    # COMPARISON AND ASSESSMENT
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON: OLD vs NEW")
    print("="*70)

    print(f"\n{'Metric':<30} {'OLD':<20} {'NEW':<20} {'Improvement':<15}")
    print("-" * 85)

    def format_improvement(old_val, new_val):
        """Format improvement ratio."""
        if new_val == 0:
            return "Perfect!"
        if old_val == 0:
            return "N/A"
        ratio = old_val / new_val
        return f"{ratio:.1f}x better"

    print(f"{'Max abs error':<30} {metrics_old['max_error']:<20.4e} {metrics_new['max_error']:<20.4e} "
          f"{format_improvement(metrics_old['max_error'], metrics_new['max_error'])}")

    print(f"{'RMS error':<30} {metrics_old['rms_error']:<20.4e} {metrics_new['rms_error']:<20.4e} "
          f"{format_improvement(metrics_old['rms_error'], metrics_new['rms_error'])}")

    print(f"{'Max rel error':<30} {metrics_old['max_rel_error']:<20.2%} {metrics_new['max_rel_error']:<20.2%} "
          f"{format_improvement(metrics_old['max_rel_error'], metrics_new['max_rel_error'])}")

    # =========================================================================
    # FINAL ASSESSMENT
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)

    # Acceptance criteria from documentation
    MAX_ABS_ERROR = 1e-2  # 0.01 kcal/(mol·Å)
    MAX_REL_ERROR = 0.05  # 5%

    print(f"\nAcceptance criteria:")
    print(f"  Max absolute error < {MAX_ABS_ERROR} kcal/(mol·Å)")
    print(f"  Max relative error < {MAX_REL_ERROR:.0%}")

    # Check OLD implementation
    print(f"\nOLD implementation (gb_compute_energy_forces):")
    old_passes_abs = metrics_old['max_error'] < MAX_ABS_ERROR
    old_passes_rel = metrics_old['max_rel_error'] < MAX_REL_ERROR
    old_passes = old_passes_abs and old_passes_rel

    if old_passes_abs:
        print(f"  Max abs error: {metrics_old['max_error']:.2e} < {MAX_ABS_ERROR} - PASS")
    else:
        print(f"  Max abs error: {metrics_old['max_error']:.2e} >= {MAX_ABS_ERROR} - FAIL")

    if old_passes_rel:
        print(f"  Max rel error: {metrics_old['max_rel_error']:.2%} < {MAX_REL_ERROR:.0%} - PASS")
    else:
        print(f"  Max rel error: {metrics_old['max_rel_error']:.2%} >= {MAX_REL_ERROR:.0%} - FAIL")

    if old_passes:
        print(f"  Overall: PASS (unexpected!)")
    else:
        print(f"  Overall: FAIL (expected - missing Born radii derivatives)")

    # Check NEW implementation
    print(f"\nNEW implementation (gb_compute_forces_complete):")
    new_passes_abs = metrics_new['max_error'] < MAX_ABS_ERROR
    new_passes_rel = metrics_new['max_rel_error'] < MAX_REL_ERROR
    new_passes = new_passes_abs and new_passes_rel

    if new_passes_abs:
        print(f"  Max abs error: {metrics_new['max_error']:.2e} < {MAX_ABS_ERROR} - PASS")
    else:
        print(f"  Max abs error: {metrics_new['max_error']:.2e} >= {MAX_ABS_ERROR} - FAIL")

    if new_passes_rel:
        print(f"  Max rel error: {metrics_new['max_rel_error']:.2%} < {MAX_REL_ERROR:.0%} - PASS")
    else:
        print(f"  Max rel error: {metrics_new['max_rel_error']:.2%} >= {MAX_REL_ERROR:.0%} - FAIL")

    if new_passes:
        print(f"  Overall: PASS")
    else:
        print(f"  Overall: FAIL (unexpected!)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if new_passes:
        print("\nSUCCESS: The NEW complete forces implementation is CORRECT!")
        print("The analytical forces match numerical gradients within acceptable tolerance.")
        print("\nThe Born radii derivative terms have been properly implemented.")
        return True
    else:
        print("\nFAILURE: The NEW complete forces implementation has errors!")
        print("The analytical forces do NOT match numerical gradients.")
        print("\nThere may be a bug in the Born radii derivative calculation.")
        return False


if __name__ == "__main__":
    success = test_gradient_accuracy()

    if not success:
        print("\n" + "="*70)
        print("Force calculation needs debugging!")
        print("="*70)
        sys.exit(1)
    else:
        print("\n" + "="*70)
        print("All tests passed!")
        print("="*70)
        sys.exit(0)
