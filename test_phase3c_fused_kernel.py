#!/usr/bin/env python3
"""
PHASE 3C Validation: Fused GB energy/forces + dE/dR kernel

Tests that the fused kernel produces identical results to running
the two separate kernels (GB pairwise + dE/dR).

Expected speedup: ~1.10× from eliminating one kernel launch overhead
and reusing computed values.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'fennol', 'cuda'))

import fennol_cuda


def create_test_system(natoms=100, seed=42):
    """Create a simple test system with random coordinates and charges."""
    np.random.seed(seed)

    coords = np.random.randn(natoms, 3).astype(np.float64) * 10.0  # Random positions
    charges = np.random.randn(natoms).astype(np.float64) * 0.5    # Random charges
    born_radii = np.random.rand(natoms).astype(np.float64) * 2.0 + 1.0  # Born radii 1-3 Å

    return coords, charges, born_radii


def build_neighbor_list(coords, cutoff):
    """Build simple neighbor list for testing."""
    natoms = coords.shape[0]

    neighbors = []
    neighbor_counts = np.zeros(natoms, dtype=np.int32)
    neighbor_offsets = np.zeros(natoms, dtype=np.int32)

    offset = 0
    for i in range(natoms):
        neighbor_offsets[i] = offset
        xi = coords[i]

        for j in range(natoms):
            if i == j:
                continue

            xj = coords[j]
            dx = xj - xi
            r_sq = np.sum(dx**2)

            if r_sq <= cutoff**2:
                neighbors.append(j)
                neighbor_counts[i] += 1

        offset += neighbor_counts[i]

    neighbor_atoms = np.array(neighbors, dtype=np.int32)

    return neighbor_atoms, neighbor_counts, neighbor_offsets


def test_fused_kernel_validation():
    """Test that fused kernel matches separate kernels."""
    print("\n" + "="*70)
    print("PHASE 3C VALIDATION TEST")
    print("="*70)
    print("\nComparing: FUSED kernel vs Separate kernels")
    print("-" * 70)

    # Test parameters
    natoms = 100
    dielectric = 78.3
    cutoff = 12.0

    print(f"\nTest system:")
    print(f"  Atoms: {natoms}")
    print(f"  Cutoff: {cutoff} Å")
    print(f"  Dielectric: {dielectric}")

    # Create test system
    coords, charges, born_radii = create_test_system(natoms)

    # Build neighbor list
    print("\nBuilding neighbor list...")
    neighbor_atoms, neighbor_counts, neighbor_offsets = build_neighbor_list(coords, cutoff)

    avg_neighbors = np.mean(neighbor_counts)
    total_neighbors = len(neighbor_atoms)

    print(f"  Average neighbors: {avg_neighbors:.1f}")
    print(f"  Total pairs: {total_neighbors}")

    # =================================================================
    # METHOD 1: Run separate kernels (Phase 2 + Phase 3B)
    # =================================================================
    print("\n1. Running SEPARATE kernels (Phase 2 + Phase 3B)...")

    # GB pairwise energy/forces (expects 2D coords)
    energy_sep, forces_sep = fennol_cuda.gb_compute_gb_energy_forces_neighborlist_mixed(
        coords, charges, born_radii, dielectric, cutoff,  # coords is 2D
        neighbor_atoms, neighbor_counts, neighbor_offsets
    )

    # dE/dR computation (expects 1D coords)
    dE_dR_sep = fennol_cuda.gb_compute_dE_dR_neighborlist_mixed(
        coords.flatten(), charges, born_radii, dielectric, cutoff,  # Flatten coords to 1D
        neighbor_atoms, neighbor_counts, neighbor_offsets
    )

    print(f"   Energy: {energy_sep[0]:.6e}")
    print(f"   Mean |force|: {np.mean(np.abs(forces_sep)):.6e}")
    print(f"   Mean |dE/dR|: {np.mean(np.abs(dE_dR_sep)):.6e}")

    # =================================================================
    # METHOD 2: Run fused kernel (Phase 3C)
    # =================================================================
    print("\n2. Running FUSED kernel (Phase 3C)...")

    energy_fused, forces_fused, dE_dR_fused = fennol_cuda.gb_compute_gb_and_dE_dR_fused_mixed(
        coords.flatten(), charges, born_radii, dielectric, cutoff,  # Flatten coords to 1D
        neighbor_atoms, neighbor_counts, neighbor_offsets
    )

    print(f"   Energy: {energy_fused[0]:.6e}")
    print(f"   Mean |force|: {np.mean(np.abs(forces_fused)):.6e}")
    print(f"   Mean |dE/dR|: {np.mean(np.abs(dE_dR_fused)):.6e}")

    # =================================================================
    # COMPARE RESULTS
    # =================================================================
    print("\n3. Comparing results...")

    # Compare energy
    energy_diff = np.abs(energy_sep[0] - energy_fused[0])
    energy_rel_err = energy_diff / max(abs(energy_sep[0]), 1e-10)

    print(f"\n   Energy:")
    print(f"     Separate: {energy_sep[0]:.10e}")
    print(f"     Fused:    {energy_fused[0]:.10e}")
    print(f"     Abs diff: {energy_diff:.6e}")
    print(f"     Rel err:  {energy_rel_err:.6e}")

    # Compare forces (reshape to same format - both to 1D)
    forces_sep_flat = forces_sep.flatten()
    forces_diff = np.abs(forces_sep_flat - forces_fused)
    max_force_diff = np.max(forces_diff)
    mean_force_diff = np.mean(forces_diff)

    force_denom = np.maximum(np.abs(forces_sep_flat), 1e-10)
    force_rel_err = forces_diff / force_denom
    max_force_rel_err = np.max(force_rel_err)
    mean_force_rel_err = np.mean(force_rel_err)

    print(f"\n   Forces:")
    print(f"     Max abs diff: {max_force_diff:.6e}")
    print(f"     Mean abs diff: {mean_force_diff:.6e}")
    print(f"     Max rel err: {max_force_rel_err:.6e}")
    print(f"     Mean rel err: {mean_force_rel_err:.6e}")

    # Compare dE/dR
    dE_dR_diff = np.abs(dE_dR_sep - dE_dR_fused)
    max_dE_dR_diff = np.max(dE_dR_diff)
    mean_dE_dR_diff = np.mean(dE_dR_diff)

    dE_dR_denom = np.maximum(np.abs(dE_dR_sep), 1e-10)
    dE_dR_rel_err = dE_dR_diff / dE_dR_denom
    max_dE_dR_rel_err = np.max(dE_dR_rel_err)
    mean_dE_dR_rel_err = np.mean(dE_dR_rel_err)

    print(f"\n   dE/dR:")
    print(f"     Max abs diff: {max_dE_dR_diff:.6e}")
    print(f"     Mean abs diff: {mean_dE_dR_diff:.6e}")
    print(f"     Max rel err: {max_dE_dR_rel_err:.6e}")
    print(f"     Mean rel err: {mean_dE_dR_rel_err:.6e}")

    # =================================================================
    # CHECK TOLERANCE
    # =================================================================
    tolerance = 1e-6

    passed = (
        energy_rel_err < tolerance and
        max_force_rel_err < tolerance and
        max_dE_dR_rel_err < tolerance
    )

    print(f"\n4. Validation result:")
    print(f"   Tolerance: {tolerance:.0e} (relative)")
    print(f"   Energy: {'✅ PASS' if energy_rel_err < tolerance else '❌ FAIL'}")
    print(f"   Forces: {'✅ PASS' if max_force_rel_err < tolerance else '❌ FAIL'}")
    print(f"   dE/dR: {'✅ PASS' if max_dE_dR_rel_err < tolerance else '❌ FAIL'}")
    print(f"   Overall: {'✅ PASS' if passed else '❌ FAIL'}")

    if not passed:
        print(f"\n   ERROR: Some results exceed tolerance")
        return False

    print("\n" + "="*70)
    print("PHASE 3C VALIDATION: ✅ SUCCESS")
    print("="*70)
    print("\nFused kernel produces identical results to separate kernels!")
    print("Expected performance gain: ~1.10× from kernel fusion")

    return True


def test_large_system():
    """Test with larger system to verify scaling."""
    print("\n" + "="*70)
    print("LARGE SYSTEM TEST")
    print("="*70)
    print("\nTesting fused kernel on larger system")
    print("-" * 70)

    natoms = 500
    cutoff = 12.0
    dielectric = 78.3

    print(f"\nSystem:")
    print(f"  Atoms: {natoms}")
    print(f"  Cutoff: {cutoff} Å")

    # Create system
    coords, charges, born_radii = create_test_system(natoms, seed=456)
    neighbor_atoms, neighbor_counts, neighbor_offsets = build_neighbor_list(coords, cutoff)

    avg_neighbors = np.mean(neighbor_counts)
    print(f"  Average neighbors: {avg_neighbors:.1f}")

    # Run fused kernel
    print("\nRunning fused kernel...")
    energy, forces, dE_dR = fennol_cuda.gb_compute_gb_and_dE_dR_fused_mixed(
        coords.flatten(), charges, born_radii, dielectric, cutoff,  # Flatten coords to 1D
        neighbor_atoms, neighbor_counts, neighbor_offsets
    )

    print(f"  Energy: {energy[0]:.6e}")
    print(f"  Mean |force|: {np.mean(np.abs(forces)):.6e}")
    print(f"  Mean |dE/dR|: {np.mean(np.abs(dE_dR)):.6e}")

    # Check for NaN or Inf
    has_nan = np.any(np.isnan(forces)) or np.any(np.isnan(dE_dR)) or np.isnan(energy[0])
    has_inf = np.any(np.isinf(forces)) or np.any(np.isinf(dE_dR)) or np.isinf(energy[0])

    print(f"\n  No NaN values: {'✅ PASS' if not has_nan else '❌ FAIL'}")
    print(f"  No Inf values: {'✅ PASS' if not has_inf else '❌ FAIL'}")

    if has_nan or has_inf:
        print("\n  ERROR: Found NaN or Inf values!")
        return False

    print("\n" + "="*70)
    print("LARGE SYSTEM TEST: ✅ PASS")
    print("="*70)

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 3C: Fused GB + dE/dR Kernel")
    print("="*70)
    print("\nThis fuses GB pairwise forces and dE/dR into a single kernel.")
    print("Benefits:")
    print("  - Eliminates one kernel launch overhead")
    print("  - Reuses computed values (r, f_GB, derivatives)")
    print("  - Single neighbor list traversal")
    print("Expected performance gain: ~1.10×")

    success = True

    # Test 1: Validation
    if not test_fused_kernel_validation():
        success = False

    # Test 2: Large system
    if not test_large_system():
        success = False

    if success:
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\nPhase 3C implementation is correct!")
        print("Fused kernel produces identical results to separate kernels.")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ❌")
        print("="*70)
        sys.exit(1)
