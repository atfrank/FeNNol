#!/usr/bin/env python
"""
Test CUDA implicit solvent on DHFR protein (2,499 atoms)

This demonstrates:
1. Loading DHFR structure without water
2. Computing solvation energy and forces with CUDA
3. Timing performance
"""

import numpy as np
import jax.numpy as jnp
import time
from pathlib import Path

# Import implicit solvent model
from fennol.models.physics.implicit_solvent import OBC
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX

def read_tinker_xyz(filename):
    """
    Read Tinker XYZ format file.

    Format:
    natoms [comment]
    idx element x y z [connectivity...]
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # First line: number of atoms
    natoms = int(lines[0].strip().split()[0])

    # Parse atoms
    coords = []
    elements = []

    for line in lines[1:natoms+1]:
        parts = line.strip().split()
        idx = int(parts[0])
        element = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

        coords.append([x, y, z])
        elements.append(element)

    return np.array(coords), elements

def get_atomic_numbers(elements):
    """Convert element symbols to atomic numbers."""
    return np.array([PERIODIC_TABLE_REV_IDX[e] for e in elements])

def estimate_charges_simple(elements):
    """
    Simple charge estimation for proteins.

    For a real simulation, you'd need proper charge assignment
    based on force field (AMBER, CHARMM, etc.)

    For testing purposes, we'll use rough estimates:
    - N (terminal): -0.3
    - O (carbonyl): -0.5
    - H: +0.3 (on N) or 0.0 (on C)
    - C: varies, use ~0.0 to 0.5
    - S: ~0.0
    """
    charges = []
    for elem in elements:
        if elem == 'N':
            charges.append(-0.3)
        elif elem == 'O':
            charges.append(-0.5)
        elif elem == 'H':
            charges.append(0.3)
        elif elem == 'C':
            charges.append(0.1)
        elif elem == 'S':
            charges.append(0.0)
        else:
            charges.append(0.0)

    # Normalize to ensure neutrality
    charges = np.array(charges)
    net_charge = charges.sum()
    charges -= net_charge / len(charges)

    return charges

def main():
    print("=" * 70)
    print("DHFR Protein Implicit Solvent Test (CUDA)")
    print("=" * 70)
    print()

    # Load DHFR structure
    xyz_file = Path("dhfr2_nowat.xyz")
    if not xyz_file.exists():
        print(f"Error: {xyz_file} not found")
        print("Please run this script from the examples/md/dhfr directory")
        return 1

    print(f"Loading structure: {xyz_file}")
    coords, elements = read_tinker_xyz(xyz_file)
    natoms = len(coords)
    print(f"  Atoms: {natoms}")
    print(f"  Dimensions: {coords.min(axis=0)} to {coords.max(axis=0)}")
    print(f"  Box size: ~{coords.max(axis=0) - coords.min(axis=0)} Å")
    print()

    # Get atomic numbers
    atomic_numbers = get_atomic_numbers(elements)
    print(f"Atomic composition:")
    from collections import Counter
    elem_counts = Counter(elements)
    for elem, count in sorted(elem_counts.items()):
        print(f"  {elem}: {count}")
    print()

    # Estimate charges (rough approximation)
    charges = estimate_charges_simple(elements)
    print(f"Charge estimation:")
    print(f"  Net charge: {charges.sum():.6f} e")
    print(f"  Charge range: [{charges.min():.3f}, {charges.max():.3f}] e")
    print()

    # Convert to JAX arrays
    coords_jax = jnp.array(coords)
    charges_jax = jnp.array(charges)
    atomic_numbers_jax = jnp.array(atomic_numbers)

    # Initialize OBC model
    print("Initializing OBC Generalized Born model...")
    model = OBC({
        "dielectric": 80.0,
        "cutoff": 12.0,
        "surface_tension": 0.005,
        "probe_radius": 1.4,
        "include_nonpolar": True
    })
    print(f"  Backend: {'CUDA' if model.has_cuda else 'JAX'}")
    print()

    # Compute solvation energy and forces
    print("Computing solvation energy and forces...")

    # Warm-up run (JIT compilation)
    print("  Warm-up (JIT compilation)...")
    energy, forces = model(coords_jax, charges_jax, atomic_numbers_jax)
    print(f"    Done: E = {energy:.4f} kcal/mol")
    print()

    # Timed runs
    n_runs = 5
    print(f"  Timing {n_runs} evaluations...")
    times = []
    for i in range(n_runs):
        t0 = time.time()
        energy, forces = model(coords_jax, charges_jax, atomic_numbers_jax)
        # Ensure computation completes (block until ready)
        forces.block_until_ready()
        t1 = time.time()
        times.append(t1 - t0)
        print(f"    Run {i+1}: {(t1-t0)*1000:.2f} ms")

    print()
    print(f"  Average time: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
    print(f"  Throughput: {natoms / np.mean(times):.0f} atoms/second")
    print()

    # Results
    print("Results:")
    print(f"  Solvation energy: {energy:.4f} kcal/mol")
    print(f"  Energy per atom: {energy/natoms:.6f} kcal/mol")
    print(f"  Force magnitude: {np.linalg.norm(forces):.4f} kcal/mol/Å")
    print(f"  Max force: {np.abs(forces).max():.4f} kcal/mol/Å")
    print(f"  RMS force: {np.sqrt(np.mean(forces**2)):.4f} kcal/mol/Å")
    print()

    # Memory estimate
    print("Memory usage estimate:")
    coords_mem = coords.nbytes / 1024**2
    forces_mem = forces.nbytes / 1024**2
    print(f"  Coordinates: {coords_mem:.2f} MB")
    print(f"  Forces: {forces_mem:.2f} MB")
    print(f"  Total arrays: ~{(coords_mem + forces_mem)*2:.2f} MB")
    print()

    print("=" * 70)
    print("Test completed successfully! ✓")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
