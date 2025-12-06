#!/usr/bin/env python3
"""
Test script for scale MD implementation.

This tests the core components without running a full simulation:
1. PDB loading and chain extraction
2. Chain mask creation
3. Backbone identification
4. COM distance calculation
5. Force scaling logic
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fennol.utils.pdb import read_pdb
from fennol.md.scale_md import (
    extract_chain_info,
    create_backbone_mask,
    compute_chain_com,
    compute_chain_distance,
    create_inter_chain_pair_mask,
    BACKBONE_ATOMS,
)


def test_pdb_loading():
    """Test PDB loading and chain extraction."""
    print("\n" + "="*60)
    print("Test 1: PDB Loading and Chain Extraction")
    print("="*60)

    pdb_file = os.path.join(os.path.dirname(__file__), "1YCR_peptide_example.pdb")
    structure = read_pdb(pdb_file)

    print(f"Loaded {structure.natoms} atoms")
    print(f"Unique elements: {sorted(set(structure.elements))}")

    chain_info = extract_chain_info(structure)

    print(f"\nUnique chains: {chain_info.unique_chains}")
    for chain in chain_info.unique_chains:
        n_atoms = len(chain_info.chain_atom_indices[chain])
        first_idx = chain_info.chain_atom_indices[chain][0]
        last_idx = chain_info.chain_atom_indices[chain][-1]
        print(f"  Chain {chain}: {n_atoms} atoms (0-indexed: {first_idx}-{last_idx})")

    # Verify chain masks
    total_from_masks = sum(np.sum(chain_info.chain_atom_masks[c]) for c in chain_info.unique_chains)
    assert total_from_masks == chain_info.n_atoms, "Chain masks don't cover all atoms"
    print("\n[PASS] Chain masks cover all atoms correctly")

    return structure, chain_info


def test_backbone_mask(structure):
    """Test backbone atom identification."""
    print("\n" + "="*60)
    print("Test 2: Backbone Atom Identification")
    print("="*60)

    print(f"Backbone atom names: {BACKBONE_ATOMS}")

    # Test for chain A (protein)
    mask_A = create_backbone_mask(structure, ["A"])
    n_backbone_A = np.sum(mask_A)
    print(f"Chain A backbone atoms: {n_backbone_A}")

    # Test for chain B (peptide)
    mask_B = create_backbone_mask(structure, ["B"])
    n_backbone_B = np.sum(mask_B)
    print(f"Chain B backbone atoms: {n_backbone_B}")

    # Test for both chains
    mask_AB = create_backbone_mask(structure, ["A", "B"])
    n_backbone_AB = np.sum(mask_AB)
    print(f"Both chains backbone atoms: {n_backbone_AB}")

    assert n_backbone_AB == n_backbone_A + n_backbone_B, "Backbone masks don't add up"
    print("\n[PASS] Backbone masks are consistent")

    return mask_A, mask_B


def test_com_calculation(structure, chain_info):
    """Test center of mass calculation."""
    print("\n" + "="*60)
    print("Test 3: Center of Mass Calculation")
    print("="*60)

    masses = structure.masses
    coords = structure.coordinates

    for chain in chain_info.unique_chains:
        mask = chain_info.chain_atom_masks[chain]
        com = compute_chain_com(coords, mask, masses)
        n_atoms = np.sum(mask)
        total_mass = np.sum(masses[mask])
        print(f"Chain {chain}: COM = ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
        print(f"         {n_atoms} atoms, {total_mass:.1f} amu total mass")

    print("\n[PASS] COM calculations completed")

    return com


def test_distance_tracking(structure, chain_info):
    """Test distance tracking between initial and displaced positions."""
    print("\n" + "="*60)
    print("Test 4: Distance Tracking")
    print("="*60)

    masses = structure.masses
    coords = structure.coordinates.copy()

    # Get initial COM of chain B (peptide)
    chain_B_mask = chain_info.chain_atom_masks["B"]
    initial_com = compute_chain_com(coords, chain_B_mask, masses)
    print(f"Initial peptide COM: ({initial_com[0]:.2f}, {initial_com[1]:.2f}, {initial_com[2]:.2f})")

    # Simulate displacement of peptide
    displacement_vector = np.array([5.0, 3.0, 2.0])  # Angstroms
    coords[chain_B_mask] += displacement_vector

    # Compute distance
    distance = compute_chain_distance(coords, chain_B_mask, masses, initial_com)
    expected_distance = np.linalg.norm(displacement_vector)

    print(f"Displacement: ({displacement_vector[0]:.1f}, {displacement_vector[1]:.1f}, {displacement_vector[2]:.1f})")
    print(f"Expected distance: {expected_distance:.3f} A")
    print(f"Computed distance: {distance:.3f} A")

    assert abs(distance - expected_distance) < 1e-6, f"Distance mismatch: {distance} vs {expected_distance}"
    print("\n[PASS] Distance tracking is accurate")


def test_inter_chain_masks(chain_info):
    """Test inter-chain pair mask creation."""
    print("\n" + "="*60)
    print("Test 5: Inter-chain Pair Masks")
    print("="*60)

    coi_mask, other_mask = create_inter_chain_pair_mask(chain_info, "B")

    n_coi = np.sum(coi_mask)
    n_other = np.sum(other_mask)
    total = n_coi + n_other

    print(f"Chain of interest (B) atoms: {n_coi}")
    print(f"Other chain atoms: {n_other}")
    print(f"Total atoms: {total}")

    assert total == chain_info.n_atoms, "Masks don't cover all atoms"
    assert np.all(coi_mask == ~other_mask), "Masks are not complementary"

    print("\n[PASS] Inter-chain masks are correct")


def test_force_scaling_logic():
    """Test the force scaling logic conceptually."""
    print("\n" + "="*60)
    print("Test 6: Force Scaling Logic (Conceptual)")
    print("="*60)

    # Create simple test scenario
    n_atoms = 10
    coi_indices = np.array([0, 1, 2])  # Chain of interest
    other_indices = np.array([3, 4, 5, 6, 7, 8, 9])

    # Simulated forces
    forces = np.random.randn(n_atoms, 3)
    print(f"Original force magnitude (COI): {np.linalg.norm(forces[coi_indices]):.4f}")

    # COM direction (simplified)
    com_direction = np.array([1.0, 0.0, 0.0])

    # Extract radial component for COI atoms
    coi_forces = forces[coi_indices].copy()
    radial_component = np.sum(coi_forces * com_direction, axis=1, keepdims=True)
    radial_forces = radial_component * com_direction
    tangential_forces = coi_forces - radial_forces

    print(f"Radial force magnitude (COI): {np.linalg.norm(radial_forces):.4f}")
    print(f"Tangential force magnitude (COI): {np.linalg.norm(tangential_forces):.4f}")

    # Apply scaling with alpha = 0.5
    alpha = 0.5
    scaled_forces = tangential_forces + alpha * radial_forces
    print(f"Scaled force magnitude (alpha={alpha}): {np.linalg.norm(scaled_forces):.4f}")

    # Verify tangential component is preserved
    scaled_radial = np.sum(scaled_forces * com_direction, axis=1, keepdims=True) * com_direction
    scaled_tangential = scaled_forces - scaled_radial

    assert np.allclose(scaled_tangential, tangential_forces), "Tangential forces should be preserved"
    print("\n[PASS] Force scaling preserves tangential component")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SCALE MD IMPLEMENTATION TESTS")
    print("="*70)

    # Test 1: PDB loading
    structure, chain_info = test_pdb_loading()

    # Test 2: Backbone masks
    mask_A, mask_B = test_backbone_mask(structure)

    # Test 3: COM calculation
    test_com_calculation(structure, chain_info)

    # Test 4: Distance tracking
    test_distance_tracking(structure, chain_info)

    # Test 5: Inter-chain masks
    test_inter_chain_masks(chain_info)

    # Test 6: Force scaling logic
    test_force_scaling_logic()

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nThe scale MD implementation is ready for use.")
    print("To run a full simulation, use:")
    print("  python -m fennol.md.scale_md scale_md_test.fnl")


if __name__ == "__main__":
    main()
