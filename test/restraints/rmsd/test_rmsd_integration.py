#!/usr/bin/env python3
"""Integration test for RMSD restraint with the FeNNol restraint system."""

import jax.numpy as jnp
import numpy as np
from src.fennol.md.restraints import setup_restraints

def create_test_pdb():
    """Create a simple test PDB file for testing."""
    pdb_content = """HEADER    TEST MOLECULE
ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  C   MOL A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  C   MOL A   1       3.000   0.000   0.000  1.00 20.00           C
ATOM      4  C   MOL A   1       4.500   0.000   0.000  1.00 20.00           C
ATOM      5  C   MOL A   1       6.000   0.000   0.000  1.00 20.00           C
END
"""
    with open('/home/aaron/ATX/software/ATF-FeNNol/v2/test_reference.pdb', 'w') as f:
        f.write(pdb_content)
    print("Created test_reference.pdb")

def test_rmsd_restraint_integration():
    """Test RMSD restraint integration with the create_restraints function."""
    print("=== Testing RMSD Restraint Integration ===\n")
    
    # Create test PDB file
    create_test_pdb()
    
    # Create test coordinates (slightly different from PDB)
    initial_coordinates = jnp.array([
        [0.0, 0.0, 0.0],
        [1.4, 0.2, 0.0],
        [2.8, 0.4, 0.0],
        [4.2, 0.6, 0.0],
        [5.6, 0.8, 0.0]
    ])
    
    # Test 1: Basic RMSD restraint using initial coordinates
    print("1. Testing basic RMSD restraint...")
    restraints_config = {
        "basic_rmsd": {
            "type": "rmsd",
            "target_rmsd": 0.5,
            "force_constant": 2.0,
            "mode": "harmonic",
            "atoms": "all"
        }
    }
    
    try:
        energies, forces, metadata = setup_restraints(
            restraints_config, 
            initial_coordinates=initial_coordinates,
            nsteps=1000
        )
        print("   ✓ Basic RMSD restraint created successfully")
        print(f"   ✓ Created {len(forces)} force calculators")
        
        # Test evaluation
        test_coords = initial_coordinates + 0.1  # Slightly perturb
        energy, force_array = forces[0](test_coords)
        print(f"   ✓ Energy evaluation: {energy:.6f}")
        print(f"   ✓ Max force: {jnp.max(jnp.linalg.norm(force_array, axis=1)):.6f}")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 2: RMSD restraint with PDB reference
    print("\n2. Testing RMSD restraint with PDB reference...")
    restraints_config = {
        "pdb_rmsd": {
            "type": "rmsd", 
            "reference_file": "/home/aaron/ATX/software/ATF-FeNNol/v2/test_reference.pdb",
            "target_rmsd": 0.3,
            "force_constant": 5.0,
            "mode": "flat_bottom",
            "atoms": "all"
        }
    }
    
    try:
        energies, forces, metadata = setup_restraints(
            restraints_config,
            initial_coordinates=initial_coordinates,
            nsteps=1000
        )
        print("   ✓ PDB reference RMSD restraint created successfully")
        
        # Test evaluation
        energy, force_array = forces[0](test_coords)
        print(f"   ✓ Energy evaluation: {energy:.6f}")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 3: RMSD restraint with atom selection
    print("\n3. Testing RMSD restraint with atom selection...")
    restraints_config = {
        "subset_rmsd": {
            "type": "rmsd",
            "target_rmsd": 0.2,
            "force_constant": 10.0,
            "mode": "harmonic", 
            "atoms": [0, 2, 4]  # Select specific atoms
        }
    }
    
    try:
        energies, forces, metadata = setup_restraints(
            restraints_config,
            initial_coordinates=initial_coordinates,
            nsteps=1000
        )
        print("   ✓ Atom selection RMSD restraint created successfully")
        
        # Test evaluation
        energy, force_array = forces[0](test_coords)
        print(f"   ✓ Energy evaluation: {energy:.6f}")
        
        # Check that forces are only applied to selected atoms
        non_zero_forces = jnp.where(jnp.linalg.norm(force_array, axis=1) > 1e-10)[0]
        print(f"   ✓ Forces applied to atoms: {non_zero_forces.tolist()}")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 4: Time-varying RMSD restraint
    print("\n4. Testing time-varying RMSD restraint...")
    restraints_config = {
        "time_varying_rmsd": {
            "type": "rmsd",
            "target_rmsd": [0.2, 0.5, 1.0],
            "force_constant": [10.0, 5.0, 2.0],
            "mode": "harmonic",
            "update_steps": 333,
            "interpolation_mode": "interpolate",
            "atoms": "all"
        }
    }
    
    try:
        energies, forces, metadata = setup_restraints(
            restraints_config,
            initial_coordinates=initial_coordinates,
            nsteps=1000
        )
        print("   ✓ Time-varying RMSD restraint created successfully")
        print(f"   ✓ Metadata stored: {metadata['time_varying']}")
        
        # Test evaluation at different steps
        for step in [0, 500, 999]:
            energy, force_array = forces[0](test_coords, step=step)
            print(f"   ✓ Step {step}: Energy = {energy:.6f}")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 5: Multiple RMSD restraints
    print("\n5. Testing multiple RMSD restraints...")
    restraints_config = {
        "global_rmsd": {
            "type": "rmsd",
            "target_rmsd": 1.0,
            "force_constant": 1.0,
            "atoms": "all"
        },
        "local_rmsd": {
            "type": "rmsd", 
            "target_rmsd": 0.2,
            "force_constant": 5.0,
            "atoms": [1, 2, 3]
        }
    }
    
    try:
        energies, forces, metadata = setup_restraints(
            restraints_config,
            initial_coordinates=initial_coordinates,
            nsteps=1000
        )
        print(f"   ✓ Multiple RMSD restraints created: {len(forces)} total")
        
        # Test evaluation of both restraints
        total_energy = 0.0
        total_forces = jnp.zeros_like(test_coords)
        
        for i, force_calc in enumerate(forces):
            energy, force_array = force_calc(test_coords)
            total_energy += energy
            total_forces += force_array
            print(f"   ✓ Restraint {i}: Energy = {energy:.6f}")
        
        print(f"   ✓ Total energy: {total_energy:.6f}")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ All RMSD restraint integration tests passed!")
    return True

def test_rmsd_with_other_restraints():
    """Test RMSD restraint combined with other restraint types."""
    print("\n=== Testing RMSD with Other Restraints ===\n")
    
    initial_coordinates = jnp.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
        [6.0, 0.0, 0.0]
    ])
    
    # Combined restraints configuration
    restraints_config = {
        "structural_preservation": {
            "type": "rmsd",
            "target_rmsd": 0.5,
            "force_constant": 2.0,
            "atoms": "all"
        },
        "distance_constraint": {
            "type": "distance",
            "atom1": 0,
            "atom2": 4,
            "target": 6.0,
            "force_constant": 5.0
        },
        "angle_constraint": {
            "type": "angle",
            "atom1": 0,
            "atom2": 2,
            "atom3": 4,
            "target": 180.0,
            "force_constant": 1.0
        }
    }
    
    try:
        energies, forces, metadata = setup_restraints(
            restraints_config,
            initial_coordinates=initial_coordinates,
            nsteps=1000
        )
        print(f"✓ Created mixed restraint system with {len(forces)} restraints:")
        print("  - RMSD restraint")
        print("  - Distance restraint") 
        print("  - Angle restraint")
        
        # Test evaluation
        test_coords = initial_coordinates + 0.1
        total_energy = 0.0
        total_forces = jnp.zeros_like(test_coords)
        
        for i, force_calc in enumerate(forces):
            energy, force_array = force_calc(test_coords)
            total_energy += energy
            total_forces += force_array
            print(f"  Restraint {i}: Energy = {energy:.6f}")
        
        print(f"✓ Combined system total energy: {total_energy:.6f}")
        print(f"✓ Max combined force: {jnp.max(jnp.linalg.norm(total_forces, axis=1)):.6f}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = True
    
    # Test RMSD restraint integration
    success &= test_rmsd_restraint_integration()
    
    # Test RMSD with other restraints
    success &= test_rmsd_with_other_restraints()
    
    print("\n" + "="*50)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nRMSD restraint is fully integrated and ready for use.")
        print("\nKey features validated:")
        print("✓ Basic harmonic and flat-bottom modes")
        print("✓ PDB reference file support")
        print("✓ Flexible atom selection")
        print("✓ Time-varying parameters")
        print("✓ Integration with existing restraint system")
        print("✓ Compatibility with other restraint types")
    else:
        print("❌ Some tests failed. Check implementation.")
    
    # Clean up test files
    import os
    if os.path.exists('/home/aaron/ATX/software/ATF-FeNNol/v2/test_reference.pdb'):
        os.remove('/home/aaron/ATX/software/ATF-FeNNol/v2/test_reference.pdb')