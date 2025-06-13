#!/usr/bin/env python3
"""Test that the parsing fix works with the provided input configuration."""

import jax.numpy as jnp
from fennol.md.restraints import setup_restraints

def test_parsing_fix():
    """Test that we can parse restraints with time-varying target angles."""
    
    # This is the restraints section from the failing input
    restraints_def = {
        "backside_attack_restraint": {
            "type": "backside_attack",
            "nucleophile": 52,
            "carbon": 245,
            "leaving_group": 249,
            "style": "flat_bottom",
            "prevent_inversion": True,
            "inversion_penalty_factor": 1.0,
            "target": [90.0, 120.0, 150.0, 180.0],  # List of angles
            "angle_force_constant": [0.008, 0.010, 0.012],
            "target_distance": 5,
            "distance_force_constant": 0.01,
            "update_steps": 200
        },
        "keep_restraint1": {
            "type": "distance",
            "atom1": 52,
            "atom2": 245,
            "target": 6.0,  # Single float
            "force_constant": 0.02,
            "style": "flat_bottom",
            "tolerance": 0.2,
            "update_steps": 500
        },
        "keep_restraint2": {
            "type": "distance",
            "atom1": 52,
            "atom2": 249,
            "target": 7.0,  # Single float
            "force_constant": 0.02,
            "style": "flat_bottom",
            "tolerance": 0.2,
            "update_steps": 500
        }
    }
    
    try:
        # This should now work without errors
        restraint_energies, restraint_forces, restraint_metadata = setup_restraints(restraints_def)
        
        print("✅ Parsing successful!")
        print(f"Number of restraints created: {len(restraint_forces)}")
        print(f"Time-varying restraints: {restraint_metadata['time_varying']}")
        
        # Check that backside attack restraint has time-varying metadata
        if "backside_attack_restraint" in restraint_metadata["restraints"]:
            meta = restraint_metadata["restraints"]["backside_attack_restraint"]
            print(f"\nBackside attack restraint metadata:")
            print(f"  Target angles: {meta.get('target_angles', 'N/A')}")
            print(f"  Angle force constants: {meta.get('angle_force_constants', 'N/A')}")
            print(f"  Update steps: {meta.get('update_steps', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parsing_fix()
    exit(0 if success else 1)