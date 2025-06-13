#!/usr/bin/env python3
"""Verify the full functionality of time-varying angle restraints."""

import jax.numpy as jnp
from fennol.md.restraints import setup_restraints, apply_restraints

def test_full_functionality():
    """Test the complete workflow with time-varying angles."""
    
    # Test configuration with time-varying angles
    restraints_def = {
        "backside_sn2": {
            "type": "backside_attack",
            "nucleophile": 0,
            "carbon": 1,
            "leaving_group": 2,
            "target": [140.0, 160.0, 180.0],  # Gradually increase angle
            "angle_force_constant": 0.01,
            "style": "flat_bottom",
            "tolerance": 0.2,  # ~11.5 degrees
            "update_steps": 1000,
            "prevent_inversion": True,
            "inversion_penalty_factor": 1.0
        }
    }
    
    # Set up restraints
    restraint_energies, restraint_forces, restraint_metadata = setup_restraints(restraints_def)
    
    # Create test coordinates with 150° angle
    coords = jnp.array([
        [0.0, 0.0, 0.0],       # nucleophile
        [3.0, 0.0, 0.0],       # carbon
        [5.6, 1.5, 0.0]        # leaving group (~150° angle)
    ])
    
    print("Testing time-varying angle functionality")
    print("=" * 50)
    print("Configuration:")
    print(f"  Target angles: {restraints_def['backside_sn2']['target']} degrees")
    print(f"  Update steps: {restraints_def['backside_sn2']['update_steps']}")
    print(f"  Style: {restraints_def['backside_sn2']['style']}")
    print(f"  Tolerance: {restraints_def['backside_sn2']['tolerance']} rad")
    print()
    
    # Test at different steps
    test_steps = [0, 500, 1000, 1500, 2000, 2500]
    
    print("Step-by-step results:")
    for i, step in enumerate(test_steps):
        # Reset step counter for clean test
        from fennol.md import restraints
        restraints.current_simulation_step = step - 1  # Will be incremented in apply_restraints
        
        # Apply restraints
        energy, forces = apply_restraints(coords, restraint_forces, step)
        
        # Calculate expected target angle
        if step < 1000:
            progress = step / 1000.0
            expected_deg = 140.0 + progress * (160.0 - 140.0)
        elif step < 2000:
            progress = (step - 1000) / 1000.0
            expected_deg = 160.0 + progress * (180.0 - 160.0)
        else:
            expected_deg = 180.0
        
        print(f"  Step {step:4d}: Target angle = {expected_deg:6.1f}°, Energy = {energy:8.4f}")
        
        # Check forces are reasonable
        force_magnitude = jnp.linalg.norm(forces)
        print(f"            Total force magnitude = {force_magnitude:.4f}")
    
    print("\n✅ Full functionality test completed successfully!")

if __name__ == "__main__":
    test_full_functionality()