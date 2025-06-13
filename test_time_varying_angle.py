#!/usr/bin/env python3
"""Test script to verify time-varying target angle for backside attack restraint."""

import jax.numpy as jnp
import numpy as np
from fennol.md.restraints import (
    setup_restraints, apply_restraints, time_varying_force_constant,
    colvar_angle, colvar_distance
)

def test_time_varying_angle():
    """Test that the target angle changes over time as expected."""
    
    # Define test restraint with time-varying target angle
    restraint_def = {
        "test_backside": {
            "type": "backside_attack",
            "nucleophile": 0,
            "carbon": 1,
            "leaving_group": 2,
            "target": [140.0, 150.0, 160.0, 170.0, 180.0],  # degrees
            "angle_force_constant": 0.01,
            "update_steps": 1000,
            "style": "harmonic",
            "prevent_inversion": False  # Simplify for testing
        }
    }
    
    # Set up restraints
    restraint_energies, restraint_forces, restraint_metadata = setup_restraints(restraint_def)
    
    # Create test coordinates (linear arrangement)
    coords = jnp.array([
        [0.0, 0.0, 0.0],    # nucleophile
        [3.0, 0.0, 0.0],    # carbon
        [6.0, 0.0, 0.0]     # leaving group
    ])
    
    print("Testing time-varying target angle for backside attack restraint")
    print("=" * 60)
    print(f"Target angles: {restraint_def['test_backside']['target']} degrees")
    print(f"Update steps: {restraint_def['test_backside']['update_steps']}")
    print()
    
    # Test at various steps
    test_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    
    for step in test_steps:
        # Calculate expected target angle
        angles_rad = [angle * np.pi / 180.0 for angle in restraint_def['test_backside']['target']]
        expected_angle = time_varying_force_constant(
            angles_rad, 
            restraint_def['test_backside']['update_steps'],
            step,
            name="test_angle"
        )
        expected_angle_deg = expected_angle * 180.0 / np.pi
        
        # Apply restraints
        energy, forces = apply_restraints(coords, restraint_forces, step)
        
        # Calculate actual angle
        actual_angle = colvar_angle(coords, 0, 1, 2)
        actual_angle_deg = actual_angle * 180.0 / np.pi
        
        print(f"Step {step:5d}: Target = {expected_angle_deg:6.1f}°, "
              f"Actual = {actual_angle_deg:6.1f}°, Energy = {energy:8.4f}")
    
    print("\nTest completed successfully!")

def test_combined_time_varying():
    """Test combined time-varying angle and force constant."""
    
    # Define test restraint with both time-varying
    restraint_def = {
        "combined_test": {
            "type": "backside_attack",
            "nucleophile": 0,
            "carbon": 1,
            "leaving_group": 2,
            "target": [140.0, 160.0, 180.0],  # degrees
            "angle_force_constant": [0.01, 0.05, 0.1],  # increasing force
            "update_steps": 1000,
            "style": "harmonic",
            "prevent_inversion": False
        }
    }
    
    # Set up restraints
    restraint_energies, restraint_forces, restraint_metadata = setup_restraints(restraint_def)
    
    # Create test coordinates with 150° angle
    coords = jnp.array([
        [0.0, 0.0, 0.0],       # nucleophile
        [3.0, 0.0, 0.0],       # carbon
        [5.6, 1.5, 0.0]        # leaving group (150° angle)
    ])
    
    print("\n\nTesting combined time-varying angle and force constant")
    print("=" * 60)
    print(f"Target angles: {restraint_def['combined_test']['target']} degrees")
    print(f"Force constants: {restraint_def['combined_test']['angle_force_constant']}")
    print(f"Update steps: {restraint_def['combined_test']['update_steps']}")
    print()
    
    # Test at various steps
    test_steps = [0, 500, 1000, 1500, 2000, 2500, 3000]
    
    for step in test_steps:
        # Apply restraints
        energy, forces = apply_restraints(coords, restraint_forces, step)
        
        # Calculate actual angle
        actual_angle = colvar_angle(coords, 0, 1, 2)
        actual_angle_deg = actual_angle * 180.0 / np.pi
        
        # Get expected values
        angles_rad = [angle * np.pi / 180.0 for angle in restraint_def['combined_test']['target']]
        expected_angle = time_varying_force_constant(
            angles_rad, 
            restraint_def['combined_test']['update_steps'],
            step
        )
        expected_angle_deg = expected_angle * 180.0 / np.pi
        
        expected_fc = time_varying_force_constant(
            restraint_def['combined_test']['angle_force_constant'],
            restraint_def['combined_test']['update_steps'],
            step
        )
        
        print(f"Step {step:5d}: Target = {expected_angle_deg:6.1f}°, "
              f"Force const = {expected_fc:5.3f}, Energy = {energy:8.4f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_time_varying_angle()
    test_combined_time_varying()