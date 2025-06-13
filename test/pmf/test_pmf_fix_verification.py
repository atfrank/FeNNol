#!/usr/bin/env python3
"""
Verification script to test that PMF initialization and progress messages are working.
This tests the exact user configuration to ensure PMF messages appear.
"""

def test_pmf_fix():
    from src.fennol.md import restraints
    from src.fennol.utils.input_parser import parse_input
    import jax.numpy as jnp

    print("=== PMF Fix Verification Test ===\n")

    # Reset PMF data
    restraints.pmf_data = {}
    restraints.current_simulation_step = 0

    # User's exact configuration
    config_content = """
device cpu
model_file examples/md/ani2x.fnx

xyz_input{
  nxyz = 3
  coordinates[Angstrom] {
    O     0.0    0.0    0.0
    H1    2.5    0.0    0.0
    H2   -0.5    0.0    0.0
  }
}

restraints {
  backside_attack_restraint {
    type = backside_attack
    nucleophile = 1
    carbon = 0
    leaving_group = 2
    
    target = 180.0
    angle_force_constant = 0.0
    
    target_distance = 2.5, 2.4, 2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5
    distance_force_constant = 0.1
    
    interpolation_mode = discrete
    estimate_pmf = yes
    equilibration_ratio = 0.3
    update_steps = 300
    
    style = harmonic
  }
}

restraint_debug yes
"""

    # Write and parse config
    with open('temp_test_config.fnl', 'w') as f:
        f.write(config_content)
    
    try:
        config = parse_input('temp_test_config.fnl')
        
        # Enable debug
        restraints.restraint_debug_enabled = True

        # Set up restraints
        restraint_defs = config.get('restraints', {})
        restraint_energies, restraint_forces, restraint_metadata = restraints.setup_restraints(restraint_defs)

        # Test coordinates
        coords = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])

        print("Testing key PMF events:\n")
        
        # Test key events
        events = [
            (1, "First step - PMF initialization"),
            (90, "Start of sampling (end of equilibration)"),
            (300, "Window transition from 2.5 to 2.4 Å")
        ]
        
        for target_step, description in events:
            print(f"--- {description} (Step {target_step}) ---")
            restraints.current_simulation_step = target_step - 1
            energy, forces = restraints.apply_restraints(coords, restraint_forces, target_step)
            print(f"Energy: {energy:.6f}\n")

        # Check PMF data was created
        if 'backside_attack_restraint' in restraints.pmf_data:
            print("✅ SUCCESS: PMF data structure created")
            pmf_info = restraints.pmf_data['backside_attack_restraint']
            print(f"   Windows: {len(pmf_info['windows'])}")
            print(f"   Samples in window 0: {len(pmf_info['samples'][0])}")
            print(f"   Sampling started: {pmf_info['sampling_started']}")
        else:
            print("❌ FAILED: PMF data structure not created")
            
    finally:
        # Cleanup
        import os
        if os.path.exists('temp_test_config.fnl'):
            os.remove('temp_test_config.fnl')

    print("\n=== Test Complete ===")
    print("If you see PMF INITIALIZATION and progress messages above, the fix is working!")

if __name__ == "__main__":
    test_pmf_fix()