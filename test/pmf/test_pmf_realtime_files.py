#!/usr/bin/env python3
"""
Test PMF real-time file creation
"""

def test_pmf_realtime_files():
    from src.fennol.md import restraints
    from src.fennol.utils.input_parser import parse_input
    import jax.numpy as jnp
    import os

    print("=== Testing PMF Real-time File Creation ===\n")

    # Reset PMF data
    restraints.pmf_data = {}
    restraints.current_simulation_step = 0

    # Create test config with real-time output enabled
    config_content = """
restraints {
  test_realtime {
    type = backside_attack
    nucleophile = 1
    carbon = 0
    leaving_group = 2
    
    target = 180.0
    angle_force_constant = 0.0
    
    target_distance = 2.5, 2.4
    distance_force_constant = 0.1
    
    interpolation_mode = discrete
    estimate_pmf = yes
    equilibration_ratio = 0.3
    update_steps = 50
    write_realtime_pmf = yes  # Enable real-time output
    
    style = harmonic
  }
}

restraint_debug yes
"""

    # Write and parse config
    with open('temp_realtime_test.fnl', 'w') as f:
        f.write(config_content)
    
    try:
        config = parse_input('temp_realtime_test.fnl')
        
        # Enable debug
        restraints.restraint_debug_enabled = True

        # Set up restraints
        restraint_defs = config.get('restraints', {})
        restraint_energies, restraint_forces, restraint_metadata = restraints.setup_restraints(restraint_defs)

        # Test coordinates
        coords = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])

        print("Simulating PMF collection with real-time output...\n")
        
        # Simulate collecting data across 2 windows
        for step in range(1, 101):  # 2 complete windows of 50 steps each
            restraints.current_simulation_step = step - 1
            energy, forces = restraints.apply_restraints(coords, restraint_forces, step)
            
            # Print less frequently to reduce output
            if step in [1, 15, 50, 65, 100]:
                print(f"Step {restraints.current_simulation_step}: Energy = {energy:.6f}")

        print("\n=== Checking Real-time Files ===")
        
        # Check for real-time files
        realtime_files = [f for f in os.listdir('.') if f.startswith('pmf_') and f.endswith('_realtime.dat')]
        
        if realtime_files:
            print(f"✅ Real-time files created: {realtime_files}")
            
            # Show content of real-time file
            for filename in realtime_files:
                print(f"\nContents of {filename}:")
                print("-" * 50)
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    # Show header and first few data lines
                    for i, line in enumerate(lines):
                        if i < 10 or i >= len(lines) - 5:  # First 10 and last 5 lines
                            print(line.rstrip())
                        elif i == 10:
                            print("... (middle lines omitted) ...")
                print("-" * 50)
        else:
            print("❌ No real-time files created")
            
        # Also create final PMF output
        print("\n=== Creating Final PMF Output ===")
        restraints.write_pmf_output("test_realtime_pmf_output.dat", temperature=300.0)
        
        if os.path.exists("test_realtime_pmf_output.dat"):
            print("✅ Final PMF output file created")
            
    finally:
        # Cleanup
        cleanup_files = ['temp_realtime_test.fnl', 'test_realtime_pmf_output.dat']
        for f in cleanup_files:
            if os.path.exists(f):
                os.remove(f)
        
        # Clean up any real-time files
        for f in os.listdir('.'):
            if f.startswith('pmf_') and f.endswith('_realtime.dat'):
                os.remove(f)

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pmf_realtime_files()