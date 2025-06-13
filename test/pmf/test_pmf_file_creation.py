#!/usr/bin/env python3
"""
Test PMF file creation functionality
"""

def test_pmf_file_output():
    from src.fennol.md import restraints
    from src.fennol.utils.input_parser import parse_input
    import jax.numpy as jnp
    import os

    print("=== Testing PMF File Creation ===\n")

    # Reset PMF data
    restraints.pmf_data = {}
    restraints.current_simulation_step = 0

    # Create test config
    config_content = """
restraints {
  test_restraint {
    type = backside_attack
    nucleophile = 1
    carbon = 0
    leaving_group = 2
    
    target = 180.0
    angle_force_constant = 0.0
    
    target_distance = 2.5, 2.4, 2.3
    distance_force_constant = 0.1
    
    interpolation_mode = discrete
    estimate_pmf = yes
    equilibration_ratio = 0.3
    update_steps = 100
    
    style = harmonic
  }
}

restraint_debug yes
"""

    # Write and parse config
    with open('temp_pmf_test.fnl', 'w') as f:
        f.write(config_content)
    
    try:
        config = parse_input('temp_pmf_test.fnl')
        
        # Enable debug
        restraints.restraint_debug_enabled = True

        # Set up restraints
        restraint_defs = config.get('restraints', {})
        restraint_energies, restraint_forces, restraint_metadata = restraints.setup_restraints(restraint_defs)

        # Test coordinates
        coords = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])

        print("Simulating PMF collection...\n")
        
        # Simulate collecting data across multiple windows
        for step in range(1, 301):  # 3 complete windows of 100 steps each
            restraints.current_simulation_step = step - 1
            energy, forces = restraints.apply_restraints(coords, restraint_forces, step)
            
            # Print progress at key points
            if step in [1, 30, 100, 130, 200, 230, 300]:
                print(f"Step {restraints.current_simulation_step}: Energy = {energy:.6f}")

        print("\n=== Testing PMF Output Functions ===\n")
        
        # Test PMF status
        restraints.print_pmf_status()
        
        # Test PMF output file creation
        print("Creating PMF output file...")
        restraints.write_pmf_output("test_pmf_output.dat", temperature=300.0)
        
        # Test final summary
        restraints.print_pmf_final_summary(temperature=300.0)
        
        # Check if files were created
        print("\n=== File Creation Check ===")
        if os.path.exists("test_pmf_output.dat"):
            print("✅ PMF output file created: test_pmf_output.dat")
            with open("test_pmf_output.dat", 'r') as f:
                content = f.read()
                print("\nFile contents:")
                print("-" * 40)
                print(content)
                print("-" * 40)
        else:
            print("❌ PMF output file NOT created")
            
        # Check real-time files if enabled
        realtime_files = [f for f in os.listdir('.') if f.startswith('pmf_') and f.endswith('_realtime.dat')]
        if realtime_files:
            print(f"\n✅ Real-time files created: {realtime_files}")
        else:
            print("\n⚠️  No real-time files (expected if write_realtime_pmf=no)")
            
    finally:
        # Cleanup
        for f in ['temp_pmf_test.fnl', 'test_pmf_output.dat']:
            if os.path.exists(f):
                os.remove(f)
        
        # Clean up any real-time files
        for f in os.listdir('.'):
            if f.startswith('pmf_') and f.endswith('_realtime.dat'):
                os.remove(f)

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pmf_file_output()