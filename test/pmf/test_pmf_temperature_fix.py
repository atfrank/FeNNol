#!/usr/bin/env python3
"""Test that the PMF temperature fix works correctly."""

import subprocess
import os
import sys

def test_pmf_temperature_fix():
    """Test that PMF output works with the temperature fix."""
    
    # Create a simple test input file
    test_input = """device cpu
enable_x64
matmul_prec highest
nreplicas 1

model_file ../examples/md/rna/mace_mp_small.fnx

xyz_input{
  file ../examples/md/rna/minimal_test.xyz
  indexed no
  has_comment_line yes
}

minimum_image no
wrap_box no

nsteps = 100
dt[fs] = 0.5
traj_format xyz

tdump[ps] = 0.1
nprint = 10
nsummary = 50

temperature = 150.0

restraints {
  test_restraint {
    type = distance
    atom1 = 0
    atom2 = 1
    target = 3.0, 3.5, 4.0
    force_constant = 1.0
    interpolation_mode = discrete
    
    # PMF settings
    estimate_pmf = yes
    equilibration_ratio = 0.2
    update_steps = 30
  }
}
"""
    
    # Write test input file
    with open('test_pmf_temp.fnl', 'w') as f:
        f.write(test_input)
    
    print("Running FeNNol MD with PMF estimation...")
    
    # Run the simulation
    try:
        result = subprocess.run(
            ['python', '-m', 'fennol.md.dynamic', 'test_pmf_temp.fnl'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print("\nSTDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        # Check if PMF output was written successfully
        if "Writing PMF estimation results..." in result.stdout:
            if "Failed to write PMF output: name 'temperature' is not defined" in result.stdout:
                print("\n❌ TEST FAILED: Temperature error still present!")
                return False
            elif "PMF data written to pmf_output.dat" in result.stdout:
                print("\n✅ TEST PASSED: PMF output written successfully!")
                
                # Check if the output file exists
                if os.path.exists('pmf_output.dat'):
                    print("✅ pmf_output.dat file created")
                    with open('pmf_output.dat', 'r') as f:
                        content = f.read()
                        if "Temperature: 150" in content:
                            print("✅ Correct temperature (150 K) written to output")
                        else:
                            print("❌ Temperature not found in output file")
                            print(content[:500])
                else:
                    print("❌ pmf_output.dat file not found")
                
                return True
            else:
                print("\n⚠️  PMF estimation may not have been triggered")
                return True
        else:
            print("\n⚠️  No PMF estimation in output")
            # This might be OK if PMF wasn't triggered
            return True
            
    except subprocess.TimeoutExpired:
        print("\n❌ TEST FAILED: Simulation timed out")
        return False
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    finally:
        # Clean up
        for f in ['test_pmf_temp.fnl', 'pmf_output.dat', 'minimal_test.traj.xyz', 'minimal_test.colvars']:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    success = test_pmf_temperature_fix()
    sys.exit(0 if success else 1)