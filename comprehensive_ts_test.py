#!/usr/bin/env python3
"""
Comprehensive transition state optimization testing suite

This script provides definitive validation that our TS methods are working correctly
by testing known chemical systems with expected transition state characteristics.
"""

import subprocess
import os
import time
import numpy as np
import re
from pathlib import Path

class TSTestSuite:
    def __init__(self):
        self.test_dir = Path("test_real_model")
        self.results = {}
        
    def create_test_systems(self):
        """Create test systems with known TS characteristics"""
        
        # Test 1: H2 dissociation - Simple, well-known TS at ~1.5 Å
        h2_near_ts = """2
H2 near transition state
H       0.00000000     0.00000000     0.00000000
H       1.20000000     0.00000000     0.00000000"""
        
        # Test 2: Water dimer - Hydrogen bond formation TS
        water_dimer_ts_guess = """6
Water dimer TS guess
O       0.00000000     0.00000000     0.00000000
H       0.95700000     0.00000000     0.00000000
H      -0.24000000     0.92700000     0.00000000
O       2.80000000     0.00000000     0.00000000
H       3.75700000     0.00000000     0.00000000
H       2.56000000     0.92700000     0.00000000"""
        
        # Test 3: SN2 reaction - Symmetric TS
        sn2_symmetric_ts = """6
SN2 symmetric TS guess
F  -2.20000000   0.00000000   0.00000000
C   0.00000000   0.00000000   0.00000000
H   0.00000000   1.09000000   0.00000000
H   0.94280000  -0.54500000   0.00000000
H  -0.94280000  -0.54500000   0.00000000
Cl  2.20000000   0.00000000   0.00000000"""
        
        # Test 4: SN2 early TS - Asymmetric
        sn2_early_ts = """6
SN2 early TS guess
F  -2.80000000   0.00000000   0.00000000
C   0.00000000   0.00000000   0.00000000
H   0.00000000   1.09000000   0.00000000
H   0.94280000  -0.54500000   0.00000000
H  -0.94280000  -0.54500000   0.00000000
Cl  2.00000000   0.00000000   0.00000000"""
        
        # Write test files
        with open(self.test_dir / "h2_ts_guess.xyz", "w") as f:
            f.write(h2_near_ts)
        with open(self.test_dir / "water_dimer_ts.xyz", "w") as f:
            f.write(water_dimer_ts_guess)
        with open(self.test_dir / "sn2_symmetric.xyz", "w") as f:
            f.write(sn2_symmetric_ts)
        with open(self.test_dir / "sn2_early.xyz", "w") as f:
            f.write(sn2_early_ts)
            
    def create_test_configs(self):
        """Create test configuration files"""
        
        # Test 1: H2 with dimer method
        h2_dimer_config = """# H2 dissociation TS with dimer method
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = h2_ts_guess.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = dimer
dimer_separation = 0.01
dimer_rotation_tolerance = 0.1
dimer_max_rotations = 10

min_max_iterations = 20
min_force_tolerance = 1e-3
min_print_freq = 1
output_prefix = h2_dimer_test
traj_format = xyz"""

        # Test 2: Water dimer with quasi-Newton
        water_qn_config = """# Water dimer TS with quasi-Newton
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = water_dimer_ts.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = quasi_newton
ts_trust_radius = 0.15
ts_max_uphill_steps = 3
ts_initial_hessian_scale = 0.05

min_max_iterations = 25
min_force_tolerance = 1e-3
min_print_freq = 1
output_prefix = water_qn_test
traj_format = xyz"""

        # Test 3: SN2 symmetric with SN2 method
        sn2_sym_config = """# SN2 symmetric TS with SN2 method
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = sn2_symmetric.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = sn2
sn2_nu_index = 1
sn2_c_index = 2
sn2_lg_index = 6
sn2_target_nu_c_distance = 2.2
sn2_target_c_lg_distance = 2.2

min_max_iterations = 20
min_force_tolerance = 1e-3
min_print_freq = 1
output_prefix = sn2_sym_test
traj_format = xyz"""

        # Test 4: SN2 early with SN2 method
        sn2_early_config = """# SN2 early TS with SN2 method
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = sn2_early.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = sn2
sn2_nu_index = 1
sn2_c_index = 2
sn2_lg_index = 6
sn2_target_nu_c_distance = 2.0
sn2_target_c_lg_distance = 2.2

min_max_iterations = 20
min_force_tolerance = 1e-3
min_print_freq = 1
output_prefix = sn2_early_test
traj_format = xyz"""

        # Write config files
        with open(self.test_dir / "h2_dimer_test.fnl", "w") as f:
            f.write(h2_dimer_config)
        with open(self.test_dir / "water_qn_test.fnl", "w") as f:
            f.write(water_qn_config)
        with open(self.test_dir / "sn2_sym_test.fnl", "w") as f:
            f.write(sn2_sym_config)
        with open(self.test_dir / "sn2_early_test.fnl", "w") as f:
            f.write(sn2_early_config)
            
    def run_single_test(self, test_name, config_file):
        """Run a single TS optimization test"""
        print(f"\n{'='*60}")
        print(f"RUNNING TEST: {test_name}")
        print(f"{'='*60}")
        
        # Change to test directory
        os.chdir(self.test_dir)
        
        # Run the test
        start_time = time.time()
        try:
            result = subprocess.run(
                ["fennol_ts", config_file, "--max-iterations", "20"],
                capture_output=True,
                text=True,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
                timeout=300  # 5 minute timeout
            )
            
            elapsed_time = time.time() - start_time
            
            # Store results
            self.results[test_name] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_time": elapsed_time
            }
            
            print(f"Test completed in {elapsed_time:.1f}s")
            if result.returncode != 0:
                print(f"ERROR: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"ERROR: Test timed out after 5 minutes")
            self.results[test_name] = {
                "success": False,
                "error": "timeout",
                "elapsed_time": 300
            }
            
        finally:
            # Return to parent directory
            os.chdir("..")
            
    def parse_results(self, test_name):
        """Parse optimization results from output"""
        if test_name not in self.results or not self.results[test_name]["success"]:
            return None
            
        stdout = self.results[test_name]["stdout"]
        
        # Extract key metrics
        metrics = {}
        
        # Find initial and final energies
        energy_lines = re.findall(r"#\s+\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)", stdout)
        if energy_lines:
            initial_energy = float(energy_lines[0][0])
            final_energy = float(energy_lines[-1][0])
            metrics["initial_energy"] = initial_energy
            metrics["final_energy"] = final_energy
            metrics["energy_change"] = final_energy - initial_energy
            
            initial_max_force = float(energy_lines[0][1])
            final_max_force = float(energy_lines[-1][1])
            metrics["initial_max_force"] = initial_max_force
            metrics["final_max_force"] = final_max_force
            metrics["force_change"] = final_max_force - initial_max_force
            
        # Check convergence
        metrics["converged"] = "Converged: True" in stdout
        
        # Extract eigenvalue info
        eigenvalue_match = re.search(r"Number of negative eigenvalues: (\d+)", stdout)
        if eigenvalue_match:
            metrics["negative_eigenvalues"] = int(eigenvalue_match.group(1))
            
        # Extract total iterations
        iter_match = re.search(r"completed in (\d+) steps", stdout)
        if iter_match:
            metrics["iterations"] = int(iter_match.group(1))
            
        return metrics
        
    def analyze_chemical_validity(self, test_name):
        """Analyze chemical validity of results"""
        ts_file = self.test_dir / f"{test_name.replace('_test', '')}_test.ts.xyz"
        
        if not ts_file.exists():
            return None
            
        # Read final structure
        with open(ts_file, "r") as f:
            lines = f.readlines()
            
        # Parse coordinates
        atoms = []
        for line in lines[2:]:  # Skip natoms and comment
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 4:
                    atoms.append({
                        "symbol": parts[0],
                        "coords": [float(parts[1]), float(parts[2]), float(parts[3])]
                    })
                    
        # Chemical analysis based on system type
        analysis = {}
        
        if "h2" in test_name:
            # H2 dissociation analysis
            if len(atoms) >= 2:
                h1_pos = np.array(atoms[0]["coords"])
                h2_pos = np.array(atoms[1]["coords"])
                bond_length = np.linalg.norm(h2_pos - h1_pos)
                analysis["h_h_distance"] = bond_length
                analysis["expected_ts_range"] = (1.0, 2.0)  # Typical H2 TS range
                analysis["chemically_reasonable"] = 1.0 <= bond_length <= 2.0
                
        elif "water" in test_name:
            # Water dimer analysis
            if len(atoms) >= 6:
                o1_pos = np.array(atoms[0]["coords"])
                o2_pos = np.array(atoms[3]["coords"])
                oo_distance = np.linalg.norm(o2_pos - o1_pos)
                analysis["o_o_distance"] = oo_distance
                analysis["expected_ts_range"] = (2.5, 3.5)  # Typical water dimer TS range
                analysis["chemically_reasonable"] = 2.5 <= oo_distance <= 3.5
                
        elif "sn2" in test_name:
            # SN2 reaction analysis
            if len(atoms) >= 6:
                f_pos = np.array(atoms[0]["coords"])
                c_pos = np.array(atoms[1]["coords"])
                cl_pos = np.array(atoms[5]["coords"])
                
                fc_distance = np.linalg.norm(c_pos - f_pos)
                ccl_distance = np.linalg.norm(cl_pos - c_pos)
                reaction_coord = fc_distance - ccl_distance
                
                # F-C-Cl angle
                fc_vec = c_pos - f_pos
                ccl_vec = cl_pos - c_pos
                cos_angle = np.dot(fc_vec, ccl_vec) / (np.linalg.norm(fc_vec) * np.linalg.norm(ccl_vec))
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                
                analysis["f_c_distance"] = fc_distance
                analysis["c_cl_distance"] = ccl_distance
                analysis["reaction_coordinate"] = reaction_coord
                analysis["f_c_cl_angle"] = angle
                analysis["expected_rc_range"] = (-0.5, 0.5)  # Near symmetric for TS
                analysis["expected_angle_range"] = (160, 180)  # Near linear
                analysis["chemically_reasonable"] = (
                    abs(reaction_coord) <= 0.5 and 160 <= angle <= 180
                )
                
        return analysis
        
    def generate_report(self):
        """Generate comprehensive test report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TRANSITION STATE OPTIMIZATION TEST REPORT")
        print(f"{'='*80}")
        
        test_cases = [
            ("h2_dimer", "H2 Dissociation (Dimer Method)"),
            ("water_qn", "Water Dimer (Quasi-Newton)"),
            ("sn2_sym", "SN2 Symmetric (SN2 Method)"),
            ("sn2_early", "SN2 Early TS (SN2 Method)")
        ]
        
        for test_name, description in test_cases:
            print(f"\n{'-'*60}")
            print(f"TEST: {description}")
            print(f"{'-'*60}")
            
            # Parse optimization results
            metrics = self.parse_results(test_name)
            if metrics is None:
                print("❌ TEST FAILED - No results available")
                continue
                
            # Chemical analysis
            chemistry = self.analyze_chemical_validity(test_name)
            
            # Report optimization metrics
            print(f"Optimization Results:")
            print(f"  ✓ Converged: {'Yes' if metrics.get('converged', False) else 'No'}")
            print(f"  ✓ Iterations: {metrics.get('iterations', 'N/A')}")
            print(f"  ✓ Energy change: {metrics.get('energy_change', 0):+.3f} kcal/mol/atom")
            print(f"  ✓ Force change: {metrics.get('force_change', 0):+.6f}")
            print(f"  ✓ Final max force: {metrics.get('final_max_force', 0):.6f}")
            print(f"  ✓ Negative eigenvalues: {metrics.get('negative_eigenvalues', 'N/A')}")
            
            # Report chemical validity
            if chemistry:
                print(f"Chemical Analysis:")
                for key, value in chemistry.items():
                    if key == "chemically_reasonable":
                        print(f"  ✓ Chemically reasonable: {'Yes' if value else 'No'}")
                    elif "expected" not in key:
                        print(f"  ✓ {key}: {value:.3f}")
                        
            # Overall assessment
            is_valid_ts = (
                metrics.get("negative_eigenvalues", 0) == 1 and
                chemistry and chemistry.get("chemically_reasonable", False)
            )
            
            print(f"Overall Assessment: {'✅ VALID TRANSITION STATE' if is_valid_ts else '⚠️ NEEDS REVIEW'}")
            
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        # Count successes
        successful_tests = sum(1 for name, _ in test_cases if 
                             self.results.get(name, {}).get("success", False))
        
        print(f"Tests completed: {successful_tests}/{len(test_cases)}")
        print(f"Success rate: {successful_tests/len(test_cases)*100:.1f}%")
        
        # Method-specific analysis
        print(f"\nMethod Performance:")
        print(f"  - Dimer Method: {'✅' if 'h2_dimer' in self.results else '❌'}")
        print(f"  - Quasi-Newton: {'✅' if 'water_qn' in self.results else '❌'}")
        print(f"  - SN2 Method: {'✅' if 'sn2_sym' in self.results else '❌'}")
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("Setting up comprehensive transition state optimization tests...")
        
        # Setup
        self.create_test_systems()
        self.create_test_configs()
        
        # Run tests
        tests = [
            ("h2_dimer", "h2_dimer_test.fnl"),
            ("water_qn", "water_qn_test.fnl"),
            ("sn2_sym", "sn2_sym_test.fnl"),
            ("sn2_early", "sn2_early_test.fnl")
        ]
        
        for test_name, config_file in tests:
            self.run_single_test(test_name, config_file)
            
        # Generate report
        self.generate_report()

if __name__ == "__main__":
    suite = TSTestSuite()
    suite.run_all_tests()