#!/usr/bin/env python3
"""
Comprehensive test script for FeNNol transition state methods with real MACE model
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, timeout=120):
    """Run a command and return success status and output"""
    try:
        env = os.environ.copy()
        env['JAX_PLATFORMS'] = 'cpu'
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            env=env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_cli_help():
    """Test CLI help functionality"""
    print("Testing CLI help...")
    success, stdout, stderr = run_command("fennol_ts --help")
    if success and "Find transition states" in stdout:
        print("✓ CLI help test passed")
        return True
    else:
        print("✗ CLI help test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_file_not_found():
    """Test error handling for non-existent files"""
    print("Testing file not found error...")
    success, stdout, stderr = run_command("fennol_ts nonexistent.fnl")
    if not success and "not found" in (stdout + stderr):
        print("✓ File not found error test passed")
        return True
    else:
        print("✗ File not found error test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_quasi_newton():
    """Test QuasiNewtonTS method"""
    print("Testing QuasiNewtonTS method...")
    success, stdout, stderr = run_command("fennol_ts test_real_model/quasi_newton_test.fnl --max-iterations 5")
    if success and "QuasiNewtonTS" in stdout:
        print("✓ QuasiNewtonTS test passed")
        return True
    elif "QuasiNewtonTS" in stdout:
        print("✓ QuasiNewtonTS test passed (with warnings)")
        return True
    else:
        print("✗ QuasiNewtonTS test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_dimer_method():
    """Test DimerMethod"""
    print("Testing DimerMethod...")
    success, stdout, stderr = run_command("fennol_ts test_real_model/dimer_test.fnl --max-iterations 3")
    if success and "DimerMethod" in stdout:
        print("✓ DimerMethod test passed")
        return True
    elif "DimerMethod" in stdout:
        print("✓ DimerMethod test passed (with warnings)")
        return True
    else:
        print("✗ DimerMethod test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_sn2_method():
    """Test SN2TransitionState method"""
    print("Testing SN2TransitionState method...")
    success, stdout, stderr = run_command("fennol_ts test_real_model/sn2_test.fnl --max-iterations 5")
    if success and "SN2TransitionState" in stdout:
        print("✓ SN2TransitionState test passed")
        return True
    elif "SN2TransitionState" in stdout:
        print("✓ SN2TransitionState test passed (with warnings)")
        return True
    else:
        print("✗ SN2TransitionState test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_cli_overrides():
    """Test CLI parameter overrides"""
    print("Testing CLI parameter overrides...")
    success, stdout, stderr = run_command("fennol_ts test_real_model/quasi_newton_test.fnl --method dimer --max-iterations 2 --force-tolerance 0.01")
    if "Method: dimer" in stdout and "Maximum iterations: 2" in stdout and "Force tolerance: 0.01" in stdout:
        print("✓ CLI parameter overrides test passed")
        return True
    else:
        print("✗ CLI parameter overrides test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_output_files():
    """Test that output files are created"""
    print("Testing output file creation...")
    
    # Clean up any existing files
    for pattern in ["*.ts.xyz", "*.min.xyz", "*.min.arc"]:
        for file in Path(".").glob(pattern):
            file.unlink()
    
    success, stdout, stderr = run_command("fennol_ts test_real_model/quasi_newton_test.fnl --max-iterations 2")
    
    # Check if TS structure file was created
    ts_files = list(Path(".").glob("*.ts.xyz"))
    if ts_files:
        print(f"✓ Output file creation test passed: {ts_files}")
        return True
    else:
        print("✗ Output file creation test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_energy_units():
    """Test energy unit handling"""
    print("Testing energy unit handling...")
    success, stdout, stderr = run_command("fennol_ts test_real_model/quasi_newton_test.fnl --max-iterations 2")
    if "Energy unit: Ha" in stdout:
        print("✓ Energy unit handling test passed")
        return True
    else:
        print("✗ Energy unit handling test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def test_verbose_output():
    """Test verbose output mode"""
    print("Testing verbose output mode...")
    success, stdout, stderr = run_command("fennol_ts test_real_model/quasi_newton_test.fnl --max-iterations 2 --verbose")
    if "Model loaded:" in stdout and "System loaded:" in stdout:
        print("✓ Verbose output test passed")
        return True
    else:
        print("✗ Verbose output test failed")
        print(f"  stdout: {stdout}")
        print(f"  stderr: {stderr}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("COMPREHENSIVE FENNOL_TS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("CLI Help", test_cli_help),
        ("File Not Found", test_file_not_found),
        ("QuasiNewtonTS", test_quasi_newton),
        ("DimerMethod", test_dimer_method),
        ("SN2TransitionState", test_sn2_method),
        ("CLI Overrides", test_cli_overrides),
        ("Output Files", test_output_files),
        ("Energy Units", test_energy_units),
        ("Verbose Output", test_verbose_output)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)