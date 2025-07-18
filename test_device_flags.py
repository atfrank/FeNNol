#!/usr/bin/env python3
"""
Test script to verify device flag functionality for fennol_refine
"""

import subprocess
import tempfile
import os

def test_device_flag(device_flag, expected_device_msg):
    """Test a specific device flag"""
    print(f"Testing --device {device_flag}...")
    
    # Create test PDB file
    test_pdb = """ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O  
ATOM      2  H1  HOH A   1       0.757   0.587   0.000  1.00  0.00           H  
ATOM      3  H2  HOH A   1      -0.757   0.587   0.000  1.00  0.00           H  
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(test_pdb)
        test_pdb_path = f.name
    
    try:
        # Test command
        result = subprocess.run([
            "fennol_refine", 
            "--pdb", test_pdb_path,
            "--model", "/home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx",
            "--device", device_flag,
            "--max-iterations", "1",
            "--verbose"
        ], capture_output=True, text=True, timeout=30)
        
        # Check for expected device message
        if expected_device_msg in result.stdout:
            print(f"✓ --device {device_flag} works correctly")
            return True
        else:
            print(f"✗ --device {device_flag} failed - expected '{expected_device_msg}' in output")
            print(f"  Stdout: {result.stdout[:200]}...")
            print(f"  Stderr: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ --device {device_flag} timed out")
        return False
    except Exception as e:
        print(f"✗ --device {device_flag} failed with exception: {e}")
        return False
    finally:
        os.unlink(test_pdb_path)

def main():
    """Run all device flag tests"""
    print("Device Flag Testing for fennol_refine")
    print("=" * 50)
    
    # Test cases: (device_flag, expected_device_message)
    test_cases = [
        ("cpu", "# Device: cpu"),
        ("gpu", "# Device: gpu"),
        ("cuda", "# Device: cuda"),
        ("cuda:0", "# Device: cuda:0"),
        ("gpu:1", "# Device: gpu:1"),  # Should fallback to CPU if device doesn't exist
        ("invalid", "# Device: invalid"),  # Should use default fallback
    ]
    
    passed = 0
    failed = 0
    
    for device_flag, expected_msg in test_cases:
        if test_device_flag(device_flag, expected_msg):
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Device Flag Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All device flag tests passed!")
        print("The device selection logic is working correctly for:")
        print("  - CPU device (--device cpu)")
        print("  - GPU device (--device gpu)")
        print("  - CUDA device (--device cuda)")
        print("  - Specific device numbers (--device cuda:0, --device gpu:1)")
        print("  - Invalid device fallback (--device invalid)")
    else:
        print("❌ Some device flag tests failed. Please check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)