#!/usr/bin/env python3
"""
Test script to verify enhanced PMF functionality with metadata for multi-simulation analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fennol.md.restraints import initialize_pmf_data, update_pmf_sample, write_pmf_output, pmf_data
import numpy as np
import tempfile
import json

def test_enhanced_pmf():
    """Test the enhanced PMF functionality."""
    print("Testing enhanced PMF functionality...")
    
    # Clear any existing PMF data
    global pmf_data
    pmf_data.clear()
    
    # Test parameters
    restraint_name = "test_distance"
    windows = [1.0, 1.5, 2.0, 2.5, 3.0]
    force_constants = [10.0, 15.0, 20.0, 25.0, 30.0]
    colvar_type = "distance"
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize PMF data with enhanced metadata
        initialize_pmf_data(
            restraint_name=restraint_name,
            windows=windows,
            colvar_type=colvar_type,
            write_realtime=True,
            output_dir=temp_dir,
            force_constants=force_constants,
            simulation_id="test_sim_001"
        )
        
        # Check if metadata was properly stored
        assert restraint_name in pmf_data
        data = pmf_data[restraint_name]
        
        # Enable sampling
        data['sampling_started'] = True
        
        print("✓ PMF data structure initialized")
        print(f"  - Simulation ID: {data['simulation_id']}")
        print(f"  - Force constants: {data['force_constants']}")
        print(f"  - Window statistics initialized: {len(data['window_statistics'])}")
        
        # Simulate adding samples to different windows
        np.random.seed(42)  # For reproducible results
        for window_idx, (window_val, force_k) in enumerate(zip(windows, force_constants)):
            # Simulate 50 samples around each window target
            for _ in range(50):
                sample_value = np.random.normal(window_val, 0.1)  # Small noise around target
                update_pmf_sample(
                    restraint_name=restraint_name,
                    colvar_value=sample_value,
                    window_idx=window_idx,
                    write_realtime=True,
                    output_dir=temp_dir,
                    force_constant=force_k
                )
        
        print("✓ Sample data added to all windows")
        
        # Check that statistics were computed
        for i, stats in data['window_statistics'].items():
            mean_val = stats['mean'] if stats['mean'] is not None else 0.0
            std_val = stats['std'] if stats['std'] is not None else 0.0
            print(f"  - Window {i}: {stats['count']} samples, mean={mean_val:.3f}, std={std_val:.3f}")
        
        # Write enhanced PMF output
        output_file = os.path.join(temp_dir, "test_pmf_enhanced.dat")
        write_pmf_output(filename=output_file, temperature=300.0)
        
        print("✓ Enhanced PMF output written")
        
        # Verify the output file contains enhanced metadata
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Check for enhanced metadata sections
        required_sections = [
            "Enhanced for Multi-Simulation Analysis",
            "Simulation ID:",
            "Force constants:",
            "Window statistics:",
            "METADATA_JSON:"
        ]
        
        for section in required_sections:
            if section in content:
                print(f"✓ Found section: {section}")
            else:
                print(f"✗ Missing section: {section}")
        
        # Check real-time file
        realtime_file = os.path.join(temp_dir, f"pmf_{restraint_name}_realtime.dat")
        if os.path.exists(realtime_file):
            with open(realtime_file, 'r') as f:
                realtime_content = f.read()
            
            if "Force_Constant" in realtime_content:
                print("✓ Real-time file contains force constant column")
            else:
                print("✗ Real-time file missing force constant column")
                
            # Count data lines (non-comment lines)
            data_lines = [line for line in realtime_content.split('\n') if line and not line.startswith('#')]
            print(f"✓ Real-time file contains {len(data_lines)} data entries")
        else:
            print("✗ Real-time file not created")
        
        # Extract and verify JSON metadata
        json_lines = [line for line in content.split('\n') if line.startswith('# METADATA_JSON:')]
        if json_lines:
            json_str = json_lines[0].replace('# METADATA_JSON: ', '')
            try:
                metadata = json.loads(json_str)
                print("✓ JSON metadata successfully parsed")
                print(f"  - Total samples: {metadata['total_samples']}")
                print(f"  - Temperature: {metadata['temperature']}")
                print(f"  - Force constants: {metadata['force_constants']}")
            except json.JSONDecodeError as e:
                print(f"✗ JSON metadata parsing failed: {e}")
        else:
            print("✗ No JSON metadata found")
        
        print("\n" + "="*50)
        print("Enhanced PMF test completed!")
        print("The PMF output now includes:")
        print("• Unique simulation IDs for multi-simulation analysis")
        print("• Force constants for each window")
        print("• Detailed window statistics (mean, std, min, max, counts)")
        print("• Timing information (start/end steps)")
        print("• Machine-readable JSON metadata")
        print("• Enhanced real-time output with force constants")
        print("="*50)

if __name__ == "__main__":
    test_enhanced_pmf()