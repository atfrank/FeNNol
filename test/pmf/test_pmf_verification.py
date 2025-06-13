#!/usr/bin/env python3
"""
Script to verify PMF is active and working correctly
"""

from fennol.md.restraints import pmf_data, pmf_enabled, restraint_debug_enabled
from fennol.utils.input_parser import parse_input

def check_pmf_status(input_file):
    """Check if PMF will be active based on input file settings"""
    
    print("=== PMF Status Check ===\n")
    
    # Parse input file
    params = parse_input(input_file)
    
    # Check global debug setting
    restraint_debug = params.get("restraint_debug", False)
    print(f"1. Restraint debug enabled: {restraint_debug}")
    if not restraint_debug:
        print("   ⚠️  WARNING: Set 'restraint_debug yes' to see PMF progress!\n")
    
    # Check restraints
    restraints = params.get("restraints", {})
    if not restraints:
        print("2. No restraints defined - PMF not possible\n")
        return
    
    print(f"2. Found {len(restraints)} restraint(s):\n")
    
    pmf_active = False
    for name, restraint in restraints.items():
        print(f"   Restraint: {name}")
        print(f"   - Type: {restraint.get('type', 'distance')}")
        
        # Check interpolation mode
        interp_mode = restraint.get("interpolation_mode", "interpolate")
        print(f"   - Interpolation mode: {interp_mode}")
        
        # Check estimate_pmf
        estimate_pmf = restraint.get("estimate_pmf", False)
        print(f"   - Estimate PMF: {estimate_pmf}")
        
        # Check for time-varying parameters
        time_varying = False
        time_varying_params = []
        
        # Check force constants
        fc = restraint.get("force_constant", restraint.get("angle_force_constant"))
        if isinstance(fc, list):
            time_varying = True
            time_varying_params.append("force_constant")
        
        # Check targets
        target = restraint.get("target")
        if isinstance(target, list):
            time_varying = True
            time_varying_params.append("target angle")
            
        target_dist = restraint.get("target_distance")
        if isinstance(target_dist, list):
            time_varying = True
            time_varying_params.append("target_distance")
        
        print(f"   - Time-varying: {time_varying}")
        if time_varying:
            print(f"     Parameters: {', '.join(time_varying_params)}")
        
        # Check if PMF will be active
        pmf_enabled_for_restraint = (
            interp_mode == "discrete" and 
            estimate_pmf and 
            time_varying
        )
        
        if pmf_enabled_for_restraint:
            print(f"   ✓ PMF ACTIVE for this restraint")
            pmf_active = True
            
            # Show PMF details
            equil_ratio = restraint.get("equilibration_ratio", 0.2)
            update_steps = restraint.get("update_steps", 1000)
            print(f"     - Update steps: {update_steps}")
            print(f"     - Equilibration: {int(update_steps * equil_ratio)} steps ({equil_ratio*100:.0f}%)")
            print(f"     - Sampling: {int(update_steps * (1-equil_ratio))} steps ({(1-equil_ratio)*100:.0f}%)")
            
            # Calculate total windows
            if isinstance(target_dist, list):
                n_windows = len(target_dist)
                print(f"     - Windows: {n_windows} (scanning target_distance)")
                print(f"     - Values: {target_dist}")
            elif isinstance(fc, list):
                n_windows = len(fc)
                print(f"     - Windows: {n_windows} (scanning force_constant)")
        else:
            print(f"   ✗ PMF NOT active")
            missing = []
            if interp_mode != "discrete":
                missing.append("interpolation_mode must be 'discrete'")
            if not estimate_pmf:
                missing.append("estimate_pmf must be 'yes'")
            if not time_varying:
                missing.append("need time-varying parameters (list values)")
            if missing:
                print(f"     Missing: {'; '.join(missing)}")
        
        print()
    
    print("\n=== Summary ===")
    if pmf_active:
        print("✓ PMF is ACTIVE for at least one restraint")
        if not restraint_debug:
            print("⚠️  But you won't see progress without 'restraint_debug yes'")
    else:
        print("✗ PMF is NOT active for any restraint")
        print("\nTo enable PMF, you need ALL of:")
        print("  1. interpolation_mode = discrete")
        print("  2. estimate_pmf = yes")
        print("  3. Time-varying parameters (e.g., target_distance = 3.0 2.5 2.0)")
        print("  4. restraint_debug yes (to see progress)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_pmf_verification.py input.fnl")
        sys.exit(1)
    
    check_pmf_status(sys.argv[1])