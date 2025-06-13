#!/usr/bin/env python3
"""
Debug script to check PMF issue with user's configuration
"""

def debug_user_pmf():
    """Debug the user's PMF configuration"""
    
    # Simulate the user's restraint configuration
    restraint_def = {
        "type": "backside_attack",
        "nucleophile": 119,
        "carbon": 536,
        "leaving_group": 540,
        "target": 180.0,
        "angle_force_constant": 0.0,
        "target_distance": [2.5, 2.4, 2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
        "distance_force_constant": 0.1,
        "interpolation_mode": "discrete",
        "estimate_pmf": True,
        "equilibration_ratio": 0.3,
        "update_steps": 300,
        "style": "harmonic"
    }
    
    print("=== User's PMF Configuration Analysis ===")
    
    # Check all conditions for PMF activation
    target_distance = restraint_def.get("target_distance")
    estimate_pmf = restraint_def.get("estimate_pmf", False)
    interpolation_mode = restraint_def.get("interpolation_mode", "interpolate")
    
    print(f"target_distance: {target_distance}")
    print(f"  Type: {type(target_distance)}")
    print(f"  Is list: {isinstance(target_distance, list)}")
    print(f"  Length: {len(target_distance) if isinstance(target_distance, list) else 'N/A'}")
    
    print(f"\nestimate_pmf: {estimate_pmf}")
    print(f"interpolation_mode: {interpolation_mode}")
    
    # Check if target_distance_time_varying would be True
    target_distance_time_varying = isinstance(target_distance, list)
    print(f"\ntarget_distance_time_varying: {target_distance_time_varying}")
    
    # Check if PMF would be enabled
    pmf_enabled = (
        estimate_pmf and 
        interpolation_mode == "discrete" and 
        target_distance_time_varying
    )
    
    print(f"\nPMF would be enabled: {pmf_enabled}")
    
    if pmf_enabled:
        update_steps = restraint_def.get("update_steps", 300)
        equilibration_ratio = restraint_def.get("equilibration_ratio", 0.3)
        equilibration_steps = int(update_steps * equilibration_ratio)
        sampling_steps = update_steps - equilibration_steps
        
        print(f"\nPMF Details:")
        print(f"  Windows: {len(target_distance)}")
        print(f"  Update steps: {update_steps}")
        print(f"  Equilibration: {equilibration_steps} steps")
        print(f"  Sampling: {sampling_steps} steps")
        print(f"  First sampling at step: {equilibration_steps}")
        print(f"  Window transitions at steps: {[i * update_steps for i in range(1, len(target_distance))]}")
        
    print("\n=== Expected Behavior ===")
    if pmf_enabled:
        print("✓ PMF should initialize when first sample is collected (around step 90)")
        print("✓ Progress messages should appear every 500 steps during sampling")
        print("✓ Window transition messages at steps 300, 600, 900, etc.")
    else:
        print("✗ PMF should NOT be active")
    
    return pmf_enabled

if __name__ == "__main__":
    debug_user_pmf()