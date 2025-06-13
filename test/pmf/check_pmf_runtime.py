#!/usr/bin/env python3
"""
Check PMF status during runtime - call this from your simulation
"""

def check_pmf_active():
    """Check if PMF is currently active and collecting data"""
    try:
        from fennol.md.restraints import pmf_data, restraint_debug_enabled, current_simulation_step
        
        print("\n=== PMF Runtime Status ===")
        print(f"Current simulation step: {current_simulation_step}")
        print(f"Restraint debug enabled: {restraint_debug_enabled}")
        
        if not pmf_data:
            print("PMF data collection: NOT ACTIVE (no PMF data)")
            return False
        
        print(f"PMF data collection: ACTIVE")
        print(f"Number of restraints with PMF: {len(pmf_data)}")
        
        for restraint_name, data in pmf_data.items():
            print(f"\nRestraint: {restraint_name}")
            print(f"  - Collective variable: {data['colvar_type']}")
            print(f"  - Number of windows: {len(data['windows'])}")
            print(f"  - Windows: {data['windows']}")
            print(f"  - Sampling started: {data['sampling_started']}")
            
            # Count samples
            total_samples = sum(len(samples) for samples in data['samples'].values())
            windows_with_data = sum(1 for samples in data['samples'].values() if len(samples) > 0)
            print(f"  - Windows with data: {windows_with_data}/{len(data['windows'])}")
            print(f"  - Total samples collected: {total_samples}")
            
            # Show current window
            if 'current_window' in data:
                print(f"  - Current window index: {data['current_window']}")
        
        return True
        
    except ImportError:
        print("ERROR: Could not import restraints module")
        return False
    except Exception as e:
        print(f"ERROR checking PMF status: {e}")
        return False

def get_pmf_indicators():
    """Get simple indicators of PMF activity"""
    try:
        from fennol.md.restraints import pmf_data, current_simulation_step
        
        if not pmf_data:
            return {
                "active": False,
                "n_restraints": 0,
                "total_samples": 0,
                "current_step": current_simulation_step
            }
        
        total_samples = 0
        for data in pmf_data.values():
            total_samples += sum(len(samples) for samples in data['samples'].values())
        
        return {
            "active": True,
            "n_restraints": len(pmf_data),
            "total_samples": total_samples,
            "current_step": current_simulation_step
        }
    except:
        return {"active": False, "error": True}

# Quick one-liner check
def is_pmf_active():
    """Simple True/False check"""
    try:
        from fennol.md.restraints import pmf_data
        return len(pmf_data) > 0
    except:
        return False

if __name__ == "__main__":
    # Run diagnostic
    active = check_pmf_active()
    
    print("\n=== Quick Check ===")
    print(f"is_pmf_active(): {is_pmf_active()}")
    
    indicators = get_pmf_indicators()
    print(f"PMF indicators: {indicators}")