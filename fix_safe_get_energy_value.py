#!/usr/bin/env python3
"""
This script fixes the _safe_get_energy_value method in the Minimizer class
"""

def fix_safe_get_energy_value():
    """Fix the _safe_get_energy_value method in the Minimizer class"""
    with open('src/fennol/md/minimize.py', 'r') as f:
        content = f.read()
    
    # Replace the _safe_get_energy_value method
    old_method = """    def _safe_get_energy_value(self, energy):
        \"\"\"Safely extract scalar energy value from either array or scalar input\"\"\"
        if isinstance(energy, jnp.ndarray) and energy.size > 0:
            return energy[0]
        return energy"""
    
    new_method = """    def _safe_get_energy_value(self, energy):
        \"\"\"Safely extract scalar energy value from either array or scalar input\"\"\"
        try:
            if hasattr(energy, "shape") and getattr(energy, "shape", None) and len(energy.shape) > 0:
                # It's an array with at least one dimension
                return float(energy[0])
            else:
                # It's a scalar
                return float(energy)
        except (IndexError, TypeError):
            # Fallback for any unexpected issues
            return float(energy) if not isinstance(energy, tuple) else float(energy[0])"""
    
    new_content = content.replace(old_method, new_method)
    
    with open('src/fennol/md/minimize.py', 'w') as f:
        f.write(new_content)
    
    print("Fixed _safe_get_energy_value method in Minimizer class")

if __name__ == "__main__":
    fix_safe_get_energy_value()