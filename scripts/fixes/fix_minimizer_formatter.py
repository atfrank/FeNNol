#!/usr/bin/env python3
"""
This script fixes the _format_energy method in the Minimizer class to properly handle scalar values
"""

def fix_minimizer_formatter():
    """Fix the _format_energy method in the Minimizer class"""
    with open('src/fennol/md/minimize.py', 'r') as f:
        content = f.read()
    
    # Replace the _format_energy method
    old_method = """    def _format_energy(self, energy_value):
        \"\"\"Format energy for display in MD-compatible units\"\"\"
        # Convert energy to appropriate units and per-atom if needed
        energy_unit_str = self.params.get("energy_unit", "kcal/mol")
        energy_unit = au.get_multiplier(energy_unit_str)
        per_atom_energy = self.params.get("per_atom_energy", True)
        nat = self.system_data["nat"]
        
        # Convert energy to chosen units
        if isinstance(energy_value, jnp.ndarray) and energy_value.size > 0:
            display_energy = energy_value[0] * energy_unit
        else:
            display_energy = energy_value * energy_unit
        
        # Convert to per-atom if needed
        if per_atom_energy:
            display_energy = display_energy / nat
            energy_display_str = f"{energy_unit_str}/atom"
        else:
            energy_display_str = energy_unit_str
            
        return display_energy, energy_display_str"""
    
    new_method = """    def _format_energy(self, energy_value):
        \"\"\"Format energy for display in MD-compatible units\"\"\"
        # Convert energy to appropriate units and per-atom if needed
        energy_unit_str = self.params.get("energy_unit", "kcal/mol")
        energy_unit = au.get_multiplier(energy_unit_str)
        per_atom_energy = self.params.get("per_atom_energy", True)
        nat = self.system_data["nat"]
        
        # IMPORTANT: Ensure energy_value is a float scalar
        try:
            if hasattr(energy_value, "shape") and getattr(energy_value, "shape", None) and len(energy_value.shape) > 0:
                # It's an array with at least one dimension
                energy_scalar = float(energy_value[0])
            else:
                # It's a scalar
                energy_scalar = float(energy_value)
        except (IndexError, TypeError):
            # Fallback for any unexpected issues
            energy_scalar = float(energy_value) if not isinstance(energy_value, tuple) else float(energy_value[0])
        
        # Convert energy to chosen units
        display_energy = energy_scalar * energy_unit
        
        # Convert to per-atom if needed
        if per_atom_energy:
            display_energy = display_energy / nat
            energy_display_str = f"{energy_unit_str}/atom"
        else:
            energy_display_str = energy_unit_str
            
        return display_energy, energy_display_str"""
    
    new_content = content.replace(old_method, new_method)
    
    with open('src/fennol/md/minimize.py', 'w') as f:
        f.write(new_content)
    
    print("Fixed _format_energy method in Minimizer class to handle scalar values correctly")

if __name__ == "__main__":
    fix_minimizer_formatter()