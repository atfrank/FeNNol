#!/usr/bin/env python
"""
Script to fix energy display formatting in the minimization module
"""
import os
import re

def update_file(file_path):
    """
    Update the energy display in the specified file
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # First, update the _format_energy method to use the imported function
    content = re.sub(
        r'def _format_energy\(self, energy_value\):.*?return display_energy, energy_display_str',
        'def _format_energy(self, energy_value):\n        """Format energy for display in MD-compatible units"""\n        return format_energy_for_display(energy_value, self.system_data, self.params)',
        content, 
        flags=re.DOTALL
    )
    
    # Next update all instances of final energy display
    pattern = r'print\("#" \+ "-" \* 78\)\s+print\(f"# Minimization completed in {(\w+)} steps"\)\s+print\(f"# Final energy: {([^}]+):.8f}"\)\s+print\(f"# Max force magnitude: {[^}]+:.8f}"\)\s+print\(f"# RMS force: {[^}]+:.8f}"\)'
    
    replacement = 'print("#" + "-" * 78)\n        print(f"# Minimization completed in {\\1} steps")\n        \n        # Get force metrics\n        max_force = jnp.max(jnp.abs(forces))\n        rms_force = jnp.sqrt(jnp.mean(forces**2))\n        \n        # Print formatted energy statistics\n        update_final_energy_display(\\2, self.system_data, self.params, max_force, rms_force)'
    
    content = re.sub(pattern, replacement, content)
    
    # Handle Simple Steepest Descent Minimizer separately
    pattern = r'print\("#" \+ "-" \* 78\)\s+print\(f"# Minimization completed in {(\w+)} steps"\)\s+print\(f"# Final energy: {(\w+):.8f}"\)\s+print\(f"# Max force magnitude: {[^}]+:.8f}"\)\s+print\(f"# RMS force: {[^}]+:.8f}"\)'
    
    replacement = 'print("#" + "-" * 78)\n        print(f"# Minimization completed in {\\1} steps")\n        \n        # Get force metrics\n        max_force = jnp.max(jnp.abs(forces))\n        rms_force = jnp.sqrt(jnp.mean(forces**2))\n        \n        # Print formatted energy statistics\n        update_final_energy_display(\\2, self.system_data, self.params, max_force, rms_force)'
    
    content = re.sub(pattern, replacement, content)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
        
    print(f"Updated {file_path}")


if __name__ == "__main__":
    # Update the minimize.py file
    update_file(os.path.join('src', 'fennol', 'md', 'minimize.py'))