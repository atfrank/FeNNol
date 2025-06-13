#!/usr/bin/env python3
"""
Quick fix for the energy indexing error in SimpleSteepestDescentMinimizer
"""

def fix_error():
    with open('src/fennol/md/minimize.py', 'r') as f:
        content = f.read()
    
    # Add this utility function to the SimpleSteepestDescentMinimizer class
    content = content.replace(
        'def _direct_eval_ef(self, coords):',
        '''def _safe_extract_energy(self, energy):
        """Safely extract energy regardless of array or scalar"""
        if hasattr(energy, "shape") and len(getattr(energy, "shape", [])) > 0:
            # It's an array with at least 1 dimension, return first element
            return float(energy[0])
        # It's a scalar, return as is
        return float(energy)
        
    def _direct_eval_ef(self, coords):''')
    
    # Replace this line in the run() method where the error happens
    content = content.replace(
        'display_energy, energy_unit_str = format_energy_for_display(energy, self.system_data, self.params)',
        'energy_val = self._safe_extract_energy(energy) if hasattr(self, "_safe_extract_energy") else energy\n'
        '            display_energy, energy_unit_str = format_energy_for_display(energy_val, self.system_data, self.params)')
    
    # Also replace any other instances of energy[0] in the class
    content = content.replace(
        'epot = self.model._total_energy(self.model.variables, conformation)[0] / self.model_energy_unit',
        'raw_epot = self.model._total_energy(self.model.variables, conformation)\n'
        '                if isinstance(raw_epot, tuple) and len(raw_epot) > 0:\n'
        '                    epot = raw_epot[0] / self.model_energy_unit\n'
        '                else:\n'
        '                    epot = raw_epot / self.model_energy_unit')
    
    # Fix the jax.grad computation
    content = content.replace(
        'forces = -jax.grad(lambda c: self.model._total_energy(\n'
        '                        self.model.variables, {**self.conformation, "coordinates": c.reshape(-1, 3)}\n'
        '                    )[0] / self.model_energy_unit)(coords.reshape(-1))',
        'def energy_fn(c):\n'
        '                        raw_result = self.model._total_energy(\n'
        '                            self.model.variables, {**self.conformation, "coordinates": c.reshape(-1, 3)}\n'
        '                        )\n'
        '                        if isinstance(raw_result, tuple) and len(raw_result) > 0:\n'
        '                            return raw_result[0] / self.model_energy_unit\n'
        '                        else:\n'
        '                            return raw_result / self.model_energy_unit\n'
        '                    forces = -jax.grad(energy_fn)(coords.reshape(-1))')
    
    # Write back to the file
    with open('src/fennol/md/minimize.py', 'w') as f:
        f.write(content)
    
    print("Applied fixes to src/fennol/md/minimize.py")

if __name__ == '__main__':
    fix_error()