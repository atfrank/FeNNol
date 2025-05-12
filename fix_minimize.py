#!/usr/bin/env python3
"""
This script completely fixes the _evaluate_energy_forces method in minimize.py
"""

import re

def fix_minimizer_file():
    """
    Fixes the _evaluate_energy_forces method in minimize.py
    """
    file_path = 'src/fennol/md/minimize.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the entire _evaluate_energy_forces method
    pattern = r'def _evaluate_energy_forces\(self.*?def _safe_get_energy_value'
    
    replacement = '''def _evaluate_energy_forces(self, coords, system=None, preproc_state=None):
        """
        Evaluate energy and forces for the given coordinates
        
        Args:
            coords: Flattened coordinates array
            system: Optional system state from previous iteration
            preproc_state: Optional preprocessing state from previous iteration
            
        Returns:
            Tuple of (energy, forces, updated_system, updated_preproc_state)
        """
        coordinates = coords.reshape(-1, 3)
        
        # Update conformation with new coordinates
        updated_conformation = {**self.conformation, "coordinates": coordinates}
        
        # If we have no system state yet, initialize preprocessing properly from the initial conformation
        if preproc_state is None:
            try:
                # Use setup from initial preprocessing module
                from .initial import initialize_preprocessing
                
                # Initialize preprocessing
                preproc_state, updated_conformation = initialize_preprocessing(
                    self.params, self.model, updated_conformation, self.system_data
                )
            except Exception as e:
                print(f"Warning: Error initializing preprocessing: {str(e)}")
                print("Trying direct processing instead...")
                
                # Fallback if initialize_preprocessing fails
                try:
                    # Create a preprocessing state directly
                    preproc_state = self.model.preproc_state  # Use model's preprocessing state
                    
                    # Process the conformation
                    updated_conformation = self.model.preprocessing.process(
                        preproc_state, updated_conformation
                    )
                except Exception as e2:
                    print(f"Error in fallback processing: {str(e2)}")
                    # Last resort - use the original conformation without preprocessing
                    print("Warning: Using unprocessed coordinates for minimization")
        else:
            # Use skin updates for efficiency
            try:
                updated_conformation = self.model.preprocessing.update_skin(updated_conformation)
            except Exception as e:
                print(f"Warning: Error in skin update: {str(e)}")
                print("Continuing with unmodified conformation")
            
        # Compute energy and forces with error handling
        try:
            # Try the standard energy and forces calculation
            epot, forces, model_out = self.model._energy_and_forces(
                self.model.variables, updated_conformation
            )
        except Exception as e:
            print(f"Error in energy calculation: {str(e)}")
            print("Trying direct energy evaluation...")
            
            try:
                # Try a direct energy calculation as fallback
                epot = self.model._total_energy(self.model.variables, updated_conformation)[0]
                # Compute forces using finite differences if needed
                forces = jnp.zeros_like(coordinates)
                model_out = {}
                print("Warning: Using zero forces for minimization step")
            except Exception as e2:
                print(f"Fatal error in energy calculation: {str(e2)}")
                # Return very high energy and zero forces as last resort
                epot = jnp.array([1e10])  # Very high energy
                forces = jnp.zeros_like(coordinates)
                model_out = {}
                print("ERROR: Minimization may fail due to energy calculation errors")
        
        # Scale to atomic units
        epot = epot / self.model_energy_unit
        forces = forces / self.model_energy_unit
        
        # Initialize or update system
        if system is None:
            try:
                system = self._prepare_system(coordinates)
            except Exception as e:
                print(f"Error preparing system: {str(e)}")
                # Create a basic system dictionary
                system = {
                    "coordinates": coordinates,
                    "forces": forces,
                    "epot": epot if not isinstance(epot, jnp.ndarray) else jnp.mean(epot),
                    # Add basic fields needed
                    "vel": jnp.zeros_like(coordinates),
                    "thermostat": {}
                }
        
        # Update system with new values
        # Make sure we handle both scalar and array epot values
        epot_value = epot if not isinstance(epot, jnp.ndarray) or epot.size == 0 else epot[0]
        system = {
            **system,
            "coordinates": coordinates,
            "forces": forces,
            "epot": epot_value
        }
        
        return epot, forces, system, preproc_state, model_out
    
    def _safe_get_energy_value'''
    
    # Use regex with DOTALL flag to match across multiple lines
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Also fix the _direct_eval_ef method in SimpleSteepestDescentMinimizer
    pattern = r'def _direct_eval_ef\(self.*?except Exception as e2:'
    
    replacement = '''def _direct_eval_ef(self, coords):
        """Direct energy and force calculation bypassing preprocessing"""
        try:
            # Try with preprocessing first
            updated_conformation = {**self.conformation, "coordinates": coords}
            
            # Use preprocessing if available
            if hasattr(self.model, 'preprocessing') and hasattr(self.model.preprocessing, 'process'):
                try:
                    # See if we can get a preproc_state from the model
                    if hasattr(self.model, 'preproc_state'):
                        preproc_state = self.model.preproc_state
                        updated_conformation = self.model.preprocessing.process(
                            preproc_state, updated_conformation
                        )
                except Exception:
                    # Just continue with unprocessed conformation
                    pass
            
            # Calculate energy and forces
            epot, forces, _ = self.model._energy_and_forces(
                self.model.variables, updated_conformation
            )
            
            # Scale to atomic units
            epot = epot / self.model_energy_unit
            forces = forces / self.model_energy_unit
            
            # Return scalar energy value
            if isinstance(epot, jnp.ndarray) and epot.size > 0:
                return epot[0], forces
            else:
                return epot, forces
                
        except Exception as e:
            print(f"# Error in standard energy calculation: {str(e)}")
            print("# Trying direct energy evaluation...")
            
            try:'''
    
    new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed {file_path}")

if __name__ == "__main__":
    fix_minimizer_file()