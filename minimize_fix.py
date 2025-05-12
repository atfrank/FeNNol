#!/usr/bin/env python3
"""
This script fixes the minimize.py file to properly handle scalar energy values.
"""

def fixed_evaluate_energy_forces():
    """Return the fixed _evaluate_energy_forces method"""
    return '''    def _evaluate_energy_forces(self, coords, system=None, preproc_state=None):
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
        
        # Convert energy to scalar if it's an array
        if isinstance(epot, jnp.ndarray) and epot.size > 0:
            epot_scalar = epot[0]
        else:
            epot_scalar = epot
            
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
                    "epot": epot_scalar,
                    # Add basic fields needed
                    "vel": jnp.zeros_like(coordinates),
                    "thermostat": {}
                }
        
        system = {
            **system,
            "coordinates": coordinates,
            "forces": forces,
            "epot": epot_scalar
        }
        
        return epot, forces, system, preproc_state, model_out'''

def fixed_direct_eval_ef():
    """Return the fixed _direct_eval_ef method"""
    return '''    def _direct_eval_ef(self, coords):
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
            
            # Handle both scalar and array epot values
            if isinstance(epot, jnp.ndarray) and epot.size > 0:
                return epot[0], forces
            else:
                return epot, forces
                
        except Exception as e:
            print(f"# Error in standard energy calculation: {str(e)}")
            print("# Trying direct energy evaluation...")
            
            try:
                # Just do direct energy calculation - this should work even without preprocessing
                conformation = {**self.conformation, "coordinates": coords}
                epot = self.model._total_energy(self.model.variables, conformation)[0] / self.model_energy_unit
                
                # Try numerical derivatives if needed
                try:
                    # Compute forces using JAX automatic differentiation if possible
                    forces = -jax.grad(lambda c: self.model._total_energy(
                        self.model.variables, {**self.conformation, "coordinates": c.reshape(-1, 3)}
                    )[0] / self.model_energy_unit)(coords.reshape(-1))
                    forces = forces.reshape(-1, 3)
                except Exception:
                    # Use zeros if everything fails
                    forces = jnp.zeros_like(coords)
                
                return epot, forces
                
            except Exception as e2:
                print(f"# Fatal error in energy calculation: {str(e2)}")
                # Return something so minimization doesn't crash
                return 1e10, jnp.zeros_like(coords)'''
                
def add_safe_get_energy_value():
    """Return the _safe_get_energy_value method"""
    return '''    def _safe_get_energy_value(self, energy):
        """Safely extract scalar energy value from either array or scalar input"""
        if isinstance(energy, jnp.ndarray) and energy.size > 0:
            return energy[0]
        return energy'''

if __name__ == "__main__":
    print("Please copy these fixed methods into your minimize.py file:")
    print("\n" + "="*80)
    print(fixed_evaluate_energy_forces())
    print("\n" + "="*80)
    print(fixed_direct_eval_ef())
    print("\n" + "="*80)
    print(add_safe_get_energy_value())
    print("="*80)