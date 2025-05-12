import sys, os, io
import argparse
import time
from pathlib import Path
import math

import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple, List
from collections import defaultdict
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

from flax.core import freeze, unfreeze

from ..utils.io import write_xyz_frame, write_extxyz_frame, write_arc_frame, human_time_duration
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .initial import initialize_system
from .energy_formatter import format_energy_for_display, update_final_energy_display

class Minimizer:
    """Base class for energy minimization algorithms"""
    
    def __init__(self, 
                 model, 
                 system_data: Dict[str, Any], 
                 conformation: Dict[str, Any],
                 simulation_parameters: Dict[str, Any],
                 fprec: str):
        """
        Initialize the minimizer
        
        Args:
            model: The model to use for energy and force calculations
            system_data: Dictionary containing system information
            conformation: Dictionary with atomic coordinates and other data
            simulation_parameters: Dictionary with simulation parameters
            fprec: Precision to use ('float32' or 'float64')
        """
        self.model = model
        self.system_data = system_data
        self.conformation = conformation
        self.params = simulation_parameters
        self.fprec = fprec
        
        # Extract common parameters
        self.model_energy_unit = au.get_multiplier(model.energy_unit)
        self.nat = system_data["nat"]
        
        # Setup minimization parameters
        self.max_iterations = int(simulation_parameters.get("min_max_iterations", 1000))
        self.energy_tolerance = simulation_parameters.get("min_energy_tolerance", 1e-6)
        self.force_tolerance = simulation_parameters.get("min_force_tolerance", 1e-4)
        self.displacement_tolerance = simulation_parameters.get("min_displacement_tolerance", 1e-3)
        self.max_step = simulation_parameters.get("min_max_step", 0.2)  # Maximum step size in Angstroms
        self.print_freq = int(simulation_parameters.get("min_print_freq", 10))
        self.history_size = int(simulation_parameters.get("min_history_size", 10))
        
        # Determine output options
        self.output_prefix = system_data["name"]
        
    def _prepare_system(self, coordinates):
        """Helper method to create a system dictionary with the given coordinates"""
        try:
            # Try to use the initialize_system function first
            return initialize_system(self.conformation, None, self.model, self.system_data, self.fprec)
        except Exception as e:
            print(f"Error initializing system: {str(e)}")
            print("Creating minimal system dictionary...")
            
            # Create a minimal system dictionary
            vel = jnp.zeros_like(coordinates)
            return {
                "coordinates": coordinates,
                "vel": vel,
                "forces": jnp.zeros_like(coordinates),
                "epot": 0.0,
                "ek": 0.0,
                "ek_tensor": jnp.zeros((3, 3)),
                "thermostat": {}
            }
    
    def _evaluate_energy_forces(self, coords, system=None, preproc_state=None):
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
        
        return epot, forces, system, preproc_state, model_out
    
    def _safe_get_energy_value(self, energy):
        """Safely extract scalar energy value from either array or scalar input"""
        try:
            if hasattr(energy, "shape") and getattr(energy, "shape", None) and len(energy.shape) > 0:
                # It's an array with at least one dimension
                return float(energy[0])
            else:
                # It's a scalar
                return float(energy)
        except (IndexError, TypeError):
            # Fallback for any unexpected issues
            return float(energy) if not isinstance(energy, tuple) else float(energy[0])
    
    def run(self):
        """Run the minimization - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement the run method")
    
    def _format_energy(self, energy_value):
        """Format energy for display in MD-compatible units"""
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
            
        return display_energy, energy_display_str
        
    def print_header(self):
        """Print the minimization header"""
        print("#" + "=" * 78)
        print(f"# Starting geometry optimization using {self.__class__.__name__}")
        
        # Get model energy unit and determine if displaying per-atom energy
        energy_unit_str = self.params.get("energy_unit", "kcal/mol")
        per_atom_energy = self.params.get("per_atom_energy", True)
        nat = self.system_data["nat"]
        atom_energy_unit_str = energy_unit_str
        
        if per_atom_energy:
            atom_energy_unit_str = f"{energy_unit_str}/atom"
            print(f"# Energy displayed per atom in {energy_unit_str}/atom")
        else:
            print(f"# Energy displayed in {energy_unit_str}")
            
        print(f"# Number of atoms: {nat}")
        print(f"# Maximum iterations: {self.max_iterations}")
        print(f"# Energy tolerance: {self.energy_tolerance}")
        print(f"# Force tolerance: {self.force_tolerance}")
        print(f"# Displacement tolerance: {self.displacement_tolerance}")
        print(f"# Maximum step size: {self.max_step} Å")
        print("#" + "=" * 78)
        print(f"# Iter      Energy[{atom_energy_unit_str}]   Max Force     RMS Force      Max Disp      Time/step")
        print("#" + "-" * 78)
    
    def print_step(self, iteration, energy, forces, max_displacement=None, step_time=None):
        """Print information about the current minimization step"""
        if iteration % self.print_freq != 0 and iteration != 1:
            return
        
        # Format energy for display
        display_energy, _ = self._format_energy(energy)
            
        max_force = jnp.max(jnp.abs(forces))
        rms_force = jnp.sqrt(jnp.mean(forces**2))
        
        line = f"# {iteration:4d}  {display_energy:14.6f}  {max_force:12.6f}  {rms_force:12.6f}"
        
        if max_displacement is not None:
            line += f"  {max_displacement:12.6f}"
        else:
            line += "              "
            
        if step_time is not None:
            line += f"  {step_time:10.4f}s"
            
        print(line)
        
    def save_trajectory(self, coords, iteration, energy, forces=None, properties=None):
        """Save the current structure to the trajectory file"""
        # Determine trajectory format
        traj_format = self.params.get("traj_format", "arc").lower()
        
        if traj_format == "xyz":
            traj_ext = ".min.xyz"
            write_frame = write_xyz_frame
        elif traj_format == "extxyz":
            traj_ext = ".min.extxyz"
            write_frame = write_extxyz_frame
        else:  # Default to arc
            traj_ext = ".min.arc"
            write_frame = write_arc_frame
            
        # Get cell if PBC is being used
        cell = None
        if self.system_data.get("pbc") is not None:
            if "cell" in self.system_data["pbc"]:
                cell = self.system_data["pbc"]["cell"]
            elif "cells" in self.conformation:
                cell = self.conformation["cells"][0]
                
        # Prepare properties
        if properties is None:
            properties = {}
            
        properties["energy"] = float(energy) * self.model_energy_unit
        properties["step"] = iteration
        properties["energy_unit"] = self.model.energy_unit
        
        # Open file in append mode
        with open(f"{self.output_prefix}{traj_ext}", "a") as f:
            coords_reshaped = coords.reshape(-1, 3)
            write_frame(
                f,
                self.system_data["symbols"],
                np.asarray(coords_reshaped),
                cell=cell,
                properties=properties,
                forces=None if forces is None else np.asarray(forces) * self.model_energy_unit
            )


class SteepestDescentMinimizer(Minimizer):
    """Simple steepest descent minimizer with adaptive step size"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_step_size = self.params.get("min_initial_step", 0.01)  # Initial step size in Angstroms
        self.step_size = self.initial_step_size
        self.step_increase = self.params.get("min_step_increase", 1.2)  # Factor to increase step size
        self.step_decrease = self.params.get("min_step_decrease", 0.5)  # Factor to decrease step size
        
    def run(self):
        """Run steepest descent minimization"""
        self.print_header()
        
        # Get initial coordinates and evaluate energy/forces
        coords = self.conformation["coordinates"]
        
        if isinstance(coords, jnp.ndarray):
            coords = coords.reshape(-1)
        else:
            coords = jnp.array(coords).reshape(-1)
            
        # Initial system state is None
        system = None
        preproc_state = None
        model_out = None
        
        # Initial evaluation
        t_start = time.time()
        energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(coords, system, preproc_state)
        forces_flat = forces.reshape(-1)
        
        # Print initial state
        energy_val = self._safe_get_energy_value(energy)
        self.print_step(0, energy_val, forces)
        self.save_trajectory(coords, 0, energy_val, forces)
        
        # Minimization loop
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Calculate step
            force_mag = jnp.linalg.norm(forces_flat)
            
            # If forces are very small, we've converged
            if force_mag < self.force_tolerance:
                print(f"# Convergence achieved: Force magnitude {force_mag} < tolerance {self.force_tolerance}")
                break
                
            # Calculate step as normalized force vector * step size
            normalized_forces = forces_flat / force_mag
            step = normalized_forces * self.step_size
            
            # Limit maximum atomic displacement
            max_displacement = jnp.max(jnp.abs(step))
            if max_displacement > self.max_step:
                step = step * (self.max_step / max_displacement)
                max_displacement = self.max_step
            
            # Take step in the direction of the force
            new_coords = coords + step
            
            # Evaluate energy at new position
            new_energy, new_forces, new_system, new_preproc_state, new_model_out = self._evaluate_energy_forces(
                new_coords, system, preproc_state
            )
            new_forces_flat = new_forces.reshape(-1)
            
            # Determine if step was successful based on energy decrease
            new_energy_val = self._safe_get_energy_value(new_energy)
            energy_val = self._safe_get_energy_value(energy)
            
            if new_energy_val < energy_val:
                # Step successful - increase step size for next iteration
                self.step_size = min(self.step_size * self.step_increase, self.max_step)
                
                # Update state
                coords = new_coords
                energy = new_energy
                forces = new_forces
                forces_flat = new_forces_flat
                system = new_system
                preproc_state = new_preproc_state
                model_out = new_model_out
            else:
                # Step unsuccessful - decrease step size and try again
                self.step_size = self.step_size * self.step_decrease
                
                # If step size becomes very small, we're stuck
                if self.step_size < 1e-6:
                    print("# Minimization stalled: step size too small")
                    break
                    
                # Don't update state, continue with current position
                
            # Calculate time per step
            step_time = time.time() - iter_start
            
            # Print status and save trajectory
            energy_val = self._safe_get_energy_value(energy)
            self.print_step(iteration, energy_val, forces, max_displacement, step_time)
            
            if iteration % self.print_freq == 0:
                self.save_trajectory(coords, iteration, energy_val, forces)
                
        # Save final structure
        energy_val = self._safe_get_energy_value(energy)
        self.save_trajectory(coords, iteration, energy_val, forces)
        
        # Final statistics
        total_time = time.time() - t_start
        
        print("#" + "-" * 78)
        print(f"# Minimization completed in {iteration} steps")
        energy_val = self._safe_get_energy_value(energy)
        print(f"# Final energy: {energy_val:.8f}")
        print(f"# Max force magnitude: {jnp.max(jnp.abs(forces)):.8f}")
        print(f"# RMS force: {jnp.sqrt(jnp.mean(forces**2)):.8f}")
        print(f"# Total minimization time: {human_time_duration(total_time)}")
        print("#" + "=" * 78)
        
        # Return minimized system state
        return {
            "coordinates": coords.reshape(-1, 3),
            "energy": self._safe_get_energy_value(energy),
            "forces": forces,
            "system": system,
            "model_output": model_out
        }


class LBFGSMinimizer(Minimizer):
    """L-BFGS minimizer using JAX's optimization implementation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _energy_fn(self, coords):
        """Function that returns energy for L-BFGS optimizer (not JIT compiled due to FrozenDict issues)"""
        try:
            # Use the same _evaluate_energy_forces method for consistency
            epot, _, _, _, _ = self._evaluate_energy_forces(coords)
            # Return scalar value
            return self._safe_get_energy_value(epot)
        except Exception as e:
            print(f"Error in L-BFGS energy function: {str(e)}")
            # Return high energy on error
            return 1e10
    
    def _value_and_grad(self, coords):
        """Function that returns both energy and gradient for L-BFGS optimizer"""
        try:
            return jax.value_and_grad(self._energy_fn)(coords)
        except Exception as e:
            print(f"Error calculating gradient: {str(e)}")
            print("Using approximation...")
            # Return high energy and zero gradient as fallback
            return 1e10, jnp.zeros_like(coords)
    
    def run(self):
        """Run L-BFGS minimization"""
        self.print_header()
        
        # Get initial coordinates
        coords = self.conformation["coordinates"]
        
        if isinstance(coords, jnp.ndarray):
            coords = coords.reshape(-1)
        else:
            coords = jnp.array(coords).reshape(-1)
        
        # Prepare for tracking
        best_coords = coords
        best_energy = float('inf')
        best_forces = None
        
        # Initial system state for evaluation
        system = None
        preproc_state = None
        
        # Initial evaluation for display
        t_start = time.time()
        energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(coords, system, preproc_state)
        
        # Print initial state
        energy_val = self._safe_get_energy_value(energy)
        self.print_step(0, energy_val, forces)
        self.save_trajectory(coords, 0, energy_val, forces)
        
        # Setup callback to track progress
        iteration = [0]
        step_times = []
        
        def callback(state):
            # Declare nonlocal variables first
            nonlocal best_coords, best_energy, best_forces
            
            iter_start = time.time()
            iteration[0] += 1
            
            # Extract info from the state
            this_coords = state.x
            this_energy = state.value
            this_gradient = state.g
            
            # Calculate step size from previous iteration
            if iteration[0] > 1:
                step = this_coords - best_coords
                max_displacement = jnp.max(jnp.abs(step))
            else:
                max_displacement = None
                
            # Reshape gradient to match forces format (negate gradient to get forces)
            this_forces = -this_gradient.reshape(-1, 3)
            
            # Update best values
            if this_energy < best_energy:
                best_coords = this_coords
                best_energy = this_energy
                best_forces = this_forces
            
            # Calculate time for this step
            step_time = time.time() - iter_start
            step_times.append(step_time)
            
            # Print progress
            self.print_step(iteration[0], this_energy, this_forces, max_displacement, step_time)
            
            # Save trajectory periodically
            if iteration[0] % self.print_freq == 0:
                self.save_trajectory(this_coords, iteration[0], this_energy, this_forces)
                
            return state.value < self.energy_tolerance
        
        # Run the L-BFGS optimizer
        try:
            # JAX's minimize doesn't support callback in the same way, so we'll use a basic version
            # without callback
            result = minimize(
                self._value_and_grad,  # Function to minimize
                coords,                # Initial coordinates
                method='BFGS',
                options={
                    'maxiter': self.max_iterations,
                    'gtol': self.force_tolerance,
                    'norm': 'max',     # Use max norm for forces
                    'line_search_maxiter': 10,
                }
            )
            
            # Manually call callback with the final state for tracking
            final_state = type('', (), {})()  # Create an empty object
            final_state.x = result.x
            final_state.value = result.fun
            final_state.g = -result.jac  # Negative jacobian gives forces
            callback(final_state)
            
            # Extract results
            min_coords = result.x
            min_energy = result.fun
            
            # Final evaluation to get forces
            energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(min_coords)
            
            # Report convergence result
            if result.success:
                print(f"# Optimization converged: {result.message}")
            else:
                print(f"# Optimization did not converge: {result.message}")
                
        except Exception as e:
            print(f"# Error during L-BFGS minimization: {str(e)}")
            # Use best values found so far
            min_coords = best_coords
            min_energy = best_energy
            forces = best_forces
            
            # Do a final evaluation to ensure we have a valid system
            energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(min_coords)
        
        # Save final structure
        energy_val = self._safe_get_energy_value(energy)
        self.save_trajectory(min_coords, iteration[0], energy_val, forces)
        
        # Final statistics
        total_time = time.time() - t_start
        
        print("#" + "-" * 78)
        print(f"# Minimization completed in {iteration[0]} steps")
        print(f"# Final energy: {min_energy:.8f}")
        print(f"# Max force magnitude: {jnp.max(jnp.abs(forces)):.8f}")
        print(f"# RMS force: {jnp.sqrt(jnp.mean(forces**2)):.8f}")
        print(f"# Total minimization time: {human_time_duration(total_time)}")
        if step_times:
            print(f"# Average time per step: {sum(step_times)/len(step_times):.4f}s")
        print("#" + "=" * 78)
        
        # Return minimized system state
        return {
            "coordinates": min_coords.reshape(-1, 3),
            "energy": min_energy,
            "forces": forces,
            "system": system,
            "model_output": model_out
        }


class ConjugateGradientMinimizer(Minimizer):
    """Conjugate gradient minimizer with line search"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_search_method = self.params.get("min_line_search", "backtracking")
        self.max_line_search_iterations = int(self.params.get("min_max_line_search", 20))
        self.restart_threshold = int(self.params.get("min_cg_restart", 20))  # Restart CG after this many iterations
        
    def _line_search(self, coords, energy, forces, direction, iteration):
        """
        Perform a line search along the given direction
        
        Args:
            coords: Current coordinates
            energy: Current energy
            forces: Current forces
            direction: Direction to search in
            iteration: Current iteration number
            
        Returns:
            Tuple of (new_coords, new_energy, new_forces, new_system, new_preproc_state, alpha)
        """
        # Initial step size - 0.1 Angstrom divided by the magnitude of the direction
        alpha = 0.1 / jnp.max(jnp.abs(direction))
        
        # Make sure we don't exceed max_step
        if alpha * jnp.max(jnp.abs(direction)) > self.max_step:
            alpha = self.max_step / jnp.max(jnp.abs(direction))
            
        # Initial system is None - will be initialized in first evaluation
        system = None
        preproc_state = None
        
        # Keep track of best values found
        best_alpha = 0.0
        energy_val = self._safe_get_energy_value(energy)
        best_energy = energy_val
        best_coords = coords
        best_forces = forces
        best_system = None
        best_preproc_state = None
        best_model_out = None
        
        # Initial trial step
        trial_coords = coords + alpha * direction
        
        # Perform line search
        if self.line_search_method == "backtracking":
            # Simple backtracking line search
            for i in range(self.max_line_search_iterations):
                # Evaluate energy and forces at trial point
                trial_energy, trial_forces, trial_system, trial_preproc_state, trial_model_out = self._evaluate_energy_forces(
                    trial_coords, system, preproc_state
                )
                
                # Check if this is better than current best
                trial_energy_val = self._safe_get_energy_value(trial_energy)
                if trial_energy_val < best_energy:
                    best_alpha = alpha
                    best_energy = trial_energy_val
                    best_coords = trial_coords
                    best_forces = trial_forces
                    best_system = trial_system
                    best_preproc_state = trial_preproc_state
                    best_model_out = trial_model_out
                    
                    # Try a bigger step
                    alpha *= 1.5
                    trial_coords = coords + alpha * direction
                else:
                    # Step was too big, back up
                    alpha *= 0.5
                    
                    # If alpha became very small, stop the line search
                    if alpha < 1e-6:
                        break
                        
                    trial_coords = coords + alpha * direction
                    
        else:  # More sophisticated Strong Wolfe line search
            # Constants for Strong Wolfe conditions
            c1 = 1e-4  # Sufficient decrease parameter
            c2 = 0.9   # Curvature condition parameter
            
            # Initial directional derivative
            initial_slope = jnp.sum(forces.reshape(-1) * direction)
            
            for i in range(self.max_line_search_iterations):
                # Evaluate energy and forces at trial point
                trial_energy, trial_forces, trial_system, trial_preproc_state, trial_model_out = self._evaluate_energy_forces(
                    trial_coords, system, preproc_state
                )
                
                # Calculate directional derivative at this point
                trial_slope = jnp.sum(trial_forces.reshape(-1) * direction)
                
                # Check Armijo condition (sufficient decrease)
                energy_val = self._safe_get_energy_value(energy)
                trial_energy_val = self._safe_get_energy_value(trial_energy)
                armijo_violated = trial_energy_val > energy_val + c1 * alpha * initial_slope
                
                # Check curvature condition
                curvature_violated = jnp.abs(trial_slope) > -c2 * initial_slope
                
                if armijo_violated:
                    # Step is too big, reduce alpha
                    alpha *= 0.5
                elif curvature_violated:
                    # Step is too small, increase alpha
                    alpha *= 2.0
                else:
                    # Both conditions satisfied
                    best_alpha = alpha
                    best_energy = trial_energy_val
                    best_coords = trial_coords
                    best_forces = trial_forces
                    best_system = trial_system
                    best_preproc_state = trial_preproc_state
                    best_model_out = trial_model_out
                    break
                    
                # Update trial point
                trial_coords = coords + alpha * direction
                
                # If alpha became very small or too large, stop the line search
                if alpha < 1e-6 or alpha > 100.0:
                    break
        
        # Return best point found during line search
        return best_coords, best_energy, best_forces, best_system, best_preproc_state, best_model_out, best_alpha
    
    def run(self):
        """Run conjugate gradient minimization"""
        self.print_header()
        
        # Get initial coordinates and evaluate energy/forces
        coords = self.conformation["coordinates"]
        
        if isinstance(coords, jnp.ndarray):
            coords = coords.reshape(-1)
        else:
            coords = jnp.array(coords).reshape(-1)
            
        # Initial system state is None
        system = None
        preproc_state = None
        
        # Initial evaluation
        t_start = time.time()
        energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(coords, system, preproc_state)
        forces_flat = forces.reshape(-1)
        
        # Print initial state
        energy_val = self._safe_get_energy_value(energy)
        self.print_step(0, energy_val, forces)
        self.save_trajectory(coords, 0, energy_val, forces)
        
        # Initialize search direction as steepest descent (forces)
        direction = forces_flat
        old_forces_flat = forces_flat
        
        # Minimization loop
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Check convergence
            max_force = jnp.max(jnp.abs(forces_flat))
            if max_force < self.force_tolerance:
                print(f"# Convergence achieved: Force magnitude {max_force} < tolerance {self.force_tolerance}")
                break
                
            # Normalize direction for better step size control
            direction_norm = jnp.linalg.norm(direction)
            if direction_norm > 0:
                normalized_direction = direction / direction_norm
            else:
                # If direction is zero (unlikely), use forces as direction
                normalized_direction = forces_flat / jnp.linalg.norm(forces_flat)
                
            # Perform line search
            new_coords, new_energy, new_forces, new_system, new_preproc_state, new_model_out, alpha = self._line_search(
                coords, energy, forces, normalized_direction, iteration
            )
            
            new_forces_flat = new_forces.reshape(-1)
            
            # Calculate step info for printing
            step = new_coords - coords
            max_displacement = jnp.max(jnp.abs(step))
            
            # Update state for next iteration
            coords = new_coords
            energy = jnp.array([new_energy])  # Wrap scalar in array to match expected format
            forces = new_forces
            forces_flat = new_forces_flat
            system = new_system
            preproc_state = new_preproc_state
            model_out = new_model_out
            
            # Calculate beta using Polak-Ribiere formula
            # beta = max(0, dot(new_grad, new_grad - old_grad) / dot(old_grad, old_grad))
            if iteration % self.restart_threshold == 0:
                # Restart CG to avoid numerical issues
                beta = 0.0
            else:
                # Calculate gradient difference
                grad_diff = new_forces_flat - old_forces_flat
                
                # Calculate dot products
                dot_new_diff = jnp.sum(new_forces_flat * grad_diff)
                dot_old_old = jnp.sum(old_forces_flat * old_forces_flat)
                
                # Polak-Ribiere formula with restart (beta >= 0)
                if dot_old_old > 0:
                    beta = max(0.0, dot_new_diff / dot_old_old)
                else:
                    beta = 0.0
            
            # Update search direction
            direction = new_forces_flat + beta * direction
            
            # Save old forces for next beta calculation
            old_forces_flat = new_forces_flat
            
            # Calculate time for this step
            step_time = time.time() - iter_start
            
            # Print status and save trajectory
            energy_val = self._safe_get_energy_value(energy)
            self.print_step(iteration, energy_val, forces, max_displacement, step_time)
            
            if iteration % self.print_freq == 0:
                self.save_trajectory(coords, iteration, energy_val, forces)
                
        # Save final structure
        energy_val = self._safe_get_energy_value(energy)
        self.save_trajectory(coords, iteration, energy_val, forces)
        
        # Final statistics
        total_time = time.time() - t_start
        
        print("#" + "-" * 78)
        print(f"# Minimization completed in {iteration} steps")
        energy_val = self._safe_get_energy_value(energy)
        print(f"# Final energy: {energy_val:.8f}")
        print(f"# Max force magnitude: {jnp.max(jnp.abs(forces)):.8f}")
        print(f"# RMS force: {jnp.sqrt(jnp.mean(forces**2)):.8f}")
        print(f"# Total minimization time: {human_time_duration(total_time)}")
        print("#" + "=" * 78)
        
        # Return minimized system state
        return {
            "coordinates": coords.reshape(-1, 3),
            "energy": self._safe_get_energy_value(energy),
            "forces": forces,
            "system": system,
            "model_output": model_out
        }


class SimpleSteepestDescentMinimizer(Minimizer):
    """
    Simplified steepest descent minimizer that doesn't use preprocessing - more robust for complex systems
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_step_size = self.params.get("min_initial_step", 0.01)
        self.step_size = self.initial_step_size
        self.step_increase = self.params.get("min_step_increase", 1.2)
        self.step_decrease = self.params.get("min_step_decrease", 0.5)
        
    def run(self):
        """Run simplified steepest descent minimization"""
        self.print_header()
        
        # Get initial coordinates
        coords = self.conformation["coordinates"]
        if isinstance(coords, jnp.ndarray):
            coords = coords.reshape(-1)
        else:
            coords = jnp.array(coords).reshape(-1)
            
        # For direct model evaluation
        atom_coords = coords.reshape(-1, 3)
        
        # Initial energy calculation
        print("# Computing initial energy and forces")
        try:
            energy, forces = self._direct_eval_ef(atom_coords)
            # Format energy for display
            energy_val = self._safe_extract_energy(energy) if hasattr(self, "_safe_extract_energy") else energy
            display_energy, energy_unit_str = format_energy_for_display(energy_val, self.system_data, self.params)
            print(f"# Initial energy: {display_energy:.6f} {energy_unit_str}")
        except Exception as e:
            print(f"# Error computing initial energy: {str(e)}")
            energy = 1e10
            forces = jnp.zeros_like(atom_coords)
            
        # Print initial state
        self.print_step(0, energy, forces)
        self.save_trajectory(coords, 0, energy, forces)
        
        # Create a simple system
        # Ensure energy is properly stored as a scalar
        energy_val = self._safe_get_energy_value(energy) if hasattr(self, '_safe_get_energy_value') else energy
        system = {
            "coordinates": atom_coords,
            "forces": forces,
            "epot": energy_val,
            "vel": jnp.zeros_like(atom_coords),
            "thermostat": {}
        }
        
        t_start = time.time()
        
        # Minimization loop
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Reshape for convenience
            coords_flat = coords
            forces_flat = forces.reshape(-1)
            
            # Calculate step
            force_mag = jnp.linalg.norm(forces_flat)
            
            # Check for convergence
            if force_mag < self.force_tolerance:
                print(f"# Convergence achieved: Force magnitude {force_mag} < tolerance {self.force_tolerance}")
                break
                
            # Calculate step as normalized force vector * step size
            normalized_forces = forces_flat / jnp.maximum(force_mag, 1e-10)  # Avoid division by zero
            step = normalized_forces * self.step_size
            
            # Limit maximum atomic displacement
            max_displacement = jnp.max(jnp.abs(step))
            if max_displacement > self.max_step:
                step = step * (self.max_step / max_displacement)
                max_displacement = self.max_step
            
            # Take step in the direction of the force
            new_coords = coords_flat + step
            new_atom_coords = new_coords.reshape(-1, 3)
            
            # Evaluate energy at new position
            try:
                new_energy, new_forces = self._direct_eval_ef(new_atom_coords)
            except Exception as e:
                print(f"# Error computing energy: {str(e)}")
                new_energy = energy
                new_forces = forces
            
            # Determine if step was successful based on energy decrease
            if new_energy < energy:
                # Step successful - increase step size for next iteration
                self.step_size = min(self.step_size * self.step_increase, self.max_step)
                
                # Update state
                coords = new_coords
                energy = new_energy
                forces = new_forces
                
                # Update system
                system = {
                    **system,
                    "coordinates": new_atom_coords,
                    "forces": new_forces,
                    "epot": new_energy
                }
            else:
                # Step unsuccessful - decrease step size and try again
                self.step_size = self.step_size * self.step_decrease
                
                # If step size becomes very small, we're stuck
                if self.step_size < 1e-6:
                    print("# Minimization stalled: step size too small")
                    break
            
            # Calculate time per step
            step_time = time.time() - iter_start
            
            # Print status and save trajectory
            self.print_step(iteration, energy, forces, max_displacement, step_time)
            
            if iteration % self.print_freq == 0:
                self.save_trajectory(coords, iteration, energy, forces)
                
        # Save final structure
        self.save_trajectory(coords, iteration, energy, forces)
        
        # Final statistics
        total_time = time.time() - t_start
        
        print("#" + "-" * 78)
        print(f"# Minimization completed in {iteration} steps")
        print(f"# Final energy: {energy:.8f}")
        print(f"# Max force magnitude: {jnp.max(jnp.abs(forces)):.8f}")
        print(f"# RMS force: {jnp.sqrt(jnp.mean(forces**2)):.8f}")
        print(f"# Total minimization time: {human_time_duration(total_time)}")
        print("#" + "=" * 78)
        
        # Return minimized system state
        return {
            "coordinates": coords.reshape(-1, 3),
            "energy": energy,
            "forces": forces,
            "system": system,
            "model_output": {}
        }
        
    def _safe_extract_energy(self, energy):
        """Safely extract energy regardless of array or scalar"""
        if hasattr(energy, "shape") and len(getattr(energy, "shape", [])) > 0:
            # It's an array with at least 1 dimension, return first element
            return float(energy[0])
        # It's a scalar, return as is
        return float(energy)
        
    def _direct_eval_ef(self, coords):
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
            
            try:
                # Just do direct energy calculation - this should work even without preprocessing
                conformation = {**self.conformation, "coordinates": coords}
                epot_result = self.model._total_energy(self.model.variables, conformation)
                # Handle both scalar and array return types
                if isinstance(epot_result, tuple) and len(epot_result) > 0:
                    epot = epot_result[0] / self.model_energy_unit
                else:
                    epot = epot_result / self.model_energy_unit
                
                # Try numerical derivatives if needed
                try:
                    # Compute forces using JAX automatic differentiation if possible
                    def energy_fn(c):
                        result = self.model._total_energy(
                            self.model.variables, {**self.conformation, "coordinates": c.reshape(-1, 3)}
                        )
                        # Handle both scalar and array return types
                        if isinstance(result, tuple) and len(result) > 0:
                            return result[0] / self.model_energy_unit
                        else:
                            return result / self.model_energy_unit
                    
                    forces = -jax.grad(energy_fn)(coords.reshape(-1))
                    forces = forces.reshape(-1, 3)
                except Exception:
                    # Use zeros if everything fails
                    forces = jnp.zeros_like(coords)
                
                return epot, forces
                
            except Exception as e2:
                print(f"# Fatal error in energy calculation: {str(e2)}")
                # Return something so minimization doesn't crash
                return 1e10, jnp.zeros_like(coords)


def get_minimizer(method, model, system_data, conformation, simulation_parameters, fprec):
    """Factory function to create appropriate minimizer"""
    method = method.lower()
    
    if method == "sd" or method == "steepest_descent":
        # Use the robust version for complex systems
        if simulation_parameters.get("use_simple_sd", False):
            return SimpleSteepestDescentMinimizer(model, system_data, conformation, simulation_parameters, fprec)
        else:
            return SteepestDescentMinimizer(model, system_data, conformation, simulation_parameters, fprec)
    elif method == "lbfgs" or method == "bfgs":
        return LBFGSMinimizer(model, system_data, conformation, simulation_parameters, fprec)
    elif method == "cg" or method == "conjugate_gradient":
        return ConjugateGradientMinimizer(model, system_data, conformation, simulation_parameters, fprec)
    elif method == "simple_sd" or method == "simple":
        return SimpleSteepestDescentMinimizer(model, system_data, conformation, simulation_parameters, fprec)
    else:
        raise ValueError(f"Unknown minimization method: {method}. Available methods: sd, lbfgs, cg, simple_sd")


def minimize_system(model, system_data, conformation, simulation_parameters, fprec):
    """
    Main entry point for minimization
    
    Args:
        model: The model to use for energy and force calculations
        system_data: Dictionary containing system information
        conformation: Dictionary with atomic coordinates and other data
        simulation_parameters: Dictionary with simulation parameters
        fprec: Precision to use ('float32' or 'float64')
        
    Returns:
        Dictionary with minimization results
    """
    # Get minimization method
    method = simulation_parameters.get("min_method", "lbfgs")
    
    # Create appropriate minimizer
    minimizer = get_minimizer(method, model, system_data, conformation, simulation_parameters, fprec)
    
    # Run minimization
    result = minimizer.run()
    
    return result