"""
Fixed restraints implementation for FeNNol.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Callable, Tuple, List, Optional
from .colvars import colvar_distance, colvar_angle, colvar_dihedral

# Global variables to track restraint state
current_force_constants = {}
current_simulation_step = 0  # Will be incremented by apply_restraints

# Function to access or update the current step
def get_current_step():
    global current_simulation_step
    return current_simulation_step

def increment_step():
    global current_simulation_step
    current_simulation_step += 1
    return current_simulation_step

def ensure_proper_coordinates(coordinates):
    """Ensure coordinates have the correct shape [natoms, 3]"""
    if coordinates.ndim == 1:
        # If we have flattened coordinates, reshape them
        natoms = coordinates.shape[0] // 3
        return coordinates.reshape(natoms, 3)
    return coordinates

def harmonic_restraint(value: float, target: float, force_constant: float) -> Tuple[float, float]:
    """
    Calculate the harmonic restraint energy and force.
    
    Args:
        value: Current value of the collective variable
        target: Target value for the restraint
        force_constant: Force constant for the harmonic restraint (k)
    
    Returns:
        Tuple containing (energy, force)
        Energy = 0.5 * k * (value - target)^2
        Force = k * (value - target)
    """
    diff = value - target
    energy = 0.5 * force_constant * diff**2
    force = force_constant * diff
    return energy, force

def flat_bottom_restraint(value: float, target: float, force_constant: float, 
                          tolerance: float) -> Tuple[float, float]:
    """
    Calculate a flat-bottom restraint energy and force. 
    No energy penalty within tolerance, harmonic outside.
    
    Args:
        value: Current value of the collective variable
        target: Target value for the restraint
        force_constant: Force constant for the harmonic restraint (k)
        tolerance: Width of the flat region (no energy penalty within target ± tolerance)
    
    Returns:
        Tuple containing (energy, force)
    """
    # Calculate deviation from target
    diff = value - target
    abs_diff = jnp.abs(diff)
    
    # Calculate violation (how far outside the flat region)
    # If within tolerance, violation is 0
    # If outside tolerance, violation is the amount beyond tolerance
    violation = jnp.maximum(0.0, abs_diff - tolerance)
    
    # Energy is 0.5*k*violation^2 (harmonic)
    energy = 0.5 * force_constant * violation**2
    
    # Force is k*violation*sign(diff)
    # Only non-zero when outside the tolerance
    sign = jnp.sign(diff)
    force = force_constant * violation * sign
    
    return energy, force

def one_sided_harmonic_restraint(value: float, target: float, force_constant: float, 
                                side: str = "lower") -> Tuple[float, float]:
    """
    Calculate a one-sided harmonic restraint energy and force.
    Forces go to zero when the value is on the non-restrained side of the target.
    
    Args:
        value: Current value of the collective variable
        target: Target value for the restraint
        force_constant: Force constant for the harmonic restraint (k)
        side: Which side to apply the restraint on ("lower" or "upper")
            - "lower": Applies force when value < target (pulls up to target)
            - "upper": Applies force when value > target (pulls down to target)
    
    Returns:
        Tuple containing (energy, force)
    """
    # Calculate deviation from target
    diff = value - target
    
    if side == "lower":
        # Apply restraint only when value < target (pull up to target)
        violation = jnp.minimum(0.0, diff)  # Will be negative when below target, 0 when above
        
        # Energy is 0.5*k*violation^2 (harmonic)
        energy = 0.5 * force_constant * violation**2
        
        # Force is -k*violation (positive when below target, pulling up)
        # Zero when above target
        force = -force_constant * violation
        
    elif side == "upper":
        # Apply restraint only when value > target (pull down to target)
        violation = jnp.maximum(0.0, diff)  # Will be positive when above target, 0 when below
        
        # Energy is 0.5*k*violation^2 (harmonic)
        energy = 0.5 * force_constant * violation**2
        
        # Force is -k*violation (negative when above target, pulling down)
        # Zero when below target
        force = -force_constant * violation
        
    else:
        raise ValueError(f"Invalid side parameter: {side}. Must be 'lower' or 'upper'.")
    
    return energy, force

def distance_restraint_force(coordinates: jnp.ndarray, atom1: int, atom2: int, 
                            target: float, force_constant: float, 
                            restraint_function: Callable = harmonic_restraint) -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for a distance restraint.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        atom1, atom2: Indices of atoms for distance measurement
        target: Target distance
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Check coordinates shape and reshape if needed
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Get atom positions
    pos1 = coordinates[atom1]
    pos2 = coordinates[atom2]
    
    # Calculate the vector between atoms and its length
    r_vec = pos1 - pos2
    r = jnp.linalg.norm(r_vec)
    
    # Unit vector in the direction of r_vec
    # Avoid division by zero with a safe normalization
    r_safe = jnp.maximum(r, 1e-10)  # Ensure r is not zero
    unit_vec = r_vec / r_safe
    
    # Calculate restraint energy and force magnitude
    energy, force_magnitude = restraint_function(r, target, force_constant)
    
    # Forces on the atoms (equal and opposite)
    forces = jnp.zeros_like(coordinates)
    force_vec = force_magnitude * unit_vec
    
    # Update forces for the two atoms
    forces = forces.at[atom1].set(-force_vec)
    forces = forces.at[atom2].set(force_vec)
    
    return energy, forces

def angle_restraint_force(coordinates: jnp.ndarray, atom1: int, atom2: int, atom3: int,
                         target: float, force_constant: float,
                         restraint_function: Callable = harmonic_restraint) -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for an angle restraint.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        atom1, atom2, atom3: Indices of atoms defining the angle (atom2 is the vertex)
        target: Target angle in radians
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Check coordinates shape and reshape if needed
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Get atom positions
    pos1 = coordinates[atom1]
    pos2 = coordinates[atom2]
    pos3 = coordinates[atom3]
    
    # Calculate vectors
    v1 = pos1 - pos2
    v2 = pos3 - pos2
    
    # Normalize vectors
    v1_norm = jnp.linalg.norm(v1)
    v2_norm = jnp.linalg.norm(v2)
    v1_unit = v1 / jnp.maximum(v1_norm, 1e-10)
    v2_unit = v2 / jnp.maximum(v2_norm, 1e-10)
    
    # Calculate angle
    cos_angle = jnp.dot(v1_unit, v2_unit)
    # Clamp to avoid numerical issues at extreme values
    cos_angle = jnp.clip(cos_angle, -0.99999999, 0.99999999)
    angle = jnp.arccos(cos_angle)
    
    # Calculate restraint energy and force magnitude
    energy, force_magnitude = restraint_function(angle, target, force_constant)
    
    # Forces calculation based on derivative of arccos
    sin_angle = jnp.sin(angle)
    
    # Avoid division by zero
    sin_angle = jnp.where(sin_angle < 1e-8, 1e-8, sin_angle)
    
    # Components perpendicular to the vectors
    perp1 = v2_unit - cos_angle * v1_unit
    perp2 = v1_unit - cos_angle * v2_unit
    
    # Calculate forces on each atom
    forces = jnp.zeros_like(coordinates)
    
    # Force magnitudes
    f1_mag = -force_magnitude / (v1_norm * sin_angle)
    f3_mag = -force_magnitude / (v2_norm * sin_angle)
    
    # Force vectors
    f1 = f1_mag * perp1
    f3 = f3_mag * perp2
    f2 = -(f1 + f3)  # Total force must be zero
    
    # Update forces for the three atoms
    forces = forces.at[atom1].set(f1)
    forces = forces.at[atom2].set(f2)
    forces = forces.at[atom3].set(f3)
    
    return energy, forces

def dihedral_restraint_force(coordinates: jnp.ndarray, atom1: int, atom2: int, atom3: int, atom4: int,
                            target: float, force_constant: float,
                            restraint_function: Callable = harmonic_restraint) -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for a dihedral angle restraint.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        atom1, atom2, atom3, atom4: Indices of atoms defining the dihedral
        target: Target dihedral angle in radians
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Check coordinates shape and reshape if needed
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Get the dihedral angle using colvar_dihedral
    dihedral = colvar_dihedral(coordinates, atom1, atom2, atom3, atom4)
    
    # Calculate restraint energy and force magnitude
    energy, force_magnitude = restraint_function(dihedral, target, force_constant)
    
    # For dihedral forces, use automatic differentiation with JAX
    # Define a function that calculates just the dihedral angle
    def calc_dihedral(coords):
        proper_coords = ensure_proper_coordinates(coords)
        return colvar_dihedral(proper_coords, atom1, atom2, atom3, atom4)
    
    # Use JAX to compute the gradient
    grad_fn = jax.grad(calc_dihedral)
    gradients = grad_fn(coordinates)
    
    # Scale gradients by force magnitude to get forces
    forces = -force_magnitude * gradients
    
    return energy, forces

def time_varying_force_constant(force_constants: List[float], update_steps: int, current_step: int,
                             debug: bool = False, name: str = "") -> float:
    """
    Calculate the force constant for the current step based on a list of force constants
    that change over time at evenly spaced intervals.
    
    Args:
        force_constants: List of force constant values to interpolate between
        update_steps: Number of steps between force constant updates
        current_step: Current simulation step
        debug: Whether to print debug information
        name: Name of the restraint (for debugging)
        
    Returns:
        Current force constant value
    """
    # Use our global step counter instead of the parameter
    # This ensures consistency with apply_restraints
    global current_simulation_step
    true_step = current_simulation_step  # Use global step that's incremented in apply_restraints
    
    # Print debugging info about steps
    print(f"# FORCE_CONSTANT: {name} param step={current_step}, global step={true_step}")
    
    if len(force_constants) == 1:
        print(f"# FORCE_CONSTANT: {name} using single value {force_constants[0]:.4f}")
        return force_constants[0]
    
    # Determine which segment we're in and how far through it
    num_segments = len(force_constants) - 1
    total_transition_steps = num_segments * update_steps
    
    # Handle case where we've gone past the last force constant
    if true_step >= total_transition_steps:
        print(f"# FORCE_CONSTANT: {name} past end at step {true_step}, using final value {force_constants[-1]:.4f}")
        return force_constants[-1]
    
    # Calculate which segment we're in and progress through that segment
    segment_idx = min(true_step // update_steps, num_segments - 1)
    progress = (true_step % update_steps) / update_steps
    
    # Linearly interpolate between adjacent force constants
    start_k = force_constants[segment_idx]
    end_k = force_constants[segment_idx + 1]
    current_k = start_k + progress * (end_k - start_k)
    
    print(f"# FORCE_CONSTANT: {name} at step {true_step} = {current_k:.4f} (segment {segment_idx}, progress {progress:.2f}, {start_k:.4f} -> {end_k:.4f})")
    
    # Store this in the global tracking dictionary
    global current_force_constants
    if 'current_force_constants' not in globals():
        current_force_constants = {}
    current_force_constants[name] = current_k
    
    return current_k

def setup_restraints(restraint_definitions: Dict[str, Any]) -> Tuple[Dict[str, Callable], List[Callable], Dict[str, Any]]:
    """
    Set up restraint calculators based on the restraint definitions in the input file.
    
    Args:
        restraint_definitions: Dictionary containing restraint definitions from input file
        
    Returns:
        Tuple containing (restraint_energies, restraint_forces, restraint_metadata):
        - restraint_energies: Dict mapping restraint names to energy calculator functions
        - restraint_forces: List of force calculator functions for all restraints
        - restraint_metadata: Dict containing metadata for time-varying restraints
    """
    restraint_energies = {}
    restraint_forces = []
    restraint_metadata = {
        "time_varying": False,
        "restraints": {}
    }
    
    for restraint_name, restraint_def in restraint_definitions.items():
        restraint_type = restraint_def.get("type", "distance")
        target = float(restraint_def.get("target", 0.0))
        restraint_style = restraint_def.get("style", "harmonic")
        tolerance = float(restraint_def.get("tolerance", 0.0))
        side = restraint_def.get("side", "lower")  # For one-sided restraints
        
        # Handle force constant - can be a single value or a list for time-varying
        force_constant_raw = restraint_def.get("force_constant", 10.0)
        
        # Check if we have a list of force constants (for time-varying)
        time_varying = False
        if isinstance(force_constant_raw, list):
            force_constants = [float(k) for k in force_constant_raw]
            update_steps = int(restraint_def.get("update_steps", 1000))
            force_constant = force_constants[0]  # Initial value
            time_varying = True
            
            # Store metadata for this time-varying restraint
            restraint_metadata["time_varying"] = True
            restraint_metadata["restraints"][restraint_name] = {
                "force_constants": force_constants,
                "update_steps": update_steps,
                "current_idx": 0
            }
        else:
            force_constant = float(force_constant_raw)
        
        # Select the appropriate restraint function
        if restraint_style == "harmonic":
            restraint_function = harmonic_restraint
        elif restraint_style == "flat_bottom":
            restraint_function = partial(flat_bottom_restraint, tolerance=tolerance)
        elif restraint_style == "one_sided":
            restraint_function = partial(one_sided_harmonic_restraint, side=side)
        else:
            raise ValueError(f"Unknown restraint style: {restraint_style}")
            
        # Configure the appropriate force calculator based on restraint type
        if restraint_type == "distance":
            atom1 = int(restraint_def["atom1"])
            atom2 = int(restraint_def["atom2"])
            
            if time_varying:
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, a1=atom1, a2=atom2, t=target, 
                               fc_list=force_constants, upd_steps=update_steps, rname=restraint_name,
                               rf=restraint_function, rm=restraint_metadata):
                    coordinates = ensure_proper_coordinates(coordinates)
                    distance = colvar_distance(coordinates, a1, a2)
                    # Get current force constant based on step
                    fc = time_varying_force_constant(fc_list, upd_steps, step)
                    energy, _ = rf(distance, t, fc)
                    # Store current force constant value for reporting
                    if rm["time_varying"] and rname in rm["restraints"]:
                        rm["restraints"][rname]["current_fc"] = fc
                    return energy
                
                # Create time-varying force calculator
                def time_varying_force_calc(coordinates, step, a1=atom1, a2=atom2, t=target,
                                          fc_list=force_constants, upd_steps=update_steps, 
                                          rf=restraint_function, rname=restraint_name):
                    # Get current force constant based on step
                    # Get the force constant from our reliable function
                    fc = time_varying_force_constant(fc_list, upd_steps, step, name=rname)
                    current_force_constants[rname] = fc
                    
                    # Use the standard force calculator with the current force constant
                    return distance_restraint_force(coordinates, a1, a2, t, fc, rf)
                
                force_calc = time_varying_force_calc
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, a1=atom1, a2=atom2, t=target, fc=force_constant, rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    distance = colvar_distance(coordinates, a1, a2)
                    energy, _ = rf(distance, t, fc)
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    distance_restraint_force, 
                    atom1=atom1, 
                    atom2=atom2, 
                    target=target, 
                    force_constant=force_constant,
                    restraint_function=restraint_function
                )
            
        elif restraint_type == "angle":
            atom1 = int(restraint_def["atom1"])
            atom2 = int(restraint_def["atom2"])
            atom3 = int(restraint_def["atom3"])
            
            if time_varying:
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, t=target, 
                               fc_list=force_constants, upd_steps=update_steps, rname=restraint_name,
                               rf=restraint_function, rm=restraint_metadata):
                    coordinates = ensure_proper_coordinates(coordinates)
                    angle = colvar_angle(coordinates, a1, a2, a3)
                    # Get current force constant based on step
                    fc = time_varying_force_constant(fc_list, upd_steps, step)
                    energy, _ = rf(angle, t, fc)
                    # Store current force constant value for reporting
                    if rm["time_varying"] and rname in rm["restraints"]:
                        rm["restraints"][rname]["current_fc"] = fc
                    return energy
                
                # Create time-varying force calculator
                def time_varying_force_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, t=target,
                                          fc_list=force_constants, upd_steps=update_steps, 
                                          rf=restraint_function, rname=restraint_name):
                    # Get current force constant based on step
                    # Get the force constant from our reliable function
                    fc = time_varying_force_constant(fc_list, upd_steps, step, name=rname)
                    current_force_constants[rname] = fc
                    
                    # Use the standard force calculator with the current force constant
                    return angle_restraint_force(coordinates, a1, a2, a3, t, fc, rf)
                
                force_calc = time_varying_force_calc
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, a1=atom1, a2=atom2, a3=atom3, t=target, fc=force_constant, rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    angle = colvar_angle(coordinates, a1, a2, a3)
                    energy, _ = rf(angle, t, fc)
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    angle_restraint_force, 
                    atom1=atom1, 
                    atom2=atom2, 
                    atom3=atom3, 
                    target=target, 
                    force_constant=force_constant,
                    restraint_function=restraint_function
                )
            
        elif restraint_type == "dihedral":
            atom1 = int(restraint_def["atom1"])
            atom2 = int(restraint_def["atom2"])
            atom3 = int(restraint_def["atom3"])
            atom4 = int(restraint_def["atom4"])
            
            if time_varying:
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, a4=atom4, t=target, 
                               fc_list=force_constants, upd_steps=update_steps, rname=restraint_name,
                               rf=restraint_function, rm=restraint_metadata):
                    coordinates = ensure_proper_coordinates(coordinates)
                    dihedral = colvar_dihedral(coordinates, a1, a2, a3, a4)
                    # Get current force constant based on step
                    fc = time_varying_force_constant(fc_list, upd_steps, step)
                    energy, _ = rf(dihedral, t, fc)
                    # Store current force constant value for reporting
                    if rm["time_varying"] and rname in rm["restraints"]:
                        rm["restraints"][rname]["current_fc"] = fc
                    return energy
                
                # Create time-varying force calculator
                def time_varying_force_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, a4=atom4, t=target,
                                          fc_list=force_constants, upd_steps=update_steps, 
                                          rf=restraint_function, rname=restraint_name):
                    # Get current force constant based on step
                    # Get the force constant from our reliable function
                    fc = time_varying_force_constant(fc_list, upd_steps, step, name=rname)
                    current_force_constants[rname] = fc
                    
                    # Use the standard force calculator with the current force constant
                    return dihedral_restraint_force(coordinates, a1, a2, a3, a4, t, fc, rf)
                
                force_calc = time_varying_force_calc
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, a1=atom1, a2=atom2, a3=atom3, a4=atom4, t=target, fc=force_constant, rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    dihedral = colvar_dihedral(coordinates, a1, a2, a3, a4)
                    energy, _ = rf(dihedral, t, fc)
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    dihedral_restraint_force, 
                    atom1=atom1, 
                    atom2=atom2, 
                    atom3=atom3, 
                    atom4=atom4, 
                    target=target, 
                    force_constant=force_constant,
                    restraint_function=restraint_function
                )
            
        else:
            raise ValueError(f"Unknown restraint type: {restraint_type}")
        
        # Store the calculators
        restraint_energies[restraint_name] = energy_calc
        restraint_forces.append(force_calc)
    
    return restraint_energies, restraint_forces, restraint_metadata

def apply_restraints(coordinates: jnp.ndarray, restraint_forces: List[Callable], 
                  step: int = 0) -> Tuple[float, jnp.ndarray]:
    """
    Apply all restraints to calculate total restraint energy and forces.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        restraint_forces: List of force calculator functions for all restraints
        step: Current simulation step (needed for time-varying restraints)
        
    Returns:
        Tuple containing (total_energy, total_forces)
    """
    # Ensure coordinates are in the right shape
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Use our global step counter instead of the parameter
    # Increment global counter - this guarantees progression regardless of what step is passed in
    global current_simulation_step
    current_simulation_step += 1
    true_step = current_simulation_step
    
    # Print the actual step we're using
    print(f"# RESTRAINTS: Called with param step={step}, using global step={true_step}")
    
    # Pre-allocate a single forces array to reuse for all restraints
    total_energy = 0.0
    total_forces = jnp.zeros_like(coordinates)
    
    # Calculate and accumulate energy and forces from all restraints
    for i, force_calc in enumerate(restraint_forces):
        # Check if this is a time-varying restraint force calculator
        # We can't check __code__ on partial objects, so we need a safer approach
        try:
            # Try to call with step parameter - this will work for time-varying functions
            # Use our global step counter instead of the parameter
            energy, forces = force_calc(coordinates, step=true_step)
        except TypeError:
            # If it fails, call without step parameter for standard force calculators
            energy, forces = force_calc(coordinates)
            
        total_energy += energy
        total_forces += forces
        
    return total_energy, total_forces