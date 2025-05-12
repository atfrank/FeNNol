"""
Clean production version of the restraints implementation for FeNNol.
This version removes debug print statements for performance and clarity.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Callable, Tuple, List
from .colvars import colvar_distance, colvar_angle, colvar_dihedral

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
        return colvar_dihedral(coords, atom1, atom2, atom3, atom4)
    
    # Use JAX to compute the gradient
    grad_fn = jax.grad(calc_dihedral)
    gradients = grad_fn(coordinates)
    
    # Scale gradients by force magnitude to get forces
    forces = -force_magnitude * gradients
    
    return energy, forces

def setup_restraints(restraint_definitions: Dict[str, Any]) -> Tuple[Dict[str, Callable], List[Callable]]:
    """
    Set up restraint calculators based on the restraint definitions in the input file.
    
    Args:
        restraint_definitions: Dictionary containing restraint definitions from input file
        
    Returns:
        Tuple containing (restraint_energies, restraint_forces):
        - restraint_energies: Dict mapping restraint names to energy calculator functions
        - restraint_forces: List of force calculator functions for all restraints
    """
    restraint_energies = {}
    restraint_forces = []
    
    for restraint_name, restraint_def in restraint_definitions.items():
        restraint_type = restraint_def.get("type", "distance")
        force_constant = float(restraint_def.get("force_constant", 10.0))
        target = float(restraint_def.get("target", 0.0))
        restraint_style = restraint_def.get("style", "harmonic")
        tolerance = float(restraint_def.get("tolerance", 0.0))
        
        # Select the appropriate restraint function
        if restraint_style == "harmonic":
            restraint_function = harmonic_restraint
        elif restraint_style == "flat_bottom":
            restraint_function = partial(flat_bottom_restraint, tolerance=tolerance)
        else:
            raise ValueError(f"Unknown restraint style: {restraint_style}")
            
        # Configure the appropriate force calculator based on restraint type
        if restraint_type == "distance":
            atom1 = int(restraint_def["atom1"])
            atom2 = int(restraint_def["atom2"])
            
            # Create energy calculator
            def energy_calc(coordinates, a1=atom1, a2=atom2, t=target, fc=force_constant, rf=restraint_function):
                distance = colvar_distance(coordinates, a1, a2)
                energy, _ = rf(distance, t, fc)
                return energy
            
            # Create force calculator
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
            
            # Create energy calculator
            def energy_calc(coordinates, a1=atom1, a2=atom2, a3=atom3, t=target, fc=force_constant, rf=restraint_function):
                angle = colvar_angle(coordinates, a1, a2, a3)
                energy, _ = rf(angle, t, fc)
                return energy
            
            # Create force calculator
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
            
            # Create energy calculator
            def energy_calc(coordinates, a1=atom1, a2=atom2, a3=atom3, a4=atom4, t=target, fc=force_constant, rf=restraint_function):
                dihedral = colvar_dihedral(coordinates, a1, a2, a3, a4)
                energy, _ = rf(dihedral, t, fc)
                return energy
            
            # Create force calculator
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
    
    return restraint_energies, restraint_forces

def apply_restraints(coordinates: jnp.ndarray, restraint_forces: List[Callable]) -> Tuple[float, jnp.ndarray]:
    """
    Apply all restraints to calculate total restraint energy and forces.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        restraint_forces: List of force calculator functions for all restraints
        
    Returns:
        Tuple containing (total_energy, total_forces)
    """
    total_energy = 0.0
    total_forces = jnp.zeros_like(coordinates)
    
    # Calculate and accumulate energy and forces from all restraints
    for force_calc in restraint_forces:
        energy, forces = force_calc(coordinates)
        total_energy += energy
        total_forces += forces
        
    return total_energy, total_forces