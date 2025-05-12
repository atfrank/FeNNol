"""
Energy formatter module for consistent energy display across MD and minimization
"""
import jax.numpy as jnp
from ..utils.atomic_units import AtomicUnits as au

def format_energy_for_display(energy_value, system_data, params):
    """
    Format energy for display consistently with MD simulation format
    
    Args:
        energy_value: Energy value to format
        system_data: Dictionary with system information
        params: Dictionary with simulation parameters
        
    Returns:
        Tuple of (display_energy, energy_unit_str) 
    """
    # Convert energy to appropriate units and per-atom if needed
    energy_unit_str = params.get("energy_unit", "kcal/mol")
    energy_unit = au.get_multiplier(energy_unit_str)
    per_atom_energy = params.get("per_atom_energy", True)
    nat = system_data["nat"]
    
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


def update_final_energy_display(energy, system_data, params, max_force, rms_force):
    """
    Print final energy statistics in MD-compatible format
    
    Args:
        energy: Energy value to display
        system_data: Dictionary with system information
        params: Dictionary with simulation parameters
        max_force: Maximum force magnitude
        rms_force: RMS force
    """
    # Format energy for display
    display_energy, energy_unit_str = format_energy_for_display(energy, system_data, params)
    
    print(f"# Final energy: {display_energy:.8f} {energy_unit_str}")
    print(f"# Max force magnitude: {max_force:.8f}")
    print(f"# RMS force: {rms_force:.8f}")