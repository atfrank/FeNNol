import jax
import jax.numpy as jnp
from functools import partial

def ensure_proper_coordinates(coordinates):
    """Ensure coordinates have the correct shape [natoms, 3]"""
    if coordinates.ndim == 1:
        # If we have flattened coordinates, reshape them
        natoms = coordinates.shape[0] // 3
        return coordinates.reshape(natoms, 3)
    return coordinates

def colvar_distance(coordinates, atom1, atom2):
    # Ensure proper shape
    coordinates = ensure_proper_coordinates(coordinates)
    return jnp.linalg.norm(coordinates[atom1] - coordinates[atom2])

def colvar_angle(coordinates, atom1, atom2, atom3):
    # Ensure proper shape
    coordinates = ensure_proper_coordinates(coordinates)
    
    v1 = coordinates[atom1] - coordinates[atom2]
    v2 = coordinates[atom3] - coordinates[atom2]
    v1_norm = jnp.linalg.norm(v1)
    v2_norm = jnp.linalg.norm(v2)
    
    # Avoid division by zero
    v1_safe = v1 / jnp.maximum(v1_norm, 1e-10)
    v2_safe = v2 / jnp.maximum(v2_norm, 1e-10)
    
    # Ensure dot product is within valid range for arccos
    dot_product = jnp.dot(v1_safe, v2_safe)
    dot_product = jnp.clip(dot_product, -0.9999999, 0.9999999)
    
    return jnp.arccos(dot_product)

def colvar_dihedral(coordinates, atom1, atom2, atom3, atom4):
    # Ensure proper shape
    coordinates = ensure_proper_coordinates(coordinates)
    
    v1 = coordinates[atom1] - coordinates[atom2]
    v2 = coordinates[atom3] - coordinates[atom2]
    v3 = coordinates[atom4] - coordinates[atom3]
    
    n1 = jnp.cross(v1, v2)
    n2 = jnp.cross(v2, v3)
    
    n1_norm = jnp.linalg.norm(n1)
    n2_norm = jnp.linalg.norm(n2)
    
    # Avoid division by zero
    n1_safe = n1 / jnp.maximum(n1_norm, 1e-10)
    n2_safe = n2 / jnp.maximum(n2_norm, 1e-10)
    
    # Ensure dot product is within valid range for arccos
    dot_product = jnp.dot(n1_safe, n2_safe)
    dot_product = jnp.clip(dot_product, -0.9999999, 0.9999999)
    
    return jnp.arccos(dot_product)


def setup_colvars(colvars_definitions):
    colvars = {}
    for colvar_name, colvar_def in colvars_definitions.items():
        colvar_type = colvar_def.get("type", "distance")
        if colvar_type == "distance":
            atom1 = colvar_def["atom1"]
            atom2 = colvar_def["atom2"]
            colvars[colvar_name] = partial(colvar_distance, atom1=atom1, atom2=atom2)
        elif colvar_type == "angle":
            atom1 = colvar_def["atom1"]
            atom2 = colvar_def["atom2"]
            atom3 = colvar_def["atom3"]
            colvars[colvar_name] = partial(colvar_angle, atom1=atom1, atom2=atom2, atom3=atom3)
        elif colvar_type == "dihedral":
            atom1 = colvar_def["atom1"]
            atom2 = colvar_def["atom2"]
            atom3 = colvar_def["atom3"]
            atom4 = colvar_def["atom4"]
            colvars[colvar_name] = partial(colvar_dihedral, atom1=atom1, atom2=atom2, atom3=atom3, atom4=atom4)
        else:
            raise ValueError(f"Unknown colvar type {colvar_type}")
    
    return colvars