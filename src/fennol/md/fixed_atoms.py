"""
Fixed atoms support for FeNNol MD simulations.

This module provides functionality to fix (freeze) certain atoms during
MD simulation. Fixed atoms do not move but still exert forces on mobile
atoms, providing physically correct behavior.

Usage:
    # Via config file (.fnl):
    fixed_atoms {
        mode = mobile_residues
        pdb_file = protein.pdb
        selection {
            residue_numbers = 45, 46, "50-55"
        }
    }

    # Via Python API:
    from fennol.md.fixed_atoms import create_fixed_atom_mask
    fixed_mask, info = create_fixed_atom_mask(
        pdb_file="protein.pdb",
        selection={"residue_numbers": [45, 46, "50-55"]},
        mode="mobile_residues"
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union


def pdb_structure_to_dict(pdb_structure) -> Dict[str, Any]:
    """
    Convert PDBStructure dataclass to dictionary format expected by parse_atom_selection.

    Args:
        pdb_structure: PDBStructure instance from read_pdb()

    Returns:
        Dictionary with keys: symbols, coordinates, residue_numbers, residue_names,
        atom_names, chain_ids
    """
    return {
        'symbols': pdb_structure.elements,
        'coordinates': pdb_structure.coordinates,
        'residue_numbers': pdb_structure.residue_ids,
        'residue_names': pdb_structure.residue_names,
        'atom_names': [atom.name for atom in pdb_structure.atoms],
        'chain_ids': pdb_structure.chain_ids,
    }


def create_fixed_atom_mask(
    pdb_file: Optional[str] = None,
    pdb_data: Optional[Dict] = None,
    coordinates: Optional[np.ndarray] = None,
    selection: Optional[Dict] = None,
    indices: Optional[List[int]] = None,
    mode: str = "fixed_residues",
    n_atoms: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create boolean mask indicating which atoms are fixed.

    Fixed atoms do not move during simulation but STILL exert forces on mobile
    atoms, providing physically correct behavior.

    Args:
        pdb_file: Path to PDB file for residue-based selection
        pdb_data: Pre-loaded PDB data dictionary (alternative to pdb_file)
        coordinates: Atomic coordinates array [natoms, 3] for distance selection
        selection: Dictionary with selection criteria:
            - residue_numbers: List of residue numbers or ranges (e.g., [1, 2, "3-5"])
            - residue_names: List of residue names (e.g., ["ALA", "GLY"])
            - atom_names: List of atom names (e.g., ["CA", "CB"])
            - chain_ids: List of chain IDs (e.g., ["A", "B"])
            - within_distance: Dict with 'distance' and 'of' keys
            - indices: Direct list of atom indices
        indices: Direct atom indices to fix/mobilize (overrides selection)
        mode: Selection interpretation:
            - "fixed_residues": selection specifies FIXED atoms (default)
            - "mobile_residues": selection specifies MOBILE atoms
        n_atoms: Total number of atoms (inferred if not provided)

    Returns:
        Tuple of:
        - fixed_mask: np.ndarray[bool] where True = fixed, False = mobile
        - info: Dict with n_fixed, n_mobile, n_total, fixed_fraction, mobile_fraction

    Raises:
        ValueError: If all atoms would be fixed
        ValueError: If PDB file not found

    Example:
        # Fix everything except active site (residues 50-60)
        mask, info = create_fixed_atom_mask(
            pdb_file="protein.pdb",
            selection={"residue_numbers": ["50-60"]},
            mode="mobile_residues"
        )
        print(f"Mobile: {info['n_mobile']}, Fixed: {info['n_fixed']}")
    """
    from ..utils.io import parse_atom_selection
    from ..utils.pdb import read_pdb as read_pdb_file

    # Load PDB if needed
    if pdb_file is not None and pdb_data is None:
        pdb_structure = read_pdb_file(pdb_file)
        pdb_data = pdb_structure_to_dict(pdb_structure)
        if coordinates is None:
            coordinates = pdb_data['coordinates']

    # Determine number of atoms
    if n_atoms is None:
        if coordinates is not None:
            n_atoms = len(coordinates)
        elif pdb_data is not None:
            n_atoms = len(pdb_data['symbols'])
        else:
            raise ValueError("Cannot determine number of atoms. Provide n_atoms, coordinates, or pdb_data.")

    # Handle direct indices
    if indices is not None:
        indices = [int(i) for i in indices]  # Ensure integers
        if mode == "fixed_residues":
            # indices specifies fixed atoms
            fixed_mask = np.zeros(n_atoms, dtype=bool)
            fixed_mask[indices] = True
        else:  # mobile_residues
            # indices specifies mobile atoms
            fixed_mask = np.ones(n_atoms, dtype=bool)
            fixed_mask[indices] = False

    # Handle selection criteria
    elif selection is not None:
        if pdb_data is None:
            raise ValueError("pdb_file or pdb_data required for selection criteria")

        if coordinates is None and 'coordinates' in pdb_data:
            coordinates = pdb_data['coordinates']

        selected_indices = parse_atom_selection(selection, pdb_data, coordinates)

        if mode == "fixed_residues":
            # selection specifies fixed atoms
            fixed_mask = np.zeros(n_atoms, dtype=bool)
            fixed_mask[selected_indices] = True
        else:  # mobile_residues
            # selection specifies mobile atoms
            fixed_mask = np.ones(n_atoms, dtype=bool)
            fixed_mask[selected_indices] = False
    else:
        # No selection = all mobile
        fixed_mask = np.zeros(n_atoms, dtype=bool)

    # Compute statistics
    n_fixed = int(np.sum(fixed_mask))
    n_mobile = n_atoms - n_fixed

    # Validation
    if n_mobile == 0:
        raise ValueError(
            "All atoms are fixed! No atoms will move during simulation. "
            "Check your fixed_atoms selection."
        )

    info = {
        'n_fixed': n_fixed,
        'n_mobile': n_mobile,
        'n_total': n_atoms,
        'fixed_fraction': n_fixed / n_atoms if n_atoms > 0 else 0.0,
        'mobile_fraction': n_mobile / n_atoms if n_atoms > 0 else 0.0,
    }

    return fixed_mask, info


def setup_fixed_atoms(
    fixed_atoms_config: Dict[str, Any],
    system_data: Dict[str, Any],
    conformation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Set up fixed atoms from configuration and add to system_data.

    This function is called during integrator initialization to parse the
    fixed_atoms block from the .fnl config file.

    Args:
        fixed_atoms_config: Dictionary from parsed .fnl file containing:
            - mode: "fixed_residues" or "mobile_residues"
            - pdb_file: Path to PDB file (optional)
            - selection: Nested dict with selection criteria (optional)
            - indices: Direct atom indices (optional)
        system_data: System data dictionary from load_system_data()
        conformation: Conformation dictionary with coordinates

    Returns:
        Updated system_data with:
        - fixed_mask: Boolean mask array
        - n_mobile_atoms: Number of mobile atoms
        - n_fixed_atoms: Number of fixed atoms
        - reference_coordinates: Original coordinates for position restoration

    Example:
        fixed_atoms_config = simulation_parameters.get("fixed_atoms", None)
        if fixed_atoms_config is not None:
            system_data = setup_fixed_atoms(fixed_atoms_config, system_data, conformation)
    """
    mode = fixed_atoms_config.get('mode', 'fixed_residues')
    pdb_file = fixed_atoms_config.get('pdb_file', None)
    selection = fixed_atoms_config.get('selection', None)
    indices = fixed_atoms_config.get('indices', None)

    # Handle indices as list
    if indices is not None:
        if isinstance(indices, (int, float)):
            indices = [int(indices)]
        elif isinstance(indices, str):
            # Handle comma-separated string
            indices = [int(i.strip()) for i in indices.split(',')]
        else:
            indices = list(indices)

    # Get coordinates
    coords = conformation.get('coordinates')
    if hasattr(coords, '__array__'):
        coords = np.asarray(coords)

    n_atoms = system_data.get('nat', len(coords))

    # Create mask
    fixed_mask, info = create_fixed_atom_mask(
        pdb_file=pdb_file,
        coordinates=coords,
        selection=selection,
        indices=indices,
        mode=mode,
        n_atoms=n_atoms
    )

    # Store in system_data
    system_data = {
        **system_data,
        'fixed_mask': fixed_mask,
        'n_mobile_atoms': info['n_mobile'],
        'n_fixed_atoms': info['n_fixed'],
        'reference_coordinates': np.array(coords).copy(),
    }

    print(f"# Fixed atoms: {info['n_fixed']} fixed, {info['n_mobile']} mobile "
          f"({info['fixed_fraction']*100:.1f}% fixed)")

    return system_data
