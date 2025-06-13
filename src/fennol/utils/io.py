import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os


def xyz_reader(
    filename,
    has_comment_line=False,
    indexed=False,
    start=1,
    stop=-1,
    step=1,
    max_frames=None,
    stream=False,
    interval=2.0,
    sleep=time.sleep,
):
    if stop > 0 and start > stop:
        return
    pf = open(filename, "r")
    inside_frame = False
    nat_read = False
    iframe = 0
    stride = step
    nframes = 0
    if start > 1:
        print(f"skipping {start-1} frames")
    if not has_comment_line:
        comment_line = ""
    while True:
        line = pf.readline()

        if not line:
            if stream:
                sleep(interval)
                continue
            elif inside_frame or nat_read:
                raise Exception("Error: premature end of file!")
            else:
                break

        line = line.strip()

        if not inside_frame:
            if not nat_read:
                if line.startswith("#") or line == "":
                    continue
                nat = int(line.split()[0])
                nat_read = True
                iat = 0
                inside_frame = not has_comment_line
                xyz = np.zeros((nat, 3))
                symbols = []
                continue

            # box = np.array(line.split(), dtype="float")
            comment_line = line
            inside_frame = True
            continue

        if line.startswith("#") or line == "":
            raise Exception("Error: premature end of frame!")

        if not nat_read:
            raise Exception("Error: nat not read!")
        ls = line.split()
        iat, s = (int(ls[0]), 1) if indexed else (iat + 1, 0)
        symbols.append(ls[s])
        xyz[iat - 1, :] = np.array([ls[s + 1], ls[s + 2], ls[s + 3]], dtype="float")
        if iat == nat:
            iframe += 1
            inside_frame = False
            nat_read = False
            if iframe < start:
                continue
            stride += 1
            if stride >= step:
                nframes += 1
                stride = 0
                yield symbols, xyz, comment_line
            if (max_frames is not None and nframes >= max_frames) or (
                stop > 0 and iframe >= stop
            ):
                break


def read_xyz(
    filename,
    has_comment_line=False,
    indexed=False,
    start=1,
    stop=-1,
    step=1,
    max_frames=None,
):
    return [
        frame
        for frame in xyz_reader(
            filename, has_comment_line, indexed, start, stop, step, max_frames
        )
    ]


def last_xyz_frame(filename, has_comment_line=False, indexed=False):
    last_frame = None
    for frame in xyz_reader(
        filename, has_comment_line, indexed, start=-1, stop=-1, step=1, max_frames=1
    ):
        last_frame = frame
    return last_frame


def write_arc_frame(
    f,
    symbols,
    coordinates,
    types=None,
    nbonds=None,
    connectivity=None,
    cell=None,
    **kwargs,
):
    nat = len(symbols)
    f.write(f"{nat}\n")
    if cell is not None:
        f.write(" ".join([f"{x: 15.5f}" for x in cell.flatten()]) + "\n")
    # f.write(f'{axis} {axis} {axis} 90.0 90.0 90.0 \n')
    for i in range(nat):
        line = f"{i+1} {symbols[i]:3} {coordinates[i,0]: 15.3f} {coordinates[i,1]: 15.3f} {coordinates[i,2]: 15.3f}"
        if types is not None:
            line += f"   {types[i]}"
        if connectivity is not None and nbonds is not None:
            line += "  " + " ".join([str(x + 1) for x in connectivity[i, : nbonds[i]]])
        f.write(line + "\n")
    f.flush()


def write_extxyz_frame(
    f, symbols, coordinates, cell=None, properties={}, forces=None, **kwargs
):
    nat = len(symbols)
    f.write(f"{nat}\n")
    comment_line = ""
    if cell is not None:
        comment_line += (
            'Lattice="' + " ".join([f"{x:.3f}" for x in cell.flatten()]) + '" '
        )
    comment_line += "Properties=species:S:1:pos:R:3"
    if forces is not None:
        comment_line += ":forces:R:3"
    comment_line += " "
    for k, v in properties.items():
        comment_line += f"{k}={v} "
    f.write(f"{comment_line}\n")
    for i in range(nat):
        line = f"{symbols[i]:3} {coordinates[i,0]: 15.5e} {coordinates[i,1]: 15.5e} {coordinates[i,2]: 15.5e}"
        if forces is not None:
            line += f" {forces[i,0]: 15.5e} {forces[i,1]: 15.5e} {forces[i,2]: 15.5e}"
        f.write(f"{line}\n")
    f.flush()


def write_xyz_frame(f, symbols, coordinates,cell=None, **kwargs):
    nat = len(symbols)
    f.write(f"{nat}\n")
    if cell is not None:
        f.write(" ".join([f"{x:.3f}" for x in cell.flatten()]))
    f.write("\n")
    for i in range(nat):
        f.write(
            f"{symbols[i]:3} {coordinates[i,0]: 15.5e} {coordinates[i,1]: 15.5e} {coordinates[i,2]: 15.5e}\n"
        )
    f.flush()


def human_time_duration(seconds: float):
    """Convert seconds (duration) to human readable string

    from https://gist.github.com/borgstrom/936ca741e885a1438c374824efb038b3
    """

    if seconds < 1.0:
        return f"{seconds*1000:.3g} ms"
    if seconds < 10.0:
        return f"{seconds:.3g} s"

    TIME_DURATION_UNITS = (
        ("week", "s", 60 * 60 * 24 * 7),
        ("day", "s", 60 * 60 * 24),
        ("h", "", 60 * 60),
        ("min", "", 60),
        ("s", "", 1),
    )
    parts = []
    for unit, plur, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(int(seconds), div)
        if amount > 0:
            parts.append(f"{amount} {unit}{plur if amount > 1 else ''}")
    return " ".join(parts)


def read_pdb(filename: str) -> Dict[str, Union[np.ndarray, List]]:
    """
    Read a PDB file and extract atom information.
    
    Args:
        filename: Path to PDB file
        
    Returns:
        Dictionary containing:
        - coordinates: np.ndarray of shape (n_atoms, 3)
        - symbols: List of element symbols
        - atom_names: List of atom names
        - residue_names: List of residue names
        - residue_numbers: List of residue numbers
        - chain_ids: List of chain identifiers
        - occupancies: List of occupancies
        - temp_factors: List of temperature factors
    """
    coordinates = []
    symbols = []
    atom_names = []
    residue_names = []
    residue_numbers = []
    chain_ids = []
    occupancies = []
    temp_factors = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse PDB ATOM/HETATM record
                # Columns: 1-6 Record name, 7-11 Atom serial, 13-16 Atom name,
                # 17 Alt location, 18-20 Residue name, 22 Chain ID,
                # 23-26 Residue sequence, 31-38 X, 39-46 Y, 47-54 Z,
                # 55-60 Occupancy, 61-66 Temperature factor, 77-78 Element
                
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                # Optional fields
                try:
                    occupancy = float(line[54:60].strip())
                except:
                    occupancy = 1.0
                    
                try:
                    temp_factor = float(line[60:66].strip())
                except:
                    temp_factor = 0.0
                
                # Element symbol - try to get from columns 77-78 first
                try:
                    element = line[76:78].strip()
                except:
                    # Fallback: guess from atom name
                    element = ''.join([c for c in atom_name if c.isalpha()])[:2]
                    element = element[0].upper() + element[1:].lower() if len(element) > 1 else element.upper()
                
                coordinates.append([x, y, z])
                symbols.append(element)
                atom_names.append(atom_name)
                residue_names.append(res_name)
                residue_numbers.append(res_num)
                chain_ids.append(chain_id)
                occupancies.append(occupancy)
                temp_factors.append(temp_factor)
    
    return {
        'coordinates': np.array(coordinates),
        'symbols': symbols,
        'atom_names': atom_names,
        'residue_names': residue_names,
        'residue_numbers': residue_numbers,
        'chain_ids': chain_ids,
        'occupancies': occupancies,
        'temp_factors': temp_factors
    }


def parse_atom_selection(selection_dict: Dict[str, any], pdb_data: Optional[Dict] = None,
                        coordinates: Optional[np.ndarray] = None) -> List[int]:
    """
    Parse atom selection criteria and return list of atom indices.
    
    Args:
        selection_dict: Dictionary with selection criteria:
            - residue_numbers: List of residue numbers or ranges (e.g., [1, 2, "3-5", 10])
            - residue_names: List of residue names (e.g., ["ALA", "GLY"])
            - atom_names: List of atom names (e.g., ["CA", "CB"])
            - chain_ids: List of chain IDs (e.g., ["A", "B"])
            - within_distance: Dict with 'distance' and 'of' keys for distance-based selection
                             e.g., {'distance': 5.0, 'of': {'residue_numbers': [100]}}
            - indices: Direct list of atom indices
        pdb_data: PDB data dictionary from read_pdb()
        coordinates: Coordinate array (required for distance calculations)
        
    Returns:
        List of selected atom indices
    """
    if 'indices' in selection_dict:
        # Direct index specification
        return list(selection_dict['indices'])
    
    if pdb_data is None:
        raise ValueError("PDB data required for selection criteria other than direct indices")
    
    n_atoms = len(pdb_data['symbols'])
    selected = np.ones(n_atoms, dtype=bool)
    
    # Residue number selection
    if 'residue_numbers' in selection_dict:
        res_mask = np.zeros(n_atoms, dtype=bool)
        for res_spec in selection_dict['residue_numbers']:
            if isinstance(res_spec, str) and '-' in res_spec:
                # Range specification
                start, end = map(int, res_spec.split('-'))
                for i, res_num in enumerate(pdb_data['residue_numbers']):
                    if start <= res_num <= end:
                        res_mask[i] = True
            else:
                # Single residue
                res_num = int(res_spec)
                for i, rn in enumerate(pdb_data['residue_numbers']):
                    if rn == res_num:
                        res_mask[i] = True
        selected &= res_mask
    
    # Residue name selection
    if 'residue_names' in selection_dict:
        res_names = set(selection_dict['residue_names'])
        name_mask = np.array([rn in res_names for rn in pdb_data['residue_names']])
        selected &= name_mask
    
    # Atom name selection
    if 'atom_names' in selection_dict:
        atom_names = set(selection_dict['atom_names'])
        atom_mask = np.array([an in atom_names for an in pdb_data['atom_names']])
        selected &= atom_mask
    
    # Chain ID selection
    if 'chain_ids' in selection_dict:
        chain_ids = set(selection_dict['chain_ids'])
        chain_mask = np.array([ch in chain_ids for ch in pdb_data['chain_ids']])
        selected &= chain_mask
    
    # Distance-based selection
    if 'within_distance' in selection_dict:
        if coordinates is None:
            if 'coordinates' in pdb_data:
                coordinates = pdb_data['coordinates']
            else:
                raise ValueError("Coordinates required for distance-based selection")
        
        dist_dict = selection_dict['within_distance']
        distance = dist_dict['distance']
        
        # Get reference atoms using recursive selection
        ref_indices = parse_atom_selection(dist_dict['of'], pdb_data, coordinates)
        ref_coords = coordinates[ref_indices]
        
        # Calculate distances to all atoms
        dist_mask = np.zeros(n_atoms, dtype=bool)
        for i in range(n_atoms):
            # Check distance to any reference atom
            min_dist = np.min(np.linalg.norm(coordinates[i] - ref_coords, axis=1))
            if min_dist <= distance:
                dist_mask[i] = True
        
        selected &= dist_mask
    
    # Convert boolean mask to indices
    return np.where(selected)[0].tolist()


def calculate_center_of_mass(coordinates: Union[np.ndarray, 'jax.Array'], indices: List[int],
                           masses: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate center of mass for selected atoms.
    
    Args:
        coordinates: Full coordinate array (numpy or JAX array)
        indices: Indices of atoms to include
        masses: Optional mass array (if None, assumes unit mass)
        
    Returns:
        Center of mass coordinates [x, y, z]
    """
    # Convert to numpy if it's a JAX array for easier handling
    if hasattr(coordinates, '__array__'):
        coordinates = np.asarray(coordinates)
    
    selected_coords = coordinates[indices]
    
    if masses is not None:
        selected_masses = masses[indices]
        total_mass = np.sum(selected_masses)
        com = np.sum(selected_coords * selected_masses[:, np.newaxis], axis=0) / total_mass
    else:
        # Assume unit mass
        com = np.mean(selected_coords, axis=0)
    
    return com
