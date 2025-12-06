"""
PDB file reader and writer for FeNNol.

Supports reading molecular structures from PDB files and setting up
MD simulations with proper topology assignment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class Atom:
    """Single atom from PDB file."""
    index: int
    name: str
    resname: str
    chain: str
    resid: int
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    bfactor: float = 0.0
    element: str = ""


@dataclass
class PDBStructure:
    """Complete structure from PDB file."""
    atoms: List[Atom]
    coordinates: np.ndarray  # [natoms, 3]
    elements: List[str]
    residue_names: List[str]
    residue_ids: List[int]
    chain_ids: List[str]
    atomic_numbers: np.ndarray  # [natoms]
    masses: np.ndarray  # [natoms]

    @property
    def natoms(self) -> int:
        return len(self.atoms)


# Element to atomic number mapping
ELEMENT_TO_ATOMIC_NUMBER = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16,
    'F': 9, 'Cl': 17, 'Br': 35, 'I': 53,
    'Na': 11, 'Mg': 12, 'K': 19, 'Ca': 20,
    'Fe': 26, 'Zn': 30, 'Cu': 29,
}

# Atomic masses (amu)
ATOMIC_MASSES = {
    1: 1.008,    # H
    6: 12.01,    # C
    7: 14.01,    # N
    8: 16.00,    # O
    9: 19.00,    # F
    11: 22.99,   # Na
    12: 24.31,   # Mg
    15: 30.97,   # P
    16: 32.07,   # S
    17: 35.45,   # Cl
    19: 39.10,   # K
    20: 40.08,   # Ca
    26: 55.85,   # Fe
    29: 63.55,   # Cu
    30: 65.38,   # Zn
    35: 79.90,   # Br
    53: 126.90,  # I
}


def guess_element_from_atom_name(atom_name: str, resname: str = "") -> str:
    """
    Guess element from atom name in PDB file.

    PDB atom names are often formatted as:
    - " CA " for alpha carbon
    - " H1 " for hydrogen
    - "1H  " or " H1 " for hydrogens

    Args:
        atom_name: 4-character atom name from PDB
        resname: Residue name (optional, helps with disambiguation)

    Returns:
        Element symbol (e.g., "C", "H", "O")
    """
    # Strip whitespace
    atom_name = atom_name.strip()

    # Remove leading digits (e.g., "1H" -> "H")
    atom_name = re.sub(r'^[0-9]+', '', atom_name)

    # Remove trailing numbers and primes (e.g., "CA1" -> "CA", "O5'" -> "O")
    atom_name = re.sub(r"[0-9'\"]+$", '', atom_name)

    # Check two-letter elements first (Cl, Br, Ca, etc.)
    if len(atom_name) >= 2:
        two_letter = atom_name[:2]
        if two_letter in ELEMENT_TO_ATOMIC_NUMBER:
            return two_letter
        # Try capitalized version
        two_letter = two_letter[0].upper() + two_letter[1].lower()
        if two_letter in ELEMENT_TO_ATOMIC_NUMBER:
            return two_letter

    # Single letter element
    if len(atom_name) >= 1:
        one_letter = atom_name[0].upper()
        if one_letter in ELEMENT_TO_ATOMIC_NUMBER:
            return one_letter

    # Default to carbon if can't determine
    print(f"Warning: Could not determine element for atom '{atom_name}', assuming Carbon")
    return 'C'


def read_pdb(filename: str, assign_elements: bool = True) -> PDBStructure:
    """
    Read molecular structure from PDB file.

    Args:
        filename: Path to PDB file
        assign_elements: Whether to guess elements from atom names

    Returns:
        PDBStructure with coordinates and topology information

    Example:
        >>> structure = read_pdb("protein.pdb")
        >>> print(f"Loaded {structure.natoms} atoms")
        >>> print(f"Coordinates shape: {structure.coordinates.shape}")
    """
    atoms = []

    with open(filename, 'r') as f:
        for line in f:
            # Parse ATOM and HETATM records
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue

            # PDB format (fixed width):
            # ATOM     index name resname chain resid    x       y       z    occ  bfac element
            # 0-6      6-11  12-16 17-20   21    22-26 30-38   38-46   46-54 54-60 60-66 76-78

            try:
                index = int(line[6:11].strip())
                name = line[12:16]  # Keep spacing for element determination
                resname = line[17:20].strip()
                chain = line[21] if len(line) > 21 else 'A'
                resid = int(line[22:26].strip()) if line[22:26].strip() else 0
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                # Optional fields
                occupancy = float(line[54:60].strip()) if len(line) > 60 and line[54:60].strip() else 1.0
                bfactor = float(line[60:66].strip()) if len(line) > 66 and line[60:66].strip() else 0.0
                element = line[76:78].strip() if len(line) > 78 else ""

                # Guess element if not provided
                if not element and assign_elements:
                    element = guess_element_from_atom_name(name, resname)

                atoms.append(Atom(
                    index=index,
                    name=name.strip(),
                    resname=resname,
                    chain=chain,
                    resid=resid,
                    x=x, y=y, z=z,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    element=element
                ))

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line.strip()}")
                print(f"  Error: {e}")
                continue

    if not atoms:
        raise ValueError(f"No atoms found in PDB file: {filename}")

    # Extract arrays
    coordinates = np.array([[a.x, a.y, a.z] for a in atoms])
    elements = [a.element if a.element else 'C' for a in atoms]
    residue_names = [a.resname for a in atoms]
    residue_ids = [a.resid for a in atoms]
    chain_ids = [a.chain for a in atoms]

    # Convert elements to atomic numbers
    atomic_numbers = np.array([
        ELEMENT_TO_ATOMIC_NUMBER.get(elem, 6)  # Default to carbon
        for elem in elements
    ], dtype=np.int32)

    # Assign masses
    masses = np.array([
        ATOMIC_MASSES.get(z, 12.01)  # Default to carbon mass
        for z in atomic_numbers
    ])

    structure = PDBStructure(
        atoms=atoms,
        coordinates=coordinates,
        elements=elements,
        residue_names=residue_names,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        atomic_numbers=atomic_numbers,
        masses=masses
    )

    print(f"Loaded PDB: {filename}")
    print(f"  Atoms: {structure.natoms}")
    print(f"  Residues: {len(set(zip(chain_ids, residue_ids)))}")
    print(f"  Chains: {len(set(chain_ids))}")

    # Element summary
    from collections import Counter
    elem_counts = Counter(elements)
    print(f"  Elements: {', '.join(f'{e}×{c}' for e, c in sorted(elem_counts.items()))}")

    return structure


def write_pdb(structure: PDBStructure, filename: str, title: str = "Structure"):
    """
    Write molecular structure to PDB file.

    Args:
        structure: PDBStructure to write
        filename: Output PDB file path
        title: Title for PDB header
    """
    with open(filename, 'w') as f:
        # Header
        f.write(f"TITLE     {title}\n")
        f.write(f"REMARK    Generated by FeNNol\n")
        f.write(f"REMARK    {structure.natoms} atoms\n")

        # Atoms
        for i, atom in enumerate(structure.atoms):
            # Update coordinates (may have changed during simulation)
            atom.x = structure.coordinates[i, 0]
            atom.y = structure.coordinates[i, 1]
            atom.z = structure.coordinates[i, 2]

            # Format: ATOM index name resname chain resid x y z occ bfac element
            line = (
                f"ATOM  {atom.index:5d} {atom.name:^4s} {atom.resname:3s} "
                f"{atom.chain:1s}{atom.resid:4d}    "
                f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
                f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}          "
                f"{atom.element:>2s}\n"
            )
            f.write(line)

        f.write("END\n")

    print(f"Wrote PDB: {filename}")


def assign_charges(structure: PDBStructure, method: str = "amber") -> np.ndarray:
    """
    Assign partial charges to atoms based on residue templates.

    Args:
        structure: PDBStructure
        method: Charge assignment method ("amber", "opls", "zero")

    Returns:
        Array of partial charges [natoms]

    Note:
        This is a simplified version. For accurate charges, use
        a proper force field parameter assignment tool like ParmEd or OpenMM.
    """
    charges = np.zeros(structure.natoms)

    if method == "zero":
        return charges

    # Simple charge assignment based on element and residue type
    # This is approximate - for real simulations use proper force field
    for i, atom in enumerate(structure.atoms):
        elem = atom.element
        resname = atom.resname
        atomname = atom.name

        # Nucleic acids (DNA/RNA)
        if resname in ['A', 'G', 'C', 'T', 'U', 'DA', 'DG', 'DC', 'DT']:
            if elem == 'P':
                charges[i] = 1.17  # Phosphate phosphorus
            elif elem == 'O' and 'P' in atomname:
                charges[i] = -0.78  # Phosphate oxygen
            elif elem == 'O' and "'" in atomname:
                charges[i] = -0.66  # Ribose oxygen
            elif elem == 'N':
                charges[i] = -0.75  # Nitrogen in bases
            elif elem == 'C' and "'" in atomname:
                charges[i] = 0.16  # Ribose carbon
            else:
                charges[i] = 0.0  # Other atoms

        # Amino acids (proteins) - very simplified
        elif len(resname) == 3:  # Likely amino acid
            if elem == 'N' and atomname == 'N':
                charges[i] = -0.47  # Backbone N
            elif elem == 'C' and atomname == 'C':
                charges[i] = 0.60  # Backbone C=O
            elif elem == 'O' and atomname in ['O', 'OXT']:
                charges[i] = -0.57  # Backbone carbonyl O
            elif elem == 'O' and atomname.startswith('O'):
                charges[i] = -0.50  # Side chain hydroxyl
            elif elem == 'N' and atomname.startswith('N'):
                charges[i] = -0.50  # Side chain amine
            else:
                charges[i] = 0.0

        else:
            charges[i] = 0.0

    # Neutralize system
    total_charge = charges.sum()
    if abs(total_charge) > 0.01:
        charges -= total_charge / structure.natoms
        print(f"Neutralized system charge: {total_charge:.3f} e")

    return charges


if __name__ == "__main__":
    # Test PDB reading
    import sys

    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
        structure = read_pdb(pdb_file)

        print(f"\nCoordinate range:")
        print(f"  X: [{structure.coordinates[:, 0].min():.2f}, {structure.coordinates[:, 0].max():.2f}]")
        print(f"  Y: [{structure.coordinates[:, 1].min():.2f}, {structure.coordinates[:, 1].max():.2f}]")
        print(f"  Z: [{structure.coordinates[:, 2].min():.2f}, {structure.coordinates[:, 2].max():.2f}]")

        print(f"\nMass: {structure.masses.sum():.2f} amu")

        charges = assign_charges(structure)
        print(f"Total charge: {charges.sum():.6f} e")
