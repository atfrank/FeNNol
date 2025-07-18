"""
Molecular topology detection and management for structure refinement
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX

# Standard bond lengths in Angstroms for common atom pairs
STANDARD_BOND_LENGTHS = {
    ('C', 'C'): 1.54,
    ('C', 'H'): 1.09,
    ('C', 'N'): 1.47,
    ('C', 'O'): 1.43,
    ('C', 'S'): 1.82,
    ('C', 'P'): 1.84,
    ('N', 'H'): 1.01,
    ('N', 'N'): 1.45,
    ('N', 'O'): 1.40,
    ('N', 'P'): 1.78,
    ('O', 'H'): 0.96,
    ('O', 'O'): 1.48,
    ('O', 'P'): 1.63,
    ('O', 'S'): 1.70,
    ('P', 'H'): 1.42,
    ('P', 'P'): 2.21,
    ('P', 'S'): 2.10,
    ('S', 'H'): 1.34,
    ('S', 'S'): 2.05,
    # Double bonds (shorter)
    ('C', 'C', 'double'): 1.34,
    ('C', 'N', 'double'): 1.28,
    ('C', 'O', 'double'): 1.23,
    ('N', 'O', 'double'): 1.21,
    # Triple bonds (even shorter)
    ('C', 'C', 'triple'): 1.20,
    ('C', 'N', 'triple'): 1.16,
    ('N', 'N', 'triple'): 1.10,
}

# Maximum bond lengths for detection (1.4x standard length)
MAX_BOND_LENGTHS = {
    ('C', 'C'): 2.2,
    ('C', 'H'): 1.5,
    ('C', 'N'): 2.0,
    ('C', 'O'): 1.9,
    ('C', 'S'): 2.5,
    ('C', 'P'): 2.5,
    ('N', 'H'): 1.4,
    ('N', 'N'): 2.0,
    ('N', 'O'): 1.9,
    ('N', 'P'): 2.4,
    ('O', 'H'): 1.3,
    ('O', 'O'): 2.0,
    ('O', 'P'): 2.2,
    ('O', 'S'): 2.3,
    ('P', 'H'): 1.9,
    ('P', 'P'): 3.0,
    ('P', 'S'): 2.8,
    ('S', 'H'): 1.8,
    ('S', 'S'): 2.8,
}

# Standard bond angles in degrees for common atom triplets
STANDARD_BOND_ANGLES = {
    ('C', 'C', 'C'): 109.5,  # sp3 tetrahedral
    ('C', 'C', 'H'): 109.5,
    ('C', 'C', 'N'): 109.5,
    ('C', 'C', 'O'): 109.5,
    ('H', 'C', 'H'): 109.5,
    ('C', 'N', 'C'): 109.5,
    ('C', 'N', 'H'): 109.5,
    ('C', 'O', 'C'): 109.5,
    ('C', 'O', 'H'): 109.5,
    ('H', 'N', 'H'): 109.5,
    ('O', 'P', 'O'): 109.5,
    ('C', 'P', 'O'): 109.5,
    # Aromatic and sp2 angles
    ('C', 'C', 'C', 'aromatic'): 120.0,
    ('C', 'N', 'C', 'aromatic'): 120.0,
    # Common nucleic acid angles
    ('P', 'O', 'C'): 109.5,
    ('O', 'C', 'C'): 109.5,
    ('C', 'C', 'O'): 109.5,
    ('O', 'C', 'O'): 109.5,
}

# Residue-specific bond patterns for common residues
RESIDUE_BOND_PATTERNS = {
    # Nucleic acids
    'A': {  # Adenine
        'backbone': [('P', "O5'"), ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"), ("C3'", "O3'"), ("O3'", 'P')],
        'base': [("C4'", "O4'"), ("O4'", "C1'"), ("C1'", "C2'"), ("C2'", "C3'"), ("C1'", 'N9'), ('N9', 'C8'), ('C8', 'N7'), ('N7', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C4', 'N9'), ('C6', 'N6')]
    },
    'G': {  # Guanine
        'backbone': [('P', "O5'"), ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"), ("C3'", "O3'"), ("O3'", 'P')],
        'base': [("C4'", "O4'"), ("O4'", "C1'"), ("C1'", "C2'"), ("C2'", "C3'"), ("C1'", 'N9'), ('N9', 'C8'), ('C8', 'N7'), ('N7', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C4', 'N9'), ('C6', 'O6'), ('C2', 'N2')]
    },
    'C': {  # Cytosine
        'backbone': [('P', "O5'"), ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"), ("C3'", "O3'"), ("O3'", 'P')],
        'base': [("C4'", "O4'"), ("O4'", "C1'"), ("C1'", "C2'"), ("C2'", "C3'"), ("C1'", 'N1'), ('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C4', 'N4'), ('C2', 'O2')]
    },
    'U': {  # Uracil
        'backbone': [('P', "O5'"), ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"), ("C3'", "O3'"), ("O3'", 'P')],
        'base': [("C4'", "O4'"), ("O4'", "C1'"), ("C1'", "C2'"), ("C2'", "C3'"), ("C1'", 'N1'), ('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C4', 'O4'), ('C2', 'O2')]
    },
    'T': {  # Thymine
        'backbone': [('P', "O5'"), ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"), ("C3'", "O3'"), ("O3'", 'P')],
        'base': [("C4'", "O4'"), ("O4'", "C1'"), ("C1'", "C2'"), ("C2'", "C3'"), ("C1'", 'N1'), ('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C4', 'O4'), ('C2', 'O2'), ('C5', 'C7')]
    },
    # Common amino acids
    'ALA': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB')]
    },
    'GLY': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': []
    },
    'SER': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'OG')]
    },
    'THR': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'OG1'), ('CB', 'CG2')]
    },
    'CYS': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'SG')]
    },
    'MET': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE')]
    },
    'ASP': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2')]
    },
    'GLU': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2')]
    },
    'LYS': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ')]
    },
    'ARG': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2')]
    },
    'HIS': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('ND1', 'CE1'), ('CE1', 'NE2'), ('NE2', 'CD2'), ('CD2', 'CG')]
    },
    'PHE': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CD1', 'CE1'), ('CE1', 'CZ'), ('CZ', 'CE2'), ('CE2', 'CD2'), ('CD2', 'CG')]
    },
    'TYR': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CD1', 'CE1'), ('CE1', 'CZ'), ('CZ', 'CE2'), ('CE2', 'CD2'), ('CD2', 'CG'), ('CZ', 'OH')]
    },
    'TRP': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CD1', 'NE1'), ('NE1', 'CE2'), ('CE2', 'CD2'), ('CD2', 'CG'), ('CE2', 'CZ2'), ('CZ2', 'CH2'), ('CH2', 'CZ3'), ('CZ3', 'CE3'), ('CE3', 'CD2')]
    },
    'PRO': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'N')]
    },
    'VAL': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2')]
    },
    'LEU': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2')]
    },
    'ILE': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG1'), ('CG1', 'CD1'), ('CB', 'CG2')]
    },
    'ASN': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2')]
    },
    'GLN': {
        'backbone': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'N')],
        'sidechain': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'NE2')]
    },
}


class MolecularTopology:
    """Detect and manage molecular topology for structure refinement using PDB connectivity"""
    
    def __init__(self, system_data: Dict, coordinates: np.ndarray):
        import time
        
        self.system_data = system_data
        self.coordinates = coordinates
        self.nat = len(coordinates)
        self.symbols = system_data["symbols"]
        self.atoms = system_data.get("atoms", [])
        
        # Initialize topology containers
        self.bonds = []  # List of (i, j, bond_type, target_length)
        self.residues = defaultdict(list)  # residue_id -> [atom_indices]
        self.residue_bonds = defaultdict(list)  # residue_id -> [bonds]
        
        # Build topology using PDB connectivity
        start_time = time.time()
        print(f"#   Building residue map for {self.nat} atoms...")
        self._build_residue_map()
        print(f"#   Identified {len(self.residues)} residues")
        
        print("#   Extracting connectivity from PDB...")
        self._extract_pdb_connectivity()
        
        print("#   Classifying bond types...")
        self._classify_bond_types()
        
        end_time = time.time()
        print(f"#   Topology detection completed in {end_time - start_time:.2f}s")
        
    def _build_residue_map(self):
        """Build mapping of residues to atoms"""
        for i, atom in enumerate(self.atoms):
            if atom:
                resid = atom.get("resid", 1)
                chain = atom.get("chain", "A")
                residue_key = f"{chain}_{resid}"
                self.residues[residue_key].append(i)
            else:
                # Fallback for atoms without residue info
                self.residues["unknown_1"].append(i)
    
    def _get_max_bond_length(self, symbol1: str, symbol2: str) -> float:
        """Get maximum bond length for two elements"""
        key1 = tuple(sorted([symbol1, symbol2]))
        key2 = tuple(sorted([symbol2, symbol1]))
        
        return MAX_BOND_LENGTHS.get(key1, MAX_BOND_LENGTHS.get(key2, 2.5))
    
    def _get_standard_bond_length(self, symbol1: str, symbol2: str) -> float:
        """Get standard bond length for two elements"""
        key1 = tuple(sorted([symbol1, symbol2]))
        key2 = tuple(sorted([symbol2, symbol1]))
        
        return STANDARD_BOND_LENGTHS.get(key1, STANDARD_BOND_LENGTHS.get(key2, 1.5))
    
    def _extract_pdb_connectivity(self):
        """Extract connectivity information from PDB data"""
        if "pdb_data" not in self.system_data:
            print("#   Warning: No PDB data found, using fallback distance-based detection")
            self._fallback_distance_detection()
            return
        
        pdb_data = self.system_data["pdb_data"]
        
        # Extract connectivity from PDB CONECT records if available
        if "connectivity" in pdb_data and pdb_data["connectivity"]:
            print(f"#   Found connectivity information in PDB")
            self._process_pdb_connectivity(pdb_data["connectivity"])
        else:
            print("#   No CONECT records found, using intra-residue distance detection")
            self._detect_intra_residue_bonds()
    
    def _process_pdb_connectivity(self, connectivity):
        """Process PDB CONECT records to extract bonds"""
        bond_count = 0
        
        for atom_idx, connected_atoms in enumerate(connectivity):
            if connected_atoms:
                for connected_idx in connected_atoms:
                    if connected_idx > atom_idx:  # Avoid duplicate bonds
                        # Calculate current distance
                        dist = np.linalg.norm(
                            self.coordinates[atom_idx] - self.coordinates[connected_idx]
                        )
                        
                        # Add bond with current distance as target
                        bond_info = (atom_idx, connected_idx, "pdb", dist)
                        self.bonds.append(bond_info)
                        bond_count += 1
                        
                        # Add to residue bonds
                        residue_i = self._get_atom_residue(atom_idx)
                        residue_j = self._get_atom_residue(connected_idx)
                        
                        if residue_i == residue_j:
                            self.residue_bonds[residue_i].append(bond_info)
        
        print(f"#   Extracted {bond_count} bonds from PDB connectivity")
    
    def _detect_intra_residue_bonds(self):
        """Detect bonds within each residue using distance criteria"""
        bond_count = 0
        
        for residue_key, atom_indices in self.residues.items():
            residue_bonds = 0
            
            for i in range(len(atom_indices)):
                for j in range(i + 1, len(atom_indices)):
                    idx_i = atom_indices[i]
                    idx_j = atom_indices[j]
                    
                    symbol1 = self.symbols[idx_i]
                    symbol2 = self.symbols[idx_j]
                    
                    # Calculate distance
                    dist = np.linalg.norm(self.coordinates[idx_i] - self.coordinates[idx_j])
                    max_dist = self._get_max_bond_length(symbol1, symbol2)
                    
                    if dist <= max_dist:
                        bond_info = (idx_i, idx_j, "intra_residue", dist)
                        self.bonds.append(bond_info)
                        self.residue_bonds[residue_key].append(bond_info)
                        bond_count += 1
                        residue_bonds += 1
            
            if residue_bonds > 0:
                print(f"#     Residue {residue_key}: {residue_bonds} bonds")
        
        print(f"#   Found {bond_count} intra-residue bonds")
    
    def _fallback_distance_detection(self):
        """Fallback distance-based detection when no PDB data is available"""
        self._detect_intra_residue_bonds()
    
    def _detect_distance_bonds(self):
        """Detect bonds using distance criteria"""
        total_pairs = self.nat * (self.nat - 1) // 2
        processed = 0
        
        for i in range(self.nat):
            for j in range(i + 1, self.nat):
                processed += 1
                
                # Progress indicator for large systems
                if self.nat > 100 and processed % 10000 == 0:
                    print(f"#     Processed {processed}/{total_pairs} pairs ({processed/total_pairs*100:.1f}%)")
                
                # Original bond detection logic continues here...
                symbol1 = self.symbols[i]
                symbol2 = self.symbols[j]
                
                # Calculate distance
                dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                max_dist = self._get_max_bond_length(symbol1, symbol2)
                
                if dist <= max_dist:
                    # This is likely a bond
                    standard_length = self._get_standard_bond_length(symbol1, symbol2)
                    bond_type = "single"  # Default to single bond
                    
                    # Determine if this is intra or inter-residue
                    residue_i = self._get_atom_residue(i)
                    residue_j = self._get_atom_residue(j)
                    
                    bond_info = (i, j, bond_type, standard_length)
                    self.bonds.append(bond_info)
                    
                    if residue_i == residue_j:
                        # Intra-residue bond
                        self.residue_bonds[residue_i].append(bond_info)
                    else:
                        # Inter-residue bond
                        self.inter_residue_bonds.append(bond_info)
    
    def _detect_residue_bonds(self):
        """Detect bonds using residue-specific patterns"""
        for residue_key, atom_indices in self.residues.items():
            if len(atom_indices) < 2:
                continue
            
            # Get residue type
            resname = self._get_residue_name(residue_key)
            
            if resname in RESIDUE_BOND_PATTERNS:
                pattern = RESIDUE_BOND_PATTERNS[resname]
                
                # Create atom name to index mapping for this residue
                atom_name_to_idx = {}
                for idx in atom_indices:
                    if idx < len(self.atoms) and self.atoms[idx]:
                        atom_name = self.atoms[idx].get("name", "")
                        atom_name_to_idx[atom_name] = idx
                
                # Add bonds from patterns
                for bond_category, bond_list in pattern.items():
                    for atom1_name, atom2_name in bond_list:
                        if atom1_name in atom_name_to_idx and atom2_name in atom_name_to_idx:
                            i = atom_name_to_idx[atom1_name]
                            j = atom_name_to_idx[atom2_name]
                            
                            # Check if this bond already exists
                            existing_bond = any(
                                (bond[0] == i and bond[1] == j) or (bond[0] == j and bond[1] == i)
                                for bond in self.bonds
                            )
                            
                            if not existing_bond:
                                symbol1 = self.symbols[i]
                                symbol2 = self.symbols[j]
                                standard_length = self._get_standard_bond_length(symbol1, symbol2)
                                bond_type = "single"
                                
                                bond_info = (i, j, bond_type, standard_length)
                                self.bonds.append(bond_info)
                                self.residue_bonds[residue_key].append(bond_info)
    
    def _detect_inter_residue_bonds(self):
        """Detect bonds between residues (e.g., peptide bonds, phosphodiester bonds)"""
        residue_keys = list(self.residues.keys())
        
        for i, res_key1 in enumerate(residue_keys):
            for j, res_key2 in enumerate(residue_keys[i+1:], i+1):
                # Check for bonds between these residues
                atoms1 = self.residues[res_key1]
                atoms2 = self.residues[res_key2]
                
                # Look for close contacts that might be bonds
                for atom1_idx in atoms1:
                    for atom2_idx in atoms2:
                        dist = np.linalg.norm(self.coordinates[atom1_idx] - self.coordinates[atom2_idx])
                        symbol1 = self.symbols[atom1_idx]
                        symbol2 = self.symbols[atom2_idx]
                        max_dist = self._get_max_bond_length(symbol1, symbol2)
                        
                        if dist <= max_dist:
                            # Check if this is a known inter-residue bond pattern
                            if self._is_inter_residue_bond(atom1_idx, atom2_idx, res_key1, res_key2):
                                # Check if this bond already exists
                                existing_bond = any(
                                    (bond[0] == atom1_idx and bond[1] == atom2_idx) or 
                                    (bond[0] == atom2_idx and bond[1] == atom1_idx)
                                    for bond in self.bonds
                                )
                                
                                if not existing_bond:
                                    standard_length = self._get_standard_bond_length(symbol1, symbol2)
                                    bond_type = "inter_residue"
                                    bond_info = (atom1_idx, atom2_idx, bond_type, standard_length)
                                    self.bonds.append(bond_info)
                                    self.inter_residue_bonds.append(bond_info)
    
    def _is_inter_residue_bond(self, atom1_idx: int, atom2_idx: int, res1_key: str, res2_key: str) -> bool:
        """Check if two atoms form a valid inter-residue bond"""
        if atom1_idx >= len(self.atoms) or atom2_idx >= len(self.atoms):
            return False
        
        if not self.atoms[atom1_idx] or not self.atoms[atom2_idx]:
            return False
        
        atom1_name = self.atoms[atom1_idx].get("name", "")
        atom2_name = self.atoms[atom2_idx].get("name", "")
        
        # Common inter-residue bond patterns
        inter_residue_patterns = [
            # Peptide bonds
            ("C", "N"),
            # Phosphodiester bonds in nucleic acids
            ("O3'", "P"),
            ("P", "O5'"),
            # Disulfide bonds
            ("SG", "SG"),
            # Hydrogen bonds (if very close)
            ("O", "H"),
            ("N", "H"),
        ]
        
        for name1, name2 in inter_residue_patterns:
            if (atom1_name == name1 and atom2_name == name2) or \
               (atom1_name == name2 and atom2_name == name1):
                return True
        
        return False
    
    def _detect_angles(self):
        """Detect bond angles from the bond topology"""
        print("# Detecting bond angles...")
        
        # Create adjacency list from bonds
        adjacency = defaultdict(list)
        for i, j, bond_type, length in self.bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        # Count potential angles
        potential_angles = 0
        for center_atom in range(self.nat):
            neighbors = adjacency[center_atom]
            potential_angles += len(neighbors) * (len(neighbors) - 1) // 2
        
        print(f"#   Analyzing {potential_angles} potential angle configurations...")
        
        # Find all angles (atom-atom-atom triplets)
        angles_found = 0
        for center_atom in range(self.nat):
            neighbors = adjacency[center_atom]
            
            # Progress indicator for large systems
            if self.nat > 100 and center_atom % 100 == 0:
                print(f"#     Processed {center_atom}/{self.nat} atoms ({center_atom/self.nat*100:.1f}%)")
            
            # Generate all pairs of neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    angles_found += 1
                    atom1 = neighbors[i]
                    atom3 = neighbors[j]
                    
                    # Create angle triplet (atom1-center_atom-atom3)
                    symbol1 = self.symbols[atom1]
                    symbol_center = self.symbols[center_atom]
                    symbol3 = self.symbols[atom3]
                    
                    angle_type = f"{symbol1}-{symbol_center}-{symbol3}"
                    target_angle = self._get_standard_angle(symbol1, symbol_center, symbol3)
                    
                    self.angles.append((atom1, center_atom, atom3, angle_type, target_angle))
        
        print(f"#   Found {len(self.angles)} bond angles")
    
    def _get_standard_angle(self, symbol1: str, symbol_center: str, symbol3: str) -> float:
        """Get standard bond angle for three atoms"""
        key1 = (symbol1, symbol_center, symbol3)
        key2 = (symbol3, symbol_center, symbol1)
        
        angle = STANDARD_BOND_ANGLES.get(key1, STANDARD_BOND_ANGLES.get(key2, 109.5))
        return np.radians(angle)  # Convert to radians
    
    def _classify_bond_types(self):
        """Classify bonds as backbone, sidechain, or inter-residue"""
        backbone_count = 0
        sidechain_count = 0
        inter_residue_count = 0
        
        for i, (atom1, atom2, bond_type, length) in enumerate(self.bonds):
            residue1 = self._get_atom_residue(atom1)
            residue2 = self._get_atom_residue(atom2)
            
            if residue1 == residue2:
                # Intra-residue bond
                if self._is_backbone_bond(atom1, atom2):
                    self.bonds[i] = (atom1, atom2, "backbone", length)
                    backbone_count += 1
                else:
                    self.bonds[i] = (atom1, atom2, "sidechain", length)
                    sidechain_count += 1
            else:
                # Inter-residue bond
                self.bonds[i] = (atom1, atom2, "inter_residue", length)
                inter_residue_count += 1
        
        print(f"#   Bond classification: {backbone_count} backbone, {sidechain_count} sidechain, {inter_residue_count} inter-residue")
    
    def _is_backbone_bond(self, atom1_idx: int, atom2_idx: int) -> bool:
        """Check if a bond is part of the backbone"""
        if atom1_idx >= len(self.atoms) or atom2_idx >= len(self.atoms):
            return False
        
        if not self.atoms[atom1_idx] or not self.atoms[atom2_idx]:
            return False
        
        atom1_name = self.atoms[atom1_idx].get("name", "")
        atom2_name = self.atoms[atom2_idx].get("name", "")
        
        # Common backbone atoms
        backbone_atoms = {
            # Protein backbone
            "N", "CA", "C", "O",
            # Nucleic acid backbone
            "P", "O1P", "O2P", "O5'", "C5'", "C4'", "C3'", "C2'", "C1'", "O4'", "O3'", "O2'"
        }
        
        return atom1_name in backbone_atoms and atom2_name in backbone_atoms
    
    def _get_atom_residue(self, atom_idx: int) -> str:
        """Get the residue key for an atom"""
        for residue_key, atom_indices in self.residues.items():
            if atom_idx in atom_indices:
                return residue_key
        return "unknown_1"
    
    def _get_residue_name(self, residue_key: str) -> str:
        """Get the residue name from residue key"""
        if residue_key in self.residues and self.residues[residue_key]:
            first_atom_idx = self.residues[residue_key][0]
            if first_atom_idx < len(self.atoms) and self.atoms[first_atom_idx]:
                return self.atoms[first_atom_idx].get("resname", "UNK")
        return "UNK"
    
    def get_bonds_by_type(self, bond_type: str) -> List[Tuple[int, int, str, float]]:
        """Get all bonds of a specific type"""
        return [bond for bond in self.bonds if bond[2] == bond_type]
    
    def get_critical_bonds(self) -> List[Tuple[int, int, str, float]]:
        """Get bonds that are critical for maintaining covalent structure"""
        critical_bonds = []
        
        # Backbone bonds are always critical
        critical_bonds.extend(self.get_bonds_by_type("backbone"))
        
        # Inter-residue bonds are critical
        critical_bonds.extend(self.get_bonds_by_type("inter_residue"))
        
        # Some sidechain bonds may be critical (disulfide bonds, etc.)
        for bond in self.get_bonds_by_type("sidechain"):
            atom1, atom2, bond_type, length = bond
            if self.symbols[atom1] == 'S' and self.symbols[atom2] == 'S':
                critical_bonds.append(bond)
        
        return critical_bonds
    
    def get_clash_safe_bonds(self) -> List[Tuple[int, int, str, float]]:
        """Get bonds that should be checked for clashes"""
        # All bonds should be checked, but inter-residue bonds are most important
        return self.bonds
    
    def get_bond_statistics(self) -> Dict[str, int]:
        """Get statistics about detected bonds"""
        stats = defaultdict(int)
        
        for bond in self.bonds:
            bond_type = bond[2]
            stats[bond_type] += 1
        
        stats["total"] = len(self.bonds)
        stats["residues"] = len(self.residues)
        # Remove angle statistics since we don't use angles anymore
        
        return dict(stats)
    
    def print_topology_summary(self):
        """Print a summary of the detected topology"""
        stats = self.get_bond_statistics()
        
        print("# Molecular Topology Summary")
        print("#" + "=" * 50)
        print(f"# Total atoms: {self.nat}")
        print(f"# Total residues: {stats['residues']}")
        print(f"# Total bonds: {stats['total']}")
        print(f"#   - Backbone bonds: {stats.get('backbone', 0)}")
        print(f"#   - Sidechain bonds: {stats.get('sidechain', 0)}")
        print(f"#   - Inter-residue bonds: {stats.get('inter_residue', 0)}")
        # Remove angle information since we don't use angles anymore
        
        # Print residue breakdown
        print("\n# Residue breakdown:")
        for residue_key in sorted(self.residues.keys()):
            resname = self._get_residue_name(residue_key)
            atom_count = len(self.residues[residue_key])
            bond_count = len(self.residue_bonds.get(residue_key, []))
            print(f"#   {residue_key} ({resname}): {atom_count} atoms, {bond_count} bonds")
        
        print("#" + "=" * 50)


def detect_molecular_topology(system_data: Dict, coordinates: np.ndarray) -> MolecularTopology:
    """
    Detect molecular topology from PDB connectivity and coordinates
    
    Args:
        system_data: Dictionary containing system information with PDB data
        coordinates: Array of atomic coordinates
        
    Returns:
        MolecularTopology object with detected bonds (no angles)
    """
    return MolecularTopology(system_data, coordinates)


def calculate_bond_restraint_forces(coordinates: np.ndarray, topology: MolecularTopology,
                                   force_constant: float = 1000.0,
                                   bond_types: List[str] = None) -> Tuple[float, np.ndarray]:
    """
    Calculate forces to maintain bond lengths
    
    Args:
        coordinates: Current atomic coordinates
        topology: Molecular topology object
        force_constant: Force constant for bond restraints
        bond_types: List of bond types to restrain (default: all critical bonds)
        
    Returns:
        Tuple of (energy, forces)
    """
    if bond_types is None:
        bonds = topology.get_critical_bonds()
    else:
        bonds = []
        for bond_type in bond_types:
            bonds.extend(topology.get_bonds_by_type(bond_type))
    
    energy = 0.0
    forces = np.zeros_like(coordinates)
    
    for atom1, atom2, bond_type, target_length in bonds:
        # Calculate current bond vector and length
        bond_vector = coordinates[atom2] - coordinates[atom1]
        current_length = np.linalg.norm(bond_vector)
        
        # Avoid division by zero
        if current_length < 1e-8:
            continue
        
        # Calculate deviation from target length
        deviation = current_length - target_length
        
        # Harmonic restraint energy
        bond_energy = 0.5 * force_constant * deviation**2
        energy += bond_energy
        
        # Force magnitude (negative gradient)
        force_magnitude = -force_constant * deviation
        
        # Unit vector along bond
        unit_vector = bond_vector / current_length
        
        # Apply forces
        force_vector = force_magnitude * unit_vector
        forces[atom1] -= force_vector
        forces[atom2] += force_vector
    
    return energy, forces


def calculate_angle_restraint_forces(coordinates: np.ndarray, topology: MolecularTopology,
                                   force_constant: float = 100.0) -> Tuple[float, np.ndarray]:
    """
    Calculate forces to maintain bond angles
    
    Args:
        coordinates: Current atomic coordinates
        topology: Molecular topology object
        force_constant: Force constant for angle restraints
        
    Returns:
        Tuple of (energy, forces)
    """
    energy = 0.0
    forces = np.zeros_like(coordinates)
    
    for atom1, atom2, atom3, angle_type, target_angle in topology.angles:
        # Calculate vectors
        vec1 = coordinates[atom1] - coordinates[atom2]  # atom2 -> atom1
        vec2 = coordinates[atom3] - coordinates[atom2]  # atom2 -> atom3
        
        # Calculate lengths
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if len1 < 1e-8 or len2 < 1e-8:
            continue
        
        # Normalize vectors
        unit1 = vec1 / len1
        unit2 = vec2 / len2
        
        # Calculate current angle
        cos_angle = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
        current_angle = np.arccos(cos_angle)
        
        # Calculate deviation
        deviation = current_angle - target_angle
        
        # Harmonic restraint energy
        angle_energy = 0.5 * force_constant * deviation**2
        energy += angle_energy
        
        # Calculate force (negative gradient)
        if abs(np.sin(current_angle)) < 1e-8:
            continue  # Avoid singularity at 0 or 180 degrees
        
        force_magnitude = -force_constant * deviation / np.sin(current_angle)
        
        # Force directions
        force_dir1 = (unit2 - cos_angle * unit1) / len1
        force_dir2 = (unit1 - cos_angle * unit2) / len2
        
        # Apply forces
        forces[atom1] += force_magnitude * force_dir1
        forces[atom3] += force_magnitude * force_dir2
        forces[atom2] -= force_magnitude * (force_dir1 + force_dir2)
    
    return energy, forces


def check_covalent_integrity(coordinates: np.ndarray, topology: MolecularTopology,
                           max_deviation: float = 0.5) -> Dict[str, any]:
    """
    Check if covalent bonds are maintained within acceptable limits
    
    Args:
        coordinates: Current atomic coordinates
        topology: Molecular topology object
        max_deviation: Maximum allowed deviation from target bond length (in Angstroms)
        
    Returns:
        Dictionary with integrity check results
    """
    results = {
        "intact": True,
        "broken_bonds": [],
        "stretched_bonds": [],
        "compressed_bonds": [],
        "max_deviation": 0.0,
        "total_bonds": len(topology.bonds)
    }
    
    critical_bonds = topology.get_critical_bonds()
    
    for atom1, atom2, bond_type, target_length in critical_bonds:
        # Calculate current bond length
        bond_vector = coordinates[atom2] - coordinates[atom1]
        current_length = np.linalg.norm(bond_vector)
        
        # Calculate deviation
        deviation = abs(current_length - target_length)
        results["max_deviation"] = max(results["max_deviation"], deviation)
        
        # Check if bond is broken
        if deviation > max_deviation:
            results["intact"] = False
            
            bond_info = {
                "atoms": (atom1, atom2),
                "type": bond_type,
                "current_length": current_length,
                "target_length": target_length,
                "deviation": current_length - target_length,
                "symbols": (topology.symbols[atom1], topology.symbols[atom2])
            }
            
            if current_length > target_length:
                results["stretched_bonds"].append(bond_info)
            else:
                results["compressed_bonds"].append(bond_info)
            
            if deviation > 2 * max_deviation:
                results["broken_bonds"].append(bond_info)
    
    return results