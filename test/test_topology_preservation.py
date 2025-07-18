"""
Unit tests for topology preservation functionality in structure refinement
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import jax.numpy as jnp

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fennol.md.topology import (
    MolecularTopology,
    detect_molecular_topology,
    calculate_bond_restraint_forces,
    calculate_angle_restraint_forces,
    check_covalent_integrity,
    STANDARD_BOND_LENGTHS,
    MAX_BOND_LENGTHS,
    STANDARD_BOND_ANGLES
)


class TestMolecularTopology:
    """Test molecular topology detection and management"""
    
    @pytest.fixture
    def simple_molecule_data(self):
        """Create simple molecule data for testing"""
        # Simple methane molecule
        system_data = {
            "nat": 5,
            "symbols": ["C", "H", "H", "H", "H"],
            "atoms": [
                {"name": "C1", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "H1", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "H2", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "H3", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "H4", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
            ]
        }
        
        # Tetrahedral arrangement of hydrogens around carbon
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # C
            [1.09, 0.0, 0.0],     # H1
            [-0.36, 1.03, 0.0],   # H2
            [-0.36, -0.51, 0.89], # H3
            [-0.36, -0.51, -0.89] # H4
        ])
        
        return system_data, coordinates
    
    @pytest.fixture
    def nucleic_acid_data(self):
        """Create nucleic acid residue data for testing"""
        system_data = {
            "nat": 10,
            "symbols": ["P", "O", "O", "O", "C", "C", "C", "C", "O", "N"],
            "atoms": [
                {"name": "P", "resname": "G", "resid": 1, "chain": "A", "element": "P"},
                {"name": "O1P", "resname": "G", "resid": 1, "chain": "A", "element": "O"},
                {"name": "O2P", "resname": "G", "resid": 1, "chain": "A", "element": "O"},
                {"name": "O5'", "resname": "G", "resid": 1, "chain": "A", "element": "O"},
                {"name": "C5'", "resname": "G", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C4'", "resname": "G", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C3'", "resname": "G", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C2'", "resname": "G", "resid": 1, "chain": "A", "element": "C"},
                {"name": "O3'", "resname": "G", "resid": 1, "chain": "A", "element": "O"},
                {"name": "N9", "resname": "G", "resid": 1, "chain": "A", "element": "N"},
            ]
        }
        
        # Rough nucleotide backbone coordinates
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # P
            [1.0, 1.0, 0.0],      # O1P
            [-1.0, 1.0, 0.0],     # O2P
            [0.0, -1.5, 0.0],     # O5'
            [0.0, -3.0, 0.0],     # C5'
            [1.5, -3.0, 0.0],     # C4'
            [1.5, -1.5, 0.0],     # C3'
            [0.0, -1.5, 1.5],     # C2'
            [3.0, -1.5, 0.0],     # O3'
            [1.5, -4.5, 0.0],     # N9
        ])
        
        return system_data, coordinates
    
    def test_topology_initialization(self, simple_molecule_data):
        """Test topology initialization"""
        system_data, coordinates = simple_molecule_data
        
        topology = MolecularTopology(system_data, coordinates)
        
        assert topology.nat == 5
        assert len(topology.symbols) == 5
        assert len(topology.residues) == 1
        assert "A_1" in topology.residues
        assert len(topology.residues["A_1"]) == 5
    
    def test_bond_detection_simple(self, simple_molecule_data):
        """Test bond detection for simple molecule"""
        system_data, coordinates = simple_molecule_data
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Should detect 4 C-H bonds
        assert len(topology.bonds) == 4
        
        # All bonds should be between C (index 0) and H atoms (indices 1-4)
        for bond in topology.bonds:
            atom1, atom2, bond_type, length = bond
            assert (atom1 == 0 and atom2 in [1, 2, 3, 4]) or (atom2 == 0 and atom1 in [1, 2, 3, 4])
            assert bond_type == "sidechain"  # Within residue, not backbone
            assert abs(length - 1.09) < 0.01  # C-H bond length
    
    def test_bond_detection_nucleic_acid(self, nucleic_acid_data):
        """Test bond detection for nucleic acid"""
        system_data, coordinates = nucleic_acid_data
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Should detect multiple bonds
        assert len(topology.bonds) > 0
        
        # Check for some expected backbone bonds
        backbone_bonds = topology.get_bonds_by_type("backbone")
        assert len(backbone_bonds) > 0
        
        # Check bond statistics
        stats = topology.get_bond_statistics()
        assert stats["total"] > 0
        assert stats["residues"] == 1
    
    def test_residue_mapping(self, simple_molecule_data):
        """Test residue to atom mapping"""
        system_data, coordinates = simple_molecule_data
        
        topology = MolecularTopology(system_data, coordinates)
        
        assert len(topology.residues) == 1
        assert "A_1" in topology.residues
        assert topology.residues["A_1"] == [0, 1, 2, 3, 4]
    
    def test_angle_detection(self, simple_molecule_data):
        """Test bond angle detection"""
        system_data, coordinates = simple_molecule_data
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Should detect H-C-H angles
        assert len(topology.angles) > 0
        
        # All angles should have carbon as center atom
        for angle in topology.angles:
            atom1, center, atom3, angle_type, target = angle
            assert center == 0  # Carbon is center
            assert atom1 in [1, 2, 3, 4]  # Hydrogens
            assert atom3 in [1, 2, 3, 4]
            assert atom1 != atom3
            assert abs(target - np.radians(109.5)) < 0.01  # Tetrahedral angle
    
    def test_critical_bonds(self, nucleic_acid_data):
        """Test critical bond identification"""
        system_data, coordinates = nucleic_acid_data
        
        topology = MolecularTopology(system_data, coordinates)
        
        critical_bonds = topology.get_critical_bonds()
        
        # Should have backbone and inter-residue bonds as critical
        assert len(critical_bonds) > 0
        
        # Check that backbone bonds are marked as critical
        backbone_bonds = topology.get_bonds_by_type("backbone")
        for bond in backbone_bonds:
            assert bond in critical_bonds
    
    def test_inter_residue_bonds(self):
        """Test inter-residue bond detection"""
        # Create system with two residues
        system_data = {
            "nat": 6,
            "symbols": ["C", "N", "C", "C", "N", "C"],
            "atoms": [
                {"name": "C", "resname": "ALA", "resid": 1, "chain": "A", "element": "C"},
                {"name": "N", "resname": "ALA", "resid": 2, "chain": "A", "element": "N"},
                {"name": "CA", "resname": "ALA", "resid": 1, "chain": "A", "element": "C"},
                {"name": "CA", "resname": "ALA", "resid": 2, "chain": "A", "element": "C"},
                {"name": "N", "resname": "ALA", "resid": 1, "chain": "A", "element": "N"},
                {"name": "C", "resname": "ALA", "resid": 2, "chain": "A", "element": "C"},
            ]
        }
        
        # Place atoms to form peptide bond
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # C (res 1)
            [1.3, 0.0, 0.0],      # N (res 2) - peptide bond
            [-1.5, 0.0, 0.0],     # CA (res 1)
            [2.8, 0.0, 0.0],      # CA (res 2)
            [-3.0, 0.0, 0.0],     # N (res 1)
            [4.0, 0.0, 0.0],      # C (res 2)
        ])
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Should detect inter-residue bonds
        assert len(topology.inter_residue_bonds) > 0
        
        # Check that peptide bond is detected
        peptide_bond_found = False
        for bond in topology.inter_residue_bonds:
            atom1, atom2, bond_type, length = bond
            if ((atom1 == 0 and atom2 == 1) or (atom1 == 1 and atom2 == 0)):
                peptide_bond_found = True
                assert bond_type == "inter_residue"
                break
        
        assert peptide_bond_found, "Peptide bond not detected"


class TestBondRestraints:
    """Test bond length restraint calculations"""
    
    @pytest.fixture
    def simple_system(self):
        """Create simple system for testing"""
        system_data = {
            "nat": 3,
            "symbols": ["C", "C", "H"],
            "atoms": [
                {"name": "C1", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C2", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "H1", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
            ]
        }
        
        coordinates = np.array([
            [0.0, 0.0, 0.0],   # C1
            [1.54, 0.0, 0.0],  # C2 (standard C-C bond length)
            [2.63, 0.0, 0.0],  # H1 (standard C-H bond length from C2)
        ])
        
        return system_data, coordinates
    
    def test_bond_restraint_forces_equilibrium(self, simple_system):
        """Test bond restraint forces at equilibrium"""
        system_data, coordinates = simple_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        # At equilibrium, forces should be near zero
        energy, forces = calculate_bond_restraint_forces(coordinates, topology, 1000.0)
        
        assert energy < 0.01  # Very small energy
        assert np.max(np.abs(forces)) < 0.1  # Very small forces
    
    def test_bond_restraint_forces_stretched(self, simple_system):
        """Test bond restraint forces for stretched bond"""
        system_data, coordinates = simple_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Stretch the C-C bond
        stretched_coords = coordinates.copy()
        stretched_coords[1, 0] = 2.0  # Stretch from 1.54 to 2.0
        
        energy, forces = calculate_bond_restraint_forces(stretched_coords, topology, 1000.0)
        
        assert energy > 0  # Positive energy for stretched bond
        assert forces[0, 0] > 0  # Force on first atom in +x direction
        assert forces[1, 0] < 0  # Force on second atom in -x direction
    
    def test_bond_restraint_forces_compressed(self, simple_system):
        """Test bond restraint forces for compressed bond"""
        system_data, coordinates = simple_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Compress the C-C bond
        compressed_coords = coordinates.copy()
        compressed_coords[1, 0] = 1.0  # Compress from 1.54 to 1.0
        
        energy, forces = calculate_bond_restraint_forces(compressed_coords, topology, 1000.0)
        
        assert energy > 0  # Positive energy for compressed bond
        assert forces[0, 0] < 0  # Force on first atom in -x direction
        assert forces[1, 0] > 0  # Force on second atom in +x direction


class TestAngleRestraints:
    """Test bond angle restraint calculations"""
    
    @pytest.fixture
    def angle_system(self):
        """Create system with defined angle for testing"""
        system_data = {
            "nat": 3,
            "symbols": ["C", "C", "C"],
            "atoms": [
                {"name": "C1", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C2", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C3", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
            ]
        }
        
        # Create 120-degree angle
        coordinates = np.array([
            [0.0, 0.0, 0.0],      # C1
            [1.0, 0.0, 0.0],      # C2 (center)
            [1.5, 0.866, 0.0],    # C3 (120 degrees from C1-C2)
        ])
        
        return system_data, coordinates
    
    def test_angle_restraint_forces_equilibrium(self, angle_system):
        """Test angle restraint forces at equilibrium"""
        system_data, coordinates = angle_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Should have one angle detected
        assert len(topology.angles) == 1
        
        # At equilibrium, forces should be small
        energy, forces = calculate_angle_restraint_forces(coordinates, topology, 100.0)
        
        # Energy should be small (not exactly zero due to target angle difference)
        assert energy < 10.0
        assert np.max(np.abs(forces)) < 10.0
    
    def test_angle_restraint_forces_bent(self, angle_system):
        """Test angle restraint forces for bent angle"""
        system_data, coordinates = angle_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Bend the angle by moving C3
        bent_coords = coordinates.copy()
        bent_coords[2] = [1.0, 1.0, 0.0]  # 90-degree angle instead of 120
        
        energy, forces = calculate_angle_restraint_forces(bent_coords, topology, 100.0)
        
        assert energy > 0  # Positive energy for bent angle
        assert np.max(np.abs(forces)) > 0  # Non-zero forces to restore angle


class TestCovalentIntegrity:
    """Test covalent integrity checking"""
    
    @pytest.fixture
    def integrity_system(self):
        """Create system for integrity testing"""
        system_data = {
            "nat": 4,
            "symbols": ["C", "C", "C", "C"],
            "atoms": [
                {"name": "C1", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C2", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C3", "resname": "MOL", "resid": 2, "chain": "A", "element": "C"},
                {"name": "C4", "resname": "MOL", "resid": 2, "chain": "A", "element": "C"},
            ]
        }
        
        coordinates = np.array([
            [0.0, 0.0, 0.0],   # C1
            [1.54, 0.0, 0.0],  # C2 (good C-C bond)
            [3.08, 0.0, 0.0],  # C3 (good C-C bond)
            [4.62, 0.0, 0.0],  # C4 (good C-C bond)
        ])
        
        return system_data, coordinates
    
    def test_covalent_integrity_intact(self, integrity_system):
        """Test covalent integrity check for intact structure"""
        system_data, coordinates = integrity_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        results = check_covalent_integrity(coordinates, topology, max_deviation=0.5)
        
        assert results["intact"] == True
        assert len(results["broken_bonds"]) == 0
        assert len(results["stretched_bonds"]) == 0
        assert len(results["compressed_bonds"]) == 0
        assert results["max_deviation"] < 0.5
    
    def test_covalent_integrity_broken(self, integrity_system):
        """Test covalent integrity check for broken structure"""
        system_data, coordinates = integrity_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Break a bond by moving atoms apart
        broken_coords = coordinates.copy()
        broken_coords[1, 0] = 3.0  # Move C2 far from C1
        
        results = check_covalent_integrity(broken_coords, topology, max_deviation=0.5)
        
        assert results["intact"] == False
        assert len(results["broken_bonds"]) > 0 or len(results["stretched_bonds"]) > 0
        assert results["max_deviation"] > 0.5
    
    def test_covalent_integrity_compressed(self, integrity_system):
        """Test covalent integrity check for compressed bonds"""
        system_data, coordinates = integrity_system
        
        topology = MolecularTopology(system_data, coordinates)
        
        # Compress bonds
        compressed_coords = coordinates.copy()
        compressed_coords[1, 0] = 0.5  # Compress C1-C2 bond
        
        results = check_covalent_integrity(compressed_coords, topology, max_deviation=0.3)
        
        assert results["intact"] == False
        assert len(results["compressed_bonds"]) > 0
        assert results["max_deviation"] > 0.3


class TestTopologyDetection:
    """Test high-level topology detection function"""
    
    def test_detect_molecular_topology(self):
        """Test the main topology detection function"""
        system_data = {
            "nat": 3,
            "symbols": ["C", "C", "H"],
            "atoms": [
                {"name": "C1", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "C2", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "H1", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
            ]
        }
        
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [2.63, 0.0, 0.0],
        ])
        
        topology = detect_molecular_topology(system_data, coordinates)
        
        assert isinstance(topology, MolecularTopology)
        assert topology.nat == 3
        assert len(topology.bonds) > 0
        assert len(topology.angles) > 0
    
    def test_standard_bond_lengths(self):
        """Test that standard bond lengths are reasonable"""
        # Check some common bond lengths
        assert STANDARD_BOND_LENGTHS[('C', 'C')] == 1.54
        assert STANDARD_BOND_LENGTHS[('C', 'H')] == 1.09
        assert STANDARD_BOND_LENGTHS[('N', 'H')] == 1.01
        assert STANDARD_BOND_LENGTHS[('O', 'H')] == 0.96
        
        # Check that max bond lengths are larger than standard
        for bond_type in STANDARD_BOND_LENGTHS:
            if bond_type in MAX_BOND_LENGTHS:
                assert MAX_BOND_LENGTHS[bond_type] > STANDARD_BOND_LENGTHS[bond_type]
    
    def test_standard_bond_angles(self):
        """Test that standard bond angles are reasonable"""
        # Check tetrahedral angle
        assert STANDARD_BOND_ANGLES[('C', 'C', 'C')] == 109.5
        assert STANDARD_BOND_ANGLES[('H', 'C', 'H')] == 109.5
        
        # Check that angles are in degrees
        for angle in STANDARD_BOND_ANGLES.values():
            assert 0 < angle <= 180


if __name__ == "__main__":
    pytest.main([__file__, "-v"])