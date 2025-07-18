"""
Unit tests for the fennol_refine CLI tool
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

from fennol.md.refine_cli import (
    build_parameters_from_cli, 
    StructureRefiner, 
    run_refinement,
    main
)
from fennol.utils.io import read_pdb, write_xyz_frame


class TestBuildParametersFromCli:
    """Test parameter building from command line arguments"""
    
    def test_basic_parameters(self):
        """Test building basic parameters"""
        # Mock command line arguments
        args = Mock()
        args.pdb = "test.pdb"
        args.model = "test_model.pkl"
        args.method = "simulated_annealing"
        args.output_prefix = "test_output"
        args.temperature_schedule = "exponential"
        args.initial_temperature = 600.0
        args.final_temperature = 10.0
        args.temperature_steps = 100
        args.max_iterations = 1000
        args.force_tolerance = 1e-4
        args.energy_tolerance = 1e-6
        args.displacement_tolerance = 1e-3
        args.max_step = 0.2
        args.print_freq = 10
        args.rmsd_restraint = True
        args.rmsd_force_constant = 10.0
        args.rmsd_target = 0.0
        args.rmsd_atoms = "heavy"
        args.clash_cutoff = 2.0
        args.clash_force_constant = 100.0
        args.clash_iterations = 100
        args.device = "cpu"
        args.double_precision = False
        args.matmul_precision = "highest"
        args.write_trajectory = True
        args.trajectory_format = "pdb"
        args.write_multimodel = True
        args.annealing_cycles = 10
        args.annealing_hold_time = 10
        args.mc_displacement = 0.1
        args.mc_acceptance_ratio = 0.5
        
        params = build_parameters_from_cli(args)
        
        assert params["pdb_file"] == "test.pdb"
        assert params["model_file"] == "test_model.pkl"
        assert params["refinement_method"] == "simulated_annealing"
        assert params["output_prefix"] == "test_output"
        assert params["use_rmsd_restraint"] == True
        assert params["rmsd_force_constant"] == 10.0
        assert params["initial_temperature"] == 600.0
        assert params["final_temperature"] == 10.0
        assert params["temperature_steps"] == 100
        assert params["annealing_cycles"] == 10
        assert params["annealing_hold_time"] == 10
    
    def test_output_prefix_from_pdb(self):
        """Test output prefix derivation from PDB filename"""
        args = Mock()
        args.pdb = "example/test_structure.pdb"
        args.model = "test_model.pkl"
        args.method = "simulated_annealing"
        args.output_prefix = None
        args.temperature_schedule = "exponential"
        args.initial_temperature = 600.0
        args.final_temperature = 10.0
        args.temperature_steps = 100
        args.max_iterations = 1000
        args.force_tolerance = 1e-4
        args.energy_tolerance = 1e-6
        args.displacement_tolerance = 1e-3
        args.max_step = 0.2
        args.print_freq = 10
        args.rmsd_restraint = True
        args.rmsd_force_constant = 10.0
        args.rmsd_target = 0.0
        args.rmsd_atoms = "heavy"
        args.clash_cutoff = 2.0
        args.clash_force_constant = 100.0
        args.clash_iterations = 100
        args.device = "cpu"
        args.double_precision = False
        args.matmul_precision = "highest"
        args.write_trajectory = True
        args.trajectory_format = "pdb"
        args.write_multimodel = True
        args.annealing_cycles = 10
        args.annealing_hold_time = 10
        args.mc_displacement = 0.1
        args.mc_acceptance_ratio = 0.5
        
        params = build_parameters_from_cli(args)
        
        assert params["output_prefix"] == "test_structure"
    
    def test_method_specific_parameters(self):
        """Test method-specific parameter inclusion"""
        args = Mock()
        args.pdb = "test.pdb"
        args.model = "test_model.pkl"
        args.method = "monte_carlo"
        args.output_prefix = "test"
        args.temperature_schedule = "exponential"
        args.initial_temperature = 600.0
        args.final_temperature = 10.0
        args.temperature_steps = 100
        args.max_iterations = 1000
        args.force_tolerance = 1e-4
        args.energy_tolerance = 1e-6
        args.displacement_tolerance = 1e-3
        args.max_step = 0.2
        args.print_freq = 10
        args.rmsd_restraint = True
        args.rmsd_force_constant = 10.0
        args.rmsd_target = 0.0
        args.rmsd_atoms = "heavy"
        args.clash_cutoff = 2.0
        args.clash_force_constant = 100.0
        args.clash_iterations = 100
        args.device = "cpu"
        args.double_precision = False
        args.matmul_precision = "highest"
        args.write_trajectory = True
        args.trajectory_format = "pdb"
        args.write_multimodel = True
        args.annealing_cycles = 10
        args.annealing_hold_time = 10
        args.mc_displacement = 0.15
        args.mc_acceptance_ratio = 0.6
        
        params = build_parameters_from_cli(args)
        
        assert params["refinement_method"] == "monte_carlo"
        assert params["mc_displacement"] == 0.15
        assert params["mc_acceptance_ratio"] == 0.6


class TestStructureRefiner:
    """Test the StructureRefiner class"""
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing"""
        model = Mock()
        model.energy_unit = "eV"
        model.energy_and_forces.return_value = {
            "energy": 0.0,
            "forces": np.zeros((10, 3))
        }
        return model
    
    @pytest.fixture
    def mock_system_data(self):
        """Mock system data for testing"""
        return {
            "nat": 10,
            "symbols": ["C", "H", "H", "H", "N", "H", "H", "O", "H", "H"],
            "atoms": [
                {"name": "C1", "resname": "MOL", "resid": 1, "chain": "A", "element": "C"},
                {"name": "H1", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "H2", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "H3", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "N1", "resname": "MOL", "resid": 1, "chain": "A", "element": "N"},
                {"name": "H4", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "H5", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "O1", "resname": "MOL", "resid": 1, "chain": "A", "element": "O"},
                {"name": "H6", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
                {"name": "H7", "resname": "MOL", "resid": 1, "chain": "A", "element": "H"},
            ],
            "name": "test_molecule"
        }
    
    @pytest.fixture
    def mock_conformation(self):
        """Mock conformation for testing"""
        coords = np.random.randn(10, 3) * 2.0  # Random coordinates
        return {
            "coordinates": jnp.array(coords)
        }
    
    @pytest.fixture
    def mock_simulation_parameters(self):
        """Mock simulation parameters"""
        return {
            "refinement_method": "simulated_annealing",
            "output_prefix": "test_refine",
            "use_rmsd_restraint": True,
            "rmsd_force_constant": 10.0,
            "rmsd_target": 0.0,
            "rmsd_atom_selection": "heavy",
            "clash_detection_cutoff": 2.0,
            "clash_force_constant": 100.0,
            "initial_temperature": 600.0,
            "final_temperature": 10.0,
            "temperature_steps": 50,
            "temperature_schedule": "exponential",
            "annealing_cycles": 5,
            "annealing_hold_time": 5,
            "write_trajectory": False,  # Disable for testing
            "trajectory_format": "pdb",
            "write_multimodel": False,
        }
    
    def test_refiner_initialization(self, mock_model, mock_system_data, 
                                  mock_conformation, mock_simulation_parameters):
        """Test StructureRefiner initialization"""
        refiner = StructureRefiner(
            mock_model,
            mock_system_data,
            mock_conformation,
            mock_simulation_parameters,
            "float32"
        )
        
        assert refiner.model == mock_model
        assert refiner.system_data == mock_system_data
        assert refiner.nat == 10
        assert refiner.method == "simulated_annealing"
        assert refiner.use_rmsd == True
        assert refiner.rmsd_fc == 10.0
        assert refiner.rmsd_target == 0.0
        assert refiner.clash_cutoff == 2.0
        assert refiner.clash_fc == 100.0
        assert refiner.initial_temp == 600.0
        assert refiner.final_temp == 10.0
        assert refiner.temp_steps == 50
        assert refiner.temp_schedule == "exponential"
    
    def test_select_heavy_atoms(self, mock_model, mock_system_data, 
                              mock_conformation, mock_simulation_parameters):
        """Test heavy atom selection"""
        refiner = StructureRefiner(
            mock_model,
            mock_system_data,
            mock_conformation,
            mock_simulation_parameters,
            "float32"
        )
        
        heavy_atoms = refiner._select_heavy_atoms()
        
        # Should select all non-hydrogen atoms (C, N, O in this case)
        expected_heavy = [i for i, symbol in enumerate(mock_system_data["symbols"]) 
                         if symbol != "H"]
        assert list(heavy_atoms) == expected_heavy
    
    def test_parse_atom_selection(self, mock_model, mock_system_data, 
                                mock_conformation, mock_simulation_parameters):
        """Test atom selection parsing"""
        refiner = StructureRefiner(
            mock_model,
            mock_system_data,
            mock_conformation,
            mock_simulation_parameters,
            "float32"
        )
        
        # Test range selection
        selected = refiner._parse_atom_selection("1-5")
        assert list(selected) == [0, 1, 2, 3, 4]  # 1-based to 0-based
        
        # Test mixed selection
        selected = refiner._parse_atom_selection("1,3,5-7")
        assert list(selected) == [0, 2, 4, 5, 6]
    
    def test_detect_clashes(self, mock_model, mock_system_data, 
                          mock_conformation, mock_simulation_parameters):
        """Test clash detection"""
        refiner = StructureRefiner(
            mock_model,
            mock_system_data,
            mock_conformation,
            mock_simulation_parameters,
            "float32"
        )
        
        # Create coordinates with known clashes
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Close to first atom
            [5.0, 0.0, 0.0],  # Far from others
            [0.5, 0.0, 0.0],  # Very close to first atom
            [2.0, 0.0, 0.0],  # Normal distance
            [3.0, 0.0, 0.0],  # Normal distance
            [4.0, 0.0, 0.0],  # Normal distance
            [6.0, 0.0, 0.0],  # Normal distance
            [7.0, 0.0, 0.0],  # Normal distance
            [8.0, 0.0, 0.0],  # Normal distance
        ])
        
        clash_mask, dist_matrix = refiner._detect_clashes(coords)
        
        # Should detect clashes between atoms 0-1 and 0-3
        assert clash_mask[0, 1] == True  # Distance = 1.0, less than cutoff (2.0)
        assert clash_mask[0, 3] == True  # Distance = 0.5, less than cutoff
        assert clash_mask[1, 2] == False  # Distance = 4.0, greater than cutoff
    
    def test_temperature_schedule_generation(self, mock_model, mock_system_data, 
                                           mock_conformation, mock_simulation_parameters):
        """Test temperature schedule generation"""
        refiner = StructureRefiner(
            mock_model,
            mock_system_data,
            mock_conformation,
            mock_simulation_parameters,
            "float32"
        )
        
        # Test exponential schedule
        refiner.temp_schedule = "exponential"
        refiner.temp_steps = 10
        temps = refiner._get_temperature_schedule()
        
        assert len(temps) == 10
        assert temps[0] == pytest.approx(600.0, rel=1e-3)
        assert temps[-1] == pytest.approx(10.0, rel=1e-3)
        assert all(temps[i] >= temps[i+1] for i in range(len(temps)-1))
        
        # Test linear schedule
        refiner.temp_schedule = "linear"
        temps = refiner._get_temperature_schedule()
        
        assert len(temps) == 10
        assert temps[0] == 600.0
        assert temps[-1] == 10.0
        
        # Test cosine schedule
        refiner.temp_schedule = "cosine"
        temps = refiner._get_temperature_schedule()
        
        assert len(temps) == 10
        assert temps[0] == 600.0
        assert temps[-1] == 10.0
    
    def test_calculate_rmsd(self, mock_model, mock_system_data, 
                          mock_conformation, mock_simulation_parameters):
        """Test RMSD calculation"""
        refiner = StructureRefiner(
            mock_model,
            mock_system_data,
            mock_conformation,
            mock_simulation_parameters,
            "float32"
        )
        
        # Test with identical coordinates
        coords = refiner.reference_coords.copy()
        rmsd = refiner._calculate_rmsd(coords)
        assert rmsd == pytest.approx(0.0, abs=1e-6)
        
        # Test with displaced coordinates
        coords_displaced = refiner.reference_coords + 1.0
        rmsd = refiner._calculate_rmsd(coords_displaced)
        assert rmsd == pytest.approx(np.sqrt(3.0), rel=1e-3)  # sqrt(3) for unit displacement in 3D
    
    @patch('builtins.open', new_callable=MagicMock)
    def test_save_final_structure(self, mock_open, mock_model, mock_system_data, 
                                 mock_conformation, mock_simulation_parameters):
        """Test saving final structure"""
        refiner = StructureRefiner(
            mock_model,
            mock_system_data,
            mock_conformation,
            mock_simulation_parameters,
            "float32"
        )
        
        # Mock the force calculation
        refiner._calculate_forces = Mock(return_value=(0.0, np.zeros((10, 3))))
        
        coords = np.random.randn(10, 3)
        
        # Test saving
        refiner._save_final_structure(coords)
        
        # Check that files were opened for writing
        assert mock_open.call_count == 2  # PDB and XYZ files
        
        # Check file names
        call_args = [call[0][0] for call in mock_open.call_args_list]
        assert "test_refine_refined.pdb" in call_args
        assert "test_refine_refined.xyz" in call_args


class TestIntegration:
    """Integration tests for the refinement process"""
    
    @pytest.fixture
    def sample_pdb_file(self):
        """Create a sample PDB file for testing"""
        pdb_content = """ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C  
ATOM      2  H   MOL A   1       1.000   0.000   0.000  1.00  0.00           H  
ATOM      3  H   MOL A   1       0.000   1.000   0.000  1.00  0.00           H  
ATOM      4  H   MOL A   1       0.000   0.000   1.000  1.00  0.00           H  
ATOM      5  N   MOL A   1       2.000   0.000   0.000  1.00  0.00           N  
ATOM      6  H   MOL A   1       2.500   0.500   0.000  1.00  0.00           H  
ATOM      7  H   MOL A   1       2.500  -0.500   0.000  1.00  0.00           H  
ATOM      8  O   MOL A   1       0.000   2.000   0.000  1.00  0.00           O  
ATOM      9  H   MOL A   1       0.000   2.500   0.500  1.00  0.00           H  
ATOM     10  H   MOL A   1       0.000   2.500  -0.500  1.00  0.00           H  
END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            return f.name
    
    def test_pdb_reading(self, sample_pdb_file):
        """Test PDB file reading"""
        pdb_data = read_pdb(sample_pdb_file)
        
        assert len(pdb_data["symbols"]) == 10
        assert pdb_data["symbols"][0] == "C"
        assert pdb_data["symbols"][4] == "N"
        assert pdb_data["symbols"][7] == "O"
        
        assert pdb_data["atom_names"][0] == "C"
        assert pdb_data["residue_names"][0] == "MOL"
        assert pdb_data["chain_ids"][0] == "A"
        
        # Check coordinates
        assert pdb_data["coordinates"][0, 0] == pytest.approx(0.0)
        assert pdb_data["coordinates"][1, 0] == pytest.approx(1.0)
        assert pdb_data["coordinates"][4, 0] == pytest.approx(2.0)
        
        # Clean up
        os.unlink(sample_pdb_file)
    
    @patch('fennol.md.refine_cli.load_model')
    @patch('fennol.md.refine_cli.initialize_preprocessing')
    def test_run_refinement_integration(self, mock_init_preprocessing, mock_load_model, sample_pdb_file):
        """Test integration of refinement process"""
        # Mock model
        mock_model = Mock()
        mock_model.energy_unit = "eV"
        mock_model.__class__.__name__ = "TestModel"
        mock_load_model.return_value = mock_model
        
        # Mock preprocessing
        mock_init_preprocessing.return_value = (None, {"coordinates": jnp.zeros((10, 3))})
        
        # Create simulation parameters
        simulation_parameters = {
            "pdb_file": sample_pdb_file,
            "refinement_method": "gradient_descent",
            "output_prefix": "test_integration",
            "use_rmsd_restraint": True,
            "rmsd_force_constant": 5.0,
            "rmsd_target": 0.0,
            "rmsd_atom_selection": "heavy",
            "clash_detection_cutoff": 2.0,
            "clash_force_constant": 50.0,
            "initial_temperature": 100.0,
            "final_temperature": 10.0,
            "temperature_steps": 5,
            "temperature_schedule": "linear",
            "write_trajectory": False,
            "trajectory_format": "pdb",
            "write_multimodel": False,
        }
        
        # Mock the refiner's refine method to avoid actual computation
        with patch('fennol.md.refine_cli.StructureRefiner.refine') as mock_refine:
            mock_refine.return_value = np.random.randn(10, 3)
            
            # Run refinement
            result = run_refinement(simulation_parameters, "float32", verbose=False)
            
            # Check that the process completed successfully
            assert result == True
            
            # Check that model was loaded
            mock_load_model.assert_called_once()
            
            # Check that preprocessing was initialized
            mock_init_preprocessing.assert_called_once()
            
            # Check that refiner was called
            mock_refine.assert_called_once()
        
        # Clean up
        os.unlink(sample_pdb_file)


class TestCLIArguments:
    """Test command line argument parsing"""
    
    @patch('sys.argv', ['fennol_refine', '--pdb', 'test.pdb', '--model', 'test.pkl', '--help'])
    def test_help_display(self):
        """Test that help is displayed correctly"""
        with pytest.raises(SystemExit):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_parse.side_effect = SystemExit(0)
                main()
    
    @patch('fennol.md.refine_cli.run_refinement')
    @patch('fennol.md.refine_cli.Path')
    @patch('sys.argv', ['fennol_refine', '--pdb', 'test.pdb', '--model', 'test.pkl'])
    def test_minimal_arguments(self, mock_path, mock_run_refinement):
        """Test minimal required arguments"""
        # Mock path existence checks
        mock_path.return_value.exists.return_value = True
        mock_run_refinement.return_value = True
        
        # Mock jax devices
        with patch('jax.devices') as mock_devices:
            mock_devices.return_value = [Mock()]
            
            # Mock jax config
            with patch('jax.config.update'):
                # This should not raise an error
                main()
                
                # Check that refinement was called
                mock_run_refinement.assert_called_once()
    
    @patch('fennol.md.refine_cli.Path')
    @patch('sys.argv', ['fennol_refine', '--pdb', 'nonexistent.pdb', '--model', 'test.pkl'])
    def test_missing_pdb_file(self, mock_path):
        """Test error handling for missing PDB file"""
        # Mock path existence - PDB doesn't exist
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        with pytest.raises(SystemExit):
            with patch('builtins.print') as mock_print:
                main()
                mock_print.assert_called_with("Error: PDB file 'nonexistent.pdb' not found")
    
    @patch('fennol.md.refine_cli.Path')
    @patch('sys.argv', ['fennol_refine', '--pdb', 'test.pdb', '--model', 'nonexistent.pkl'])
    def test_missing_model_file(self, mock_path):
        """Test error handling for missing model file"""
        # Mock path existence - model doesn't exist
        def mock_exists(self):
            return str(self).endswith('.pdb')  # Only PDB exists
        
        mock_path_instance = Mock()
        mock_path_instance.exists = mock_exists
        mock_path.return_value = mock_path_instance
        
        with pytest.raises(SystemExit):
            with patch('builtins.print') as mock_print:
                main()
                mock_print.assert_called_with("Error: Model file 'nonexistent.pkl' not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])