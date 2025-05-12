import os
import sys
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import tempfile
from pathlib import Path

# Add parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fennol.md.minimize import (
    SteepestDescentMinimizer,
    LBFGSMinimizer,
    ConjugateGradientMinimizer,
    get_minimizer
)
from src.fennol.md.initial import load_model, load_system_data, initialize_preprocessing

class TestMinimization(unittest.TestCase):
    """Test cases for energy minimization"""

    def setUp(self):
        """Setup test data - create a simple test system"""
        # Set JAX config
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_platform_name', 'cpu')
        
        # Create a simple water molecule xyz file
        self.xyz_file = tempfile.NamedTemporaryFile(suffix='.xyz', delete=False)
        self.xyz_file.write(b"""3
Simple water molecule
O      0.000   0.000   0.000
H      0.758   0.586   0.000
H     -0.758   0.586   0.000
""")
        self.xyz_file.close()
        
        # Create basic simulation parameters
        self.simulation_parameters = {
            "coordinates": self.xyz_file.name,
            "model_type": "ani2x",  # Use ANI model
            "double_precision": True,
            "device": "cpu",
            # Minimization parameters
            "min_max_iterations": 10,  # Keep small for test
            "min_force_tolerance": 1e-4,
            "min_energy_tolerance": 1e-6,
            "min_displacement_tolerance": 1e-3,
            "min_print_freq": 5,
            "traj_format": "xyz"
        }
        
        # Load model and system
        self.model = None
        self.system_data = None
        self.conformation = None
        self.fprec = "float64"
        
        # Setup minimizer output locations
        self.output_files = []
        
    def tearDown(self):
        """Clean up test files"""
        # Remove xyz file
        os.unlink(self.xyz_file.name)
        
        # Remove output files
        for f in self.output_files:
            if os.path.exists(f):
                os.unlink(f)
    
    def _load_system(self):
        """Helper to load model and system if not already loaded"""
        if self.model is None:
            try:
                self.model = load_model(self.simulation_parameters)
                self.system_data, self.conformation = load_system_data(self.simulation_parameters, self.fprec)
                _, self.conformation = initialize_preprocessing(
                    self.simulation_parameters, self.model, self.conformation, self.system_data
                )
            except Exception as e:
                self.skipTest(f"Failed to load model or system: {str(e)}")
    
    def test_get_minimizer(self):
        """Test that get_minimizer factory function works correctly"""
        self._load_system()
        
        # Test each minimizer type
        sd_minimizer = get_minimizer("sd", self.model, self.system_data, self.conformation, 
                                    self.simulation_parameters, self.fprec)
        self.assertIsInstance(sd_minimizer, SteepestDescentMinimizer)
        
        lbfgs_minimizer = get_minimizer("lbfgs", self.model, self.system_data, self.conformation, 
                                        self.simulation_parameters, self.fprec)
        self.assertIsInstance(lbfgs_minimizer, LBFGSMinimizer)
        
        cg_minimizer = get_minimizer("cg", self.model, self.system_data, self.conformation, 
                                    self.simulation_parameters, self.fprec)
        self.assertIsInstance(cg_minimizer, ConjugateGradientMinimizer)
        
        # Test alternate names
        sd_alt = get_minimizer("steepest_descent", self.model, self.system_data, self.conformation, 
                              self.simulation_parameters, self.fprec)
        self.assertIsInstance(sd_alt, SteepestDescentMinimizer)
        
        lbfgs_alt = get_minimizer("bfgs", self.model, self.system_data, self.conformation, 
                                  self.simulation_parameters, self.fprec)
        self.assertIsInstance(lbfgs_alt, LBFGSMinimizer)
        
        cg_alt = get_minimizer("conjugate_gradient", self.model, self.system_data, self.conformation, 
                              self.simulation_parameters, self.fprec)
        self.assertIsInstance(cg_alt, ConjugateGradientMinimizer)
        
        # Test invalid name
        with self.assertRaises(ValueError):
            get_minimizer("invalid", self.model, self.system_data, self.conformation, 
                         self.simulation_parameters, self.fprec)
    
    def test_steepest_descent(self):
        """Test steepest descent minimization"""
        self._load_system()
        
        # Set minimizer type
        self.simulation_parameters["min_method"] = "sd"
        
        # Setup output file
        self.output_files.append(f"{self.system_data['name']}.min.xyz")
        
        # Create minimizer
        minimizer = SteepestDescentMinimizer(
            self.model, self.system_data, self.conformation, 
            self.simulation_parameters, self.fprec
        )
        
        # Run minimization
        result = minimizer.run()
        
        # Check result structure
        self.assertIn("coordinates", result)
        self.assertIn("energy", result)
        self.assertIn("forces", result)
        self.assertIn("system", result)
        
        # Check output file was created
        self.assertTrue(os.path.exists(self.output_files[0]))
        self.assertTrue(os.path.getsize(self.output_files[0]) > 0)
    
    def test_lbfgs(self):
        """Test L-BFGS minimization"""
        self._load_system()
        
        # Set minimizer type
        self.simulation_parameters["min_method"] = "lbfgs"
        
        # Setup output file
        self.output_files.append(f"{self.system_data['name']}.min.xyz")
        
        # Create minimizer
        minimizer = LBFGSMinimizer(
            self.model, self.system_data, self.conformation, 
            self.simulation_parameters, self.fprec
        )
        
        # Run minimization
        result = minimizer.run()
        
        # Check result structure
        self.assertIn("coordinates", result)
        self.assertIn("energy", result)
        self.assertIn("forces", result)
        self.assertIn("system", result)
        
        # Check output file was created
        self.assertTrue(os.path.exists(self.output_files[0]))
        self.assertTrue(os.path.getsize(self.output_files[0]) > 0)
    
    def test_conjugate_gradient(self):
        """Test conjugate gradient minimization"""
        self._load_system()
        
        # Set minimizer type
        self.simulation_parameters["min_method"] = "cg"
        
        # Setup output file
        self.output_files.append(f"{self.system_data['name']}.min.xyz")
        
        # Create minimizer
        minimizer = ConjugateGradientMinimizer(
            self.model, self.system_data, self.conformation, 
            self.simulation_parameters, self.fprec
        )
        
        # Run minimization
        result = minimizer.run()
        
        # Check result structure
        self.assertIn("coordinates", result)
        self.assertIn("energy", result)
        self.assertIn("forces", result)
        self.assertIn("system", result)
        
        # Check output file was created
        self.assertTrue(os.path.exists(self.output_files[0]))
        self.assertTrue(os.path.getsize(self.output_files[0]) > 0)
    
    def test_minimization_improves_energy(self):
        """Test that minimization actually improves the energy"""
        self._load_system()
        
        # Distort the water molecule to create a high-energy structure
        coords = self.conformation["coordinates"]
        # Move H atoms farther out
        coords = coords.at[1, 0].set(coords[1, 0] * 1.5)  # Stretch first H-O bond
        coords = coords.at[2, 0].set(coords[2, 0] * 1.5)  # Stretch second H-O bond
        self.conformation = {**self.conformation, "coordinates": coords}
        
        # Compute initial energy
        _, conf = initialize_preprocessing(
            self.simulation_parameters, self.model, self.conformation, self.system_data
        )
        initial_energy, _, _ = self.model._energy_and_forces(self.model.variables, conf)
        initial_energy = initial_energy[0] / self.model.energy_unit_multiplier
        
        # Run minimization (use SD for simplicity)
        self.simulation_parameters["min_method"] = "sd"
        minimizer = SteepestDescentMinimizer(
            self.model, self.system_data, self.conformation, 
            self.simulation_parameters, self.fprec
        )
        result = minimizer.run()
        
        # Check that energy improved
        self.assertLess(result["energy"], initial_energy)


if __name__ == '__main__':
    unittest.main()