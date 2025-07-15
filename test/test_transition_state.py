"""
Test transition state search functionality
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np

def test_transition_state_import():
    """Test that transition state module can be imported"""
    try:
        from fennol.md.transition_state import (
            QuasiNewtonTS, 
            DimerMethod, 
            get_ts_optimizer,
            find_transition_state
        )
        print("✓ Successfully imported transition state modules")
        return True
    except ImportError as e:
        print(f"✗ Failed to import transition state modules: {e}")
        return False

def test_ts_optimizer_creation():
    """Test that TS optimizers can be created"""
    try:
        from fennol.md.transition_state import get_ts_optimizer
        
        # Mock parameters
        model = type('MockModel', (), {
            'energy_unit': 'kcal/mol',
            'variables': {},
            '_energy_and_forces': lambda *args: (np.array([0.0]), np.zeros((3, 3)), {}),
            '_total_energy': lambda *args: np.array([0.0]),
            'preprocessing': type('MockPreproc', (), {'process': lambda *args: args[1]})(),
            'preproc_state': {}
        })()
        
        system_data = {
            'nat': 3,
            'name': 'test',
            'symbols': ['H', 'C', 'N']
        }
        
        conformation = {
            'coordinates': np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        }
        
        simulation_parameters = {
            'ts_method': 'quasi_newton',
            'min_force_tolerance': 1e-4
        }
        
        # Test quasi-Newton optimizer
        qn_optimizer = get_ts_optimizer('quasi_newton', model, system_data, 
                                       conformation, simulation_parameters, 'float64')
        assert qn_optimizer is not None
        print("✓ Successfully created quasi-Newton TS optimizer")
        
        # Test dimer optimizer
        dimer_optimizer = get_ts_optimizer('dimer', model, system_data, 
                                          conformation, simulation_parameters, 'float64')
        assert dimer_optimizer is not None
        print("✓ Successfully created dimer method optimizer")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create TS optimizers: {e}")
        return False

def test_input_file_parsing():
    """Test that TS parameters are correctly parsed from input files"""
    try:
        from fennol.utils.input_parser import parse_input
        
        # Create a temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fnl', delete=False) as f:
            f.write("""
# Test transition state input
transition_state = True
ts_only = True
ts_method = quasi_newton
ts_hessian_update = bfgs
ts_trust_radius = 0.3
ts_eigenvalue_tolerance = 1e-4
min_force_tolerance = 1e-4
""")
            temp_file = f.name
        
        # Parse the file
        params = parse_input(temp_file)
        
        # Check parameters
        assert params.get('transition_state') == True
        assert params.get('ts_only') == True
        assert params.get('ts_method') == 'quasi_newton'
        assert params.get('ts_hessian_update') == 'bfgs'
        assert abs(params.get('ts_trust_radius') - 0.3) < 1e-6
        assert abs(params.get('ts_eigenvalue_tolerance') - 1e-4) < 1e-8
        
        print("✓ Successfully parsed TS parameters from input file")
        
        # Clean up
        os.unlink(temp_file)
        return True
        
    except Exception as e:
        print(f"✗ Failed to parse TS parameters: {e}")
        if 'temp_file' in locals():
            os.unlink(temp_file)
        return False

def test_hessian_operations():
    """Test Hessian initialization and update operations"""
    try:
        import jax.numpy as jnp
        from fennol.md.transition_state import QuasiNewtonTS
        
        # Create mock objects
        model = type('MockModel', (), {'energy_unit': 'kcal/mol'})()
        system_data = {'nat': 3, 'name': 'test'}
        conformation = {'coordinates': np.zeros((3, 3))}
        simulation_parameters = {
            'ts_hessian_update': 'bfgs',
            'ts_initial_hessian_scale': -0.1,
            'min_force_tolerance': 1e-4,
            'min_max_iterations': 100,
            'min_print_freq': 10
        }
        
        # Create optimizer
        optimizer = QuasiNewtonTS(model, system_data, conformation, 
                                 simulation_parameters, 'float64')
        
        # Test Hessian initialization
        coords = jnp.array([0., 0., 0., 1., 0., 0., 2., 0., 0.])
        optimizer._initialize_hessian(coords)
        
        assert optimizer.hessian is not None
        assert optimizer.hessian.shape == (9, 9)
        assert jnp.allclose(jnp.diag(optimizer.hessian), -0.1)
        print("✓ Successfully initialized Hessian")
        
        # Test Hessian update
        gradient1 = jnp.array([0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        optimizer._update_hessian(coords, gradient1)
        
        coords2 = coords + 0.1 * gradient1
        gradient2 = jnp.array([0.05, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        optimizer._update_hessian(coords2, gradient2)
        
        # Check that Hessian was updated (no longer diagonal)
        assert not jnp.allclose(optimizer.hessian, jnp.diag(jnp.diag(optimizer.hessian)))
        print("✓ Successfully updated Hessian")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed Hessian operations test: {e}")
        return False

def test_dimer_rotation():
    """Test dimer rotation functionality"""
    try:
        import jax.numpy as jnp
        from fennol.md.transition_state import DimerMethod
        
        # Create mock model that returns predictable forces
        def mock_energy_forces(coords, *args):
            # Simple quadratic potential with saddle at origin
            c = coords.reshape(-1, 3)
            x, y = c[0, 0], c[0, 1]
            energy = x**2 - y**2
            forces = jnp.zeros_like(c)
            forces = forces.at[0, 0].set(-2*x)
            forces = forces.at[0, 1].set(2*y)
            return jnp.array([energy]), forces, None, None, {}
        
        model = type('MockModel', (), {
            'energy_unit': 'kcal/mol',
            '_energy_and_forces': mock_energy_forces
        })()
        
        system_data = {'nat': 1, 'name': 'test'}
        conformation = {'coordinates': np.array([[0.1, 0.1, 0.0]])}
        simulation_parameters = {
            'dimer_separation': 0.01,
            'dimer_rotation_tolerance': 0.1,
            'min_force_tolerance': 1e-4,
            'min_max_iterations': 100,
            'min_print_freq': 10,
            'min_initial_step': 0.01
        }
        
        # Create dimer optimizer
        optimizer = DimerMethod(model, system_data, conformation, 
                               simulation_parameters, 'float64')
        
        # Initialize dimer
        coords = jnp.array([0.1, 0.1, 0.0])
        optimizer._initialize_dimer(coords)
        
        assert optimizer.dimer_vector is not None
        assert jnp.abs(jnp.linalg.norm(optimizer.dimer_vector) - 1.0) < 1e-6
        print("✓ Successfully initialized dimer")
        
        # Test rotation
        energy = jnp.array([0.0])
        forces = jnp.array([[0.0, 0.0, 0.0]])
        optimizer._evaluate_energy_forces = mock_energy_forces
        
        vector, curvature = optimizer._rotate_dimer(coords, energy, forces, None, None)
        
        assert vector is not None
        assert isinstance(curvature, float)
        print("✓ Successfully rotated dimer")
        print(f"  Curvature: {curvature}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed dimer rotation test: {e}")
        return False

def test_sn2_optimizer():
    """Test SN2-specific transition state optimizer"""
    try:
        import jax.numpy as jnp
        from fennol.md.transition_state import SN2TransitionState
        
        # Create mock objects for SN2 system (Cl- + CH3Br -> ClCH3 + Br-)
        model = type('MockModel', (), {
            'energy_unit': 'kcal/mol',
            '_energy_and_forces': lambda *args: (np.array([0.0]), np.zeros((8, 3)), {})
        })()
        
        system_data = {'nat': 8, 'name': 'sn2_test', 'symbols': ['Cl', 'C', 'H', 'H', 'H', 'Br', 'H', 'H']}
        
        # SN2 initial guess: Cl- ... CH3 ... Br-
        conformation = {
            'coordinates': np.array([
                [-2.5, 0.0, 0.0],  # Cl (nucleophile)
                [0.0, 0.0, 0.0],   # C (center)
                [0.0, 1.09, 0.0],  # H
                [0.944, -0.545, 0.0],  # H
                [-0.944, -0.545, 0.0], # H
                [2.5, 0.0, 0.0],   # Br (leaving group)
                [0.0, 0.0, 1.5],   # H
                [0.0, 0.0, -1.5]   # H
            ])
        }
        
        simulation_parameters = {
            'sn2_nu_index': 1,    # Cl (1-based)
            'sn2_c_index': 2,     # C (1-based)
            'sn2_lg_index': 6,    # Br (1-based)
            'sn2_target_nu_c_distance': 2.0,
            'sn2_target_c_lg_distance': 2.0,
            'min_force_tolerance': 1e-3,
            'min_max_iterations': 10,
            'min_print_freq': 5
        }
        
        # Create SN2 optimizer
        optimizer = SN2TransitionState(model, system_data, conformation, 
                                      simulation_parameters, 'float64')
        
        # Test reaction coordinate calculation
        coords = conformation['coordinates'].reshape(-1)
        rc_value, rc_gradient, nu_c_dist, c_lg_dist = optimizer._compute_sn2_reaction_coordinate(coords)
        
        print(f"✓ SN2 reaction coordinate: {rc_value:.3f}")
        print(f"  Nu-C distance: {nu_c_dist:.3f} Å")
        print(f"  C-LG distance: {c_lg_dist:.3f} Å")
        
        # Check that gradient has correct shape
        assert rc_gradient.shape == coords.shape
        assert isinstance(rc_value, (float, jnp.ndarray))
        assert nu_c_dist > 0 and c_lg_dist > 0
        
        print("✓ Successfully created and tested SN2 optimizer")
        return True
        
    except Exception as e:
        print(f"✗ Failed SN2 optimizer test: {e}")
        return False

def test_sn2_input_parsing():
    """Test SN2-specific input parameter parsing"""
    try:
        from fennol.utils.input_parser import parse_input
        
        # Create a temporary SN2 input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fnl', delete=False) as f:
            f.write("""
# Test SN2 transition state input
transition_state = True
ts_method = sn2
sn2_nu_index = 1
sn2_c_index = 2  
sn2_lg_index = 6
sn2_target_nu_c_distance = 2.0
sn2_constraint_strength = 0.1
""")
            temp_file = f.name
        
        # Parse the file
        params = parse_input(temp_file)
        
        # Check SN2 parameters
        assert params.get('ts_method') == 'sn2'
        assert params.get('sn2_nu_index') == 1
        assert params.get('sn2_c_index') == 2
        assert params.get('sn2_lg_index') == 6
        assert abs(params.get('sn2_target_nu_c_distance') - 2.0) < 1e-6
        assert abs(params.get('sn2_constraint_strength') - 0.1) < 1e-6
        
        print("✓ Successfully parsed SN2 parameters from input file")
        
        # Clean up
        os.unlink(temp_file)
        return True
        
    except Exception as e:
        print(f"✗ Failed to parse SN2 parameters: {e}")
        if 'temp_file' in locals():
            os.unlink(temp_file)
        return False

def run_all_tests():
    """Run all transition state tests"""
    print("Running transition state implementation tests...\n")
    
    tests = [
        ("Import test", test_transition_state_import),
        ("Optimizer creation", test_ts_optimizer_creation),
        ("Input parsing", test_input_file_parsing),
        ("Hessian operations", test_hessian_operations),
        ("Dimer rotation", test_dimer_rotation),
        ("SN2 optimizer", test_sn2_optimizer),
        ("SN2 input parsing", test_sn2_input_parsing)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"{'='*50}\n")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)