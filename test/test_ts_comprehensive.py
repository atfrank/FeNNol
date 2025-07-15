"""
Comprehensive tests for transition state search methods
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import Mock, patch
import tempfile
import os


class TestTransitionStateBase:
    """Base class for transition state tests"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = Mock()
        model.energy_unit = 'kcal/mol'
        model.variables = {}
        
        # Simple harmonic potential with saddle point
        def mock_energy_forces(coords, *args):
            coords_3d = coords.reshape(-1, 3)
            # Create a saddle at origin: x^2 + y^2 - z^2
            x, y, z = coords_3d[0]
            energy = x**2 + y**2 - z**2
            forces = jnp.zeros_like(coords_3d)
            forces = forces.at[0].set(jnp.array([-2*x, -2*y, 2*z]))
            return jnp.array([energy]), forces, None, None, {}
        
        model._energy_and_forces = mock_energy_forces
        return model
    
    @pytest.fixture
    def simple_system_data(self):
        """Simple system data for testing"""
        return {
            'nat': 1,
            'name': 'test_atom',
            'symbols': ['H']
        }
    
    @pytest.fixture
    def simple_conformation(self):
        """Simple conformation for testing"""
        return {
            'coordinates': jnp.array([[0.1, 0.1, 0.1]])
        }
    
    @pytest.fixture
    def basic_simulation_params(self):
        """Basic simulation parameters"""
        return {
            'min_force_tolerance': 1e-4,
            'min_max_iterations': 50,
            'min_print_freq': 10,
            'min_max_step': 0.1,
            'min_initial_step': 0.01
        }


class TestQuasiNewtonTS(TestTransitionStateBase):
    """Test quasi-Newton transition state optimizer"""
    
    def test_initialization(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test QuasiNewtonTS initialization"""
        from fennol.md.transition_state import QuasiNewtonTS
        
        params = basic_simulation_params.copy()
        params.update({
            'ts_hessian_update': 'bfgs',
            'ts_trust_radius': 0.3,
            'ts_eigenvalue_tolerance': 1e-4
        })
        
        optimizer = QuasiNewtonTS(mock_model, simple_system_data, simple_conformation, params, 'float64')
        
        assert optimizer.hessian_update_scheme == 'bfgs'
        assert optimizer.trust_radius == 0.3
        assert optimizer.eigenvalue_tolerance == 1e-4
        assert optimizer.target_n_negative == 1
    
    def test_hessian_initialization(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test Hessian initialization"""
        from fennol.md.transition_state import QuasiNewtonTS
        
        params = basic_simulation_params.copy()
        params['ts_initial_hessian_scale'] = -0.2
        
        optimizer = QuasiNewtonTS(mock_model, simple_system_data, simple_conformation, params, 'float64')
        
        coords = jnp.array([0., 0., 0.])
        optimizer._initialize_hessian(coords)
        
        assert optimizer.hessian is not None
        assert optimizer.hessian.shape == (3, 3)
        assert jnp.allclose(jnp.diag(optimizer.hessian), -0.2)
    
    def test_bfgs_update(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test BFGS Hessian update"""
        from fennol.md.transition_state import QuasiNewtonTS
        
        params = basic_simulation_params.copy()
        params['ts_hessian_update'] = 'bfgs'
        
        optimizer = QuasiNewtonTS(mock_model, simple_system_data, simple_conformation, params, 'float64')
        
        coords = jnp.array([0., 0., 0.])
        optimizer._initialize_hessian(coords)
        
        # First update
        gradient1 = jnp.array([0.1, 0.1, -0.1])
        optimizer._update_hessian(coords, gradient1)
        
        # Second update
        coords2 = coords + 0.1 * gradient1
        gradient2 = jnp.array([0.05, 0.05, -0.05])
        optimizer._update_hessian(coords2, gradient2)
        
        # Hessian should have been updated (not diagonal anymore)
        assert not jnp.allclose(optimizer.hessian, jnp.diag(jnp.diag(optimizer.hessian)))
    
    def test_sr1_update(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test SR1 Hessian update"""
        from fennol.md.transition_state import QuasiNewtonTS
        
        params = basic_simulation_params.copy()
        params['ts_hessian_update'] = 'sr1'
        
        optimizer = QuasiNewtonTS(mock_model, simple_system_data, simple_conformation, params, 'float64')
        
        coords = jnp.array([0., 0., 0.])
        optimizer._initialize_hessian(coords)
        
        gradient1 = jnp.array([0.1, 0.1, -0.1])
        optimizer._update_hessian(coords, gradient1)
        
        coords2 = coords + 0.1 * gradient1
        gradient2 = jnp.array([0.05, 0.05, -0.05])
        optimizer._update_hessian(coords2, gradient2)
        
        # Should be updated
        assert optimizer.hessian is not None
    
    def test_search_direction_computation(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test search direction computation"""
        from fennol.md.transition_state import QuasiNewtonTS
        
        optimizer = QuasiNewtonTS(mock_model, simple_system_data, simple_conformation, basic_simulation_params, 'float64')
        
        # Create a simple test case
        gradient = jnp.array([1., 0., 0.])
        eigenvalues = jnp.array([-1., 1., 2.])  # One negative eigenvalue
        eigenvectors = jnp.eye(3)  # Identity matrix for simplicity
        
        search_direction, sorted_eigenvalues = optimizer._compute_search_direction(gradient, eigenvalues, eigenvectors)
        
        assert search_direction is not None
        assert search_direction.shape == gradient.shape
        assert sorted_eigenvalues[0] < 0  # Most negative eigenvalue first
    
    def test_trust_radius_constraint(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test trust radius constraint"""
        from fennol.md.transition_state import QuasiNewtonTS
        
        params = basic_simulation_params.copy()
        params['ts_trust_radius'] = 0.1
        
        optimizer = QuasiNewtonTS(mock_model, simple_system_data, simple_conformation, params, 'float64')
        
        # Large search direction
        large_direction = jnp.array([1., 1., 1.])  # Norm = sqrt(3) > 0.1
        
        constrained_step = optimizer._trust_radius_step(large_direction)
        
        assert jnp.linalg.norm(constrained_step) <= 0.1 + 1e-6


class TestDimerMethod(TestTransitionStateBase):
    """Test dimer method for transition state search"""
    
    def test_initialization(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test DimerMethod initialization"""
        from fennol.md.transition_state import DimerMethod
        
        params = basic_simulation_params.copy()
        params.update({
            'dimer_separation': 0.01,
            'dimer_rotation_tolerance': 0.1,
            'dimer_max_rotations': 10
        })
        
        optimizer = DimerMethod(mock_model, simple_system_data, simple_conformation, params, 'float64')
        
        assert optimizer.dimer_separation == 0.01
        assert optimizer.rotation_tolerance == 0.1
        assert optimizer.max_rotation_iterations == 10
    
    def test_dimer_initialization(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test dimer vector initialization"""
        from fennol.md.transition_state import DimerMethod
        
        optimizer = DimerMethod(mock_model, simple_system_data, simple_conformation, basic_simulation_params, 'float64')
        
        coords = jnp.array([0., 0., 0.])
        optimizer._initialize_dimer(coords)
        
        assert optimizer.dimer_vector is not None
        assert optimizer.dimer_vector.shape == coords.shape
        assert jnp.abs(jnp.linalg.norm(optimizer.dimer_vector) - 1.0) < 1e-6
    
    def test_dimer_forces_computation(self, mock_model, simple_system_data, simple_conformation, basic_simulation_params):
        """Test dimer force computation"""
        from fennol.md.transition_state import DimerMethod
        
        optimizer = DimerMethod(mock_model, simple_system_data, simple_conformation, basic_simulation_params, 'float64')
        
        coords = jnp.array([0., 0., 0.])
        optimizer._initialize_dimer(coords)
        
        forces_center = jnp.array([[1., 0., 0.]])
        effective_forces = optimizer._compute_dimer_forces(coords, forces_center)
        
        assert effective_forces is not None
        assert effective_forces.shape == forces_center.shape


class TestSN2TransitionState(TestTransitionStateBase):
    """Test SN2-specific transition state optimizer"""
    
    @pytest.fixture
    def sn2_system_data(self):
        """SN2 system data"""
        return {
            'nat': 8,
            'name': 'sn2_test',
            'symbols': ['Cl', 'C', 'H', 'H', 'H', 'Br', 'H', 'H']
        }
    
    @pytest.fixture
    def sn2_conformation(self):
        """SN2 conformation"""
        return {
            'coordinates': jnp.array([
                [-2.5, 0.0, 0.0],     # Cl
                [0.0, 0.0, 0.0],      # C
                [0.0, 1.09, 0.0],     # H
                [0.944, -0.545, 0.0], # H
                [-0.944, -0.545, 0.0], # H
                [2.5, 0.0, 0.0],      # Br
                [0.0, 0.0, 1.5],      # H
                [0.0, 0.0, -1.5]      # H
            ])
        }
    
    @pytest.fixture
    def sn2_params(self, basic_simulation_params):
        """SN2 simulation parameters"""
        params = basic_simulation_params.copy()
        params.update({
            'sn2_nu_index': 1,    # Cl (1-based)
            'sn2_c_index': 2,     # C (1-based)
            'sn2_lg_index': 6,    # Br (1-based)
            'sn2_target_nu_c_distance': 2.0,
            'sn2_target_c_lg_distance': 2.0,
            'sn2_constraint_strength': 0.1
        })
        return params
    
    def test_initialization(self, mock_model, sn2_system_data, sn2_conformation, sn2_params):
        """Test SN2TransitionState initialization"""
        from fennol.md.transition_state import SN2TransitionState
        
        optimizer = SN2TransitionState(mock_model, sn2_system_data, sn2_conformation, sn2_params, 'float64')
        
        # Check indices are converted to 0-based
        assert optimizer.nu_index == 0  # Cl
        assert optimizer.c_index == 1   # C
        assert optimizer.lg_index == 5  # Br
        assert optimizer.sn2_target_nu_c_distance == 2.0
        assert optimizer.sn2_constraint_strength == 0.1
    
    def test_missing_indices(self, mock_model, sn2_system_data, sn2_conformation, basic_simulation_params):
        """Test that missing SN2 indices raise error"""
        from fennol.md.transition_state import SN2TransitionState
        
        with pytest.raises(ValueError, match="SN2 method requires"):
            SN2TransitionState(mock_model, sn2_system_data, sn2_conformation, basic_simulation_params, 'float64')
    
    def test_reaction_coordinate_calculation(self, mock_model, sn2_system_data, sn2_conformation, sn2_params):
        """Test SN2 reaction coordinate calculation"""
        from fennol.md.transition_state import SN2TransitionState
        
        optimizer = SN2TransitionState(mock_model, sn2_system_data, sn2_conformation, sn2_params, 'float64')
        
        coords = sn2_conformation['coordinates'].reshape(-1)
        rc_value, rc_gradient, nu_c_dist, c_lg_dist = optimizer._compute_sn2_reaction_coordinate(coords)
        
        # Check return types and shapes
        assert isinstance(rc_value, (float, jnp.ndarray))
        assert rc_gradient.shape == coords.shape
        assert nu_c_dist > 0
        assert c_lg_dist > 0
        
        # Check that RC = d(Nu-C) - d(C-LG)
        expected_rc = nu_c_dist - c_lg_dist
        assert jnp.abs(rc_value - expected_rc) < 1e-6
    
    def test_constraint_application(self, mock_model, sn2_system_data, sn2_conformation, sn2_params):
        """Test SN2 constraint application"""
        from fennol.md.transition_state import SN2TransitionState
        
        optimizer = SN2TransitionState(mock_model, sn2_system_data, sn2_conformation, sn2_params, 'float64')
        
        coords = sn2_conformation['coordinates'].reshape(-1)
        forces = jnp.ones_like(coords) * 0.1  # Small uniform forces
        
        constrained_forces, rc_value, nu_c_dist, c_lg_dist = optimizer._apply_sn2_constraints(coords, forces)
        
        assert constrained_forces.shape == forces.shape
        assert isinstance(rc_value, (float, jnp.ndarray))
        assert nu_c_dist > 0
        assert c_lg_dist > 0
        
        # Constraints should modify forces
        assert not jnp.allclose(constrained_forces, forces)
    
    def test_mode_initialization(self, mock_model, sn2_system_data, sn2_conformation, sn2_params):
        """Test SN2 initial mode generation"""
        from fennol.md.transition_state import SN2TransitionState
        
        optimizer = SN2TransitionState(mock_model, sn2_system_data, sn2_conformation, sn2_params, 'float64')
        
        coords = sn2_conformation['coordinates'].reshape(-1)
        initial_mode = optimizer._initialize_sn2_mode(coords)
        
        assert initial_mode.shape == coords.shape
        assert jnp.abs(jnp.linalg.norm(initial_mode) - 1.0) < 1e-6


class TestTSOptimizerFactory:
    """Test transition state optimizer factory functions"""
    
    def test_get_ts_optimizer_quasi_newton(self):
        """Test getting quasi-Newton optimizer"""
        from fennol.md.transition_state import get_ts_optimizer, QuasiNewtonTS
        
        model = Mock()
        model.energy_unit = 'kcal/mol'
        system_data = {'nat': 1, 'symbols': ['H'], 'name': 'test'}
        conformation = {'coordinates': jnp.array([[0., 0., 0.]])}
        params = {'min_force_tolerance': 1e-4, 'min_max_iterations': 100, 'min_print_freq': 10}
        
        optimizer = get_ts_optimizer('quasi_newton', model, system_data, conformation, params, 'float64')
        assert isinstance(optimizer, QuasiNewtonTS)
        
        # Test aliases
        optimizer = get_ts_optimizer('qn', model, system_data, conformation, params, 'float64')
        assert isinstance(optimizer, QuasiNewtonTS)
    
    def test_get_ts_optimizer_dimer(self):
        """Test getting dimer optimizer"""
        from fennol.md.transition_state import get_ts_optimizer, DimerMethod
        
        model = Mock()
        model.energy_unit = 'kcal/mol'
        system_data = {'nat': 1, 'symbols': ['H'], 'name': 'test'}
        conformation = {'coordinates': jnp.array([[0., 0., 0.]])}
        params = {'min_force_tolerance': 1e-4, 'min_max_iterations': 100, 'min_print_freq': 10}
        
        optimizer = get_ts_optimizer('dimer', model, system_data, conformation, params, 'float64')
        assert isinstance(optimizer, DimerMethod)
    
    def test_get_ts_optimizer_sn2(self):
        """Test getting SN2 optimizer"""
        from fennol.md.transition_state import get_ts_optimizer, SN2TransitionState
        
        model = Mock()
        model.energy_unit = 'kcal/mol'
        system_data = {'nat': 8, 'symbols': ['Cl', 'C', 'H', 'H', 'H', 'Br', 'H', 'H'], 'name': 'test'}
        conformation = {'coordinates': jnp.zeros((8, 3))}
        params = {
            'min_force_tolerance': 1e-4, 
            'min_max_iterations': 100, 
            'min_print_freq': 10,
            'sn2_nu_index': 1,
            'sn2_c_index': 2,
            'sn2_lg_index': 6
        }
        
        optimizer = get_ts_optimizer('sn2', model, system_data, conformation, params, 'float64')
        assert isinstance(optimizer, SN2TransitionState)
    
    def test_get_ts_optimizer_invalid_method(self):
        """Test invalid method raises error"""
        from fennol.md.transition_state import get_ts_optimizer
        
        model = Mock()
        system_data = {'nat': 1, 'symbols': ['H'], 'name': 'test'}
        conformation = {'coordinates': jnp.array([[0., 0., 0.]])}
        params = {}
        
        with pytest.raises(ValueError, match="Unknown TS optimization method"):
            get_ts_optimizer('invalid_method', model, system_data, conformation, params, 'float64')


class TestFindTransitionState:
    """Test main transition state finding function"""
    
    def test_find_transition_state_quasi_newton(self):
        """Test find_transition_state with quasi-Newton"""
        from fennol.md.transition_state import find_transition_state
        
        model = Mock()
        model.energy_unit = 'kcal/mol'
        
        def mock_energy_forces(coords, *args):
            # Simple potential
            energy = jnp.sum(coords**2)
            forces = -2 * coords.reshape(-1, 3)
            return jnp.array([energy]), forces, None, None, {}
        
        model._energy_and_forces = mock_energy_forces
        
        system_data = {'nat': 1, 'name': 'test', 'symbols': ['H']}
        conformation = {'coordinates': jnp.array([[0.1, 0.1, 0.1]])}
        params = {
            'ts_method': 'quasi_newton',
            'min_force_tolerance': 1e-2,  # Loose tolerance for test
            'min_max_iterations': 5,
            'min_print_freq': 1
        }
        
        # Mock the _evaluate_energy_forces method for the optimizer
        with patch('fennol.md.transition_state.QuasiNewtonTS._evaluate_energy_forces') as mock_eval:
            mock_eval.return_value = (jnp.array([0.01]), jnp.array([[0.01, 0.01, 0.01]]), None, None, {})
            
            result = find_transition_state(model, system_data, conformation, params, 'float64')
            
            assert 'coordinates' in result
            assert 'energy' in result
            assert 'forces' in result
            assert 'converged' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])