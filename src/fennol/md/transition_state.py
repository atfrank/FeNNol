"""
Transition state searching methods for FeNNol

This module implements various transition state optimization algorithms including:
- Quasi-Newton methods for finding first-order saddle points
- Dimer method
- Nudged Elastic Band (NEB) method
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, List
from functools import partial

from ..utils.io import write_xyz_frame, write_extxyz_frame, write_arc_frame, human_time_duration
from ..utils.atomic_units import AtomicUnits as au
from .minimize import Minimizer
from .energy_formatter import format_energy_for_display


class TransitionStateOptimizer(Minimizer):
    """Base class for transition state optimization algorithms"""
    
    def __init__(self, 
                 model, 
                 system_data: Dict[str, Any], 
                 conformation: Dict[str, Any],
                 simulation_parameters: Dict[str, Any],
                 fprec: str):
        """
        Initialize the transition state optimizer
        
        Args:
            model: The model to use for energy and force calculations
            system_data: Dictionary containing system information
            conformation: Dictionary with atomic coordinates and other data
            simulation_parameters: Dictionary with simulation parameters
            fprec: Precision to use ('float32' or 'float64')
        """
        super().__init__(model, system_data, conformation, simulation_parameters, fprec)
        
        # TS-specific parameters
        self.ts_method = simulation_parameters.get("ts_method", "quasi_newton")
        self.hessian_update_scheme = simulation_parameters.get("ts_hessian_update", "bfgs")
        self.max_uphill_steps = int(simulation_parameters.get("ts_max_uphill_steps", 5))
        self.eigenvalue_tolerance = simulation_parameters.get("ts_eigenvalue_tolerance", 1e-4)
        self.trust_radius = simulation_parameters.get("ts_trust_radius", 0.3)
        
        # For tracking negative eigenvalues
        self.target_n_negative = 1  # First-order saddle point
        
    def print_header(self):
        """Print the transition state optimization header"""
        print("#" + "=" * 78)
        print(f"# Starting transition state optimization using {self.__class__.__name__}")
        
        # Get model energy unit and determine if displaying per-atom energy
        energy_unit_str = self.params.get("energy_unit", "kcal/mol")
        per_atom_energy = self.params.get("per_atom_energy", True)
        nat = self.system_data["nat"]
        atom_energy_unit_str = energy_unit_str
        
        if per_atom_energy:
            atom_energy_unit_str = f"{energy_unit_str}/atom"
            print(f"# Energy displayed per atom in {energy_unit_str}/atom")
        else:
            print(f"# Energy displayed in {energy_unit_str}")
            
        print(f"# Number of atoms: {nat}")
        print(f"# Maximum iterations: {self.max_iterations}")
        print(f"# Force tolerance: {self.force_tolerance}")
        print(f"# Eigenvalue tolerance: {self.eigenvalue_tolerance}")
        print(f"# Trust radius: {self.trust_radius} Å")
        print(f"# Target number of negative eigenvalues: {self.target_n_negative}")
        print("#" + "=" * 78)
        print(f"# Iter      Energy[{atom_energy_unit_str}]   Max Force     RMS Force    N_Neg_Eval     Time/step")
        print("#" + "-" * 78)
        
    def print_step(self, iteration, energy, forces, n_negative_eigenvalues=None, step_time=None):
        """Print information about the current TS optimization step"""
        if iteration % self.print_freq != 0 and iteration != 1:
            return
        
        # Format energy for display
        display_energy, _ = self._format_energy(energy)
            
        max_force = jnp.max(jnp.abs(forces))
        rms_force = jnp.sqrt(jnp.mean(forces**2))
        
        line = f"# {iteration:4d}  {display_energy:14.6f}  {max_force:12.6f}  {rms_force:12.6f}"
        
        if n_negative_eigenvalues is not None:
            line += f"  {n_negative_eigenvalues:12d}"
        else:
            line += "              "
            
        if step_time is not None:
            line += f"  {step_time:10.4f}s"
            
        print(line)


class QuasiNewtonTS(TransitionStateOptimizer):
    """
    Quasi-Newton transition state optimizer
    
    Uses a quasi-Newton approach with Hessian updates to find first-order saddle points.
    The algorithm maximizes along the mode with the most negative eigenvalue while
    minimizing along all other modes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian = None
        self.prev_coords = None
        self.prev_gradient = None
        
    def _initialize_hessian(self, coords):
        """Initialize the Hessian matrix"""
        n = len(coords)
        
        # Start with a scaled identity matrix
        # Negative diagonal for initial uphill direction
        initial_scale = self.params.get("ts_initial_hessian_scale", -0.1)
        self.hessian = jnp.eye(n) * initial_scale
        
    def _update_hessian(self, coords, gradient):
        """Update the Hessian using BFGS or SR1 update scheme"""
        if self.prev_coords is None or self.prev_gradient is None:
            self.prev_coords = coords
            self.prev_gradient = gradient
            return
            
        # Calculate differences
        s = coords - self.prev_coords  # Step
        y = gradient - self.prev_gradient  # Gradient difference
        
        # Check if update is numerically stable
        sy = jnp.dot(s, y)
        
        if self.hessian_update_scheme.lower() == "bfgs":
            # BFGS update for transition states
            if jnp.abs(sy) > 1e-8:
                Hs = jnp.dot(self.hessian, s)
                sHs = jnp.dot(s, Hs)
                
                # BFGS formula
                self.hessian = (self.hessian + 
                               jnp.outer(y, y) / sy - 
                               jnp.outer(Hs, Hs) / sHs)
        
        elif self.hessian_update_scheme.lower() == "sr1":
            # SR1 update (Symmetric Rank-1)
            Hs = jnp.dot(self.hessian, s)
            v = y - Hs
            vTs = jnp.dot(v, s)
            
            if jnp.abs(vTs) > 1e-8 * jnp.linalg.norm(v) * jnp.linalg.norm(s):
                self.hessian = self.hessian + jnp.outer(v, v) / vTs
                
        # Update stored values
        self.prev_coords = coords
        self.prev_gradient = gradient
        
    def _compute_search_direction(self, gradient, eigenvalues, eigenvectors):
        """
        Compute the search direction for TS optimization
        
        For a first-order saddle point:
        - Maximize along the eigenvector with the most negative eigenvalue
        - Minimize along all other eigenvectors
        """
        # Sort eigenvalues and eigenvectors
        idx = jnp.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project gradient onto eigenvector basis
        gradient_eigen = jnp.dot(eigenvectors.T, gradient)
        
        # Compute search direction in eigenvector basis
        search_eigen = jnp.zeros_like(gradient_eigen)
        
        # For the most negative eigenvalue(s), we want to maximize (go uphill)
        # For all others, we want to minimize (go downhill)
        for i in range(len(eigenvalues)):
            if i < self.target_n_negative:
                # Uphill direction for negative eigenvalues
                search_eigen = search_eigen.at[i].set(-gradient_eigen[i])
            else:
                # Downhill direction for positive eigenvalues
                search_eigen = search_eigen.at[i].set(gradient_eigen[i])
                
        # Transform back to Cartesian coordinates
        search_direction = jnp.dot(eigenvectors, search_eigen)
        
        return search_direction, eigenvalues
        
    def _trust_radius_step(self, search_direction):
        """Apply trust radius constraint to the step"""
        step_norm = jnp.linalg.norm(search_direction)
        
        if step_norm > self.trust_radius:
            # Scale step to trust radius
            search_direction = search_direction * (self.trust_radius / step_norm)
            
        return search_direction
        
    def run(self):
        """Run quasi-Newton transition state optimization"""
        self.print_header()
        
        # Get initial coordinates
        coords = self.conformation["coordinates"]
        if isinstance(coords, jnp.ndarray):
            coords = coords.reshape(-1)
        else:
            coords = jnp.array(coords).reshape(-1)
            
        # Initialize Hessian
        self._initialize_hessian(coords)
        
        # Initial system state
        system = None
        preproc_state = None
        
        # Initial evaluation
        t_start = time.time()
        energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(coords, system, preproc_state)
        gradient = -forces.reshape(-1)  # Gradient is negative of forces
        
        # Print initial state
        energy_val = self._safe_get_energy_value(energy)
        self.print_step(0, energy_val, forces, n_negative_eigenvalues=0)
        self.save_trajectory(coords, 0, energy_val, forces)
        
        # Main optimization loop
        n_uphill_steps = 0
        converged = False
        
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Update Hessian
            self._update_hessian(coords, gradient)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = jnp.linalg.eigh(self.hessian)
            n_negative = jnp.sum(eigenvalues < -self.eigenvalue_tolerance)
            
            # Check convergence
            max_force = jnp.max(jnp.abs(forces))
            if max_force < self.force_tolerance and n_negative == self.target_n_negative:
                print(f"# Convergence achieved: Force magnitude {max_force} < tolerance {self.force_tolerance}")
                print(f"# Number of negative eigenvalues: {n_negative} (target: {self.target_n_negative})")
                converged = True
                break
                
            # Compute search direction
            search_direction, eigenvalues = self._compute_search_direction(gradient, eigenvalues, eigenvectors)
            
            # Apply trust radius
            step = self._trust_radius_step(search_direction)
            
            # Take step
            new_coords = coords + step
            
            # Evaluate at new position
            new_energy, new_forces, new_system, new_preproc_state, new_model_out = self._evaluate_energy_forces(
                new_coords, system, preproc_state
            )
            new_gradient = -new_forces.reshape(-1)
            
            # Check if we should accept the step
            new_energy_val = self._safe_get_energy_value(new_energy)
            energy_val = self._safe_get_energy_value(energy)
            
            # For TS search, we may accept uphill steps
            accept_step = True
            if new_energy_val > energy_val:
                n_uphill_steps += 1
                if n_uphill_steps > self.max_uphill_steps:
                    # Reduce trust radius
                    self.trust_radius *= 0.5
                    accept_step = False
                    n_uphill_steps = 0
            else:
                n_uphill_steps = 0
                
            if accept_step:
                # Update state
                coords = new_coords
                energy = new_energy
                forces = new_forces
                gradient = new_gradient
                system = new_system
                preproc_state = new_preproc_state
                model_out = new_model_out
                
                # Possibly increase trust radius
                if jnp.linalg.norm(step) < 0.8 * self.trust_radius:
                    self.trust_radius = min(self.trust_radius * 1.5, self.max_step)
                    
            # Calculate time per step
            step_time = time.time() - iter_start
            
            # Print status and save trajectory
            energy_val = self._safe_get_energy_value(energy)
            self.print_step(iteration, energy_val, forces, n_negative, step_time)
            
            if iteration % self.print_freq == 0:
                self.save_trajectory(coords, iteration, energy_val, forces)
                
        # Save final structure
        energy_val = self._safe_get_energy_value(energy)
        self.save_trajectory(coords, iteration, energy_val, forces)
        
        # Final statistics
        total_time = time.time() - t_start
        
        print("#" + "-" * 78)
        print(f"# Transition state optimization completed in {iteration} steps")
        print(f"# Converged: {converged}")
        energy_val = self._safe_get_energy_value(energy)
        print(f"# Final energy: {energy_val:.8f}")
        print(f"# Max force magnitude: {jnp.max(jnp.abs(forces)):.8f}")
        print(f"# RMS force: {jnp.sqrt(jnp.mean(forces**2)):.8f}")
        print(f"# Number of negative eigenvalues: {n_negative}")
        if n_negative > 0:
            print(f"# Lowest eigenvalue: {eigenvalues[0]:.6f}")
        print(f"# Total optimization time: {human_time_duration(total_time)}")
        print("#" + "=" * 78)
        
        # Return optimized system state
        return {
            "coordinates": coords.reshape(-1, 3),
            "energy": self._safe_get_energy_value(energy),
            "forces": forces,
            "system": system,
            "model_output": model_out,
            "n_negative_eigenvalues": int(n_negative),
            "lowest_eigenvalue": float(eigenvalues[0]) if n_negative > 0 else None,
            "converged": converged
        }


class DimerMethod(TransitionStateOptimizer):
    """
    Dimer method for finding transition states
    
    The dimer method uses two images separated by a small distance to
    estimate the lowest curvature mode and climb uphill along it while
    minimizing in all orthogonal directions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Dimer-specific parameters
        self.dimer_separation = self.params.get("dimer_separation", 0.01)  # Angstroms
        self.rotation_tolerance = self.params.get("dimer_rotation_tolerance", 0.1)
        self.max_rotation_iterations = int(self.params.get("dimer_max_rotations", 10))
        self.rotation_step_size = self.params.get("dimer_rotation_step", 0.1)
        
        # Initialize dimer orientation randomly or from input
        self.dimer_vector = None
        
    def _initialize_dimer(self, coords):
        """Initialize the dimer orientation"""
        n = len(coords)
        
        # Try to get initial mode from input, otherwise use random
        initial_mode = self.params.get("dimer_initial_mode", None)
        
        if initial_mode is not None:
            self.dimer_vector = jnp.array(initial_mode).reshape(-1)
        else:
            # Random initialization
            key = jax.random.PRNGKey(42)
            self.dimer_vector = jax.random.normal(key, shape=(n,))
            
        # Normalize
        self.dimer_vector = self.dimer_vector / jnp.linalg.norm(self.dimer_vector)
        
    def _rotate_dimer(self, coords, energy_center, forces_center, system, preproc_state):
        """
        Rotate the dimer to find the lowest curvature mode
        
        Returns:
            Tuple of (rotated_dimer_vector, curvature)
        """
        best_vector = self.dimer_vector
        lowest_curvature = float('inf')
        
        for rot_iter in range(self.max_rotation_iterations):
            # Calculate energies at dimer endpoints
            coords1 = coords + self.dimer_separation * self.dimer_vector
            coords2 = coords - self.dimer_separation * self.dimer_vector
            
            # Evaluate energies
            energy1, forces1, _, _, _ = self._evaluate_energy_forces(coords1, system, preproc_state)
            energy2, forces2, _, _, _ = self._evaluate_energy_forces(coords2, system, preproc_state)
            
            # Calculate curvature using finite differences
            energy1_val = self._safe_get_energy_value(energy1)
            energy2_val = self._safe_get_energy_value(energy2)
            energy_center_val = self._safe_get_energy_value(energy_center)
            
            curvature = (energy1_val + energy2_val - 2 * energy_center_val) / (self.dimer_separation ** 2)
            
            # Calculate rotation direction using modified Newton's method
            force_parallel = jnp.dot(forces_center.reshape(-1), self.dimer_vector)
            force1_parallel = jnp.dot(forces1.reshape(-1), self.dimer_vector)
            force2_parallel = jnp.dot(forces2.reshape(-1), self.dimer_vector)
            
            # Perpendicular component of force difference
            force_perp = ((forces1 - forces2).reshape(-1) - 
                         (force1_parallel - force2_parallel) * self.dimer_vector)
            
            # Check convergence
            rotation_force = jnp.linalg.norm(force_perp) / (2 * self.dimer_separation)
            if rotation_force < self.rotation_tolerance:
                break
                
            # Rotation direction
            if jnp.linalg.norm(force_perp) > 1e-10:
                rotation_direction = force_perp / jnp.linalg.norm(force_perp)
                
                # Calculate optimal rotation angle
                theta = self.rotation_step_size
                
                # Rotate dimer vector
                new_vector = (jnp.cos(theta) * self.dimer_vector + 
                             jnp.sin(theta) * rotation_direction)
                self.dimer_vector = new_vector / jnp.linalg.norm(new_vector)
                
            # Update best if this has lower curvature
            if curvature < lowest_curvature:
                lowest_curvature = curvature
                best_vector = self.dimer_vector
                
        self.dimer_vector = best_vector
        return self.dimer_vector, lowest_curvature
        
    def _compute_dimer_forces(self, coords, forces_center):
        """Compute the effective force for dimer translation"""
        # Project out component along dimer direction
        force_parallel = jnp.dot(forces_center.reshape(-1), self.dimer_vector)
        
        # For climbing, reverse the parallel component if curvature is negative
        # This is done after rotation, so we know the curvature
        forces_effective = (forces_center.reshape(-1) - 
                           2 * force_parallel * self.dimer_vector)
        
        return forces_effective.reshape(-1, 3)
        
    def run(self):
        """Run dimer method for transition state search"""
        self.print_header()
        
        # Get initial coordinates
        coords = self.conformation["coordinates"]
        if isinstance(coords, jnp.ndarray):
            coords = coords.reshape(-1)
        else:
            coords = jnp.array(coords).reshape(-1)
            
        # Initialize dimer
        self._initialize_dimer(coords)
        
        # Initial system state
        system = None
        preproc_state = None
        
        # Initial evaluation
        t_start = time.time()
        energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(coords, system, preproc_state)
        
        # Print initial state
        energy_val = self._safe_get_energy_value(energy)
        self.print_step(0, energy_val, forces)
        self.save_trajectory(coords, 0, energy_val, forces)
        
        # Main optimization loop
        converged = False
        step_size = self.initial_step_size
        
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Rotate dimer to find lowest curvature mode
            self.dimer_vector, curvature = self._rotate_dimer(coords, energy, forces, system, preproc_state)
            
            # Compute effective forces for translation
            effective_forces = self._compute_dimer_forces(coords, forces)
            
            # Check convergence
            max_force = jnp.max(jnp.abs(effective_forces))
            if max_force < self.force_tolerance:
                print(f"# Convergence achieved: Effective force magnitude {max_force} < tolerance {self.force_tolerance}")
                print(f"# Lowest curvature: {curvature:.6f}")
                converged = True
                break
                
            # Take step
            force_norm = jnp.linalg.norm(effective_forces.reshape(-1))
            if force_norm > 1e-10:
                step_direction = effective_forces.reshape(-1) / force_norm
                step = step_direction * step_size
                
                # Limit step size
                max_displacement = jnp.max(jnp.abs(step))
                if max_displacement > self.max_step:
                    step = step * (self.max_step / max_displacement)
                    
                new_coords = coords + step
                
                # Evaluate at new position
                new_energy, new_forces, new_system, new_preproc_state, new_model_out = self._evaluate_energy_forces(
                    new_coords, system, preproc_state
                )
                
                # Simple line search - accept if energy increases (climbing)
                new_energy_val = self._safe_get_energy_value(new_energy)
                energy_val = self._safe_get_energy_value(energy)
                
                if curvature < 0:  # We're at a saddle point region
                    # Accept step
                    coords = new_coords
                    energy = new_energy
                    forces = new_forces
                    system = new_system
                    preproc_state = new_preproc_state
                    model_out = new_model_out
                    
                    # Adjust step size
                    if new_energy_val > energy_val:
                        step_size = min(step_size * 1.2, self.max_step)
                    else:
                        step_size = max(step_size * 0.5, 1e-6)
                else:
                    # In positive curvature region, minimize
                    if new_energy_val < energy_val:
                        coords = new_coords
                        energy = new_energy
                        forces = new_forces
                        system = new_system
                        preproc_state = new_preproc_state
                        model_out = new_model_out
                        step_size = min(step_size * 1.2, self.max_step)
                    else:
                        step_size = max(step_size * 0.5, 1e-6)
                        
            # Calculate time per step
            step_time = time.time() - iter_start
            
            # For display, estimate number of negative eigenvalues from curvature
            n_negative = 1 if curvature < -self.eigenvalue_tolerance else 0
            
            # Print status and save trajectory
            energy_val = self._safe_get_energy_value(energy)
            self.print_step(iteration, energy_val, forces, n_negative, step_time)
            
            if iteration % self.print_freq == 0:
                self.save_trajectory(coords, iteration, energy_val, forces)
                
        # Save final structure
        energy_val = self._safe_get_energy_value(energy)
        self.save_trajectory(coords, iteration, energy_val, forces)
        
        # Final statistics
        total_time = time.time() - t_start
        
        print("#" + "-" * 78)
        print(f"# Dimer method completed in {iteration} steps")
        print(f"# Converged: {converged}")
        energy_val = self._safe_get_energy_value(energy)
        print(f"# Final energy: {energy_val:.8f}")
        print(f"# Max force magnitude: {jnp.max(jnp.abs(forces)):.8f}")
        print(f"# Final curvature along dimer: {curvature:.6f}")
        print(f"# Total optimization time: {human_time_duration(total_time)}")
        print("#" + "=" * 78)
        
        # Return optimized system state
        return {
            "coordinates": coords.reshape(-1, 3),
            "energy": self._safe_get_energy_value(energy),
            "forces": forces,
            "system": system,
            "model_output": model_out,
            "lowest_curvature": float(curvature),
            "dimer_orientation": self.dimer_vector.reshape(-1, 3),
            "converged": converged
        }


class SN2TransitionState(TransitionStateOptimizer):
    """
    SN2-specific transition state optimizer
    
    This optimizer is designed specifically for SN2 reactions where the user
    provides the indices of the nucleophile (Nu), carbon center (C), and
    leaving group (LG). The method uses these indices to define the reaction
    coordinate and applies specialized constraints and initial modes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get SN2-specific parameters
        self.nu_index = self.params.get("sn2_nu_index", None)
        self.c_index = self.params.get("sn2_c_index", None)
        self.lg_index = self.params.get("sn2_lg_index", None)
        
        # Validate indices
        if self.nu_index is None or self.c_index is None or self.lg_index is None:
            raise ValueError("SN2 method requires sn2_nu_index, sn2_c_index, and sn2_lg_index")
            
        # Convert to 0-based indexing if necessary
        if isinstance(self.nu_index, int):
            self.nu_index -= 1  # Assume 1-based input
        if isinstance(self.c_index, int):
            self.c_index -= 1
        if isinstance(self.lg_index, int):
            self.lg_index -= 1
            
        # SN2-specific parameters
        self.sn2_constraint_strength = self.params.get("sn2_constraint_strength", 0.1)
        self.sn2_target_nu_c_distance = self.params.get("sn2_target_nu_c_distance", 2.0)  # Angstroms
        self.sn2_target_c_lg_distance = self.params.get("sn2_target_c_lg_distance", 2.0)  # Angstroms
        self.sn2_reaction_coordinate_weight = self.params.get("sn2_reaction_coordinate_weight", 1.0)
        
        print(f"# SN2 TS optimization:")
        print(f"#   Nucleophile index: {self.nu_index + 1}")
        print(f"#   Carbon index: {self.c_index + 1}")
        print(f"#   Leaving group index: {self.lg_index + 1}")
        print(f"#   Target Nu-C distance: {self.sn2_target_nu_c_distance:.2f} Å")
        print(f"#   Target C-LG distance: {self.sn2_target_c_lg_distance:.2f} Å")
        
    def _compute_sn2_reaction_coordinate(self, coords):
        """
        Compute the SN2 reaction coordinate and its derivatives
        
        RC = d(Nu-C) - d(C-LG)
        
        For a perfect SN2 TS, RC should be close to 0
        """
        coords_3d = coords.reshape(-1, 3)
        
        # Get positions
        nu_pos = coords_3d[self.nu_index]
        c_pos = coords_3d[self.c_index]
        lg_pos = coords_3d[self.lg_index]
        
        # Calculate distances
        nu_c_vec = nu_pos - c_pos
        c_lg_vec = c_pos - lg_pos
        
        nu_c_dist = jnp.linalg.norm(nu_c_vec)
        c_lg_dist = jnp.linalg.norm(c_lg_vec)
        
        # Reaction coordinate
        reaction_coord = nu_c_dist - c_lg_dist
        
        # Derivatives
        drc_dnu = nu_c_vec / nu_c_dist
        drc_dc = -nu_c_vec / nu_c_dist + c_lg_vec / c_lg_dist
        drc_dlg = -c_lg_vec / c_lg_dist
        
        # Build full gradient
        gradient = jnp.zeros_like(coords_3d)
        gradient = gradient.at[self.nu_index].set(drc_dnu)
        gradient = gradient.at[self.c_index].set(drc_dc)
        gradient = gradient.at[self.lg_index].set(drc_dlg)
        
        return reaction_coord, gradient.reshape(-1), nu_c_dist, c_lg_dist
        
    def _apply_sn2_constraints(self, coords, forces):
        """
        Apply SN2-specific constraints to guide the optimization
        """
        rc_value, rc_gradient, nu_c_dist, c_lg_dist = self._compute_sn2_reaction_coordinate(coords)
        
        # Constraint forces to maintain reasonable geometry
        constraint_forces = jnp.zeros_like(forces)
        
        # Constraint 1: Keep Nu-C distance reasonable
        if nu_c_dist > self.sn2_target_nu_c_distance:
            excess = nu_c_dist - self.sn2_target_nu_c_distance
            constraint_strength = self.sn2_constraint_strength * excess
            
            coords_3d = coords.reshape(-1, 3)
            nu_c_vec = coords_3d[self.nu_index] - coords_3d[self.c_index]
            nu_c_unit = nu_c_vec / jnp.linalg.norm(nu_c_vec)
            
            # Pull nucleophile toward carbon
            constraint_forces_3d = constraint_forces.reshape(-1, 3)
            constraint_forces_3d = constraint_forces_3d.at[self.nu_index].add(-constraint_strength * nu_c_unit)
            constraint_forces_3d = constraint_forces_3d.at[self.c_index].add(constraint_strength * nu_c_unit)
            constraint_forces = constraint_forces_3d.reshape(-1)
        
        # Constraint 2: Keep C-LG distance reasonable
        if c_lg_dist > self.sn2_target_c_lg_distance:
            excess = c_lg_dist - self.sn2_target_c_lg_distance
            constraint_strength = self.sn2_constraint_strength * excess
            
            coords_3d = coords.reshape(-1, 3)
            c_lg_vec = coords_3d[self.c_index] - coords_3d[self.lg_index]
            c_lg_unit = c_lg_vec / jnp.linalg.norm(c_lg_vec)
            
            # Pull carbon toward leaving group
            constraint_forces_3d = constraint_forces.reshape(-1, 3)
            constraint_forces_3d = constraint_forces_3d.at[self.c_index].add(-constraint_strength * c_lg_unit)
            constraint_forces_3d = constraint_forces_3d.at[self.lg_index].add(constraint_strength * c_lg_unit)
            constraint_forces = constraint_forces_3d.reshape(-1)
        
        return forces + constraint_forces, rc_value, nu_c_dist, c_lg_dist
        
    def _initialize_sn2_mode(self, coords):
        """Initialize search direction based on SN2 reaction coordinate"""
        _, rc_gradient, _, _ = self._compute_sn2_reaction_coordinate(coords)
        
        # The reaction coordinate gradient gives us the direction of the reaction
        # Normalize it to get the initial search direction
        rc_norm = jnp.linalg.norm(rc_gradient)
        if rc_norm > 1e-10:
            initial_mode = rc_gradient / rc_norm
        else:
            # Fallback to random if RC gradient is zero
            key = jax.random.PRNGKey(42)
            initial_mode = jax.random.normal(key, shape=coords.shape)
            initial_mode = initial_mode / jnp.linalg.norm(initial_mode)
            
        return initial_mode
        
    def run(self):
        """Run SN2-specific transition state optimization"""
        self.print_header()
        
        # Get initial coordinates
        coords = self.conformation["coordinates"]
        if isinstance(coords, jnp.ndarray):
            coords = coords.reshape(-1)
        else:
            coords = jnp.array(coords).reshape(-1)
            
        # Initial system state
        system = None
        preproc_state = None
        
        # Check initial geometry
        rc_value, _, nu_c_dist, c_lg_dist = self._compute_sn2_reaction_coordinate(coords)
        print(f"# Initial SN2 geometry:")
        print(f"#   Nu-C distance: {nu_c_dist:.3f} Å")
        print(f"#   C-LG distance: {c_lg_dist:.3f} Å") 
        print(f"#   Reaction coordinate: {rc_value:.3f}")
        
        # Initial evaluation
        t_start = time.time()
        energy, forces, system, preproc_state, model_out = self._evaluate_energy_forces(coords, system, preproc_state)
        
        # Apply SN2 constraints
        forces, rc_value, nu_c_dist, c_lg_dist = self._apply_sn2_constraints(coords, forces)
        
        # Initialize search direction based on reaction coordinate
        search_direction = self._initialize_sn2_mode(coords)
        
        # Print initial state
        energy_val = self._safe_get_energy_value(energy)
        self.print_step(0, energy_val, forces)
        self.save_trajectory(coords, 0, energy_val, forces)
        
        # Variables for adaptive step size
        step_size = self.params.get("sn2_initial_step", 0.05)
        max_step = self.params.get("min_max_step", 0.2)
        
        # Main optimization loop
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Check convergence
            max_force = jnp.max(jnp.abs(forces))
            if max_force < self.force_tolerance and abs(rc_value) < 0.1:
                print(f"# Convergence achieved:")
                print(f"#   Force magnitude: {max_force:.6f} < {self.force_tolerance}")
                print(f"#   Reaction coordinate: {rc_value:.6f}")
                converged = True
                break
                
            # Compute step direction
            # For SN2, we want to follow the reaction coordinate while minimizing perpendicular forces
            forces_flat = forces.reshape(-1)
            rc_value, rc_gradient, nu_c_dist, c_lg_dist = self._compute_sn2_reaction_coordinate(coords)
            
            # Project forces onto and perpendicular to reaction coordinate
            rc_norm = jnp.linalg.norm(rc_gradient)
            if rc_norm > 1e-10:
                rc_unit = rc_gradient / rc_norm
                force_parallel = jnp.dot(forces_flat, rc_unit)
                force_perpendicular = forces_flat - force_parallel * rc_unit
                
                # Step: minimize perpendicular forces, follow RC if needed
                if abs(rc_value) > 0.1:
                    # Far from TS, move along RC toward TS
                    rc_step = -jnp.sign(rc_value) * self.sn2_reaction_coordinate_weight * rc_unit
                else:
                    # Close to TS, minimize perpendicular forces
                    rc_step = jnp.zeros_like(rc_unit)
                    
                step_direction = force_perpendicular / jnp.maximum(jnp.linalg.norm(force_perpendicular), 1e-10) + rc_step
            else:
                # Fallback to steepest descent
                step_direction = forces_flat / jnp.maximum(jnp.linalg.norm(forces_flat), 1e-10)
            
            # Apply step size
            step = step_direction * step_size
            
            # Limit step size
            step_norm = jnp.linalg.norm(step)
            if step_norm > max_step:
                step = step * (max_step / step_norm)
                
            # Take step
            new_coords = coords + step
            
            # Evaluate at new position
            new_energy, new_forces, new_system, new_preproc_state, new_model_out = self._evaluate_energy_forces(
                new_coords, system, preproc_state
            )
            
            # Apply constraints
            new_forces, new_rc_value, new_nu_c_dist, new_c_lg_dist = self._apply_sn2_constraints(new_coords, new_forces)
            
            # Accept step if energy decreases or constraint satisfaction improves
            new_energy_val = self._safe_get_energy_value(new_energy)
            energy_val = self._safe_get_energy_value(energy)
            
            constraint_improvement = abs(new_rc_value) < abs(rc_value)
            energy_improvement = new_energy_val < energy_val
            
            if energy_improvement or constraint_improvement:
                # Accept step
                coords = new_coords
                energy = new_energy
                forces = new_forces
                system = new_system
                preproc_state = new_preproc_state
                model_out = new_model_out
                rc_value = new_rc_value
                nu_c_dist = new_nu_c_dist
                c_lg_dist = new_c_lg_dist
                
                # Increase step size
                step_size = min(step_size * 1.1, max_step)
            else:
                # Reject step, reduce step size
                step_size = max(step_size * 0.5, 1e-6)
                
            # Calculate time per step
            step_time = time.time() - iter_start
            
            # Print status
            energy_val = self._safe_get_energy_value(energy)
            if iteration % self.print_freq == 0:
                self.print_step(iteration, energy_val, forces, step_time=step_time)
                print(f"#     RC: {rc_value:.4f}, Nu-C: {nu_c_dist:.3f}, C-LG: {c_lg_dist:.3f}")
                
            if iteration % self.print_freq == 0:
                self.save_trajectory(coords, iteration, energy_val, forces)
                
        # Save final structure
        energy_val = self._safe_get_energy_value(energy)
        self.save_trajectory(coords, iteration, energy_val, forces)
        
        # Final statistics
        total_time = time.time() - t_start
        
        print("#" + "-" * 78)
        print(f"# SN2 TS optimization completed in {iteration} steps")
        print(f"# Converged: {converged}")
        print(f"# Final energy: {energy_val:.8f}")
        print(f"# Max force magnitude: {jnp.max(jnp.abs(forces)):.8f}")
        print(f"# RMS force: {jnp.sqrt(jnp.mean(forces**2)):.8f}")
        print(f"# Final reaction coordinate: {rc_value:.6f}")
        print(f"# Final Nu-C distance: {nu_c_dist:.3f} Å")
        print(f"# Final C-LG distance: {c_lg_dist:.3f} Å")
        print(f"# Total optimization time: {human_time_duration(total_time)}")
        print("#" + "=" * 78)
        
        # Return optimized system state
        return {
            "coordinates": coords.reshape(-1, 3),
            "energy": self._safe_get_energy_value(energy),
            "forces": forces,
            "system": system,
            "model_output": model_out,
            "reaction_coordinate": float(rc_value),
            "nu_c_distance": float(nu_c_dist),
            "c_lg_distance": float(c_lg_dist),
            "converged": converged
        }


def get_ts_optimizer(method, model, system_data, conformation, simulation_parameters, fprec):
    """Factory function to create appropriate TS optimizer"""
    method = method.lower()
    
    if method in ["quasi_newton", "qn", "eigenvector_following", "ef"]:
        return QuasiNewtonTS(model, system_data, conformation, simulation_parameters, fprec)
    elif method in ["dimer", "dimer_method"]:
        return DimerMethod(model, system_data, conformation, simulation_parameters, fprec)
    elif method in ["sn2", "sn2_ts"]:
        return SN2TransitionState(model, system_data, conformation, simulation_parameters, fprec)
    else:
        raise ValueError(f"Unknown TS optimization method: {method}. Available methods: quasi_newton, dimer, sn2")


def find_transition_state(model, system_data, conformation, simulation_parameters, fprec):
    """
    Main entry point for transition state optimization
    
    Args:
        model: The model to use for energy and force calculations
        system_data: Dictionary containing system information
        conformation: Dictionary with atomic coordinates and other data
        simulation_parameters: Dictionary with simulation parameters
        fprec: Precision to use ('float32' or 'float64')
        
    Returns:
        Dictionary with TS optimization results
    """
    # Get TS method
    method = simulation_parameters.get("ts_method", "quasi_newton")
    
    # Create appropriate optimizer
    optimizer = get_ts_optimizer(method, model, system_data, conformation, simulation_parameters, fprec)
    
    # Run optimization
    result = optimizer.run()
    
    return result