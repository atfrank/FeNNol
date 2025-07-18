#!/usr/bin/env python3
"""
Command-line interface for structure refinement using FeNNol with enhanced sampling
"""

import sys
import os
import io
import argparse
import time
from pathlib import Path
import math

import jax
import numpy as np
import jax.numpy as jnp

from ..utils.input_parser import parse_input
from ..utils.io import read_pdb, read_xyz, write_xyz_frame, write_pdb_frame
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX
from .initial import load_model, load_system_data, initialize_preprocessing
from .minimize import Minimizer, get_minimizer
from .restraints import rmsd_restraint_force
from .topology import (
    detect_molecular_topology, 
    calculate_bond_restraint_forces, 
    check_covalent_integrity
)


def build_parameters_from_cli(args):
    """Build simulation parameters dictionary from CLI arguments"""
    
    # Determine output prefix from XYZ filename if not provided
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = Path(args.xyz).stem
    
    # Base parameters
    params = {
        # Model and system
        "model_file": args.model,
        "pdb_file": args.pdb,
        "xyz_file": args.xyz,
        
        # Refinement method
        "refinement_method": args.method,
        
        # Enhanced sampling parameters
        "temperature_schedule": args.temperature_schedule,
        "initial_temperature": args.initial_temperature,
        "final_temperature": args.final_temperature,
        "temperature_steps": args.temperature_steps,
        
        # Minimization parameters
        "min_max_iterations": args.max_iterations,
        "min_force_tolerance": args.force_tolerance,
        "min_energy_tolerance": args.energy_tolerance,
        "min_displacement_tolerance": args.displacement_tolerance,
        "min_max_step": args.max_step,
        "min_print_freq": args.print_freq,
        
        # RMSD restraint parameters
        "use_rmsd_restraint": args.rmsd_restraint,
        "rmsd_force_constant": args.rmsd_force_constant,
        "rmsd_target": args.rmsd_target,
        "rmsd_atom_selection": args.rmsd_atoms,
        
        # Clash removal parameters
        "clash_detection_cutoff": args.clash_cutoff,
        "clash_force_constant": args.clash_force_constant,
        "clash_removal_iterations": args.clash_iterations,
        
        # Covalent preservation parameters
        "preserve_covalent_bonds": args.preserve_covalent_bonds,
        "bond_force_constant": args.bond_force_constant,
        "angle_force_constant": args.angle_force_constant,
        "bond_types_to_preserve": args.bond_types_to_preserve,
        "max_bond_deviation": args.max_bond_deviation,
        "detect_topology": args.detect_topology,
        
        # Device and precision
        "device": args.device or "cpu",
        "enable_x64": args.double_precision,
        "matmul_prec": args.matmul_precision,
        
        # Output options
        "output_prefix": output_prefix,
        "write_trajectory": args.write_trajectory,
    }
    
    # Method-specific parameters
    if args.method == "simulated_annealing":
        params.update({
            "annealing_cycles": args.annealing_cycles,
            "annealing_hold_time": args.annealing_hold_time,
        })
    
    elif args.method == "monte_carlo":
        params.update({
            "mc_displacement": args.mc_displacement,
            "mc_acceptance_ratio": args.mc_acceptance_ratio,
        })
    
    return params


class StructureRefiner:
    """Enhanced structure refinement with clash removal and interaction optimization"""
    
    def __init__(self, model, system_data, conformation, simulation_parameters, fprec):
        self.model = model
        self.system_data = system_data
        self.conformation = conformation
        self.params = simulation_parameters
        self.fprec = fprec
        
        # Extract parameters
        self.nat = system_data["nat"]
        self.output_prefix = simulation_parameters.get("output_prefix", "refined")
        self.method = simulation_parameters.get("refinement_method", "simulated_annealing")
        
        # RMSD restraint parameters
        self.use_rmsd = simulation_parameters.get("use_rmsd_restraint", True)
        self.rmsd_fc = simulation_parameters.get("rmsd_force_constant", 10.0)
        self.rmsd_target = simulation_parameters.get("rmsd_target", 0.0)
        self.reference_coords = conformation["coordinates"].copy()
        
        # Parse atom selection for RMSD
        atom_selection = simulation_parameters.get("rmsd_atom_selection", "all")
        if atom_selection == "all":
            self.rmsd_atoms = None
        elif atom_selection == "backbone":
            # Select backbone atoms (CA, C, N, O for proteins)
            backbone_names = ["CA", "C", "N", "O", "P", "O5'", "O3'", "C5'", "C3'"]
            self.rmsd_atoms = self._select_atoms_by_name(backbone_names)
        elif atom_selection == "heavy":
            # Select all non-hydrogen atoms
            self.rmsd_atoms = self._select_heavy_atoms()
        else:
            # Parse custom selection (e.g., "1-100,150-200")
            self.rmsd_atoms = self._parse_atom_selection(atom_selection)
        
        # Clash detection parameters
        self.clash_cutoff = simulation_parameters.get("clash_detection_cutoff", 2.0)
        self.clash_fc = simulation_parameters.get("clash_force_constant", 100.0)
        
        # Covalent preservation parameters
        self.preserve_covalent = simulation_parameters.get("preserve_covalent_bonds", True)
        self.bond_fc = simulation_parameters.get("bond_force_constant", 1000.0)
        self.angle_fc = simulation_parameters.get("angle_force_constant", 100.0)
        self.bond_types_to_preserve = simulation_parameters.get("bond_types_to_preserve", ["backbone", "inter_residue"])
        self.max_bond_deviation = simulation_parameters.get("max_bond_deviation", 0.5)
        self.detect_topology = simulation_parameters.get("detect_topology", True)
        
        # Temperature schedule for simulated annealing
        self.initial_temp = simulation_parameters.get("initial_temperature", 600.0)
        self.final_temp = simulation_parameters.get("final_temperature", 10.0)
        self.temp_steps = simulation_parameters.get("temperature_steps", 100)
        self.temp_schedule = simulation_parameters.get("temperature_schedule", "exponential")
        
        # Output options
        self.write_trajectory = simulation_parameters.get("write_trajectory", True)
        
        # Initialize trajectory files
        if self.write_trajectory:
            self._init_trajectory_files()
        
        # Detect molecular topology for covalent preservation
        self.topology = None
        if self.preserve_covalent and self.detect_topology:
            print("# Detecting molecular topology...")
            self.topology = detect_molecular_topology(self.system_data, self.conformation["coordinates"])
            if hasattr(self.topology, 'print_topology_summary'):
                self.topology.print_topology_summary()
    
    def _select_atoms_by_name(self, atom_names):
        """Select atoms by their names"""
        selected = []
        for i, atom in enumerate(self.system_data.get("atoms", [])):
            if atom.get("name", "") in atom_names:
                selected.append(i)
        return np.array(selected, dtype=np.int32) if selected else None
    
    def _select_heavy_atoms(self):
        """Select all non-hydrogen atoms"""
        selected = []
        for i, symbol in enumerate(self.system_data["symbols"]):
            if symbol != "H":
                selected.append(i)
        return np.array(selected, dtype=np.int32)
    
    def _parse_atom_selection(self, selection_str):
        """Parse atom selection string like '1-100,150-200'"""
        selected = []
        for part in selection_str.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                selected.extend(range(start-1, end))  # Convert to 0-based
            else:
                selected.append(int(part) - 1)  # Convert to 0-based
        return np.array(selected, dtype=np.int32)
    
    def _init_trajectory_files(self):
        """Initialize trajectory output files"""
        # Always create both PDB and XYZ trajectory files
        self.traj_pdb_file = open(f"{self.output_prefix}_traj.pdb", "w")
        self.traj_xyz_file = open(f"{self.output_prefix}_traj.xyz", "w")
        
        # Multi-model files (same as trajectory but with MODEL/ENDMDL records)
        self.multimodel_pdb_file = open(f"{self.output_prefix}_multimodel.pdb", "w")
        self.multimodel_xyz_file = open(f"{self.output_prefix}_multimodel.xyz", "w")
    
    def _detect_clashes(self, coordinates):
        """Detect atomic clashes based on distance cutoff"""
        # Calculate pairwise distances
        dist_matrix = jnp.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], 
            axis=-1
        )
        
        # Mask out diagonal and use upper triangle
        mask = jnp.triu(jnp.ones_like(dist_matrix, dtype=bool), k=1)
        clash_mask = (dist_matrix < self.clash_cutoff) & mask
        
        return clash_mask, dist_matrix
    
    def _apply_clash_removal_forces(self, coordinates, forces):
        """Apply repulsive forces to remove clashes"""
        clash_mask, dist_matrix = self._detect_clashes(coordinates)
        
        # Calculate repulsive forces for clashing atoms
        if jnp.any(clash_mask):
            # Get indices of clashing pairs
            i_indices, j_indices = jnp.where(clash_mask)
            
            for i, j in zip(i_indices, j_indices):
                # Calculate repulsive force
                r_vec = coordinates[i] - coordinates[j]
                r_dist = dist_matrix[i, j]
                
                # Soft-core repulsion to avoid singularities
                safe_dist = jnp.maximum(r_dist, 0.1)
                force_mag = self.clash_fc * (self.clash_cutoff - r_dist) / safe_dist
                
                # Apply equal and opposite forces
                force_vec = force_mag * r_vec / safe_dist
                forces = forces.at[i].add(force_vec)
                forces = forces.at[j].add(-force_vec)
        
        return forces, jnp.sum(clash_mask)
    
    def _get_temperature_schedule(self):
        """Generate temperature schedule for simulated annealing"""
        if self.temp_schedule == "linear":
            temps = np.linspace(self.initial_temp, self.final_temp, self.temp_steps)
        elif self.temp_schedule == "exponential":
            temps = np.logspace(
                np.log10(self.initial_temp), 
                np.log10(self.final_temp), 
                self.temp_steps
            )
        elif self.temp_schedule == "cosine":
            # Cosine annealing schedule
            t = np.linspace(0, np.pi, self.temp_steps)
            temps = self.final_temp + (self.initial_temp - self.final_temp) * (1 + np.cos(t)) / 2
        else:
            raise ValueError(f"Unknown temperature schedule: {self.temp_schedule}")
        
        return temps
    
    def refine(self):
        """Main refinement routine"""
        print("#" + "=" * 60)
        print("# STRUCTURE REFINEMENT")
        print("#" + "=" * 60)
        print(f"# Method: {self.method}")
        print(f"# Number of atoms: {self.nat}")
        print(f"# RMSD restraint: {'ON' if self.use_rmsd else 'OFF'}")
        if self.use_rmsd:
            atom_desc = "all" if self.rmsd_atoms is None else f"{len(self.rmsd_atoms)} atoms"
            print(f"#   - Target RMSD: {self.rmsd_target:.3f} Å")
            print(f"#   - Force constant: {self.rmsd_fc:.1f}")
            print(f"#   - Atom selection: {atom_desc}")
        print(f"# Clash removal cutoff: {self.clash_cutoff:.2f} Å")
        print(f"# Covalent preservation: {'ON' if self.preserve_covalent else 'OFF'}")
        if self.preserve_covalent:
            print(f"#   - Bond force constant: {self.bond_fc:.1f}")
            print(f"#   - Angle force constant: {self.angle_fc:.1f}")
            print(f"#   - Max bond deviation: {self.max_bond_deviation:.2f} Å")
            if self.topology:
                stats = self.topology.get_bond_statistics()
                print(f"#   - Detected bonds: {stats.get('total', 0)}")
                print(f"#   - Critical bonds: {len(self.topology.get_critical_bonds())}")
        print("#" + "=" * 60)
        
        start_time = time.time()
        
        if self.method == "simulated_annealing":
            final_coords = self._simulated_annealing()
        elif self.method == "monte_carlo":
            final_coords = self._monte_carlo_refinement()
        elif self.method == "gradient_descent":
            final_coords = self._gradient_descent_refinement()
        else:
            raise ValueError(f"Unknown refinement method: {self.method}")
        
        # Final minimization at low temperature
        print("\n# Final minimization...")
        final_coords = self._final_minimization(final_coords)
        
        # Calculate final RMSD
        if self.use_rmsd:
            final_rmsd = self._calculate_rmsd(final_coords)
            print(f"\n# Final RMSD from reference: {final_rmsd:.3f} Å")
        
        # Check for remaining clashes
        clash_mask, _ = self._detect_clashes(final_coords)
        n_clashes = jnp.sum(clash_mask)
        print(f"# Remaining clashes: {n_clashes}")
        
        # Check covalent integrity
        if self.preserve_covalent and self.topology:
            integrity_results = check_covalent_integrity(
                final_coords, self.topology, self.max_bond_deviation
            )
            print(f"# Covalent integrity: {'INTACT' if integrity_results['intact'] else 'COMPROMISED'}")
            if not integrity_results['intact']:
                print(f"#   - Broken bonds: {len(integrity_results['broken_bonds'])}")
                print(f"#   - Stretched bonds: {len(integrity_results['stretched_bonds'])}")
                print(f"#   - Compressed bonds: {len(integrity_results['compressed_bonds'])}")
                print(f"#   - Max deviation: {integrity_results['max_deviation']:.3f} Å")
        
        # Save final structure
        self._save_final_structure(final_coords)
        
        total_time = time.time() - start_time
        print(f"\n# Total refinement time: {total_time:.2f} seconds")
        print("#" + "=" * 60)
        
        # Close trajectory files
        if self.write_trajectory:
            self.traj_pdb_file.close()
            self.traj_xyz_file.close()
            self.multimodel_pdb_file.close()
            self.multimodel_xyz_file.close()
        
        return final_coords
    
    def _simulated_annealing(self):
        """Perform simulated annealing refinement"""
        print("\n# Starting simulated annealing...")
        
        temps = self._get_temperature_schedule()
        coordinates = self.conformation["coordinates"].copy()
        
        # Number of iterations per temperature
        n_cycles = self.params.get("annealing_cycles", 10)
        hold_time = self.params.get("annealing_hold_time", 10)
        
        step = 0
        for i, temp in enumerate(temps):
            print(f"\n# Temperature step {i+1}/{len(temps)}: T = {temp:.1f} K")
            
            # Update temperature for force calculations
            kT = temp * 0.0019872041  # Convert to kcal/mol
            
            for cycle in range(n_cycles):
                # Calculate forces with RMSD restraint and clash removal
                energy, forces = self._calculate_forces(coordinates)
                
                # Add thermal noise
                noise = np.random.normal(0, np.sqrt(kT), coordinates.shape)
                
                # Update positions with damping
                damping = 0.95
                dt = 0.001  # Small timestep
                coordinates += dt * (forces + noise) * damping
                
                # Write trajectory frame
                if self.write_trajectory and step % 10 == 0:
                    self._write_trajectory_frame(coordinates, energy, temp, step)
                
                step += 1
                
                # Print progress
                if cycle % max(1, n_cycles // 5) == 0:
                    rmsd = self._calculate_rmsd(coordinates) if self.use_rmsd else 0.0
                    print(f"  Cycle {cycle}: E = {float(energy.item()):.3f}, RMSD = {rmsd:.3f}")
        
        return coordinates
    
    def _monte_carlo_refinement(self):
        """Perform Monte Carlo refinement"""
        print("\n# Starting Monte Carlo refinement...")
        
        coordinates = self.conformation["coordinates"].copy()
        displacement = self.params.get("mc_displacement", 0.1)
        target_acceptance = self.params.get("mc_acceptance_ratio", 0.5)
        
        # Temperature schedule
        temps = self._get_temperature_schedule()
        
        n_accepted = 0
        n_total = 0
        
        for i, temp in enumerate(temps):
            kT = temp * 0.0019872041  # Convert to kcal/mol
            
            # Adjust displacement to maintain acceptance ratio
            if i > 0 and i % 100 == 0:
                acceptance_ratio = n_accepted / max(1, n_total)
                if acceptance_ratio < target_acceptance - 0.1:
                    displacement *= 0.9
                elif acceptance_ratio > target_acceptance + 0.1:
                    displacement *= 1.1
                print(f"  Acceptance ratio: {acceptance_ratio:.3f}, displacement: {displacement:.3f}")
            
            # MC cycles at this temperature
            for cycle in range(100):
                # Pick random atom
                atom = np.random.randint(0, self.nat)
                
                # Save old position
                old_pos = coordinates[atom].copy()
                old_energy, _ = self._calculate_forces(coordinates)
                
                # Make random displacement
                coordinates = coordinates.at[atom].add(
                    np.random.normal(0, displacement, 3)
                )
                
                # Calculate new energy
                new_energy, _ = self._calculate_forces(coordinates)
                
                # Metropolis criterion
                delta_E = new_energy - old_energy
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / kT):
                    n_accepted += 1
                else:
                    # Reject move
                    coordinates = coordinates.at[atom].set(old_pos)
                
                n_total += 1
                
                # Write trajectory
                if self.write_trajectory and n_total % 100 == 0:
                    self._write_trajectory_frame(coordinates, new_energy, temp, n_total)
        
        print(f"\n# Final acceptance ratio: {n_accepted/n_total:.3f}")
        return coordinates
    
    def _gradient_descent_refinement(self):
        """Simple gradient descent refinement"""
        print("\n# Starting gradient descent refinement...")
        
        # Create modified parameters for minimizer
        min_params = self.params.copy()
        min_params["minimize_with_restraints"] = True
        
        # Create minimizer instance
        minimizer = get_minimizer(
            "lbfgs",
            self.model, 
            self.system_data, 
            self.conformation,
            min_params,
            self.fprec
        )
        
        # Add custom force evaluation that includes RMSD and clash removal
        original_eval = minimizer._evaluate_energy_forces
        
        def custom_eval(coords, system=None, preproc_state=None):
            # Get base forces from model
            energy, forces, system, preproc = original_eval(coords, system, preproc_state)
            
            # Reshape coordinates
            coords_3d = coords.reshape(-1, 3)
            
            # Add RMSD restraint
            if self.use_rmsd:
                # Ensure rmsd_atoms is JAX int32 array
                rmsd_atoms = None if self.rmsd_atoms is None else jnp.array(self.rmsd_atoms, dtype=jnp.int32)
                rmsd_energy, rmsd_forces = rmsd_restraint_force(
                    coords_3d,
                    self.reference_coords,
                    self.rmsd_fc,
                    self.rmsd_target,
                    rmsd_atoms
                )
                energy += rmsd_energy
                forces += rmsd_forces.flatten()
            
            # Add clash removal forces
            forces_3d = forces.reshape(-1, 3)
            forces_3d, n_clashes = self._apply_clash_removal_forces(coords_3d, forces_3d)
            forces = forces_3d.flatten()
            
            return energy, forces, system, preproc
        
        minimizer._evaluate_energy_forces = custom_eval
        
        # Run minimization
        result = minimizer.run()
        
        return result["coordinates"]
    
    def _final_minimization(self, coordinates):
        """Final minimization at low temperature"""
        # Create a minimizer for final refinement
        min_params = self.params.copy()
        min_params["min_max_iterations"] = 100
        min_params["min_force_tolerance"] = 1e-5
        
        # Update conformation with current coordinates
        self.conformation["coordinates"] = coordinates
        
        # Use LBFGS for final minimization
        minimizer = get_minimizer(
            "lbfgs",
            self.model,
            self.system_data,
            self.conformation,
            min_params,
            self.fprec
        )
        
        # Add RMSD restraint to minimizer
        if self.use_rmsd:
            original_eval = minimizer._evaluate_energy_forces
            
            def rmsd_eval(coords, system=None, preproc_state=None):
                energy, forces, system, preproc = original_eval(coords, system, preproc_state)
                coords_3d = coords.reshape(-1, 3)
                # Ensure rmsd_atoms is JAX int32 array
                rmsd_atoms = None if self.rmsd_atoms is None else jnp.array(self.rmsd_atoms, dtype=jnp.int32)
                rmsd_energy, rmsd_forces = rmsd_restraint_force(
                    coords_3d,
                    self.reference_coords,
                    self.rmsd_fc * 0.1,  # Weaker restraint for final minimization
                    self.rmsd_target,
                    rmsd_atoms
                )
                energy += rmsd_energy
                forces += rmsd_forces.flatten()
                return energy, forces, system, preproc
            
            minimizer._evaluate_energy_forces = rmsd_eval
        
        result = minimizer.run()
        return result["coordinates"]
    
    def _calculate_forces(self, coordinates):
        """Calculate total forces including model forces and restraints"""
        # Update conformation
        self.conformation["coordinates"] = coordinates
        
        # Get forces from model
        conformation = {"coordinates": coordinates}
        system = {
            "coordinates": coordinates, 
            "vel": jnp.zeros_like(coordinates)
        }
        
        # Evaluate energy and forces
        energy_forces = self.model.energy_and_forces(
            coordinates=coordinates, 
            species=jnp.array(self.system_data["species"], dtype=jnp.int32),
            neighbors=self.system_data.get("neighbors", None)
        )
        
        # energy_and_forces returns (energy, forces, output)
        energy, forces, output = energy_forces
        
        # Add RMSD restraint
        if self.use_rmsd:
            # Ensure rmsd_atoms is JAX int32 array
            rmsd_atoms = None if self.rmsd_atoms is None else jnp.array(self.rmsd_atoms, dtype=jnp.int32)
            rmsd_energy, rmsd_forces = rmsd_restraint_force(
                coordinates,
                self.reference_coords,
                self.rmsd_fc,
                self.rmsd_target,
                rmsd_atoms
            )
            energy += rmsd_energy
            forces += rmsd_forces
        
        # Add clash removal forces
        forces, n_clashes = self._apply_clash_removal_forces(coordinates, forces)
        
        # Add covalent restraint forces
        if self.preserve_covalent and self.topology:
            # Bond length restraints
            bond_energy, bond_forces = calculate_bond_restraint_forces(
                coordinates, self.topology, self.bond_fc, self.bond_types_to_preserve
            )
            energy += bond_energy
            forces += bond_forces
            
            # No angle restraints - only using distance restraints for simplicity
        
        return energy, forces
    
    def _calculate_rmsd(self, coordinates):
        """Calculate RMSD from reference structure"""
        if self.rmsd_atoms is None:
            # Use all atoms
            diff = coordinates - self.reference_coords
        else:
            # Use selected atoms
            diff = coordinates[self.rmsd_atoms] - self.reference_coords[self.rmsd_atoms]
        
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    def _write_trajectory_frame(self, coordinates, energy, temperature, step):
        """Write a frame to both PDB and XYZ trajectory files"""
        # Write to PDB trajectory file
        self.traj_pdb_file.write(f"MODEL     {step+1:4d}\n")
        self.traj_pdb_file.write(f"REMARK   Energy: {float(energy.item()):.3f}  Temperature: {temperature:.1f}  Step: {step}\n")
        
        # Write atoms to PDB
        for i, (symbol, coord) in enumerate(zip(self.system_data["symbols"], coordinates)):
            atom_info = self.system_data.get("atoms", [{}] * self.nat)[i]
            atom_name = atom_info.get("name", symbol)
            resname = atom_info.get("resname", "UNK")
            resid = atom_info.get("resid", 1)
            chain = atom_info.get("chain", "A")
            
            self.traj_pdb_file.write(
                f"ATOM  {i+1:5d} {atom_name:4s} {resname:3s} {chain:1s}{resid:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        
        self.traj_pdb_file.write("ENDMDL\n")
        self.traj_pdb_file.flush()
        
        # Write to XYZ trajectory file
        write_xyz_frame(
            self.traj_xyz_file,
            self.system_data["symbols"],
            coordinates,
            properties={"energy": energy, "temperature": temperature, "step": step}
        )
        self.traj_xyz_file.flush()
        
        # Write to multi-model PDB file (same as trajectory)
        self.multimodel_pdb_file.write(f"MODEL     {step+1:4d}\n")
        self.multimodel_pdb_file.write(f"REMARK   Energy: {float(energy.item()):.3f}  Temperature: {temperature:.1f}  Step: {step}\n")
        
        for i, (symbol, coord) in enumerate(zip(self.system_data["symbols"], coordinates)):
            atom_info = self.system_data.get("atoms", [{}] * self.nat)[i]
            atom_name = atom_info.get("name", symbol)
            resname = atom_info.get("resname", "UNK")
            resid = atom_info.get("resid", 1)
            chain = atom_info.get("chain", "A")
            
            self.multimodel_pdb_file.write(
                f"ATOM  {i+1:5d} {atom_name:4s} {resname:3s} {chain:1s}{resid:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        
        self.multimodel_pdb_file.write("ENDMDL\n")
        self.multimodel_pdb_file.flush()
        
        # Write to multi-model XYZ file
        write_xyz_frame(
            self.multimodel_xyz_file,
            self.system_data["symbols"],
            coordinates,
            properties={"energy": energy, "temperature": temperature, "step": step, "model": step+1}
        )
        self.multimodel_xyz_file.flush()
    
    def _save_final_structure(self, coordinates):
        """Save the final refined structure"""
        # Save as PDB
        with open(f"{self.output_prefix}_refined.pdb", "w") as f:
            f.write("REMARK   Refined structure from FeNNol\n")
            f.write(f"REMARK   Final RMSD: {self._calculate_rmsd(coordinates):.3f} Å\n")
            
            for i, (symbol, coord) in enumerate(zip(self.system_data["symbols"], coordinates)):
                atom_info = self.system_data.get("atoms", [{}] * self.nat)[i]
                atom_name = atom_info.get("name", symbol)
                resname = atom_info.get("resname", "UNK")
                resid = atom_info.get("resid", 1)
                chain = atom_info.get("chain", "A")
                
                f.write(
                    f"ATOM  {i+1:5d} {atom_name:4s} {resname:3s} {chain:1s}{resid:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {symbol:>2s}\n"
                )
            f.write("END\n")
        
        # Also save as XYZ
        with open(f"{self.output_prefix}_refined.xyz", "w") as f:
            energy, _ = self._calculate_forces(coordinates)
            write_xyz_frame(
                f,
                self.system_data["symbols"],
                coordinates,
                properties={
                    "energy": energy,
                    "rmsd": self._calculate_rmsd(coordinates),
                    "refined": True
                }
            )
        
        print(f"\n# Refined structures saved:")
        print(f"#   - {self.output_prefix}_refined.pdb")
        print(f"#   - {self.output_prefix}_refined.xyz")
        
        if self.write_trajectory:
            print(f"# Trajectory files saved:")
            print(f"#   - {self.output_prefix}_traj.pdb")
            print(f"#   - {self.output_prefix}_traj.xyz") 
            print(f"#   - {self.output_prefix}_multimodel.pdb")
            print(f"#   - {self.output_prefix}_multimodel.xyz")


def main():
    """Main entry point for structure refinement CLI"""
    
    # Set up unbuffered output
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    
    # Pre-parse to check for device setting
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--device", type=str)
    pre_args, _ = pre_parser.parse_known_args()
    
    # Set JAX platform early if CPU is specified
    if pre_args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="fennol_refine",
        description="Refine molecular structures from XYZ files using FeNNol with enhanced sampling methods"
    )
    
    # Required arguments
    parser.add_argument("--pdb", type=str, required=True,
                       help="Input PDB structure file (for topology)")
    parser.add_argument("--xyz", type=str, required=True,
                       help="Input XYZ structure file (for coordinates)")
    parser.add_argument("--model", type=str, 
                       default="/home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx",
                       help="FeNNol model file path")
    
    # Refinement method selection
    parser.add_argument("--method", choices=["simulated_annealing", "monte_carlo", "gradient_descent"],
                       default="simulated_annealing",
                       help="Refinement method (default: simulated_annealing)")
    
    # Temperature schedule parameters
    parser.add_argument("--temperature-schedule", choices=["linear", "exponential", "cosine"],
                       default="exponential",
                       help="Temperature schedule for annealing (default: exponential)")
    parser.add_argument("--initial-temperature", type=float, default=600.0,
                       help="Initial temperature in K (default: 600)")
    parser.add_argument("--final-temperature", type=float, default=10.0,
                       help="Final temperature in K (default: 10)")
    parser.add_argument("--temperature-steps", type=int, default=100,
                       help="Number of temperature steps (default: 100)")
    
    # Simulated annealing specific
    parser.add_argument("--annealing-cycles", type=int, default=10,
                       help="Number of cycles per temperature (default: 10)")
    parser.add_argument("--annealing-hold-time", type=int, default=10,
                       help="Steps to hold at each temperature (default: 10)")
    
    # Monte Carlo specific
    parser.add_argument("--mc-displacement", type=float, default=0.1,
                       help="Initial MC displacement in Angstroms (default: 0.1)")
    parser.add_argument("--mc-acceptance-ratio", type=float, default=0.5,
                       help="Target MC acceptance ratio (default: 0.5)")
    
    # RMSD restraint parameters
    parser.add_argument("--rmsd-restraint", action="store_true", default=True,
                       help="Use RMSD restraint to maintain structure (default: True)")
    parser.add_argument("--no-rmsd-restraint", dest="rmsd_restraint", action="store_false",
                       help="Disable RMSD restraint")
    parser.add_argument("--rmsd-force-constant", type=float, default=10.0,
                       help="RMSD restraint force constant (default: 10.0)")
    parser.add_argument("--rmsd-target", type=float, default=0.0,
                       help="Target RMSD from initial structure (default: 0.0)")
    parser.add_argument("--rmsd-atoms", type=str, default="heavy",
                       choices=["all", "backbone", "heavy"],
                       help="Atoms to include in RMSD calculation (default: heavy)")
    
    # Clash removal parameters
    parser.add_argument("--clash-cutoff", type=float, default=2.0,
                       help="Distance cutoff for clash detection in Angstroms (default: 2.0)")
    parser.add_argument("--clash-force-constant", type=float, default=100.0,
                       help="Force constant for clash removal (default: 100.0)")
    parser.add_argument("--clash-iterations", type=int, default=100,
                       help="Maximum iterations for clash removal (default: 100)")
    
    # Covalent preservation parameters
    parser.add_argument("--preserve-covalent-bonds", action="store_true", default=True,
                       help="Preserve covalent bonds during refinement (default: True)")
    parser.add_argument("--no-preserve-covalent-bonds", dest="preserve_covalent_bonds", action="store_false",
                       help="Disable covalent bond preservation")
    parser.add_argument("--bond-force-constant", type=float, default=1000.0,
                       help="Force constant for bond length restraints (default: 1000.0)")
    parser.add_argument("--angle-force-constant", type=float, default=100.0,
                       help="Force constant for bond angle restraints (default: 100.0)")
    parser.add_argument("--bond-types-to-preserve", nargs="+", 
                       choices=["backbone", "sidechain", "inter_residue", "all"],
                       default=["backbone", "inter_residue"],
                       help="Types of bonds to preserve (default: backbone inter_residue)")
    parser.add_argument("--max-bond-deviation", type=float, default=0.5,
                       help="Maximum allowed bond length deviation in Angstroms (default: 0.5)")
    parser.add_argument("--detect-topology", action="store_true", default=True,
                       help="Automatically detect molecular topology (default: True)")
    parser.add_argument("--no-detect-topology", dest="detect_topology", action="store_false",
                       help="Disable automatic topology detection")
    
    # General optimization parameters
    parser.add_argument("--max-iterations", type=int, default=1000,
                       help="Maximum optimization iterations (default: 1000)")
    parser.add_argument("--force-tolerance", type=float, default=1e-4,
                       help="Force convergence tolerance (default: 1e-4)")
    parser.add_argument("--energy-tolerance", type=float, default=1e-6,
                       help="Energy convergence tolerance (default: 1e-6)")
    parser.add_argument("--displacement-tolerance", type=float, default=1e-3,
                       help="Displacement convergence tolerance (default: 1e-3)")
    parser.add_argument("--max-step", type=float, default=0.2,
                       help="Maximum step size in Angstroms (default: 0.2)")
    
    # Device and precision
    parser.add_argument("--device", type=str,
                       help="Device to run on (cpu, gpu, cuda, or gpu:N/cuda:N for specific device)")
    parser.add_argument("--double-precision", action="store_true",
                       help="Use double precision (float64)")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "float32"],
                       default="highest", help="Matrix multiplication precision")
    
    # Output options
    parser.add_argument("--output-prefix", type=str,
                       help="Output file prefix (default: input filename)")
    parser.add_argument("--write-trajectory", action="store_true", default=True,
                       help="Write trajectory files (both PDB and XYZ) (default: True)")
    parser.add_argument("--no-trajectory", dest="write_trajectory", action="store_false",
                       help="Disable trajectory writing")
    parser.add_argument("--print-freq", type=int, default=10,
                       help="Print frequency for optimization steps (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.pdb).exists():
        print(f"Error: PDB file '{args.pdb}' not found")
        sys.exit(1)
    
    if not Path(args.xyz).exists():
        print(f"Error: XYZ file '{args.xyz}' not found")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)
    
    # Build simulation parameters
    simulation_parameters = build_parameters_from_cli(args)
    
    # Set up device
    device = simulation_parameters.get("device", "cpu").lower()
    original_device = device  # Preserve for display
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        platform = "cpu"
    elif device.startswith("cuda") or device.startswith("gpu"):
        if ":" in device:
            num = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = num
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        platform = "gpu"
    else:
        platform = "cpu"  # Default fallback
    
    try:
        _device = jax.devices(platform)[0]
        jax.config.update("jax_default_device", _device)
    except Exception as e:
        print(f"Warning: Could not set device to {platform}, falling back to CPU: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        _device = jax.devices("cpu")[0]
        jax.config.update("jax_default_device", _device)
        # Don't modify original_device for display purposes
    
    # Set precision
    enable_x64 = simulation_parameters.get("enable_x64", False)
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"
    
    # Set matrix multiplication precision
    matmul_precision = simulation_parameters.get("matmul_prec", "highest").lower()
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    
    if args.verbose:
        print(f"# Device: {original_device}")
        print(f"# Precision: {fprec}")
        print(f"# Matrix multiplication precision: {matmul_precision}")
    
    # Run refinement
    try:
        run_refinement(simulation_parameters, fprec, args.verbose)
    except Exception as e:
        print(f"Error during refinement: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_refinement(simulation_parameters, fprec, verbose=False):
    """Run the structure refinement"""
    
    start_time = time.time()
    
    if verbose:
        print("# Loading model...")
    
    # Load model
    model = load_model(simulation_parameters)
    
    if verbose:
        print(f"# Model loaded: {model.__class__.__name__}")
        print(f"# Energy unit: {model.energy_unit}")
    
    # Load PDB for topology information
    pdb_file = simulation_parameters["pdb_file"]
    pdb_data = read_pdb(pdb_file)
    
    # Load XYZ for coordinates
    xyz_file = simulation_parameters["xyz_file"]
    xyz_frames = read_xyz(xyz_file, has_comment_line=True)
    
    if not xyz_frames:
        raise ValueError(f"No frames found in XYZ file: {xyz_file}")
    
    # Use the first frame for coordinates
    xyz_frame = xyz_frames[0]
    symbols, coordinates, comment_line = xyz_frame
    
    # Verify that PDB and XYZ have the same number of atoms
    if len(pdb_data["symbols"]) != len(symbols):
        raise ValueError(f"PDB has {len(pdb_data['symbols'])} atoms but XYZ has {len(symbols)} atoms")
    
    # Create system data using PDB topology but XYZ coordinates
    system_data = {
        "nat": len(symbols),
        "symbols": symbols,
        "species": np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32),
        "atoms": [
            {
                "name": pdb_data["atom_names"][i],
                "resname": pdb_data["residue_names"][i],
                "resid": pdb_data["residue_numbers"][i],
                "chain": pdb_data["chain_ids"][i],
                "element": symbols[i]
            }
            for i in range(len(symbols))
        ],
        "name": simulation_parameters.get("output_prefix", "refined"),
        "pdb_data": pdb_data,  # Store PDB data for topology
    }
    
    # Create conformation with XYZ coordinates and species
    conformation = {
        "coordinates": jnp.array(coordinates, dtype=fprec),
        "species": system_data["species"]
    }
    
    if verbose:
        print(f"# System loaded: {system_data['nat']} atoms")
        print(f"# System name: {system_data['name']}")
    
    # Initialize preprocessing
    preproc_state, conformation = initialize_preprocessing(
        simulation_parameters, model, conformation, system_data
    )
    
    if verbose:
        print("# Preprocessing initialized")
    
    # Create refiner
    refiner = StructureRefiner(
        model, system_data, conformation, simulation_parameters, fprec
    )
    
    # Run refinement
    refined_coords = refiner.refine()
    
    total_time = time.time() - start_time
    print(f"\n# Total time: {total_time:.2f} seconds")
    
    return True




if __name__ == "__main__":
    main()