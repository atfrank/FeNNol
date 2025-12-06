"""
Scale MD simulation for studying unbinding kinetics.

This module implements a scaled molecular dynamics method where inter-chain
forces between a chain of interest and other chains are scaled by a parameter
alpha. This allows accelerated sampling of unbinding events.

Key features:
- Inter-chain force scaling by alpha parameter
- Backbone fixing for specified chains
- Distance tracking of chain of interest from initial position
- Loop over multiple alpha values in single simulation
- Early stopping when distance threshold is reached

Usage:
    python -m fennol.md.scale_md config.fnl

Config file example:
    scale_md {
        pdb_file = "protein_peptide.pdb"
        chain_of_interest = "B"           # Chain to track (1-indexed PDB chain ID)
        alpha_values = [0.1, 0.2, 0.5, 1.0]

        fix_backbone {
            enabled = true
            chains = ["A"]                # Fix backbone of these chains
        }

        early_stopping {
            enabled = true
            distance_threshold = 20.0     # Angstroms
            min_steps = 1000
            check_frequency = 100
        }

        output {
            trajectory_prefix = "traj"    # Will create traj_alpha_0.1.xyz, etc.
            distance_file = "distances.dat"
        }
    }
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp

from ..utils.pdb import read_pdb, PDBStructure, write_pdb
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from ..utils.io import write_xyz_frame, write_extxyz_frame, write_arc_frame, human_time_duration
from ..models import FENNIX

from .initial import load_model, initialize_preprocessing
from .thermostats import get_thermostat


# Backbone atom names for proteins
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'H', 'HA'}


@dataclass
class ChainInfo:
    """Information about chains in the system."""
    chain_ids: List[str]                # Chain ID for each atom (0-indexed)
    unique_chains: List[str]            # List of unique chain IDs
    chain_atom_indices: Dict[str, np.ndarray]  # Chain ID -> atom indices (0-indexed)
    chain_atom_masks: Dict[str, np.ndarray]    # Chain ID -> boolean mask
    n_atoms: int


@dataclass
class ScaleMDConfig:
    """Configuration for scale MD simulation."""
    pdb_file: str
    chain_of_interest: str              # Chain to track (1-indexed PDB ID)
    alpha_values: List[float]           # List of scaling factors

    # Backbone fixing
    fix_backbone_enabled: bool = False
    fix_backbone_chains: List[str] = field(default_factory=list)

    # Early stopping
    early_stopping_enabled: bool = True
    distance_threshold: float = 20.0    # Angstroms
    min_steps: int = 1000
    check_frequency: int = 100

    # Output
    trajectory_prefix: str = "traj"
    distance_file: str = "distances.dat"
    overwrite: bool = True

    # Inter-chain force reporting (expensive - off by default)
    report_inter_chain_forces: bool = False

    # MD parameters
    temperature: float = 300.0
    timestep: float = 1.0               # fs
    nsteps_per_alpha: int = 10000
    thermostat: str = "langevin"
    gamma: float = 1.0                  # ps^-1 for Langevin

    # Minimization
    minimize_first: bool = True
    min_steps: int = 500


def extract_chain_info(pdb_structure: PDBStructure) -> ChainInfo:
    """
    Extract chain information from PDB structure.

    Note: PDB uses 1-indexed residues, but atom arrays are 0-indexed.
    Chain IDs are strings (e.g., 'A', 'B') and remain unchanged.

    Args:
        pdb_structure: PDBStructure from read_pdb

    Returns:
        ChainInfo with chain membership for each atom
    """
    chain_ids = pdb_structure.chain_ids
    n_atoms = len(chain_ids)
    unique_chains = sorted(set(chain_ids))

    chain_atom_indices = {}
    chain_atom_masks = {}

    for chain in unique_chains:
        # 0-indexed atom indices for this chain
        indices = np.array([i for i, c in enumerate(chain_ids) if c == chain], dtype=np.int32)
        mask = np.array([c == chain for c in chain_ids], dtype=bool)
        chain_atom_indices[chain] = indices
        chain_atom_masks[chain] = mask

    print(f"# Chain information:")
    for chain in unique_chains:
        n = len(chain_atom_indices[chain])
        print(f"#   Chain {chain}: {n} atoms (indices {chain_atom_indices[chain][0]}-{chain_atom_indices[chain][-1]})")

    return ChainInfo(
        chain_ids=chain_ids,
        unique_chains=unique_chains,
        chain_atom_indices=chain_atom_indices,
        chain_atom_masks=chain_atom_masks,
        n_atoms=n_atoms
    )


def create_backbone_mask(pdb_structure: PDBStructure, chains: List[str]) -> np.ndarray:
    """
    Create mask for backbone atoms of specified chains.

    Args:
        pdb_structure: PDBStructure from read_pdb
        chains: List of chain IDs to fix backbone for

    Returns:
        Boolean mask where True = backbone atom in specified chains (should be fixed)
    """
    n_atoms = pdb_structure.natoms
    mask = np.zeros(n_atoms, dtype=bool)

    for i, atom in enumerate(pdb_structure.atoms):
        if atom.chain in chains and atom.name in BACKBONE_ATOMS:
            mask[i] = True

    n_fixed = np.sum(mask)
    print(f"# Backbone fixing: {n_fixed} atoms fixed in chains {chains}")

    return mask


def compute_chain_com(coordinates: np.ndarray, chain_mask: np.ndarray,
                      masses: np.ndarray) -> np.ndarray:
    """
    Compute center of mass of a chain.

    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        chain_mask: Boolean mask for chain atoms
        masses: Atomic masses [n_atoms]

    Returns:
        Center of mass [3]
    """
    chain_coords = coordinates[chain_mask]
    chain_masses = masses[chain_mask]
    total_mass = np.sum(chain_masses)
    com = np.sum(chain_coords * chain_masses[:, None], axis=0) / total_mass
    return com


def compute_chain_distance(coordinates: np.ndarray, chain_mask: np.ndarray,
                           masses: np.ndarray, reference_com: np.ndarray) -> float:
    """
    Compute distance of chain COM from reference position.

    Args:
        coordinates: Current atomic coordinates [n_atoms, 3]
        chain_mask: Boolean mask for chain of interest
        masses: Atomic masses [n_atoms]
        reference_com: Reference COM position [3]

    Returns:
        Distance from reference COM in Angstroms
    """
    current_com = compute_chain_com(coordinates, chain_mask, masses)
    distance = np.linalg.norm(current_com - reference_com)
    return float(distance)


def create_inter_chain_pair_mask(chain_info: ChainInfo,
                                  chain_of_interest: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create masks for identifying inter-chain atom pairs involving chain of interest.

    This creates two 1D masks that can be used with neighbor lists to identify
    pairs where one atom is in chain_of_interest and the other is not.

    Args:
        chain_info: ChainInfo object
        chain_of_interest: Chain ID of interest

    Returns:
        Tuple of (chain_of_interest_mask, other_chains_mask) as boolean arrays
    """
    coi_mask = chain_info.chain_atom_masks[chain_of_interest]
    other_mask = ~coi_mask

    return coi_mask, other_mask


def compute_inter_chain_force_magnitude(
    coordinates: np.ndarray,
    forces: np.ndarray,
    coi_mask: np.ndarray,
    masses: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute the magnitude of inter-chain forces on the chain of interest.

    This estimates inter-chain forces by decomposing forces on chain of interest
    atoms into radial (toward/away from other chains) and tangential components.
    The radial component represents the inter-chain interaction.

    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        forces: Forces on all atoms [n_atoms, 3]
        coi_mask: Boolean mask for chain of interest atoms
        masses: Atomic masses [n_atoms]

    Returns:
        Tuple of (total_inter_chain_force_magnitude, radial_component, tangential_component)
        All in force units (Hartree/Bohr or model units)
    """
    other_mask = ~coi_mask

    # Compute COMs
    coi_com = compute_chain_com(coordinates, coi_mask, masses)
    other_com = compute_chain_com(coordinates, other_mask, masses)

    # Direction from COI to other chains (inter-chain direction)
    com_direction = other_com - coi_com
    com_distance = np.linalg.norm(com_direction)
    if com_distance > 1e-6:
        com_direction = com_direction / com_distance
    else:
        # Chains are overlapping - use arbitrary direction
        com_direction = np.array([1.0, 0.0, 0.0])

    # Get forces on chain of interest atoms
    coi_forces = forces[coi_mask]

    # Decompose into radial (inter-chain) and tangential (intra-chain) components
    # For each atom, project force onto COM direction
    radial_components = np.sum(coi_forces * com_direction, axis=1)  # scalar per atom
    radial_forces = radial_components[:, None] * com_direction  # vector per atom
    tangential_forces = coi_forces - radial_forces

    # Sum up the radial forces (these represent inter-chain interactions)
    # The net radial force on the chain
    total_radial_force = np.sum(radial_forces, axis=0)
    total_tangential_force = np.sum(tangential_forces, axis=0)

    # Magnitudes
    radial_magnitude = np.linalg.norm(total_radial_force)
    tangential_magnitude = np.linalg.norm(total_tangential_force)
    total_magnitude = np.linalg.norm(np.sum(coi_forces, axis=0))

    return float(total_magnitude), float(radial_magnitude), float(tangential_magnitude)


class ScaleMDSimulation:
    """
    Scale MD simulation for studying unbinding kinetics.

    This class manages the simulation loop with scaled inter-chain forces.
    """

    def __init__(self, config: ScaleMDConfig, simulation_parameters: Dict):
        """
        Initialize the scale MD simulation.

        Args:
            config: ScaleMDConfig object
            simulation_parameters: Parsed simulation parameters from .fnl file
        """
        self.config = config
        self.simulation_parameters = simulation_parameters

        # Load PDB structure
        print(f"# Loading PDB: {config.pdb_file}")
        self.pdb_structure = read_pdb(config.pdb_file)
        self.chain_info = extract_chain_info(self.pdb_structure)

        # Validate chain of interest exists
        if config.chain_of_interest not in self.chain_info.unique_chains:
            raise ValueError(
                f"Chain of interest '{config.chain_of_interest}' not found. "
                f"Available chains: {self.chain_info.unique_chains}"
            )

        # Get masses
        self.masses = self.pdb_structure.masses.copy()

        # Create chain masks
        self.coi_mask = self.chain_info.chain_atom_masks[config.chain_of_interest]
        self.other_mask = ~self.coi_mask

        # Initial COM of chain of interest
        self.initial_com = compute_chain_com(
            self.pdb_structure.coordinates,
            self.coi_mask,
            self.masses
        )
        print(f"# Initial COM of chain {config.chain_of_interest}: {self.initial_com}")

        # Backbone fixing mask
        if config.fix_backbone_enabled:
            self.backbone_mask = create_backbone_mask(
                self.pdb_structure,
                config.fix_backbone_chains
            )
            self.mobile_mask = ~self.backbone_mask
        else:
            self.backbone_mask = np.zeros(self.pdb_structure.natoms, dtype=bool)
            self.mobile_mask = np.ones(self.pdb_structure.natoms, dtype=bool)

        # Reference coordinates for fixed atoms
        self.reference_coords = self.pdb_structure.coordinates.copy()

        # Initialize model and system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the FeNNol model and simulation system."""
        # Set precision
        fprec = "float32"
        if self.simulation_parameters.get("double_precision", False):
            fprec = "float64"
        self.fprec = fprec

        # Load model
        print("# Loading model...")
        self.model = load_model(self.simulation_parameters)
        self.model_energy_unit = au.get_multiplier(self.model.energy_unit)

        # Prepare system data
        nat = self.pdb_structure.natoms
        species = self.pdb_structure.atomic_numbers
        coordinates = self.pdb_structure.coordinates.astype(fprec)

        mass_amu = self.masses.astype(fprec)
        mass = mass_amu * (au.MPROT * (au.FS / au.BOHR) ** 2)

        temperature = self.config.temperature
        kT = temperature / au.KELVIN

        self.system_data = {
            "name": Path(self.config.pdb_file).stem,
            "nat": nat,
            "symbols": self.pdb_structure.elements,
            "species": species,
            "mass": mass,
            "mass_amu": mass_amu,
            "temperature": temperature,
            "kT": kT,
            "totmass_amu": mass_amu.sum() / 6.02214129e-1,
            "pbc": None,
            "nreplicas": 1,
        }

        # Store number of mobile atoms
        self.n_mobile = int(np.sum(self.mobile_mask))
        self.system_data["n_mobile_atoms"] = self.n_mobile

        # Create conformation
        self.initial_conformation = {
            "species": species,
            "coordinates": coordinates,
            "batch_index": np.zeros(nat, dtype=np.int32),
            "natoms": np.array([nat], dtype=np.int32),
        }

        # Initialize preprocessing
        print("# Initializing preprocessing...")
        self.preproc_state, self.conformation = initialize_preprocessing(
            self.simulation_parameters,
            self.model,
            self.initial_conformation,
            self.system_data
        )

        # Store initial minimized coordinates
        self.minimized_coords = None

    def _assign_velocities(self, temperature: float) -> np.ndarray:
        """
        Assign Maxwell-Boltzmann velocities.

        Args:
            temperature: Temperature in Kelvin

        Returns:
            Velocities [n_atoms, 3] in atomic units
        """
        masses = self.system_data["mass"]
        nat = self.system_data["nat"]
        kT = temperature / au.KELVIN

        # Generate random velocities
        velocities = np.random.randn(nat, 3)

        # Scale by sqrt(kT/m) for each atom
        for i in range(nat):
            sigma = np.sqrt(kT / masses[i])
            velocities[i] *= sigma

        # Zero velocities for fixed atoms
        velocities[self.backbone_mask] = 0.0

        # Remove COM motion for mobile atoms
        mobile_masses = masses[self.mobile_mask]
        mobile_vel = velocities[self.mobile_mask]
        total_momentum = np.sum(mobile_masses[:, None] * mobile_vel, axis=0)
        total_mass = np.sum(mobile_masses)
        velocities[self.mobile_mask] -= total_momentum[None, :] / total_mass

        return velocities.astype(self.fprec)

    def _minimize(self, coordinates: np.ndarray, max_steps: int = 500) -> np.ndarray:
        """
        Simple energy minimization using steepest descent.

        Args:
            coordinates: Initial coordinates
            max_steps: Maximum minimization steps

        Returns:
            Minimized coordinates
        """
        print(f"# Running energy minimization (max {max_steps} steps)...")

        coords = coordinates.copy()
        step_size = 0.01  # Angstroms

        # Create temporary conformation
        conformation = {**self.conformation, "coordinates": coords}

        for step in range(max_steps):
            # Compute forces
            preproc_state, conformation = self.model.preprocessing(
                self.preproc_state, conformation
            )
            epot, forces, _ = self.model._energy_and_forces(
                self.model.variables, conformation
            )

            forces = np.array(forces) / self.model_energy_unit
            epot = float(np.mean(epot)) / self.model_energy_unit

            # Zero forces for fixed atoms
            forces[self.backbone_mask] = 0.0

            max_force = np.max(np.abs(forces))

            if step % 50 == 0:
                print(f"#   Step {step:4d}: E = {epot:.4f} Ha, F_max = {max_force:.4f}")

            # Convergence check
            if max_force < 0.01:
                print(f"#   Converged at step {step}")
                break

            # Adaptive step size
            adaptive_step = min(step_size, 0.1 / max_force)

            # Update positions (only mobile atoms)
            coords[self.mobile_mask] += forces[self.mobile_mask] * adaptive_step

            # Restore fixed atom positions
            coords[self.backbone_mask] = self.reference_coords[self.backbone_mask]

            conformation = {**conformation, "coordinates": coords}

        return coords

    def _compute_forces_with_scaling(self, coordinates: np.ndarray,
                                      alpha: float,
                                      compute_inter_chain_forces: bool = False
                                      ) -> Tuple[float, np.ndarray, Optional[Tuple[float, float, float]]]:
        """
        Compute forces with inter-chain scaling.

        For pairs where one atom is in chain_of_interest and the other is not,
        the force is scaled by alpha.

        This is implemented by:
        1. Computing full forces
        2. Identifying inter-chain contributions (approximation: scale forces on
           chain of interest atoms that point toward/away from other chains)

        A more rigorous implementation would require modifying the force
        calculation at the neighbor list level, but this approximation works
        well for the unbinding kinetics use case.

        Args:
            coordinates: Atomic coordinates [n_atoms, 3]
            alpha: Scaling factor for inter-chain forces
            compute_inter_chain_forces: If True, compute and return inter-chain
                force magnitudes (expensive)

        Returns:
            Tuple of (potential energy, forces [n_atoms, 3], inter_chain_force_info)
            where inter_chain_force_info is None if not requested, or
            (total_force, radial_force, tangential_force) if requested
        """
        # Create conformation
        conformation = {**self.conformation, "coordinates": coordinates}

        # Preprocess
        preproc_state, conformation = self.model.preprocessing(
            self.preproc_state, conformation
        )

        # Compute forces
        epot, forces, _ = self.model._energy_and_forces(
            self.model.variables, conformation
        )

        forces = np.array(forces) / self.model_energy_unit
        epot = float(np.mean(epot)) / self.model_energy_unit

        # Compute inter-chain force magnitudes BEFORE scaling (raw interaction strength)
        inter_chain_force_info = None
        if compute_inter_chain_forces:
            inter_chain_force_info = compute_inter_chain_force_magnitude(
                coordinates, forces, self.coi_mask, self.masses
            )

        if alpha != 1.0:
            # Scale inter-chain forces
            # Strategy: For atoms in chain of interest, scale the component of
            # force that points toward/away from atoms in other chains

            # Get COMs
            coi_com = compute_chain_com(coordinates, self.coi_mask, self.masses)
            other_com = compute_chain_com(coordinates, self.other_mask, self.masses)

            # Direction from COI to other chains
            com_direction = other_com - coi_com
            com_distance = np.linalg.norm(com_direction)
            if com_distance > 1e-6:
                com_direction = com_direction / com_distance
            else:
                com_direction = np.array([1.0, 0.0, 0.0])

            # For chain of interest atoms: scale the radial component of force
            coi_forces = forces[self.coi_mask].copy()

            # Project forces onto COM direction
            radial_component = np.sum(coi_forces * com_direction, axis=1, keepdims=True)
            radial_forces = radial_component * com_direction
            tangential_forces = coi_forces - radial_forces

            # Scale only the radial (inter-chain) component
            scaled_forces = tangential_forces + alpha * radial_forces
            forces[self.coi_mask] = scaled_forces

            # For other chain atoms: apply Newton's third law adjustment
            other_forces = forces[self.other_mask].copy()
            radial_component = np.sum(other_forces * (-com_direction), axis=1, keepdims=True)
            radial_forces = radial_component * (-com_direction)
            tangential_forces = other_forces - radial_forces
            scaled_forces = tangential_forces + alpha * radial_forces
            forces[self.other_mask] = scaled_forces

        # Zero forces for fixed atoms
        forces[self.backbone_mask] = 0.0

        return epot, forces, inter_chain_force_info

    def run_single_alpha(self, alpha: float,
                         initial_coords: np.ndarray,
                         trajectory_file: str,
                         distance_writer) -> Tuple[np.ndarray, bool]:
        """
        Run simulation for a single alpha value.

        Args:
            alpha: Force scaling parameter
            initial_coords: Starting coordinates
            trajectory_file: Path to trajectory output file
            distance_writer: File handle for distance output

        Returns:
            Tuple of (final_coordinates, early_stopped)
        """
        print(f"\n{'='*60}")
        print(f"# Running simulation with alpha = {alpha}")
        print(f"# Trajectory: {trajectory_file}")
        print(f"{'='*60}")

        # Initialize
        coords = initial_coords.copy()
        velocities = self._assign_velocities(self.config.temperature)

        dt = self.config.timestep * au.FS
        dt2 = 0.5 * dt
        mass = self.system_data["mass"]
        dt2m = dt2 / mass[:, None]
        kT = self.system_data["kT"]

        # Langevin thermostat parameters
        gamma = self.config.gamma / 1000.0  # Convert ps^-1 to fs^-1
        gamma_au = gamma * au.FS
        c1 = np.exp(-gamma_au * dt)
        c2 = np.sqrt((1 - c1**2) * kT / mass[:, None])

        # Check if inter-chain force reporting is enabled
        report_forces = self.config.report_inter_chain_forces

        # Open trajectory file
        with open(trajectory_file, 'w') as traj_file:
            early_stopped = False

            for step in range(1, self.config.nsteps_per_alpha + 1):
                # Velocity Verlet with Langevin thermostat

                # Half step velocity update
                epot, forces, _ = self._compute_forces_with_scaling(coords, alpha)
                velocities = velocities + forces * dt2m

                # Zero velocities for fixed atoms
                velocities[self.backbone_mask] = 0.0

                # Position update
                coords = coords + velocities * dt

                # Restore fixed atom positions
                coords[self.backbone_mask] = self.reference_coords[self.backbone_mask]

                # Compute new forces (with optional inter-chain force computation)
                epot, forces, inter_chain_info = self._compute_forces_with_scaling(
                    coords, alpha, compute_inter_chain_forces=report_forces
                )

                # Second half step velocity update
                velocities = velocities + forces * dt2m

                # Zero velocities for fixed atoms
                velocities[self.backbone_mask] = 0.0

                # Langevin thermostat (velocity rescaling + random kicks)
                random_forces = np.random.randn(*velocities.shape).astype(self.fprec)
                velocities = c1 * velocities + c2 * random_forces
                velocities[self.backbone_mask] = 0.0

                # Compute kinetic energy and temperature
                ek = 0.5 * np.sum(mass[:, None] * velocities**2 * self.mobile_mask[:, None])
                temp = 2 * ek / (3 * self.n_mobile) * au.KELVIN

                # Compute distance
                distance = compute_chain_distance(
                    coords, self.coi_mask, self.masses, self.initial_com
                )

                # Write distance data (with optional force info)
                sim_time = step * self.config.timestep / 1000.0  # ps
                if report_forces and inter_chain_info is not None:
                    total_f, radial_f, tangential_f = inter_chain_info
                    distance_writer.write(
                        f"{alpha:.4f} {sim_time:.6f} {distance:.6f} "
                        f"{total_f:.6f} {radial_f:.6f} {tangential_f:.6f}\n"
                    )
                else:
                    distance_writer.write(f"{alpha:.4f} {sim_time:.6f} {distance:.6f}\n")
                distance_writer.flush()

                # Print progress
                if step % 100 == 0:
                    if report_forces and inter_chain_info is not None:
                        total_f, radial_f, _ = inter_chain_info
                        print(f"  Step {step:6d}: E = {epot:.4f} Ha, T = {temp:.1f} K, "
                              f"d = {distance:.2f} A, F_inter = {radial_f:.4f}")
                    else:
                        print(f"  Step {step:6d}: E = {epot:.4f} Ha, T = {temp:.1f} K, "
                              f"d = {distance:.2f} A")

                # Write trajectory frame
                if step % 10 == 0:
                    write_xyz_frame(
                        traj_file,
                        self.system_data["symbols"],
                        coords,
                        properties={"energy": epot, "time": sim_time}
                    )

                # Early stopping check
                if (self.config.early_stopping_enabled and
                    step >= self.config.min_steps and
                    step % self.config.check_frequency == 0):

                    if distance >= self.config.distance_threshold:
                        print(f"# Early stopping: distance {distance:.2f} A >= "
                              f"threshold {self.config.distance_threshold:.2f} A")
                        early_stopped = True
                        break

        return coords, early_stopped

    def run(self):
        """
        Run the full scale MD simulation with all alpha values.
        """
        print("\n" + "="*70)
        print("# Scale MD Simulation for Unbinding Kinetics")
        print("="*70)
        print(f"# PDB file: {self.config.pdb_file}")
        print(f"# Chain of interest: {self.config.chain_of_interest}")
        print(f"# Alpha values: {self.config.alpha_values}")
        print(f"# Steps per alpha: {self.config.nsteps_per_alpha}")
        print(f"# Temperature: {self.config.temperature} K")
        print(f"# Timestep: {self.config.timestep} fs")

        # Minimize initial structure
        if self.config.minimize_first:
            print("\n# Minimizing initial structure...")
            self.minimized_coords = self._minimize(
                self.pdb_structure.coordinates.copy(),
                max_steps=self.config.min_steps
            )
        else:
            self.minimized_coords = self.pdb_structure.coordinates.copy()

        # Update reference COM after minimization
        self.initial_com = compute_chain_com(
            self.minimized_coords, self.coi_mask, self.masses
        )
        print(f"# Post-minimization COM of chain {self.config.chain_of_interest}: "
              f"{self.initial_com}")

        # Open distance output file
        with open(self.config.distance_file, 'w') as dist_file:
            if self.config.report_inter_chain_forces:
                dist_file.write("# alpha time_ps distance_A F_total F_radial F_tangential\n")
            else:
                dist_file.write("# alpha time_ps distance_A\n")

            # Run simulations for each alpha
            for alpha in self.config.alpha_values:
                traj_file = f"{self.config.trajectory_prefix}_alpha_{alpha:.4f}.xyz"

                final_coords, early_stopped = self.run_single_alpha(
                    alpha=alpha,
                    initial_coords=self.minimized_coords.copy(),
                    trajectory_file=traj_file,
                    distance_writer=dist_file
                )

                status = "EARLY STOPPED" if early_stopped else "COMPLETED"
                print(f"\n# Alpha {alpha}: {status}")

        print("\n" + "="*70)
        print("# Scale MD simulation complete!")
        print(f"# Distance data: {self.config.distance_file}")
        print("="*70)


def parse_scale_md_config(simulation_parameters: Dict) -> ScaleMDConfig:
    """
    Parse scale MD configuration from simulation parameters.

    Args:
        simulation_parameters: Parsed .fnl file

    Returns:
        ScaleMDConfig object
    """
    scale_md_params = simulation_parameters.get("scale_md", {})

    # Required parameters
    pdb_file = scale_md_params.get("pdb_file")
    if pdb_file is None:
        # Fall back to pdb_input/file
        pdb_file = simulation_parameters.get("pdb_input/file")
    if pdb_file is None:
        raise ValueError("scale_md.pdb_file is required")

    chain_of_interest = scale_md_params.get("chain_of_interest")
    if chain_of_interest is None:
        raise ValueError("scale_md.chain_of_interest is required")

    # Alpha values - handle both list and single value
    alpha_values = scale_md_params.get("alpha_values", [0.1, 0.2, 0.5, 1.0])
    if isinstance(alpha_values, (int, float)):
        alpha_values = [float(alpha_values)]
    elif isinstance(alpha_values, str):
        # Handle string like "[0.5, 1.0]" or "0.5 1.0"
        alpha_values = alpha_values.strip('[]').replace(',', ' ').split()
        alpha_values = [float(a) for a in alpha_values]
    else:
        # Handle list - clean up any bracket characters
        cleaned = []
        for a in alpha_values:
            if isinstance(a, str):
                a = a.strip('[]').replace(',', '')
                if a:  # Skip empty strings
                    cleaned.append(float(a))
            else:
                cleaned.append(float(a))
        alpha_values = cleaned

    # Backbone fixing
    fix_backbone = scale_md_params.get("fix_backbone", {})
    fix_backbone_enabled = fix_backbone.get("enabled", False)
    if isinstance(fix_backbone_enabled, str):
        fix_backbone_enabled = fix_backbone_enabled.lower() in ("true", "yes", "1")
    fix_backbone_chains = fix_backbone.get("chains", [])
    if isinstance(fix_backbone_chains, str):
        # Handle strings like '["A"]' or 'A B' or 'A'
        fix_backbone_chains = fix_backbone_chains.strip('[]"\'').replace(',', ' ').split()
    elif isinstance(fix_backbone_chains, list):
        # Clean up list items
        fix_backbone_chains = [str(c).strip('[]"\'') for c in fix_backbone_chains]

    # Early stopping
    early_stopping = scale_md_params.get("early_stopping", {})
    early_stopping_enabled = early_stopping.get("enabled", True)
    if isinstance(early_stopping_enabled, str):
        early_stopping_enabled = early_stopping_enabled.lower() in ("true", "yes", "1")
    distance_threshold = float(early_stopping.get("distance_threshold", 20.0))
    min_steps_early = int(early_stopping.get("min_steps", 1000))
    check_frequency = int(early_stopping.get("check_frequency", 100))

    # Output
    output = scale_md_params.get("output", {})
    trajectory_prefix = str(output.get("trajectory_prefix", "traj")).strip('"\'')
    distance_file = str(output.get("distance_file", "distances.dat")).strip('"\'')
    thermostat_str = str(scale_md_params.get("thermostat", "langevin")).strip('"\'')

    # Inter-chain force reporting (expensive - off by default)
    report_inter_chain_forces = scale_md_params.get("report_inter_chain_forces", False)
    if isinstance(report_inter_chain_forces, str):
        report_inter_chain_forces = report_inter_chain_forces.lower() in ("true", "yes", "1")

    # MD parameters
    temperature = float(simulation_parameters.get("temperature", 300.0))
    timestep = float(simulation_parameters.get("dt", 1.0))
    nsteps_per_alpha = int(scale_md_params.get("nsteps_per_alpha",
                                               simulation_parameters.get("nsteps", 10000)))
    gamma = float(scale_md_params.get("gamma", 1.0))

    # Minimization
    minimize_first = scale_md_params.get("minimize", True)
    if isinstance(minimize_first, str):
        minimize_first = minimize_first.lower() in ("true", "yes", "1")
    min_steps_minimize = int(scale_md_params.get("min_steps", 500))

    return ScaleMDConfig(
        pdb_file=str(pdb_file).strip('"\''),
        chain_of_interest=str(chain_of_interest).strip('"\''),
        alpha_values=alpha_values,
        fix_backbone_enabled=fix_backbone_enabled,
        fix_backbone_chains=fix_backbone_chains,
        early_stopping_enabled=early_stopping_enabled,
        distance_threshold=distance_threshold,
        min_steps=min_steps_early,
        check_frequency=check_frequency,
        trajectory_prefix=trajectory_prefix,
        distance_file=distance_file,
        report_inter_chain_forces=report_inter_chain_forces,
        temperature=temperature,
        timestep=timestep,
        nsteps_per_alpha=nsteps_per_alpha,
        thermostat=thermostat_str,
        gamma=gamma,
        minimize_first=minimize_first,
    )


def main():
    """Main entry point for scale MD simulation."""
    parser = argparse.ArgumentParser(
        prog="fennol_scale_md",
        description="Scale MD simulation for unbinding kinetics"
    )
    parser.add_argument("param_file", type=Path, help="Parameter file (.fnl)")
    args = parser.parse_args()

    # Parse parameter file
    simulation_parameters = parse_input(args.param_file)

    # Set device
    device = simulation_parameters.get("device", "cpu").lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda") or device.startswith("gpu"):
        if ":" in device:
            num = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = num
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "gpu"

    _device = jax.devices(device)[0]
    jax.config.update("jax_default_device", _device)

    # Set precision
    enable_x64 = simulation_parameters.get("double_precision", False)
    jax.config.update("jax_enable_x64", enable_x64)

    # Parse scale MD config
    config = parse_scale_md_config(simulation_parameters)

    # Run simulation
    sim = ScaleMDSimulation(config, simulation_parameters)
    sim.run()


if __name__ == "__main__":
    main()
