#!/usr/bin/env python3
"""
Example: Molecular Dynamics simulation starting from a PDB file.

This script demonstrates how to:
1. Read a PDB structure
2. Assign velocities and parameters
3. Setup implicit solvent model
4. Run MD simulation with CUDA acceleration
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fennol.utils.pdb import read_pdb, assign_charges, write_pdb
from fennol.models.physics.implicit_solvent import OBC


def assign_maxwell_boltzmann_velocities(masses: np.ndarray, temperature: float) -> np.ndarray:
    """
    Assign velocities from Maxwell-Boltzmann distribution.

    Args:
        masses: Atomic masses [natoms] in amu
        temperature: Temperature in Kelvin

    Returns:
        Velocities [natoms, 3] in Å/ps
    """
    natoms = len(masses)

    # Maxwell-Boltzmann velocity distribution
    # For each velocity component: <v_i²> = kB*T/m
    #
    # Units:
    # - kB = 0.001987204 kcal/(mol·K) (this is R/Avogadro)
    # - T in K
    # - m in amu (= g/mol)
    # - v in Å/ps
    #
    # From kinetic theory: 0.5 * m * v² = 0.5 * kB * T (per degree of freedom)
    # So: v² = kB*T/m
    # Conversion factor from kcal/(mol·amu) to (Å/ps)²:
    # 1 kcal/mol = 4.184e3 J/mol = 4.184e3 kg·m²/(s²·mol)
    # 1 amu = 1.66054e-27 kg = 1 g/mol
    # So kcal/(mol·amu) = 4.184e3 m²/s² × (10^10 Å/m)² × (10^-12 s/ps)² = 41.84 Å²/ps²

    kB = 0.001987204  # kcal/(mol·K)
    conversion = 41.84  # Convert kcal/(mol·amu) to (Å/ps)²

    # Standard deviation of velocity for each component
    sigma_v = np.sqrt(kB * temperature / masses * conversion)  # Å/ps

    # Sample velocities
    velocities = np.random.randn(natoms, 3) * sigma_v[:, np.newaxis]

    # Remove center-of-mass motion
    total_momentum = (masses[:, np.newaxis] * velocities).sum(axis=0)
    total_mass = masses.sum()
    velocities -= total_momentum[np.newaxis, :] / total_mass

    return velocities


def compute_kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    """
    Compute kinetic energy.

    Args:
        velocities: [natoms, 3] in Å/ps
        masses: [natoms] in amu

    Returns:
        Kinetic energy in kcal/mol
    """
    # KE = 0.5 * m * v^2
    # Convert from amu·Å²/ps² to kcal/mol
    # 1 amu·Å²/ps² = 0.01036427 kcal/mol
    conversion = 0.01036427

    v_squared = (velocities ** 2).sum(axis=1)  # [natoms]
    ke = 0.5 * (masses * v_squared).sum() * conversion

    return ke


def compute_temperature(velocities: np.ndarray, masses: np.ndarray) -> float:
    """
    Compute instantaneous temperature from kinetic energy.

    Args:
        velocities: [natoms, 3] in Å/ps
        masses: [natoms] in amu

    Returns:
        Temperature in Kelvin
    """
    natoms = len(masses)
    degrees_of_freedom = 3 * natoms - 3  # Remove center-of-mass motion

    ke = compute_kinetic_energy(velocities, masses)

    # T = 2*KE / (DOF * kB)
    kB = 0.001987204  # kcal/(mol·K)
    temperature = 2 * ke / (degrees_of_freedom * kB)

    return temperature


def velocity_rescale_thermostat(
    velocities: np.ndarray,
    masses: np.ndarray,
    target_temperature: float
) -> np.ndarray:
    """
    Rescale velocities to target temperature.

    Args:
        velocities: [natoms, 3] in Å/ps
        masses: [natoms] in amu
        target_temperature: Target temperature in K

    Returns:
        Rescaled velocities
    """
    current_T = compute_temperature(velocities, masses)

    if current_T > 0:
        scale_factor = np.sqrt(target_temperature / current_T)
        velocities = velocities * scale_factor

    return velocities


def minimize_energy(
    coords: np.ndarray,
    charges: np.ndarray,
    atomic_numbers: np.ndarray,
    solvent_model,
    max_steps: int = 200,
    step_size: float = 0.001,
    force_tolerance: float = 1.0
) -> np.ndarray:
    """
    Simple steepest descent energy minimization with line search.

    Args:
        coords: [natoms, 3]
        charges: [natoms]
        atomic_numbers: [natoms]
        solvent_model: Implicit solvent model
        max_steps: Maximum minimization steps
        step_size: Initial step size in Å
        force_tolerance: Convergence criterion (kcal/(mol·Å))

    Returns:
        Minimized coordinates
    """
    print(f"  Running energy minimization (max {max_steps} steps)...")

    coords = coords.copy()
    prev_energy = None

    for step in range(max_steps):
        energy, forces = solvent_model(coords, charges, atomic_numbers)

        # Check for NaN or inf
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            print(f"  Warning: Non-finite energy or forces at step {step}")
            print(f"  Reverting to previous coordinates")
            break

        # Maximum force magnitude
        max_force = np.abs(forces).max()

        if step % 20 == 0 or step == max_steps - 1:
            print(f"    Step {step:3d}: E = {energy:.2f} kcal/mol, "
                  f"F_max = {max_force:.2f} kcal/(mol·Å)")

        # Check convergence
        if max_force < force_tolerance:
            print(f"  Converged at step {step} (F_max < {force_tolerance})")
            break

        # Steepest descent with adaptive step size
        # Scale step size based on maximum force
        adaptive_step = min(step_size, 0.1 / max_force)

        # Move atoms in direction of forces
        coords += forces * adaptive_step

        # Simple line search: if energy increased, reduce step size
        if prev_energy is not None and energy > prev_energy:
            step_size *= 0.5
            print(f"    Reducing step size to {step_size:.6f}")

        prev_energy = energy

    return coords


def run_md_from_pdb(
    pdb_file: str,
    temperature: float = 300.0,
    timestep: float = 0.0005,  # ps (0.5 fs)
    n_steps: int = 1000,
    output_pdb: str = "output.pdb",
    implicit_solvent: str = "OBC",
    use_cuda: bool = True,
    minimize: bool = True,
    thermostat_interval: int = 10
):
    """
    Run MD simulation starting from a PDB file.

    Args:
        pdb_file: Input PDB file
        temperature: Temperature in Kelvin
        timestep: Integration timestep in ps
        n_steps: Number of MD steps
        output_pdb: Output PDB file
        implicit_solvent: Implicit solvent model ("OBC", "GB", "GNN", or None)
        use_cuda: Whether to use CUDA acceleration
    """
    print(f"\n{'='*70}")
    print(f"MD Simulation from PDB")
    print(f"{'='*70}\n")

    # 1. Read PDB structure
    print("Step 1: Reading PDB structure...")
    structure = read_pdb(pdb_file)

    coords = structure.coordinates.copy()
    masses = structure.masses.copy()
    atomic_numbers = structure.atomic_numbers.copy()

    # 2. Assign charges
    print("\nStep 2: Assigning partial charges...")
    charges = assign_charges(structure, method="amber")
    print(f"  Total charge: {charges.sum():.3f} e")

    # 3. Assign velocities
    print(f"\nStep 3: Assigning velocities (T = {temperature} K)...")
    velocities = assign_maxwell_boltzmann_velocities(masses, temperature)

    T_initial = compute_temperature(velocities, masses)
    KE_initial = compute_kinetic_energy(velocities, masses)
    print(f"  Initial temperature: {T_initial:.2f} K")
    print(f"  Initial kinetic energy: {KE_initial:.2f} kcal/mol")

    # 4. Setup implicit solvent model
    if implicit_solvent:
        print(f"\nStep 4: Setting up {implicit_solvent} implicit solvent...")

        if implicit_solvent == "OBC":
            solvent_model = OBC({
                "dielectric": 80.0,  # Water
                "cutoff": 12.0,      # Angstroms
                "surface_tension": 0.005,  # kcal/(mol·Ų)
                "use_cuda": use_cuda
            })
        else:
            print(f"Warning: {implicit_solvent} not yet implemented, using OBC")
            solvent_model = OBC({
                "dielectric": 80.0,
                "cutoff": 12.0,
                "surface_tension": 0.005,
                "use_cuda": use_cuda
            })

        print(f"  Model: {solvent_model.__class__.__name__}")
        print(f"  Dielectric: {solvent_model.dielectric}")
        print(f"  Cutoff: {solvent_model.cutoff} Å")
        print(f"  CUDA: {use_cuda}")
    else:
        solvent_model = None
        print("\nStep 4: No implicit solvent (vacuum)")

    # 5. Energy minimization
    if minimize and solvent_model:
        print("\nStep 5: Energy minimization...")
        coords = minimize_energy(coords, charges, atomic_numbers, solvent_model,
                                max_steps=100, step_size=0.01)

    # 6. Compute initial energy
    print("\nStep 6: Computing initial energy...")

    if solvent_model:
        energy, forces = solvent_model(coords, charges, atomic_numbers)
        print(f"  Solvation energy: {energy:.2f} kcal/mol")
        print(f"  Force magnitude: {np.linalg.norm(forces):.2f} kcal/(mol·Å)")
    else:
        energy = 0.0
        forces = np.zeros_like(coords)

    total_energy = KE_initial + energy
    print(f"  Total energy: {total_energy:.2f} kcal/mol")

    # 7. Run MD simulation
    print(f"\nStep 7: Running MD simulation...")
    print(f"  Steps: {n_steps}")
    print(f"  Timestep: {timestep} ps ({timestep*1000:.2f} fs)")
    print(f"  Total time: {n_steps * timestep:.3f} ps")
    print(f"  Thermostat: Every {thermostat_interval} steps")
    print()

    # Simple velocity Verlet integrator with thermostat
    print(f"{'Step':<8} {'Time (ps)':<12} {'T (K)':<10} {'E_tot':<12} {'E_solv':<12}")
    print("-" * 70)

    conversion = 0.01036427  # amu·Å²/ps² to kcal/mol

    for step in range(n_steps):
        # Velocity Verlet - Step A
        # v(t + dt/2) = v(t) + (F(t)/m) * (dt/2)
        # r(t + dt) = r(t) + v(t + dt/2) * dt

        # Convert forces from kcal/(mol·Å) to Å/ps²
        # F = ma → a = F/m
        # a [Å/ps²] = F [kcal/(mol·Å)] / (m [amu] * conversion)
        accel = forces / (masses[:, np.newaxis] * conversion)

        velocities += 0.5 * accel * timestep
        coords += velocities * timestep

        # Compute new forces
        if solvent_model:
            energy, forces = solvent_model(coords, charges, atomic_numbers)
        else:
            energy = 0.0
            forces = np.zeros_like(coords)

        # Velocity Verlet - Step B
        # v(t + dt) = v(t + dt/2) + (F(t + dt)/m) * (dt/2)
        accel = forces / (masses[:, np.newaxis] * conversion)
        velocities += 0.5 * accel * timestep

        # Apply thermostat periodically
        if thermostat_interval > 0 and step % thermostat_interval == 0:
            velocities = velocity_rescale_thermostat(velocities, masses, temperature)

        # Compute properties
        if step % 100 == 0 or step == n_steps - 1:
            T = compute_temperature(velocities, masses)
            KE = compute_kinetic_energy(velocities, masses)
            E_total = KE + energy

            print(f"{step:<8} {step*timestep:<12.3f} {T:<10.2f} {E_total:<12.2f} {energy:<12.2f}")

    # 8. Write output
    print(f"\nStep 7: Writing output to {output_pdb}...")
    structure.coordinates = coords
    write_pdb(structure, output_pdb, title=f"MD simulation, {n_steps} steps")

    print(f"\n{'='*70}")
    print(f"Simulation complete!")
    print(f"{'='*70}\n")

    print("Final statistics:")
    print(f"  Final temperature: {T:.2f} K")
    print(f"  Final energy: {E_total:.2f} kcal/mol")
    print(f"  Output written to: {output_pdb}")


if __name__ == "__main__":
    # Example usage
    pdb_file = sys.argv[1] if len(sys.argv) > 1 else "ternary_complex.pdb"

    run_md_from_pdb(
        pdb_file=pdb_file,
        temperature=300.0,
        timestep=0.0005,  # 0.5 fs
        n_steps=2000,
        output_pdb="output.pdb",
        implicit_solvent="OBC",
        use_cuda=True,
        minimize=True,
        thermostat_interval=10
    )
