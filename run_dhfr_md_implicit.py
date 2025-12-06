#!/usr/bin/env python3
"""
10,000-step MD Simulation of DHFR in Implicit Solvent

Uses the optimized Phase 3C fused kernel for maximum performance.

This demonstrates:
1. Loading DHFR structure (2,499 atoms)
2. Running 10,000-step MD simulation with GB implicit solvent
3. Measuring performance with Phase 3 optimizations
4. Outputting trajectory and energies
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path
import sys

# Force JAX to use CPU (let CUDA handle GB)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Import implicit solvent model
from fennol.models.physics.implicit_solvent import OBC
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX

def read_tinker_xyz(filename):
    """Read Tinker XYZ format file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # First line: number of atoms
    natoms = int(lines[0].strip().split()[0])

    # Parse atoms
    coords = []
    elements = []

    for line in lines[1:natoms+1]:
        parts = line.strip().split()
        idx = int(parts[0])
        element = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

        coords.append([x, y, z])
        elements.append(element)

    return np.array(coords), elements

def get_atomic_numbers(elements):
    """Convert element symbols to atomic numbers."""
    return np.array([PERIODIC_TABLE_REV_IDX[e] for e in elements])

def estimate_charges_simple(elements):
    """Simple charge estimation for proteins."""
    charges = []
    for elem in elements:
        if elem == 'N':
            charges.append(-0.3)
        elif elem == 'O':
            charges.append(-0.5)
        elif elem == 'H':
            charges.append(0.3)
        elif elem == 'C':
            charges.append(0.1)
        elif elem == 'S':
            charges.append(0.0)
        else:
            charges.append(0.0)

    # Normalize to ensure neutrality
    charges = np.array(charges)
    net_charge = charges.sum()
    charges -= net_charge / len(charges)

    return charges

def get_masses(elements):
    """Get atomic masses (amu)."""
    mass_dict = {
        'H': 1.008,
        'C': 12.011,
        'N': 14.007,
        'O': 15.999,
        'S': 32.065,
    }
    return np.array([mass_dict.get(e, 12.0) for e in elements])

def initialize_velocities(masses, temperature, seed=42):
    """
    Initialize velocities from Maxwell-Boltzmann distribution.

    Args:
        masses: Atomic masses [natoms] in amu
        temperature: Temperature in Kelvin
        seed: Random seed

    Returns:
        velocities: [natoms, 3] in Å/fs
    """
    np.random.seed(seed)

    # Boltzmann constant in amu·Å²/(fs²·K)
    kb = 8.314462e-3 / 1.660539e-3  # kJ/(mol·K) -> amu·Å²/(fs²·K)
    # Actually: kb = 0.831446 amu·Å²/(fs²·K)
    kb = 0.831446

    natoms = len(masses)
    sigma = np.sqrt(kb * temperature / masses[:, None])
    velocities = np.random.normal(0, sigma, size=(natoms, 3))

    # Remove center of mass motion
    total_momentum = np.sum(masses[:, None] * velocities, axis=0)
    velocities -= total_momentum / np.sum(masses)

    return velocities

def compute_temperature(velocities, masses):
    """
    Compute instantaneous temperature from velocities.

    T = (2 * KE) / (3 * N * kb)
    where KE = 0.5 * sum(m * v²)
    """
    kb = 0.831446  # amu·Å²/(fs²·K)
    ke = 0.5 * np.sum(masses[:, None] * velocities**2)
    natoms = len(masses)
    temperature = (2.0 * ke) / (3.0 * natoms * kb)
    return temperature

def velocity_verlet_step(coords, velocities, masses, forces, dt):
    """
    Velocity Verlet integration step.

    Args:
        coords: [natoms, 3] in Å
        velocities: [natoms, 3] in Å/fs
        masses: [natoms] in amu
        forces: [natoms, 3] in kcal/mol/Å
        dt: time step in fs

    Returns:
        new_coords, new_velocities
    """
    # Convert forces to acceleration: a = F/m
    # F in kcal/mol/Å, m in amu
    # a in Å/fs²
    # 1 kcal/mol = 6.95e-4 amu·Å²/fs²
    kcal_to_amu = 6.95e-4
    accel = forces * kcal_to_amu / masses[:, None]

    # Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
    new_coords = coords + velocities * dt + 0.5 * accel * dt**2

    # Half-step velocity: v(t+dt/2) = v(t) + 0.5*a(t)*dt
    velocities_half = velocities + 0.5 * accel * dt

    return new_coords, velocities_half, accel

def main():
    print("=" * 70)
    print("DHFR 10,000-Step MD Simulation with Implicit Solvent")
    print("=" * 70)
    print()

    # =====================================================================
    # SYSTEM SETUP
    # =====================================================================

    # Load DHFR structure
    xyz_file = Path("examples/md/dhfr/dhfr2_nowat.xyz")
    if not xyz_file.exists():
        print(f"Error: {xyz_file} not found")
        print("Checking alternative location...")
        xyz_file = Path("dhfr2_nowat.xyz")
        if not xyz_file.exists():
            print(f"Error: Could not find DHFR structure file")
            return 1

    print(f"Loading structure: {xyz_file}")
    coords, elements = read_tinker_xyz(xyz_file)
    natoms = len(coords)
    print(f"  Atoms: {natoms}")
    print()

    # Get atomic properties
    atomic_numbers = get_atomic_numbers(elements)
    charges = estimate_charges_simple(elements)
    masses = get_masses(elements)

    print(f"System properties:")
    print(f"  Net charge: {charges.sum():.6f} e")
    print(f"  Total mass: {masses.sum():.2f} amu")
    print()

    # Convert to JAX arrays
    coords_jax = jnp.array(coords)
    charges_jax = jnp.array(charges)
    atomic_numbers_jax = jnp.array(atomic_numbers)

    # =====================================================================
    # INITIALIZE IMPLICIT SOLVENT MODEL
    # =====================================================================

    print("Initializing OBC Generalized Born model (Phase 3C optimized)...")
    model = OBC({
        "dielectric": 78.3,
        "cutoff": 12.0,
        "surface_tension": 0.005,
        "probe_radius": 1.4,
        "include_nonpolar": True
    })
    print(f"  Backend: {'CUDA (Phase 3C fused kernel)' if model.has_cuda else 'JAX'}")
    print(f"  Dielectric: {model.dielectric}")
    print(f"  Cutoff: {model.cutoff} Å")
    print()

    # Warm-up (JIT compilation)
    print("Warm-up (JIT compilation)...")
    energy, forces_jax = model(coords_jax, charges_jax, atomic_numbers_jax)
    forces = np.array(forces_jax)
    print(f"  Initial energy: {energy:.4f} kcal/mol")
    print(f"  Max force: {np.abs(forces).max():.4f} kcal/mol/Å")
    print()

    # =====================================================================
    # MD PARAMETERS
    # =====================================================================

    dt = 1.0  # fs
    temperature = 300.0  # K
    nsteps = 10000
    output_freq = 100  # Output every 100 steps

    print(f"MD Parameters:")
    print(f"  Time step: {dt} fs")
    print(f"  Target temperature: {temperature} K")
    print(f"  Number of steps: {nsteps}")
    print(f"  Total time: {nsteps * dt / 1000:.1f} ps")
    print(f"  Output frequency: {output_freq} steps")
    print()

    # Initialize velocities
    print("Initializing velocities...")
    velocities = initialize_velocities(masses, temperature, seed=42)
    temp_initial = compute_temperature(velocities, masses)
    print(f"  Initial temperature: {temp_initial:.2f} K")
    print()

    # =====================================================================
    # RUN MD SIMULATION
    # =====================================================================

    print("Starting MD simulation...")
    print()
    print(f"{'Step':>6} {'Time':>8} {'Energy':>12} {'Temp':>8} {'Max_F':>10} {'Time/step':>12}")
    print("-" * 70)

    # Storage
    trajectory = [coords.copy()]
    energies = [float(energy)]
    temperatures_traj = [temp_initial]
    step_times = []

    # Current state
    coords_current = coords.copy()
    velocities_current = velocities.copy()

    # Start timing
    start_time = time.time()

    for step in range(nsteps):
        step_start = time.time()

        # Compute forces (CUDA optimized!)
        coords_jax = jnp.array(coords_current)
        energy, forces_jax = model(coords_jax, charges_jax, atomic_numbers_jax)
        forces = np.array(forces_jax)

        # Velocity Verlet step 1: update positions and half-step velocities
        coords_new, velocities_half, accel_old = velocity_verlet_step(
            coords_current, velocities_current, masses, forces, dt
        )

        # Compute forces at new position
        coords_jax = jnp.array(coords_new)
        energy_new, forces_jax_new = model(coords_jax, charges_jax, atomic_numbers_jax)
        forces_new = np.array(forces_jax_new)

        # Velocity Verlet step 2: complete velocity update
        kcal_to_amu = 6.95e-4
        accel_new = forces_new * kcal_to_amu / masses[:, None]
        velocities_new = velocities_half + 0.5 * accel_new * dt

        # Update state
        coords_current = coords_new
        velocities_current = velocities_new

        # Compute temperature
        temp_current = compute_temperature(velocities_current, masses)

        step_end = time.time()
        step_time = (step_end - step_start) * 1000  # ms
        step_times.append(step_time)

        # Output
        if step % output_freq == 0 or step == nsteps - 1:
            time_ps = (step + 1) * dt / 1000
            max_force = np.abs(forces_new).max()

            print(f"{step+1:6d} {time_ps:8.3f} {float(energy_new):12.4f} {temp_current:8.2f} {max_force:10.4f} {step_time:12.2f}")

            # Store trajectory
            trajectory.append(coords_current.copy())
            energies.append(float(energy_new))
            temperatures_traj.append(temp_current)

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 70)
    print()

    # =====================================================================
    # PERFORMANCE STATISTICS
    # =====================================================================

    print("Performance Statistics:")
    print(f"  Total simulation time: {total_time:.2f} s")
    print(f"  Average time per step: {np.mean(step_times):.2f} ms")
    print(f"  Std dev: {np.std(step_times):.2f} ms")
    print(f"  Min time: {np.min(step_times):.2f} ms")
    print(f"  Max time: {np.max(step_times):.2f} ms")
    print(f"  Throughput: {nsteps / total_time:.2f} steps/second")
    print(f"  Simulation speed: {nsteps * dt / total_time:.2f} fs/second")
    print(f"  Time to simulate 1 ns: {1e6 / (nsteps * dt / total_time) / 60:.2f} minutes")
    print()

    # =====================================================================
    # TRAJECTORY ANALYSIS
    # =====================================================================

    trajectory = np.array(trajectory)
    energies = np.array(energies)
    temperatures_traj = np.array(temperatures_traj)

    print("Trajectory Analysis:")
    print(f"  Frames saved: {len(trajectory)}")
    print(f"  Energy:")
    print(f"    Mean: {np.mean(energies):.4f} kcal/mol")
    print(f"    Std: {np.std(energies):.4f} kcal/mol")
    print(f"    Min: {np.min(energies):.4f} kcal/mol")
    print(f"    Max: {np.max(energies):.4f} kcal/mol")
    print(f"  Temperature:")
    print(f"    Mean: {np.mean(temperatures_traj):.2f} K")
    print(f"    Std: {np.std(temperatures_traj):.2f} K")
    print(f"    Min: {np.min(temperatures_traj):.2f} K")
    print(f"    Max: {np.max(temperatures_traj):.2f} K")
    print()

    # RMSD from initial structure
    rmsd = np.sqrt(np.mean((trajectory - trajectory[0])**2, axis=(1,2)))
    print(f"  RMSD from initial:")
    print(f"    Final: {rmsd[-1]:.4f} Å")
    print(f"    Max: {np.max(rmsd):.4f} Å")
    print()

    # =====================================================================
    # SAVE OUTPUT
    # =====================================================================

    output_dir = Path("dhfr_md_output")
    output_dir.mkdir(exist_ok=True)

    print(f"Saving output to {output_dir}/")

    # Save trajectory (XYZ format)
    with open(output_dir / "trajectory.xyz", 'w') as f:
        for i, frame in enumerate(trajectory):
            f.write(f"{natoms}\n")
            f.write(f"Frame {i}, Time = {i * output_freq * dt / 1000:.3f} ps\n")
            for j, (elem, coord) in enumerate(zip(elements, frame)):
                f.write(f"{elem} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    # Save energies
    np.savetxt(output_dir / "energies.txt", energies,
               header="Energy (kcal/mol)", fmt="%.6f")

    # Save temperatures
    np.savetxt(output_dir / "temperatures.txt", temperatures_traj,
               header="Temperature (K)", fmt="%.2f")

    # Save RMSD
    np.savetxt(output_dir / "rmsd.txt", rmsd,
               header="RMSD from initial (Å)", fmt="%.6f")

    # Save summary
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("DHFR 10,000-Step MD Simulation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"System: {natoms} atoms\n")
        f.write(f"Time step: {dt} fs\n")
        f.write(f"Number of steps: {nsteps}\n")
        f.write(f"Total time: {nsteps * dt / 1000:.1f} ps\n")
        f.write(f"\nPerformance:\n")
        f.write(f"  Total time: {total_time:.2f} s\n")
        f.write(f"  Average time/step: {np.mean(step_times):.2f} ms\n")
        f.write(f"  Throughput: {nsteps / total_time:.2f} steps/s\n")
        f.write(f"\nEnergy Statistics:\n")
        f.write(f"  Mean: {np.mean(energies):.4f} kcal/mol\n")
        f.write(f"  Std: {np.std(energies):.4f} kcal/mol\n")
        f.write(f"\nTemperature Statistics:\n")
        f.write(f"  Mean: {np.mean(temperatures_traj):.2f} K\n")
        f.write(f"  Std: {np.std(temperatures_traj):.2f} K\n")

    print("  trajectory.xyz")
    print("  energies.txt")
    print("  temperatures.txt")
    print("  rmsd.txt")
    print("  summary.txt")
    print()

    print("=" * 70)
    print("Simulation completed successfully!")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
