#!/usr/bin/env python3
"""Run MD simulation with ANI2x + GB implicit solvent."""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models import ANI2xModel
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model
from fennol.md.integrate import generate_dynamics

# Force JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Initial water molecule coordinates
coords = np.array([
    [0.000, 0.000, 0.000],  # O
    [0.757, 0.586, 0.000],  # H1
    [-0.757, 0.586, 0.000],  # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)
species = jnp.array([8, 1, 1], dtype=jnp.int32)

# Initialize ANI2x model
print("Loading ANI2x model...")
ani_model = ANI2xModel()

# Initialize GB implicit solvent
print("Initializing GB implicit solvent...")
gb_params = {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "radii_set": "mbondi",
    "include_nonpolar": False
}
gb_model = create_implicit_solvent_model("OBC", gb_params)

# Force JAX backend for GB
gb_model.has_cuda = False
print(f"GB backend: {'CUDA' if gb_model.has_cuda else 'JAX'}")

# MD parameters
dt = 0.5  # fs
temperature = 300.0  # K
nsteps = 500
output_freq = 10

print(f"\nMD Parameters:")
print(f"  Time step: {dt} fs")
print(f"  Temperature: {temperature} K")
print(f"  Number of steps: {nsteps}")
print(f"  Output frequency: {output_freq}")

# Initialize velocities (Maxwell-Boltzmann distribution)
masses = np.array([15.999, 1.008, 1.008])  # amu
kb = 8.314462e-3  # kJ/(mol·K)
sigma = np.sqrt(kb * temperature / masses[:, None])
velocities = np.random.normal(0, sigma, size=(3, 3))

# Remove center of mass motion
total_momentum = np.sum(masses[:, None] * velocities, axis=0)
velocities -= total_momentum / np.sum(masses)

print(f"\nInitial state:")
print(f"  Coordinates:\n{coords}")
print(f"  Velocities:\n{velocities}")

# Compute initial energy
ani_energy = ani_model.energy(coords, species)
gb_energy, gb_forces = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
ani_forces = -jax.grad(lambda x: ani_model.energy(x, species))(coords)

print(f"\nInitial energies:")
print(f"  ANI2x energy: {ani_energy:.6f} eV")
print(f"  GB energy: {gb_energy:.6f} kcal/mol")
print(f"  ANI2x forces (max): {np.max(np.abs(ani_forces)):.6f} eV/Å")
print(f"  GB forces (max): {np.max(np.abs(gb_forces)):.6f} kcal/mol/Å")

# Convert units: ANI gives eV, GB gives kcal/mol
# 1 eV = 23.06 kcal/mol
# 1 eV/Å = 23.06 kcal/mol/Å
eV_to_kcal = 23.06054783

print(f"\nStarting MD simulation...")
print(f"{'Step':>6} {'Time':>8} {'E_ANI':>12} {'E_GB':>12} {'E_total':>12} {'T':>8} {'Max_F':>10}")
print("-" * 80)

# Simple velocity Verlet integration
coords_traj = [coords.copy()]
energies_ani = [float(ani_energy)]
energies_gb = [float(gb_energy)]
temperatures = []

for step in range(nsteps):
    # Compute forces
    ani_forces = -jax.grad(lambda x: ani_model.energy(x, species))(coords)
    gb_energy, gb_forces = gb_model.compute_energy_forces(coords, charges, atomic_numbers)

    # Convert ANI forces from eV/Å to kcal/mol/Å
    total_forces = ani_forces * eV_to_kcal + gb_forces

    # Acceleration (kcal/mol/Å / amu -> Å/fs²)
    # 1 kcal/mol = 4.184 kJ/mol = 4.184e-3 kJ/mol per amu·Å²·fs⁻²
    # a = F/m where F is in kcal/mol/Å and m is in amu
    # Conversion: 1 kcal/mol/Å / amu = 4.184e4 Å/fs² / amu
    accel = total_forces / masses[:, None] * 4.184e4

    # Velocity Verlet
    velocities += 0.5 * accel * dt
    coords += velocities * dt

    # Recompute forces at new position
    ani_forces = -jax.grad(lambda x: ani_model.energy(x, species))(coords)
    gb_energy, gb_forces = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
    total_forces = ani_forces * eV_to_kcal + gb_forces
    accel = total_forces / masses[:, None] * 4.184e4

    velocities += 0.5 * accel * dt

    # Compute temperature from kinetic energy
    ke = 0.5 * np.sum(masses[:, None] * velocities**2)  # amu·Å²/fs²
    # Convert to K: KE = (3/2) N kb T
    # 1 amu·Å²/fs² = 1.036e-4 eV
    temp = 2.0 * ke * 1.036e-4 * eV_to_kcal / (3 * kb * 3)  # 3 atoms

    # Save trajectory
    coords_traj.append(coords.copy())
    ani_energy = ani_model.energy(coords, species)
    energies_ani.append(float(ani_energy))
    energies_gb.append(float(gb_energy))
    temperatures.append(temp)

    # Output
    if step % output_freq == 0:
        total_energy = float(ani_energy) * eV_to_kcal + float(gb_energy)
        max_force = np.max(np.abs(total_forces))
        time_ps = (step + 1) * dt / 1000.0

        # Check for NaN
        if np.isnan(total_energy) or np.isnan(max_force):
            print(f"{step+1:6d} {time_ps:8.3f} {'NaN':>12} {'NaN':>12} {'NaN':>12} {'NaN':>8} {'NaN':>10}")
            print("\n*** NaN detected! Simulation crashed. ***")
            break

        print(f"{step+1:6d} {time_ps:8.3f} {float(ani_energy)*eV_to_kcal:12.3f} {float(gb_energy):12.3f} {total_energy:12.3f} {temp:8.1f} {max_force:10.3f}")

print("\nSimulation completed!")
print(f"\nFinal coordinates:\n{coords}")
print(f"\nEnergy statistics:")
print(f"  ANI energy: {np.mean(energies_ani)*eV_to_kcal:.3f} ± {np.std(energies_ani)*eV_to_kcal:.3f} kcal/mol")
print(f"  GB energy:  {np.mean(energies_gb):.3f} ± {np.std(energies_gb):.3f} kcal/mol")
print(f"  Temperature: {np.mean(temperatures):.1f} ± {np.std(temperatures):.1f} K")

# Save trajectory
np.save("water_gb_trajectory.npy", np.array(coords_traj))
print(f"\nTrajectory saved to water_gb_trajectory.npy")
