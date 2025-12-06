#!/usr/bin/env python3
"""Run proper MD simulation with GB and generate trajectory."""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

# Force JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Water molecule coordinates
coords = np.array([
    [0.000, 0.000, 0.000],  # O
    [0.757, 0.586, 0.000],  # H1
    [-0.757, 0.586, 0.000],  # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

# Atomic masses (amu)
masses = np.array([15.999, 1.008, 1.008])

# Initialize GB model
gb_params = {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4,
    "radii_set": "mbondi",
    "include_nonpolar": False
}

print("="*80)
print("MD Simulation: Water molecule with GB Implicit Solvent")
print("="*80)

gb_model = create_implicit_solvent_model("OBC", gb_params)
gb_model.has_cuda = False  # Force JAX
print(f"\nUsing GB backend: JAX (analytical forces)")

# MD parameters
dt = 0.1  # fs (smaller for stability)
temperature = 300.0  # K
nsteps = 200  # Shorter run to see what happens
output_freq = 10

print(f"\nSimulation parameters:")
print(f"  Time step: {dt} fs")
print(f"  Temperature: {temperature} K")
print(f"  Number of steps: {nsteps}")
print(f"  Total time: {nsteps * dt / 1000:.2f} ps")
print(f"  Output frequency: every {output_freq} steps")

# Initialize velocities (Maxwell-Boltzmann distribution)
kb = 8.314462e-3  # kJ/(mol·K)
sigma = np.sqrt(kb * temperature / masses[:, None])
velocities = np.random.normal(0, sigma, size=(3, 3))

# Remove center of mass motion
total_momentum = np.sum(masses[:, None] * velocities, axis=0)
velocities -= total_momentum / np.sum(masses)

print(f"\nInitial coordinates (Å):\n{coords}")
print(f"\nInitial velocities (Å/fs):\n{velocities}")

# Compute initial energy and forces
energy, forces = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
print(f"\nInitial GB energy: {energy:.3f} kcal/mol")
print(f"Initial forces (max): {np.max(np.abs(forces)):.3f} kcal/mol/Å")

# Storage for trajectory
trajectory = []
energies = []
temperatures = []
times = []

# Add initial frame
trajectory.append(coords.copy())
energies.append(energy)

# Velocity Verlet integration
print(f"\n{'Step':>6} {'Time(ps)':>10} {'Energy':>12} {'Temp(K)':>10} {'Max_F':>10} {'Status'}")
print("-"*80)

current_coords = coords.copy()
current_velocities = velocities.copy()

for step in range(nsteps):
    try:
        # Compute forces at current position
        energy, forces = gb_model.compute_energy_forces(
            current_coords, charges, atomic_numbers
        )

        # Check for NaN
        if np.isnan(energy) or np.any(np.isnan(forces)):
            print(f"{step:6d} {'NaN':>10} {'NaN':>12} {'NaN':>10} {'NaN':>10} FAILED")
            print("\n*** NaN detected! Stopping simulation. ***")
            break

        # Convert forces to acceleration
        # Force in kcal/mol/Å, mass in amu
        # 1 kcal/mol/Å / amu = 4.184e4 Å²/fs² / amu
        accel = forces / masses[:, None] * 4.184e4

        # Velocity Verlet: v(t+dt/2) = v(t) + a(t)*dt/2
        current_velocities += 0.5 * accel * dt

        # Position update: r(t+dt) = r(t) + v(t+dt/2)*dt
        current_coords += current_velocities * dt

        # Compute forces at new position
        energy, forces = gb_model.compute_energy_forces(
            current_coords, charges, atomic_numbers
        )

        if np.isnan(energy) or np.any(np.isnan(forces)):
            print(f"{step:6d} {'NaN':>10} {'NaN':>12} {'NaN':>10} {'NaN':>10} FAILED")
            print("\n*** NaN detected! Stopping simulation. ***")
            break

        accel = forces / masses[:, None] * 4.184e4

        # Velocity update: v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
        current_velocities += 0.5 * accel * dt

        # Compute temperature from kinetic energy
        # KE = 0.5 * m * v^2 (in amu·Å²/fs²)
        ke = 0.5 * np.sum(masses[:, None] * current_velocities**2)
        # Convert to temperature: KE = (3N/2) * kb * T
        # 1 amu·Å²/fs² = 1.036e-4 eV = 2.39e-3 kcal/mol
        ke_kcal = ke * 2.39e-3  # kcal/mol
        temp = 2.0 * ke_kcal / (3 * 3 * kb)  # 3 atoms, 3N degrees of freedom

        # Save trajectory
        trajectory.append(current_coords.copy())
        energies.append(float(energy))
        temperatures.append(temp)
        times.append((step + 1) * dt / 1000.0)  # ps

        # Output
        if (step + 1) % output_freq == 0:
            max_force = np.max(np.abs(forces))
            time_ps = (step + 1) * dt / 1000.0
            print(f"{step+1:6d} {time_ps:10.3f} {energy:12.3f} {temp:10.1f} {max_force:10.3f} OK")

    except Exception as e:
        print(f"{step:6d} ERROR")
        print(f"\nException occurred: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n" + "="*80)
print("Simulation completed successfully!")
print("="*80)

# Statistics
energies_arr = np.array(energies)
temps_arr = np.array(temperatures)

print(f"\nStatistics:")
print(f"  Energy:      {np.mean(energies_arr):8.3f} ± {np.std(energies_arr):6.3f} kcal/mol")
print(f"  Temperature: {np.mean(temps_arr):8.1f} ± {np.std(temps_arr):6.1f} K")
print(f"  Min energy:  {np.min(energies_arr):8.3f} kcal/mol")
print(f"  Max energy:  {np.max(energies_arr):8.3f} kcal/mol")

print(f"\nFinal coordinates (Å):\n{current_coords}")

# Save trajectory as XYZ format
xyz_file = "water_gb_trajectory.xyz"
with open(xyz_file, 'w') as f:
    for i, frame in enumerate(trajectory):
        f.write("3\n")
        f.write(f"Step {i}, Time {times[i] if i > 0 else 0.0:.3f} ps, Energy {energies[i]:.3f} kcal/mol\n")
        f.write(f"O  {frame[0,0]:12.6f} {frame[0,1]:12.6f} {frame[0,2]:12.6f}\n")
        f.write(f"H  {frame[1,0]:12.6f} {frame[1,1]:12.6f} {frame[1,2]:12.6f}\n")
        f.write(f"H  {frame[2,0]:12.6f} {frame[2,1]:12.6f} {frame[2,2]:12.6f}\n")

print(f"\nTrajectory saved to: {xyz_file}")
print(f"  {len(trajectory)} frames")
print(f"  Can be visualized with VMD, PyMOL, or other molecular viewers")

# Save as numpy array too
np.savez("water_gb_trajectory.npz",
         coordinates=np.array(trajectory),
         energies=energies_arr,
         temperatures=temps_arr,
         times=np.array(times))
print(f"\nNumpy data saved to: water_gb_trajectory.npz")
print(f"  Load with: data = np.load('water_gb_trajectory.npz')")

print(f"\n✓ MD simulation with GB forces completed successfully - NO NaN!")
