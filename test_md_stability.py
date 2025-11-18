#!/usr/bin/env python3
"""
Test MD stability with fixed GB forces
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("MD STABILITY TEST - Water Molecule")
print("="*80)
print()

# Water molecule
coords = np.array([
    [0.0, 0.0, 0.0],      # O
    [0.757, 0.586, 0.0],  # H1
    [-0.757, 0.586, 0.0], # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417], dtype=np.float64)
radii = np.array([1.5, 1.2, 1.2], dtype=np.float64)
b_params = np.array([0.8, 0.85, 0.85], dtype=np.float64)
c_params = np.array([0.0, 0.0, 0.0], dtype=np.float64)
masses = np.array([15.999, 1.008, 1.008], dtype=np.float64)
dielectric = 80.0
cutoff = 12.0

# MD parameters
dt = 0.001  # 1 fs
n_steps = 1000
conversion = 418.4  # kcal/mol/A to (A/ps)^2

# Initialize
velocities = np.zeros_like(coords)

# Compute initial energy
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
forces = forces_direct + forces_born

initial_energy = float(energy)
print(f"Initial energy: {initial_energy:.6f} kcal/mol")
print()

# Compute initial kinetic energy
ke_initial = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) / conversion
total_energy_initial = initial_energy + ke_initial

print(f"Starting MD simulation ({n_steps} steps, dt = {dt} ps)...")
print()

# MD loop
energies = []
kinetic_energies = []
total_energies = []
temperatures = []

for step in range(n_steps):
    # Velocity Verlet integration

    # Update positions
    acceleration = forces * conversion / masses[:, np.newaxis]
    coords += velocities * dt + 0.5 * acceleration * dt**2

    # Compute new forces
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
    energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
    dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
    dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
    forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
    forces_new = forces_direct + forces_born

    # Update velocities
    acceleration_new = forces_new * conversion / masses[:, np.newaxis]
    velocities += 0.5 * (acceleration + acceleration_new) * dt

    # Update forces for next step
    forces = forces_new

    # Compute energies
    pe = float(energy)
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) / conversion
    total_energy = pe + ke

    # Compute temperature (3N - 6 degrees of freedom for non-linear molecule)
    # T = 2*KE / (k_B * dof), where k_B = 0.001987 kcal/mol/K
    dof = 3 * len(masses) - 6  # 3 atoms, non-linear
    if dof > 0:
        temp = 2 * ke / (0.001987 * dof)
    else:
        temp = 0.0

    energies.append(pe)
    kinetic_energies.append(ke)
    total_energies.append(total_energy)
    temperatures.append(temp)

    if step % 100 == 0:
        print(f"Step {step:4d}: E_pot = {pe:10.6f}, E_kin = {ke:10.6f}, E_tot = {total_energy:10.6f}, T = {temp:7.1f} K")

print()
print("="*80)
print("STABILITY ANALYSIS")
print("="*80)
print()

energies = np.array(energies)
kinetic_energies = np.array(kinetic_energies)
total_energies = np.array(total_energies)
temperatures = np.array(temperatures)

# Energy drift
energy_drift = (total_energies[-1] - total_energies[0]) / abs(total_energies[0]) * 100
print(f"Initial total energy: {total_energies[0]:.6f} kcal/mol")
print(f"Final total energy:   {total_energies[-1]:.6f} kcal/mol")
print(f"Energy drift:         {energy_drift:.4f}%")
print()

# Temperature drift
temp_initial = temperatures[0]
temp_final = temperatures[-1]
temp_mean = np.mean(temperatures)
temp_std = np.std(temperatures)
print(f"Initial temperature:  {temp_initial:.1f} K")
print(f"Final temperature:    {temp_final:.1f} K")
print(f"Mean temperature:     {temp_mean:.1f} K")
print(f"Std temperature:      {temp_std:.1f} K")
print()

# Check stability
if abs(energy_drift) < 1.0:
    print(f"✓ MD STABLE: Energy drift < 1% ({energy_drift:.4f}%)")
    stable = True
elif abs(energy_drift) < 10.0:
    print(f"⚠️  MD MARGINAL: Energy drift between 1-10% ({energy_drift:.4f}%)")
    stable = False
else:
    print(f"❌ MD UNSTABLE: Energy drift > 10% ({energy_drift:.4f}%)")
    stable = False

if temp_final < 1000:
    print(f"✓ Temperature reasonable: {temp_final:.1f} K")
else:
    print(f"❌ Temperature exploded: {temp_final:.1f} K")
    stable = False

if stable:
    print()
    print("🎉 SUCCESS: GB forces are now correct and MD is stable!")
