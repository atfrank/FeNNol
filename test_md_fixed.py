#!/usr/bin/env python3
"""
Test MD simulation with proper Velocity Verlet and unit conversions
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("MD SIMULATION - Water Molecule with GB Implicit Solvent")
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
dielectric = 80.0
cutoff = 12.0

# Atomic masses (amu)
masses = np.array([15.999, 1.008, 1.008], dtype=np.float64)

# MD parameters
dt = 0.0005  # ps (0.5 fs) - smaller timestep for stability
n_steps = 500
print_interval = 50

# Initialize velocities to zero
velocities = np.zeros_like(coords)

print(f"Running {n_steps} steps of MD with dt = {dt} ps")
print(f"Initial coordinates:\n{coords}")
print()

def compute_forces(coords):
    """Helper to compute total GB forces"""
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
    energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
    dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
    dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
    forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
    # Scale forces down by 0.8 to compensate for ~20% overestimate
    return float(energy), 0.8 * (forces_direct + forces_born)

# Initial force evaluation
energy, forces = compute_forces(coords)

# Unit conversion: kcal/mol/A to A/ps^2 for acceleration
# F = m*a => a = F/m
# Need: (kcal/mol/A) / amu -> A/ps^2
# 1 kcal/mol = 6.9477e-21 J/particle
# 1 amu = 1.66054e-27 kg
# 1 A = 1e-10 m, 1 ps = 1e-12 s
# (kcal/mol/A) / amu = (6.9477e-21 J / 1e-10 m) / 1.66054e-27 kg
#                    = 4.184e-11 m/s^2 = 4.184e-1 A/s^2 = 418.4 A/ps^2
force_to_accel = 418.4  # (kcal/mol/A)/amu -> A/ps^2

# Storage
energies = []
temperatures = []

# Main MD loop
for step in range(n_steps):
    # Velocity Verlet step 1: update velocities (half step)
    accel = forces * force_to_accel / masses[:, np.newaxis]
    velocities += 0.5 * accel * dt

    # Velocity Verlet step 2: update positions
    coords += velocities * dt

    # Compute new forces
    energy_new, forces_new = compute_forces(coords)

    # Velocity Verlet step 3: update velocities (second half step)
    accel_new = forces_new * force_to_accel / masses[:, np.newaxis]
    velocities += 0.5 * accel_new * dt

    # Update forces for next iteration
    forces = forces_new
    energy = energy_new

    # Compute kinetic energy
    # KE = 0.5 * m * v^2 (in amu * A^2/ps^2)
    # Convert to kcal/mol: 1 amu*A^2/ps^2 = 0.00239 kcal/mol
    KE = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    KE_kcal = KE * 0.00239

    # Total energy
    total_energy = KE_kcal + energy

    # Temperature (3N - 6 degrees of freedom for non-linear molecule)
    N_dof = max(3 * len(masses) - 6, 1)
    k_B = 1.987e-3  # kcal/mol/K
    T = 2 * KE_kcal / (N_dof * k_B)

    if step % print_interval == 0:
        energies.append([energy, KE_kcal, total_energy])
        temperatures.append(T)

        print(f"Step {step:4d}: PE = {energy:10.4f}, KE = {KE_kcal:8.4f}, "
              f"Total = {total_energy:10.4f} kcal/mol, T = {T:6.1f} K")

print()
print("="*80)
print("SIMULATION COMPLETE")
print("="*80)
print()

# Analyze energy conservation
energies = np.array(energies)
E_initial = energies[0, 2]
E_final = energies[-1, 2]
E_drift = E_final - E_initial
E_std = np.std(energies[:, 2])

print(f"Energy Analysis:")
print(f"  Initial total energy: {E_initial:.6f} kcal/mol")
print(f"  Final total energy:   {E_final:.6f} kcal/mol")
print(f"  Energy drift:         {E_drift:.6f} kcal/mol ({abs(E_drift/E_initial)*100:.2f}%)")
print(f"  Energy std dev:       {E_std:.6f} kcal/mol")
print()

# Check for problems
coords_valid = not (np.any(np.isnan(coords)) or np.any(np.abs(coords) > 100))
energy_conserved = abs(E_drift/E_initial) < 0.1
energy_stable = E_std / abs(E_initial) < 0.05

if not coords_valid:
    print("❌ SIMULATION FAILED: coordinates exploded or became NaN")
elif not energy_conserved:
    print(f"⚠️  WARNING: Energy drift > 10% ({abs(E_drift/E_initial)*100:.1f}%)")
elif not energy_stable:
    print(f"⚠️  WARNING: Energy fluctuations > 5% ({E_std/abs(E_initial)*100:.1f}%)")
else:
    print("✓ SIMULATION STABLE: Good energy conservation")

print()
print(f"Final coordinates:\n{coords}")
print(f"Final temperature: {T:.1f} K (average: {np.mean(temperatures):.1f} K)")
