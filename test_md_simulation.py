#!/usr/bin/env python3
"""
Test MD simulation with GB implicit solvent
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("MD SIMULATION TEST - Water Molecule with GB Implicit Solvent")
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
dt = 0.001  # ps (1 fs)
n_steps = 1000
print_interval = 100

# Initialize velocities to zero
velocities = np.zeros_like(coords)

# Storage for trajectory
trajectory = []
energies = []
temperatures = []

print(f"Running {n_steps} steps of MD with dt = {dt} ps")
print(f"Initial coordinates:\n{coords}")
print()

# Main MD loop (Velocity Verlet)
for step in range(n_steps):
    # Compute forces
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
    energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

    # Add Born radii derivative forces
    dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
    dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
    forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)

    forces_total = forces_direct + forces_born

    # Convert forces from kcal/mol/Angstrom to acceleration (Angstrom/ps^2)
    # 1 kcal/mol = 418.4 kJ/mol = 418.4e3 J/mol = 418.4e3 / 6.022e23 J/atom
    # F = m*a => a = F/m
    # Need to convert units: kcal/mol/Angstrom -> Angstrom/ps^2
    # Factor: 418.4 / masses (converts kcal/mol/A to A*amu/ps^2, then divide by mass)
    conversion = 418.4  # kcal/mol to kJ/mol * 1000 / NA gives correct units
    acceleration = forces_total * conversion / masses[:, np.newaxis]

    # Velocity Verlet integration
    # Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    coords += velocities * dt + 0.5 * acceleration * dt**2

    # Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    # For now, use half-step: v(t+dt/2) = v(t) + 0.5*a(t)*dt
    velocities += 0.5 * acceleration * dt

    # Compute new forces for velocity update
    born_radii_new, psi_sum_new = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
    energy_new, forces_direct_new = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii_new, dielectric, cutoff)
    dE_dR_new = fennol_cuda.compute_dE_dR(coords, charges, born_radii_new, dielectric, cutoff)
    dE_dpsi_new = fennol_cuda.reduce_born_force(dE_dR_new, born_radii_new, radii, b_params, c_params, psi_sum_new)
    forces_born_new = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi_new, cutoff)
    forces_total_new = forces_direct_new + forces_born_new

    acceleration_new = forces_total_new * conversion / masses[:, np.newaxis]

    # Complete velocity update: v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
    velocities += 0.5 * acceleration_new * dt

    # Compute kinetic energy (amu * Angstrom^2/ps^2)
    KE = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    # Convert to kcal/mol: (amu * A^2/ps^2) -> kcal/mol
    # 1 amu * A^2/ps^2 = 1.660539e-27 kg * 1e-20 m^2 / 1e-24 s^2 = 1.660539e-23 J
    # 1 kcal/mol = 4184 J/mol / 6.022e23 = 6.947e-21 J
    # So: amu*A^2/ps^2 -> kcal/mol: multiply by 1.660539e-23 / 6.947e-21 = 0.00239
    KE_kcal = KE * 0.00239

    # Potential energy
    PE = float(energy_new)

    # Total energy
    total_energy = KE_kcal + PE

    # Temperature (K) from kinetic energy
    # KE = 0.5 * N_dof * k_B * T
    # N_dof = 3*N_atoms - 6 (for non-linear molecule) = 9 - 6 = 3
    # k_B = 1.380649e-23 J/K = 1.987e-3 kcal/mol/K
    N_dof = 3 * len(masses) - 6
    if N_dof > 0:
        k_B = 1.987e-3  # kcal/mol/K
        T = 2 * KE_kcal / (N_dof * k_B)
    else:
        T = 0.0

    # Store
    if step % print_interval == 0:
        trajectory.append(coords.copy())
        energies.append([PE, KE_kcal, total_energy])
        temperatures.append(T)

        print(f"Step {step:4d}: PE = {PE:10.4f} kcal/mol, KE = {KE_kcal:8.4f} kcal/mol, "
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

# Check for NaN or explosion
if np.any(np.isnan(coords)) or np.any(np.abs(coords) > 100):
    print("❌ SIMULATION FAILED: coordinates exploded or became NaN")
elif abs(E_drift/E_initial) > 0.1:
    print(f"⚠️  WARNING: Energy drift > 10% (poor energy conservation)")
elif E_std / abs(E_initial) > 0.05:
    print(f"⚠️  WARNING: Energy fluctuations > 5% (unstable)")
else:
    print("✓ SIMULATION STABLE: Energy is reasonably conserved")

print()
print(f"Final coordinates:\n{coords}")
print(f"Average temperature: {np.mean(temperatures):.1f} K")
