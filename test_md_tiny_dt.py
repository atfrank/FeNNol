#!/usr/bin/env python3
"""
Test MD with very small timestep
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("MD TEST WITH TINY TIMESTEP")
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

# MD parameters - VERY small timestep
dt = 0.0001  # 0.1 fs
n_steps = 1000
conversion = 418.4  # kcal/mol/A to (A/ps)^2

# Start from rest
velocities = np.zeros_like(coords)

# Compute initial energy
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
forces = forces_direct + forces_born

initial_pe = float(energy)
initial_ke = 0.0
initial_total = initial_pe + initial_ke

print(f"Initial PE:    {initial_pe:.6f} kcal/mol")
print(f"Initial Total: {initial_total:.6f} kcal/mol")
print(f"Timestep:      {dt} ps = {dt*1000} fs")
print()

print("Starting MD from rest...")
print()

# MD loop
for step in range(n_steps):
    # Velocity Verlet
    acceleration = forces * conversion / masses[:, np.newaxis]

    # Half-step velocity update
    velocities += 0.5 * acceleration * dt

    # Position update
    coords += velocities * dt

    # Compute new forces
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
    energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
    dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
    dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
    forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
    forces_new = forces_direct + forces_born

    # Second half-step velocity update
    acceleration_new = forces_new * conversion / masses[:, np.newaxis]
    velocities += 0.5 * acceleration_new * dt
    forces = forces_new

    # Compute energies
    pe = float(energy)
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) / conversion
    total_energy = pe + ke

    # Temperature
    kb = 0.001987  # kcal/mol/K
    dof = max(3 * len(masses) - 6, 1)
    temp = 2 * ke / (kb * dof)

    if step % 100 == 0:
        drift = (total_energy - initial_total) / abs(initial_total) * 100
        print(f"Step {step:4d}: PE = {pe:10.6f}, KE = {ke:8.6f}, "
              f"Total = {total_energy:10.6f} kcal/mol, T = {temp:7.1f} K, Drift = {drift:+8.5f}%")

print()
print("="*80)
print("FINAL ANALYSIS")
print("="*80)
print()

final_pe = pe
final_ke = ke
final_total = total_energy
energy_drift = (final_total - initial_total) / abs(initial_total) * 100

print(f"Initial total energy: {initial_total:.6f} kcal/mol")
print(f"Final total energy:   {final_total:.6f} kcal/mol")
print(f"Energy drift:         {energy_drift:.6f}%")
print(f"Final temperature:    {temp:.1f} K")
print()

if abs(energy_drift) < 1.0:
    print(f"✓ STABLE: Energy drift < 1% ({abs(energy_drift):.3f}%)")
elif abs(energy_drift) < 5.0:
    print(f"⚠️  MARGINAL: Energy drift = {abs(energy_drift):.3f}%")
else:
    print(f"❌ UNSTABLE: Energy drift = {abs(energy_drift):.3f}%")
