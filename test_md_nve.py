#!/usr/bin/env python3
"""
Test NVE (microcanonical ensemble) - energy should be perfectly conserved
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("NVE MD TEST - Perfect Energy Conservation")
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

# MD parameters - use MUCH smaller timestep to test energy conservation
dt = 0.0001  # 0.1 fs - very small!
n_steps = 100
conversion = 418.4  # kcal/mol/A to (A/ps)^2

# Initialize with small random velocities
np.random.seed(42)
velocities = np.random.randn(3, 3) * 0.001  # Very small initial velocities

# Remove center of mass motion
total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
velocities -= total_momentum / np.sum(masses)

# Compute initial energy
born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
forces = forces_direct + forces_born

initial_pe = float(energy)
initial_ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) / conversion
initial_total = initial_pe + initial_ke

print(f"Initial PE: {initial_pe:.10f} kcal/mol")
print(f"Initial KE: {initial_ke:.10f} kcal/mol")
print(f"Initial Total: {initial_total:.10f} kcal/mol")
print()

print(f"Starting NVE MD ({n_steps} steps, dt = {dt} ps)...")
print()

# MD loop
energies = []
for step in range(n_steps):
    # Velocity Verlet integration
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
    forces = forces_new

    # Compute energies
    pe = float(energy)
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) / conversion
    total_energy = pe + ke
    energies.append(total_energy)

    if step % 10 == 0:
        drift = (total_energy - initial_total) / abs(initial_total) * 100
        print(f"Step {step:3d}: PE = {pe:12.8f}, KE = {ke:12.8f}, Total = {total_energy:12.8f}, Drift = {drift:+8.5f}%")

print()
print("="*80)
print("ENERGY CONSERVATION ANALYSIS")
print("="*80)
print()

energies = np.array(energies)
final_total = energies[-1]
drift = (final_total - initial_total) / abs(initial_total) * 100

print(f"Initial total energy: {initial_total:.10f} kcal/mol")
print(f"Final total energy:   {final_total:.10f} kcal/mol")
print(f"Energy drift:         {drift:.8f}%")
print(f"Max energy:           {np.max(energies):.10f} kcal/mol")
print(f"Min energy:           {np.min(energies):.10f} kcal/mol")
print(f"Std deviation:        {np.std(energies):.10f} kcal/mol")
print()

if abs(drift) < 0.01:
    print(f"✓ EXCELLENT: Energy conserved to < 0.01% (drift = {drift:.8f}%)")
elif abs(drift) < 0.1:
    print(f"✓ GOOD: Energy conserved to < 0.1% (drift = {drift:.8f}%)")
elif abs(drift) < 1.0:
    print(f"⚠️  ACCEPTABLE: Energy drift < 1% (drift = {drift:.8f}%)")
else:
    print(f"❌ POOR: Energy drift > 1% (drift = {drift:.8f}%)")
    print("This indicates a problem with force calculation or integration!")
