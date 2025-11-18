#!/usr/bin/env python3
"""
Run MD simulation and save trajectory for visualization
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

print("="*80)
print("MD SIMULATION WITH TRAJECTORY SAVING")
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
dt = 0.0001  # 0.1 fs
n_steps = 1000
conversion = 418.4  # kcal/mol/A to (A/ps)^2

# Start from rest
velocities = np.zeros_like(coords)

# Trajectory storage
trajectory = []
energies = []
temperatures = []
times = []

def compute_forces(coords):
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
    energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
    dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
    dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
    forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
    return float(energy), forces_direct + forces_born, born_radii

# Initial energy
energy, forces, born_radii = compute_forces(coords)
initial_total = float(energy)

print(f"Initial PE:    {initial_total:.6f} kcal/mol")
print(f"Timestep:      {dt} ps = {dt*1000} fs")
print(f"Total time:    {n_steps*dt} ps = {n_steps*dt*1000} fs")
print(f"Ensemble:      NVE (microcanonical - no thermostat)")
print()
print("Starting MD from rest...")
print()

# MD loop
kb = 0.001987  # kcal/mol/K
dof = max(3 * len(masses) - 6, 1)

for step in range(n_steps):
    # Velocity Verlet
    acceleration = forces * conversion / masses[:, np.newaxis]
    velocities += 0.5 * acceleration * dt
    coords += velocities * dt

    energy_new, forces_new, born_radii_new = compute_forces(coords)

    acceleration_new = forces_new * conversion / masses[:, np.newaxis]
    velocities += 0.5 * acceleration_new * dt
    forces = forces_new

    # Compute energies
    pe = energy_new
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) / conversion
    total = pe + ke
    temp = 2 * ke / (kb * dof)

    # Save trajectory every 10 steps
    if step % 10 == 0:
        trajectory.append(coords.copy())
        energies.append([pe, ke, total])
        temperatures.append(temp)
        times.append(step * dt)

    if step % 100 == 0:
        drift = (total - initial_total) / abs(initial_total) * 100
        print(f"Step {step:4d}: PE = {pe:10.6f}, KE = {ke:8.6f}, "
              f"Total = {total:10.6f} kcal/mol, T = {temp:7.1f} K, Drift = {drift:+8.5f}%")

print()
print("="*80)
print("SAVING TRAJECTORY")
print("="*80)
print()

# Convert to numpy arrays
trajectory = np.array(trajectory)
energies = np.array(energies)
temperatures = np.array(temperatures)
times = np.array(times)

# Save as XYZ file
with open('trajectory.xyz', 'w') as f:
    for i, frame in enumerate(trajectory):
        f.write("3\n")
        f.write(f"Frame {i}, Time = {times[i]:.4f} ps, T = {temperatures[i]:.1f} K, E = {energies[i,2]:.6f} kcal/mol\n")
        f.write(f"O  {frame[0,0]:12.6f} {frame[0,1]:12.6f} {frame[0,2]:12.6f}\n")
        f.write(f"H  {frame[1,0]:12.6f} {frame[1,1]:12.6f} {frame[1,2]:12.6f}\n")
        f.write(f"H  {frame[2,0]:12.6f} {frame[2,1]:12.6f} {frame[2,2]:12.6f}\n")

print(f"✓ Saved {len(trajectory)} frames to trajectory.xyz")
print()

# Save energy data
np.savetxt('energies.dat',
           np.column_stack([times, energies[:,0], energies[:,1], energies[:,2], temperatures]),
           header='Time(ps)  PE(kcal/mol)  KE(kcal/mol)  Total(kcal/mol)  T(K)',
           fmt='%12.6f')

print(f"✓ Saved energy data to energies.dat")
print()

print("="*80)
print("VISUALIZATION")
print("="*80)
print()
print("To visualize the trajectory:")
print("  - Use VMD: vmd trajectory.xyz")
print("  - Use PyMOL: pymol trajectory.xyz")
print("  - Use Ovito or other molecular viewers")
print()
print("To plot energies:")
print("  - gnuplot: plot 'energies.dat' using 1:4 with lines")
print("  - Python: import matplotlib.pyplot as plt; data = np.loadtxt('energies.dat'); plt.plot(data[:,0], data[:,3])")
