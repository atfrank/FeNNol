#!/usr/bin/env python3
"""
Debug MD - track geometry changes
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/src/fennol/cuda')
import fennol_cuda

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

dt = 0.0001  # 0.1 fs
conversion = 418.4
velocities = np.zeros_like(coords)

def compute_forces(coords):
    born_radii, psi_sum = fennol_cuda.gb_compute_born_radii_with_psi(coords, radii, b_params, c_params, cutoff)
    energy, forces_direct = fennol_cuda.gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)
    dE_dR = fennol_cuda.compute_dE_dR(coords, charges, born_radii, dielectric, cutoff)
    dE_dpsi = fennol_cuda.reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)
    forces_born = fennol_cuda.apply_born_forces(coords, radii, dE_dpsi, cutoff)
    return float(energy), forces_direct + forces_born, born_radii

energy, forces, born_radii = compute_forces(coords)
initial_total = float(energy)

print("Step    PE         KE         Total      Drift%     Temp(K)  O-H1_dist  O-H2_dist  H1-H2_dist  BornR_O  BornR_H1")
print("-" * 130)

for step in range(500):
    # Velocity Verlet
    acceleration = forces * conversion / masses[:, np.newaxis]
    velocities += 0.5 * acceleration * dt
    coords += velocities * dt

    energy_new, forces_new, born_radii_new = compute_forces(coords)

    acceleration_new = forces_new * conversion / masses[:, np.newaxis]
    velocities += 0.5 * acceleration_new * dt
    forces = forces_new

    pe = energy_new
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) / conversion
    total = pe + ke
    drift = (total - initial_total) / abs(initial_total) * 100

    kb = 0.001987
    dof = max(3 * len(masses) - 6, 1)
    temp = 2 * ke / (kb * dof)

    # Bond distances
    oh1_dist = np.linalg.norm(coords[1] - coords[0])
    oh2_dist = np.linalg.norm(coords[2] - coords[0])
    h1h2_dist = np.linalg.norm(coords[2] - coords[1])

    if step % 20 == 0 or (step >= 300 and step <= 450):
        print(f"{step:4d}  {pe:10.5f}  {ke:9.5f}  {total:10.5f}  {drift:+9.5f}  {temp:8.1f}  {oh1_dist:9.5f}  {oh2_dist:9.5f}  {h1h2_dist:10.5f}  {born_radii_new[0]:7.4f}  {born_radii_new[1]:8.4f}")
