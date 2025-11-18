#!/usr/bin/env python3
"""
Test ANI2x + Implicit Solvent (GB/OBC) combined forces
This is the FULL physics: bonded (ANI) + solvation (GB)
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import tempfile
import os

sys.path.insert(0, 'src')

from fennol.md.initial import load_model, load_system_data, initialize_preprocessing

# Configure JAX
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

print("="*80)
print("MD WITH ANI2x + IMPLICIT SOLVENT (GB/OBC)")
print("="*80)
print()

# Create temporary XYZ file for water
xyz_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False)
xyz_file.write("""3
Water molecule
O      0.000   0.000   0.000
H      0.757   0.586   0.000
H     -0.757   0.586   0.000
""")
xyz_file.close()

# Simulation parameters - combining ANI2x with implicit solvent
params = {
    "xyz_input/file": xyz_file.name,  # Path to XYZ file
    "xyz_input/indexed": False,  # XYZ file doesn't have index column
    "xyz_input/has_comment_line": True,  # XYZ has comment line
    "model_file": "examples/md/ani2x.fnx",
    "double_precision": True,
    "device": "cpu",
    "minimum_image": False,
    "wrap_box": False,
    # Implicit solvent parameters (OBC Generalized Born)
    "implicit_solvent": {
        "model": "OBC",
        "dielectric": 80.0,
        "cutoff": 12.0,
        "surface_tension": 0.005,
        "probe_radius": 1.4,
        "radii_set": "mbondi",
        "include_nonpolar": True
    }
}

print("Loading ANI2x model with implicit solvent...")
try:
    model = load_model(params)
    system_data, conformation = load_system_data(params, "float64")
    _, conformation = initialize_preprocessing(params, model, conformation, system_data)
    print("✓ ANI2x + GB implicit solvent loaded successfully")
    print()
    print(f"Model components: {model}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    os.unlink(xyz_file.name)
    sys.exit(1)

print()
print("="*80)
print("COMPUTING INITIAL ENERGY AND FORCES")
print("="*80)
print()

# Get initial coordinates
coords = conformation["coordinates"]
print(f"Initial coordinates:\n{coords}")
print(f"Conformation keys: {list(conformation.keys())}")
print()

# Compute energy and forces directly from model
# The model includes both ANI2x and GB, and computes forces analytically
# Pass as kwargs to avoid dict unpacking issues
energy, forces, info = model.energy_and_forces(**conformation)

print(f"Initial energy: {energy} Hartree")
print(f"Initial forces (Hartree/Bohr):\n{forces}")
print()

# Check if forces are reasonable
force_magnitude = jnp.linalg.norm(forces)
print(f"Total force magnitude: {force_magnitude} Hartree/Bohr")
print()

# Check for NaN
has_nan = jnp.any(jnp.isnan(forces)) or jnp.any(jnp.isnan(energy))
print(f"Contains NaN: {has_nan}")
print()

# Simple MD integration (Velocity Verlet with Langevin thermostat)
print("="*80)
print("RUNNING MD SIMULATION")
print("="*80)
print()

dt = 0.00048888  # atomic units (~0.5 fs)
n_steps = 500
temperature = 300.0  # Kelvin
kb = 3.166811563e-6  # Boltzmann constant in Hartree/K
gamma = 0.001  # friction coefficient (au)

# Convert masses to atomic units
masses_au = jnp.array([15.999, 1.008, 1.008]) * 1822.888  # amu to electron masses

# Initialize velocities at target temperature
np.random.seed(42)
velocities = jnp.array(np.random.randn(3, 3)) * jnp.sqrt(kb * temperature / masses_au[:, None])

# Remove center of mass motion
total_mom = jnp.sum(masses_au[:, None] * velocities, axis=0)
velocities = velocities - total_mom / jnp.sum(masses_au)

# Trajectory storage
trajectory = []
energies_list = []
temperatures_list = []

print(f"Timestep: {dt*0.02419} fs")
print(f"Temperature: {temperature} K")
print(f"Thermostat: Langevin (gamma = {gamma} au)")
print()
print("Step    PE(Hartree)    KE(Hartree)    Total         Temp(K)     O-H1(Å)   O-H2(Å)")
print("-"*90)

for step in range(n_steps):
    # Langevin dynamics (BAOAB integrator)
    # B: velocity half-step
    accel = forces / masses_au[:, None]
    velocities = velocities + 0.5 * accel * dt

    # A: position update
    coords = coords + velocities * dt

    # O: Ornstein-Uhlenbeck (thermostat)
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt((1 - c1**2) * kb * temperature / masses_au[:, None])
    random_force = jnp.array(np.random.randn(3, 3))
    velocities = c1 * velocities + c2 * random_force

    # Compute new forces
    energy, forces_grad = energy_and_grad(coords)
    forces = -forces_grad

    # B: velocity half-step
    accel_new = forces / masses_au[:, None]
    velocities = velocities + 0.5 * accel_new * dt

    # Compute energies
    pe = float(energy)
    ke = float(0.5 * jnp.sum(masses_au[:, None] * velocities**2))
    total = pe + ke
    temp = 2 * ke / (3 * 3 * kb)  # 3 atoms, 3 DOF each

    # Compute bond lengths (convert to Angstrom)
    bohr_to_angstrom = 0.529177
    oh1_dist = float(jnp.linalg.norm(coords[1] - coords[0])) * bohr_to_angstrom
    oh2_dist = float(jnp.linalg.norm(coords[2] - coords[0])) * bohr_to_angstrom

    # Save trajectory
    if step % 10 == 0:
        trajectory.append(np.array(coords))
        energies_list.append([pe, ke, total])
        temperatures_list.append(temp)

    if step % 50 == 0:
        print(f"{step:4d}  {pe:12.8f}  {ke:12.8f}  {total:12.8f}  {temp:8.1f}  {oh1_dist:8.4f}  {oh2_dist:8.4f}")

print()
print("="*80)
print("FINAL ANALYSIS")
print("="*80)
print()

trajectory = np.array(trajectory)
energies_arr = np.array(energies_list)
temperatures_arr = np.array(temperatures_list)

print(f"Final O-H1 distance: {oh1_dist:.4f} Å")
print(f"Final O-H2 distance: {oh2_dist:.4f} Å")
print(f"Average temperature: {np.mean(temperatures_arr):.1f} K")
print(f"Std temperature: {np.std(temperatures_arr):.1f} K")
print()

if oh1_dist < 1.2 and oh2_dist < 1.2:
    print("✓ SUCCESS! Bonds remained stable with ANI2x + GB forces")
    print("  O-H bonds stayed near equilibrium (~0.96 Å)")
else:
    print("⚠️  WARNING: Bonds stretched significantly")

# Save trajectory
bohr_to_angstrom = 0.529177
with open('trajectory_ani_gb.xyz', 'w') as f:
    for i, frame in enumerate(trajectory):
        f.write("3\n")
        f.write(f"Frame {i}, T = {temperatures_list[i]:.1f} K, E = {energies_list[i][2]:.8f} Hartree\n")
        f.write(f"O  {frame[0,0]*bohr_to_angstrom:12.6f} {frame[0,1]*bohr_to_angstrom:12.6f} {frame[0,2]*bohr_to_angstrom:12.6f}\n")
        f.write(f"H  {frame[1,0]*bohr_to_angstrom:12.6f} {frame[1,1]*bohr_to_angstrom:12.6f} {frame[1,2]*bohr_to_angstrom:12.6f}\n")
        f.write(f"H  {frame[2,0]*bohr_to_angstrom:12.6f} {frame[2,1]*bohr_to_angstrom:12.6f} {frame[2,2]*bohr_to_angstrom:12.6f}\n")

print()
print(f"✓ Saved trajectory to trajectory_ani_gb.xyz")

# Cleanup
os.unlink(xyz_file.name)
