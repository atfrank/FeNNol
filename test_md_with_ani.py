#!/usr/bin/env python3
"""
Test MD with ANI2x neural network potential
This includes proper bonded interactions!
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, 'src')

from fennol.md.initial import load_model, load_system_data, initialize_preprocessing

# Configure JAX
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

print("="*80)
print("MD WITH ANI2x NEURAL NETWORK POTENTIAL")
print("="*80)
print()

# Create temporary XYZ file for water
import tempfile
xyz_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False)
xyz_file.write("""3
Water molecule
O      0.000   0.000   0.000
H      0.757   0.586   0.000
H     -0.757   0.586   0.000
""")
xyz_file.close()

# Simulation parameters
params = {
    "coordinates": xyz_file.name,
    "model_type": "ani2x",
    "double_precision": True,
    "device": "cpu",
}

print("Loading ANI2x model...")
try:
    model = load_model(params)
    system_data, conformation = load_system_data(params, "float64")
    _, conformation = initialize_preprocessing(model, system_data, conformation, params)
    print("✓ ANI2x model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load ANI2x model: {e}")
    print()
    print("Note: ANI2x model file (ani2x.fnx) may be missing.")
    print("This is expected - we've been testing only GB forces in isolation.")
    sys.exit(1)

print()
print("Computing forces with ANI2x...")

# Get initial coordinates
coords = conformation["coordinates"]
species = conformation["species"]

# Compute energy and forces
def compute_energy_forces(coords_input):
    """Compute energy and forces using ANI model"""
    conf = {**conformation, "coordinates": coords_input}
    energy = model.energy(conf)
    return energy

# Compute forces via autodiff
energy_and_grad = jax.value_and_grad(compute_energy_forces)
energy, forces = energy_and_grad(coords)
forces = -forces  # Negative gradient gives forces

print(f"Initial energy: {float(energy):.6f} Hartree")
print(f"Initial forces:\n{forces}")
print()

# Simple MD integration (Velocity Verlet)
print("Running short MD simulation...")
print()

dt = 0.0005  # ps = 0.5 fs
n_steps = 100
masses_au = jnp.array([15.999, 1.008, 1.008]) * 1822.888  # Convert amu to atomic units

velocities = jnp.zeros_like(coords)

for step in range(n_steps):
    # Velocity Verlet
    accel = forces / masses_au[:, None]
    velocities = velocities + 0.5 * accel * dt
    coords = coords + velocities * dt

    # Compute new forces
    energy, forces = energy_and_grad(coords)
    forces = -forces

    accel_new = forces / masses_au[:, None]
    velocities = velocities + 0.5 * accel_new * dt

    # Compute kinetic energy
    ke = 0.5 * jnp.sum(masses_au[:, None] * velocities**2)
    total = energy + ke

    if step % 20 == 0:
        print(f"Step {step:3d}: E = {float(energy):.6f}, KE = {float(ke):.6f}, "
              f"Total = {float(total):.6f} Hartree")

print()
print("✓ MD with ANI2x completed successfully!")
print()
print("With ANI potential, bonds should be stable (includes bonded interactions)")

# Cleanup
import os
os.unlink(xyz_file.name)
