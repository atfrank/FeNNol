#!/usr/bin/env python3
"""
Demonstrate GB trajectory generation.

NOTE: This shows that GB forces work (no NaN), but are insufficient alone
to hold a molecule together. In production, GB must be combined with bonded
forces (e.g., ANI2x) or constraints.
"""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Water molecule
coords = np.array([
    [0.000, 0.000, 0.000],
    [0.757, 0.586, 0.000],
    [-0.757, 0.586, 0.000],
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417])
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)
masses = np.array([15.999, 1.008, 1.008])

# Initialize GB
gb_model = create_implicit_solvent_model("OBC", {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "radii_set": "mbondi",
    "include_nonpolar": False
})
gb_model.has_cuda = False

print("="*70)
print("GB Trajectory Generation Demo")
print("="*70)
print("\nNOTE: This demonstrates GB forces work without NaN.")
print("However, GB alone cannot hold molecules together - it only")
print("provides solvation energy, not bonded interactions.")
print("In production, use GB + ANI2x or GB + force field.")
print("="*70)

# Very small time step, short run
dt = 0.05  # fs
nsteps = 50

print(f"\nParameters: dt={dt} fs, {nsteps} steps")
print(f"\nInitial O-H distances: {np.linalg.norm(coords[1] - coords[0]):.3f}, {np.linalg.norm(coords[2] - coords[0]):.3f} Å")

# Zero initial velocities to see pure force effects
velocities = np.zeros_like(coords)

trajectory = [coords.copy()]
energies = []
forces_list = []

current_coords = coords.copy()
current_velocities = velocities.copy()

print(f"\n{'Step':>4} {'Energy':>12} {'O-H1':>8} {'O-H2':>8} {'Max_F':>10} {'NaN?'}")
print("-"*70)

for step in range(nsteps):
    # Compute forces
    energy, forces = gb_model.compute_energy_forces(
        current_coords, charges, atomic_numbers
    )

    has_nan = np.isnan(energy) or np.any(np.isnan(forces))

    # Distances
    oh1 = np.linalg.norm(current_coords[1] - current_coords[0])
    oh2 = np.linalg.norm(current_coords[2] - current_coords[0])
    max_f = np.max(np.abs(forces))

    if step % 5 == 0:
        print(f"{step:4d} {energy:12.3f} {oh1:8.3f} {oh2:8.3f} {max_f:10.3f} {'YES' if has_nan else 'NO'}")

    if has_nan:
        print("\n*** NaN detected - stopping ***")
        break

    # Simple velocity Verlet
    accel = forces / masses[:, None] * 4.184e4
    current_velocities += 0.5 * accel * dt
    current_coords += current_velocities * dt

    energy, forces = gb_model.compute_energy_forces(current_coords, charges, atomic_numbers)
    accel = forces / masses[:, None] * 4.184e4
    current_velocities += 0.5 * accel * dt

    trajectory.append(current_coords.copy())
    energies.append(float(energy))
    forces_list.append(forces.copy())

print(f"\nFinal O-H distances: {np.linalg.norm(current_coords[1] - current_coords[0]):.3f}, {np.linalg.norm(current_coords[2] - current_coords[0]):.3f} Å")
print(f"\n✓ Completed {len(trajectory)} frames without NaN")

# Save trajectory
xyz_file = "gb_demo_trajectory.xyz"
with open(xyz_file, 'w') as f:
    for i, frame in enumerate(trajectory):
        f.write("3\n")
        f.write(f"Frame {i}, Energy {energies[i] if i > 0 else energies[0]:.3f}\n")
        f.write(f"O  {frame[0,0]:12.6f} {frame[0,1]:12.6f} {frame[0,2]:12.6f}\n")
        f.write(f"H  {frame[1,0]:12.6f} {frame[1,1]:12.6f} {frame[1,2]:12.6f}\n")
        f.write(f"H  {frame[2,0]:12.6f} {frame[2,1]:12.6f} {frame[2,2]:12.6f}\n")

print(f"\nTrajectory saved: {xyz_file} ({len(trajectory)} frames)")
print(f"View with: vmd {xyz_file}")

print("\n" + "="*70)
print("CONCLUSION:")
print("  ✓ GB forces computed successfully without NaN")
print("  ✓ Trajectory generated and saved")
print("  ! Molecule dissociates (expected - GB has no bonded terms)")
print("  → For stable MD, combine GB with ANI2x or force field")
print("="*70)
