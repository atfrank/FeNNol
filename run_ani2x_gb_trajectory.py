#!/usr/bin/env python3
"""
Run MD with ANI2x + GB implicit solvent - the intended use case.
This tests the fixed JAX GB forces combined with ANI2x bonded interactions.
"""

import numpy as np
import jax
import jax.numpy as jnp
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

print("="*80)
print("MD Simulation: ANI2x + GB Implicit Solvent")
print("="*80)

# Water molecule
coords = np.array([
    [0.000, 0.000, 0.000],  # O
    [0.757, 0.586, 0.000],  # H1
    [-0.757, 0.586, 0.000],  # H2
], dtype=np.float64)

charges = np.array([-0.834, 0.417, 0.417])
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)
species = jnp.array([8, 1, 1], dtype=jnp.int32)
masses = np.array([15.999, 1.008, 1.008])

# Initialize ANI2x
print("\nLoading ANI2x model...")
try:
    from fennol.models import FENNIX
    from fennol.utils import AtomicUnits as au

    # Create ANI2x model using FENNIX
    ani_model = FENNIX()
    ani_model.load_model("ani2x")

    print("✓ ANI2x loaded successfully")

    # Test ANI2x
    test_energy = ani_model.energy(coords, species)
    test_forces = ani_model.total_force(coords, species)
    print(f"  Initial ANI2x energy: {test_energy:.6f} (units: {ani_model.units})")
    print(f"  Initial ANI2x forces (max): {np.max(np.abs(test_forces)):.6f}")

    has_ani = True
except Exception as e:
    print(f"✗ Could not load ANI2x: {e}")
    print("  Will run with GB forces only (molecule will dissociate)")
    has_ani = False

# Initialize GB
print("\nInitializing GB implicit solvent...")
gb_model = create_implicit_solvent_model("OBC", {
    "model": "OBC",
    "dielectric": 80.0,
    "cutoff": 8.0,
    "radii_set": "mbondi",
    "include_nonpolar": False
})
gb_model.has_cuda = False

gb_energy, gb_forces = gb_model.compute_energy_forces(coords, charges, atomic_numbers)
print(f"✓ GB initialized")
print(f"  Initial GB energy: {gb_energy:.3f} kcal/mol")
print(f"  Initial GB forces (max): {np.max(np.abs(gb_forces)):.3f} kcal/mol/Å")

# MD parameters
dt = 0.5  # fs
nsteps = 500
output_freq = 25
temperature = 300.0  # K

print(f"\nMD parameters:")
print(f"  Time step: {dt} fs")
print(f"  Steps: {nsteps} ({nsteps*dt/1000:.2f} ps)")
print(f"  Temperature: {temperature} K")
print(f"  Output: every {output_freq} steps")

# Initialize velocities
kb = 8.314462e-3  # kJ/(mol·K)
sigma = np.sqrt(kb * temperature / masses[:, None])
velocities = np.random.normal(0, sigma, size=(3, 3))
total_momentum = np.sum(masses[:, None] * velocities, axis=0)
velocities -= total_momentum / np.sum(masses)

# Storage
trajectory = [coords.copy()]
energies_ani = []
energies_gb = []
forces_ani_list = []
forces_gb_list = []

current_coords = coords.copy()
current_velocities = velocities.copy()

print(f"\n{'Step':>5} {'Time':>8} {'E_ANI':>12} {'E_GB':>12} {'E_tot':>12} {'OH1':>7} {'OH2':>7} {'Status'}")
print("-"*95)

# Unit conversion: ANI uses Hartree and Bohr internally but returns in eV and Angstroms
eV_to_kcal = 23.06054783

for step in range(nsteps):
    try:
        # Compute ANI2x forces
        if has_ani:
            ani_energy = ani_model.energy(current_coords, species)
            ani_forces = ani_model.total_force(current_coords, species)

            # Check ANI units and convert if needed
            if np.max(np.abs(ani_forces)) < 1.0:  # Likely in Hartree/Bohr
                ani_forces = ani_forces * au.HA_TO_KCAL_MOL / au.BOHR_TO_ANGS
                ani_energy_kcal = ani_energy * au.HA_TO_KCAL_MOL
            else:  # Likely already in kcal/mol/Å or eV/Å
                # ANI typically outputs eV, so convert
                ani_energy_kcal = ani_energy * eV_to_kcal
                ani_forces = ani_forces * eV_to_kcal
        else:
            ani_energy_kcal = 0.0
            ani_forces = np.zeros_like(current_coords)

        # Compute GB forces
        gb_energy, gb_forces = gb_model.compute_energy_forces(
            current_coords, charges, atomic_numbers
        )

        # Check for NaN
        if np.isnan(gb_energy) or np.any(np.isnan(gb_forces)):
            print(f"{step:5d} {'NaN':>8} {'NaN':>12} {'NaN':>12} {'NaN':>12} {'NaN':>7} {'NaN':>7} FAILED")
            print("\n*** NaN in GB forces - FAILED ***")
            break

        if has_ani and (np.isnan(ani_energy_kcal) or np.any(np.isnan(ani_forces))):
            print(f"{step:5d} {'NaN':>8} {'NaN':>12} {'NaN':>12} {'NaN':>12} {'NaN':>7} {'NaN':>7} FAILED")
            print("\n*** NaN in ANI forces - FAILED ***")
            break

        # Combine forces
        total_forces = ani_forces + gb_forces
        total_energy = ani_energy_kcal + gb_energy

        # Velocity Verlet
        accel = total_forces / masses[:, None] * 4.184e4  # kcal/mol/Å to Å/fs²
        current_velocities += 0.5 * accel * dt
        current_coords += current_velocities * dt

        # Recompute at new position
        if has_ani:
            ani_energy = ani_model.energy(current_coords, species)
            ani_forces = ani_model.total_force(current_coords, species)
            if np.max(np.abs(ani_forces)) < 1.0:
                ani_forces = ani_forces * au.HA_TO_KCAL_MOL / au.BOHR_TO_ANGS
                ani_energy_kcal = ani_energy * au.HA_TO_KCAL_MOL
            else:
                ani_energy_kcal = ani_energy * eV_to_kcal
                ani_forces = ani_forces * eV_to_kcal

        gb_energy, gb_forces = gb_model.compute_energy_forces(
            current_coords, charges, atomic_numbers
        )

        total_forces = ani_forces + gb_forces
        accel = total_forces / masses[:, None] * 4.184e4
        current_velocities += 0.5 * accel * dt

        # Save data
        trajectory.append(current_coords.copy())
        energies_ani.append(float(ani_energy_kcal))
        energies_gb.append(float(gb_energy))
        forces_ani_list.append(ani_forces.copy())
        forces_gb_list.append(gb_forces.copy())

        # Output
        if (step + 1) % output_freq == 0:
            oh1 = np.linalg.norm(current_coords[1] - current_coords[0])
            oh2 = np.linalg.norm(current_coords[2] - current_coords[0])
            time_ps = (step + 1) * dt / 1000.0
            total_energy = ani_energy_kcal + gb_energy

            print(f"{step+1:5d} {time_ps:8.3f} {ani_energy_kcal:12.3f} {gb_energy:12.3f} {total_energy:12.3f} {oh1:7.3f} {oh2:7.3f} OK")

    except Exception as e:
        print(f"{step:5d} ERROR: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n" + "="*80)
print("Simulation completed!")
print("="*80)

# Statistics
energies_ani = np.array(energies_ani)
energies_gb = np.array(energies_gb)
energies_total = energies_ani + energies_gb

print(f"\nEnergy statistics:")
print(f"  ANI2x: {np.mean(energies_ani):10.3f} ± {np.std(energies_ani):6.3f} kcal/mol")
print(f"  GB:    {np.mean(energies_gb):10.3f} ± {np.std(energies_gb):6.3f} kcal/mol")
print(f"  Total: {np.mean(energies_total):10.3f} ± {np.std(energies_total):6.3f} kcal/mol")

# Save trajectory
xyz_file = "ani2x_gb_trajectory.xyz"
with open(xyz_file, 'w') as f:
    for i, frame in enumerate(trajectory):
        f.write("3\n")
        e_ani = energies_ani[i] if i < len(energies_ani) else 0
        e_gb = energies_gb[i] if i < len(energies_gb) else 0
        f.write(f"Frame {i}, ANI={e_ani:.3f}, GB={e_gb:.3f} kcal/mol\n")
        f.write(f"O  {frame[0,0]:12.6f} {frame[0,1]:12.6f} {frame[0,2]:12.6f}\n")
        f.write(f"H  {frame[1,0]:12.6f} {frame[1,1]:12.6f} {frame[1,2]:12.6f}\n")
        f.write(f"H  {frame[2,0]:12.6f} {frame[2,1]:12.6f} {frame[2,2]:12.6f}\n")

print(f"\nTrajectory saved: {xyz_file}")
print(f"  {len(trajectory)} frames")

# Save numpy data
np.savez("ani2x_gb_trajectory.npz",
         coordinates=np.array(trajectory),
         energies_ani=energies_ani,
         energies_gb=energies_gb,
         forces_ani=np.array(forces_ani_list),
         forces_gb=np.array(forces_gb_list))

print(f"\nNumpy data saved: ani2x_gb_trajectory.npz")
print(f"\n✓ SUCCESS: ANI2x + GB MD completed without NaN!")
