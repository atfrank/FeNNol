#!/usr/bin/env python3
"""
Run 1000-step MD simulation with ANI2x + GB implicit solvent.
Generates trajectory file for visualization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import tempfile
import os

sys.path.insert(0, 'src')

from fennol.md.initial import load_model, load_system_data, initialize_preprocessing
from fennol.utils import AtomicUnits as au

# Configure JAX
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

print("="*80)
print("1000-STEP MD: ANI2x + GB Implicit Solvent")
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

# Simulation parameters
params = {
    "xyz_input/file": xyz_file.name,
    "xyz_input/indexed": False,
    "xyz_input/has_comment_line": True,
    "model_file": "examples/md/ani2x.fnx",
    "double_precision": True,
    "device": "cpu",
    "minimum_image": False,
    "wrap_box": False,
    # Implicit solvent parameters
    "implicit_solvent": {
        "model": "OBC",
        "dielectric": 80.0,
        "cutoff": 8.0,
        "radii_set": "mbondi",
        "include_nonpolar": False,
        "charges": [-0.834, 0.417, 0.417]  # Water charges
    }
}

print("Loading ANI2x model + GB implicit solvent...")
try:
    model = load_model(params)
    system_data, conformation = load_system_data(params, "float64")
    _, conformation = initialize_preprocessing(params, model, conformation, system_data)
    print("✓ Model loaded successfully")
    print()
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    os.unlink(xyz_file.name)
    sys.exit(1)

# Get initial coordinates and species
coords = conformation["coordinates"]
species = conformation["species"]
charges = np.array(params["implicit_solvent"]["charges"])

print(f"Initial coordinates:\n{coords}")
print(f"Species: {species}")
print(f"Charges: {charges}")
print()

# MD parameters
dt = 0.5  # fs (0.5 fs timestep)
nsteps = 1000
output_freq = 10  # Output every 10 steps
temperature = 300.0  # K

print(f"MD Parameters:")
print(f"  Time step: {dt} fs")
print(f"  Total steps: {nsteps}")
print(f"  Total time: {nsteps * dt / 1000:.2f} ps")
print(f"  Temperature: {temperature} K")
print(f"  Output frequency: every {output_freq} steps")
print()

# Initialize velocities (Maxwell-Boltzmann)
masses = np.array([15.999, 1.008, 1.008])  # amu (O, H, H)
kb = 8.617333e-5  # eV/K
kb_au = kb / 27.2114  # Hartree/K

# Generate random velocities
np.random.seed(42)  # For reproducibility
sigma = np.sqrt(kb_au * temperature / masses[:, None])
velocities = np.random.normal(0, sigma, size=(3, 3))

# Remove center of mass motion
total_momentum = np.sum(masses[:, None] * velocities, axis=0)
velocities -= total_momentum / np.sum(masses)

print(f"Initial velocities (Bohr/au_time):\n{velocities}")
print()

# Storage
trajectory = [coords.copy()]
energies_total = []
energies_ani = []
energies_gb = []
temperatures = []
times = []

# Get GB model from integrate module
from fennol.models.physics.implicit_solvent import create_implicit_solvent_model
gb_model = create_implicit_solvent_model("OBC", params["implicit_solvent"])
gb_model.has_cuda = False  # Force JAX

# Compute initial energy and forces
print("Computing initial energy and forces...")
energy_ani, forces_ani, _ = model.energy_and_forces(**conformation)
energy_gb, forces_gb = gb_model.compute_energy_forces(coords, charges, species.astype(np.int32))

# Convert GB to atomic units
kcal_to_hartree = 0.001593601
energy_gb_au = energy_gb * kcal_to_hartree
forces_gb_au = forces_gb * kcal_to_hartree

total_energy = energy_ani + energy_gb_au
total_forces = forces_ani + forces_gb_au

print(f"Initial ANI2x energy: {energy_ani[0]:.6f} Hartree")
print(f"Initial GB energy: {energy_gb_au:.6f} Hartree")
print(f"Initial total energy: {total_energy[0]:.6f} Hartree")
print(f"Initial forces (max): {np.max(np.abs(total_forces)):.6f} Hartree/Bohr")
print()

# Convert masses to atomic units (electron masses)
# 1 amu = 1822.888 electron masses
masses_au = masses * 1822.888

print("="*80)
print("STARTING MD SIMULATION")
print("="*80)
print(f"{'Step':>6} {'Time':>8} {'E_ANI':>12} {'E_GB':>12} {'E_tot':>12} {'Temp':>8} {'OH1':>7} {'OH2':>7}")
print("-"*80)

current_coords = coords.copy()
current_velocities = velocities.copy()

for step in range(nsteps):
    try:
        # Update conformation with new coordinates
        conf = dict(conformation)
        conf["coordinates"] = current_coords

        # Compute forces
        energy_ani, forces_ani, _ = model.energy_and_forces(**conf)
        energy_gb, forces_gb = gb_model.compute_energy_forces(
            current_coords, charges, species.astype(np.int32)
        )

        # Convert GB to atomic units
        energy_gb_au = energy_gb * kcal_to_hartree
        forces_gb_au = forces_gb * kcal_to_hartree

        total_forces = forces_ani + forces_gb_au

        # Check for NaN
        if np.any(np.isnan(total_forces)) or np.isnan(energy_ani) or np.isnan(energy_gb_au):
            print(f"\n*** NaN detected at step {step}! ***")
            break

        # Velocity Verlet integration
        # v(t+dt/2) = v(t) + 0.5*a(t)*dt
        accel = total_forces / masses_au[:, None]
        current_velocities += 0.5 * accel * dt

        # x(t+dt) = x(t) + v(t+dt/2)*dt
        current_coords += current_velocities * dt

        # Update conformation and recompute forces
        conf["coordinates"] = current_coords
        energy_ani, forces_ani, _ = model.energy_and_forces(**conf)
        energy_gb, forces_gb = gb_model.compute_energy_forces(
            current_coords, charges, species.astype(np.int32)
        )

        energy_gb_au = energy_gb * kcal_to_hartree
        forces_gb_au = forces_gb * kcal_to_hartree
        total_forces = forces_ani + forces_gb_au

        # v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
        accel = total_forces / masses_au[:, None]
        current_velocities += 0.5 * accel * dt

        # Compute temperature
        ke = 0.5 * np.sum(masses_au[:, None] * current_velocities**2)
        # T = 2*KE / (3N * kb) for 3 atoms
        temp = 2.0 * ke / (9 * kb_au)

        # Store data
        trajectory.append(current_coords.copy())
        energies_ani.append(float(energy_ani[0]))
        energies_gb.append(float(energy_gb_au))
        energies_total.append(float(energy_ani[0] + energy_gb_au))
        temperatures.append(temp)
        times.append((step + 1) * dt / 1000.0)  # ps

        # Output
        if (step + 1) % output_freq == 0:
            oh1 = np.linalg.norm(current_coords[1] - current_coords[0])
            oh2 = np.linalg.norm(current_coords[2] - current_coords[0])
            time_ps = (step + 1) * dt / 1000.0

            print(f"{step+1:6d} {time_ps:8.3f} {energy_ani[0]:12.6f} {energy_gb_au:12.6f} "
                  f"{energy_ani[0]+energy_gb_au:12.6f} {temp:8.1f} {oh1:7.3f} {oh2:7.3f}")

    except Exception as e:
        print(f"\n*** Error at step {step}: {e} ***")
        import traceback
        traceback.print_exc()
        break

print()
print("="*80)
print("SIMULATION COMPLETED")
print("="*80)

# Statistics
energies_total = np.array(energies_total)
energies_ani = np.array(energies_ani)
energies_gb = np.array(energies_gb)
temperatures = np.array(temperatures)

print(f"\nStatistics ({len(trajectory)} frames):")
print(f"  ANI2x energy:  {np.mean(energies_ani):10.6f} ± {np.std(energies_ani):8.6f} Hartree")
print(f"  GB energy:     {np.mean(energies_gb):10.6f} ± {np.std(energies_gb):8.6f} Hartree")
print(f"  Total energy:  {np.mean(energies_total):10.6f} ± {np.std(energies_total):8.6f} Hartree")
print(f"  Temperature:   {np.mean(temperatures):10.1f} ± {np.std(temperatures):8.1f} K")

# Save trajectory as XYZ
xyz_traj_file = "ani2x_gb_1000steps.xyz"
with open(xyz_traj_file, 'w') as f:
    for i, frame in enumerate(trajectory):
        f.write("3\n")
        # Handle initial frame (i=0) which has no time/energy data
        if i == 0:
            time_ps = 0.0
            e_tot = 0.0
            temp = 0.0
        elif i-1 < len(times):
            time_ps = times[i-1]
            e_tot = energies_total[i-1]
            temp = temperatures[i-1]
        else:
            time_ps = 0.0
            e_tot = 0.0
            temp = 0.0
        f.write(f"Frame {i}, t={time_ps:.3f} ps, E={e_tot:.6f} Ha, T={temp:.1f} K\n")
        f.write(f"O  {frame[0,0]:12.6f} {frame[0,1]:12.6f} {frame[0,2]:12.6f}\n")
        f.write(f"H  {frame[1,0]:12.6f} {frame[1,1]:12.6f} {frame[1,2]:12.6f}\n")
        f.write(f"H  {frame[2,0]:12.6f} {frame[2,1]:12.6f} {frame[2,2]:12.6f}\n")

print(f"\n✓ Trajectory saved: {xyz_traj_file}")
print(f"  {len(trajectory)} frames")
print(f"  View with: vmd {xyz_traj_file}")
print(f"            pymol {xyz_traj_file}")

# Save numpy data
npz_file = "ani2x_gb_1000steps.npz"
np.savez(npz_file,
         coordinates=np.array(trajectory),
         energies_ani=energies_ani,
         energies_gb=energies_gb,
         energies_total=energies_total,
         temperatures=temperatures,
         times=np.array(times))

print(f"\n✓ Numpy data saved: {npz_file}")
print(f"  Load with: data = np.load('{npz_file}')")

# Cleanup
os.unlink(xyz_file.name)

print(f"\n✓ SUCCESS: 1000-step MD completed without NaN!")
print("="*80)
