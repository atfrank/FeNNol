#!/usr/bin/env python
"""
Comprehensive benchmark: JAX vs CUDA implicit solvent

Compares:
1. Performance (timing, throughput)
2. Numerical accuracy (energy/forces comparison)
3. Dynamics quality (simple MD integration test)
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path
import json
from collections import defaultdict

from fennol.models.physics.implicit_solvent import OBC
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX

def read_tinker_xyz(filename):
    """Read Tinker XYZ format file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    natoms = int(lines[0].strip().split()[0])
    coords = []
    elements = []

    for line in lines[1:natoms+1]:
        parts = line.strip().split()
        element = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        coords.append([x, y, z])
        elements.append(element)

    return np.array(coords), elements

def get_atomic_numbers(elements):
    """Convert element symbols to atomic numbers."""
    return np.array([PERIODIC_TABLE_REV_IDX[e] for e in elements])

def estimate_charges_simple(elements):
    """Simple charge estimation for proteins."""
    charges = []
    for elem in elements:
        if elem == 'N':
            charges.append(-0.3)
        elif elem == 'O':
            charges.append(-0.5)
        elif elem == 'H':
            charges.append(0.3)
        elif elem == 'C':
            charges.append(0.1)
        elif elem == 'S':
            charges.append(0.0)
        else:
            charges.append(0.0)

    charges = np.array(charges)
    charges -= charges.sum() / len(charges)
    return charges

def simple_velocity_verlet_step(coords, velocities, forces, masses, dt):
    """
    Simple velocity Verlet integration step.

    Returns:
        new_coords, new_velocities
    """
    # v(t + dt/2) = v(t) + 0.5 * dt * f(t) / m
    velocities_half = velocities + 0.5 * dt * forces / masses[:, None]

    # x(t + dt) = x(t) + dt * v(t + dt/2)
    new_coords = coords + dt * velocities_half

    return new_coords, velocities_half

def complete_velocity_verlet_step(velocities_half, forces, masses, dt):
    """
    Complete velocity Verlet step after force evaluation.

    Returns:
        new_velocities
    """
    # v(t + dt) = v(t + dt/2) + 0.5 * dt * f(t + dt) / m
    new_velocities = velocities_half + 0.5 * dt * forces / masses[:, None]
    return new_velocities

def benchmark_performance(model, coords, charges, atomic_numbers, n_runs=10, warmup=2):
    """
    Benchmark performance of a model.

    Returns:
        dict with timing statistics
    """
    print(f"  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        energy, forces = model(coords, charges, atomic_numbers)
        forces.block_until_ready()

    print(f"  Timing {n_runs} evaluations...")
    times = []
    for i in range(n_runs):
        t0 = time.time()
        energy, forces = model(coords, charges, atomic_numbers)
        forces.block_until_ready()
        t1 = time.time()
        times.append(t1 - t0)
        if (i + 1) % 5 == 0:
            print(f"    Completed {i + 1}/{n_runs}")

    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }

def benchmark_accuracy(model_jax, model_cuda, coords, charges, atomic_numbers):
    """
    Compare numerical accuracy between JAX and CUDA.

    Returns:
        dict with accuracy metrics
    """
    print("  Computing with JAX...")
    e_jax, f_jax = model_jax(coords, charges, atomic_numbers)

    print("  Computing with CUDA...")
    e_cuda, f_cuda = model_cuda(coords, charges, atomic_numbers)

    # Convert to numpy for comparison
    e_jax_np = np.array(e_jax)
    e_cuda_np = np.array(e_cuda)
    f_jax_np = np.array(f_jax)
    f_cuda_np = np.array(f_cuda)

    # Compute differences
    e_diff = np.abs(e_jax_np - e_cuda_np)
    e_rel_diff = e_diff / np.abs(e_jax_np) if e_jax_np != 0 else 0.0

    f_diff = np.abs(f_jax_np - f_cuda_np)
    f_rel_diff = f_diff / (np.abs(f_jax_np) + 1e-10)

    return {
        'energy_jax': float(e_jax_np),
        'energy_cuda': float(e_cuda_np),
        'energy_abs_diff': float(e_diff),
        'energy_rel_diff': float(e_rel_diff),
        'forces_max_abs_diff': float(f_diff.max()),
        'forces_mean_abs_diff': float(f_diff.mean()),
        'forces_max_rel_diff': float(f_rel_diff.max()),
        'forces_mean_rel_diff': float(f_rel_diff.mean()),
        'forces_rms_diff': float(np.sqrt(np.mean(f_diff**2))),
    }

def benchmark_dynamics(model, coords, charges, atomic_numbers, masses, n_steps=100, dt=0.0005):
    """
    Run short MD simulation and track energy conservation.

    Returns:
        dict with dynamics statistics
    """
    print(f"  Running {n_steps} MD steps (dt={dt} ps)...")

    # Initialize velocities (300 K)
    kT = 0.001987 * 300.0  # kcal/mol
    velocities = np.random.randn(*coords.shape) * np.sqrt(kT / masses[:, None])

    # Remove center of mass motion
    velocities -= (velocities * masses[:, None]).sum(axis=0) / masses.sum()

    energies = []
    kinetic_energies = []
    potential_energies = []
    temperatures = []

    current_coords = jnp.array(coords)
    current_vels = velocities

    for step in range(n_steps):
        # Compute forces
        epot, forces = model(current_coords, charges, atomic_numbers)
        forces_np = np.array(forces)
        epot_np = float(epot)

        # Velocity Verlet - first half
        current_coords_np = np.array(current_coords)
        new_coords, vels_half = simple_velocity_verlet_step(
            current_coords_np, current_vels, forces_np, masses, dt
        )

        # Compute new forces
        current_coords = jnp.array(new_coords)
        epot_new, forces_new = model(current_coords, charges, atomic_numbers)
        forces_new_np = np.array(forces_new)
        epot_new_np = float(epot_new)

        # Velocity Verlet - second half
        current_vels = complete_velocity_verlet_step(vels_half, forces_new_np, masses, dt)

        # Compute kinetic energy
        ekin = 0.5 * np.sum(masses[:, None] * current_vels**2)

        # Temperature (3N-3 degrees of freedom for non-periodic)
        temp = 2.0 * ekin / ((3 * len(masses) - 3) * 0.001987)

        energies.append(epot_new_np + ekin)
        kinetic_energies.append(ekin)
        potential_energies.append(epot_new_np)
        temperatures.append(temp)

        if (step + 1) % 20 == 0:
            print(f"    Step {step + 1}/{n_steps}: E={energies[-1]:.2f}, T={temp:.1f} K")

    energies = np.array(energies)
    potential_energies = np.array(potential_energies)
    kinetic_energies = np.array(kinetic_energies)
    temperatures = np.array(temperatures)

    # Energy conservation (drift over simulation)
    energy_drift = energies[-1] - energies[0]
    energy_fluctuation = np.std(energies)

    return {
        'energies': energies.tolist(),
        'potential_energies': potential_energies.tolist(),
        'kinetic_energies': kinetic_energies.tolist(),
        'temperatures': temperatures.tolist(),
        'energy_drift': float(energy_drift),
        'energy_fluctuation': float(energy_fluctuation),
        'mean_temperature': float(temperatures.mean()),
        'std_temperature': float(temperatures.std()),
    }

def main():
    print("=" * 80)
    print("IMPLICIT SOLVENT BENCHMARK: JAX vs CUDA")
    print("=" * 80)
    print()

    # Load system
    systems = [
        {
            'name': 'water',
            'file': 'examples/md/watersmall/watersmall.xyz',
            'description': 'Small water box (216 atoms)',
        },
        {
            'name': 'dhfr',
            'file': 'examples/md/dhfr/dhfr2_nowat.xyz',
            'description': 'DHFR protein (2,499 atoms)',
        }
    ]

    results = {}

    for system_info in systems:
        system_name = system_info['name']
        xyz_file = Path(system_info['file'])

        if not xyz_file.exists():
            print(f"Skipping {system_name}: file not found ({xyz_file})")
            continue

        print("=" * 80)
        print(f"SYSTEM: {system_info['description']}")
        print("=" * 80)
        print()

        # Load structure
        print(f"Loading: {xyz_file}")
        coords, elements = read_tinker_xyz(xyz_file)
        natoms = len(coords)
        print(f"  Atoms: {natoms}")

        # Get atomic data
        atomic_numbers = get_atomic_numbers(elements)
        charges = estimate_charges_simple(elements)
        masses = np.array([
            {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06}.get(e, 12.0)
            for e in elements
        ])

        # Convert to JAX arrays
        coords_jax = jnp.array(coords)
        charges_jax = jnp.array(charges)
        atomic_numbers_jax = jnp.array(atomic_numbers)

        print()

        # Initialize models
        print("Initializing models...")

        # Force JAX backend
        model_jax = OBC({
            "dielectric": 80.0,
            "cutoff": 12.0,
            "surface_tension": 0.005,
            "probe_radius": 1.4,
            "include_nonpolar": True
        })
        model_jax.has_cuda = False  # Force JAX backend

        # Force CUDA backend
        model_cuda = OBC({
            "dielectric": 80.0,
            "cutoff": 12.0,
            "surface_tension": 0.005,
            "probe_radius": 1.4,
            "include_nonpolar": True
        })

        if not model_cuda.has_cuda:
            print("ERROR: CUDA not available!")
            continue

        print(f"  JAX backend: {model_jax}")
        print(f"  CUDA backend: {model_cuda}")
        print()

        system_results = {}

        # 1. Performance benchmark
        print("1. PERFORMANCE BENCHMARK")
        print("-" * 80)

        n_perf_runs = 10 if natoms < 500 else 5

        print("JAX Backend:")
        jax_perf = benchmark_performance(model_jax, coords_jax, charges_jax, atomic_numbers_jax, n_runs=n_perf_runs)
        print(f"  Mean time: {jax_perf['mean']*1000:.2f} ± {jax_perf['std']*1000:.2f} ms")
        print(f"  Throughput: {natoms/jax_perf['mean']:.0f} atoms/s")
        print()

        print("CUDA Backend:")
        cuda_perf = benchmark_performance(model_cuda, coords_jax, charges_jax, atomic_numbers_jax, n_runs=n_perf_runs)
        print(f"  Mean time: {cuda_perf['mean']*1000:.2f} ± {cuda_perf['std']*1000:.2f} ms")
        print(f"  Throughput: {natoms/cuda_perf['mean']:.0f} atoms/s")
        print()

        speedup = jax_perf['mean'] / cuda_perf['mean']
        print(f"SPEEDUP: {speedup:.2f}x (CUDA vs JAX)")
        print()

        system_results['performance'] = {
            'jax': jax_perf,
            'cuda': cuda_perf,
            'speedup': float(speedup),
            'natoms': natoms,
        }

        # 2. Accuracy benchmark
        print("2. NUMERICAL ACCURACY")
        print("-" * 80)

        accuracy = benchmark_accuracy(model_jax, model_cuda, coords_jax, charges_jax, atomic_numbers_jax)

        print(f"Energy:")
        print(f"  JAX:  {accuracy['energy_jax']:.6f} kcal/mol")
        print(f"  CUDA: {accuracy['energy_cuda']:.6f} kcal/mol")
        print(f"  Absolute difference: {accuracy['energy_abs_diff']:.6e} kcal/mol")
        print(f"  Relative difference: {accuracy['energy_rel_diff']:.6e}")
        print()

        print(f"Forces:")
        print(f"  Max absolute difference: {accuracy['forces_max_abs_diff']:.6e} kcal/mol/Å")
        print(f"  Mean absolute difference: {accuracy['forces_mean_abs_diff']:.6e} kcal/mol/Å")
        print(f"  RMS difference: {accuracy['forces_rms_diff']:.6e} kcal/mol/Å")
        print(f"  Max relative difference: {accuracy['forces_max_rel_diff']:.6e}")
        print()

        system_results['accuracy'] = accuracy

        # 3. Dynamics benchmark (only for smaller systems)
        if natoms < 1000:
            print("3. DYNAMICS QUALITY")
            print("-" * 80)

            n_md_steps = 100

            print("JAX Backend:")
            jax_dynamics = benchmark_dynamics(
                model_jax, coords, charges_jax, atomic_numbers_jax, masses, n_steps=n_md_steps
            )
            print(f"  Energy drift: {jax_dynamics['energy_drift']:.4f} kcal/mol")
            print(f"  Energy fluctuation (std): {jax_dynamics['energy_fluctuation']:.4f} kcal/mol")
            print(f"  Mean temperature: {jax_dynamics['mean_temperature']:.1f} ± {jax_dynamics['std_temperature']:.1f} K")
            print()

            print("CUDA Backend:")
            cuda_dynamics = benchmark_dynamics(
                model_cuda, coords, charges_jax, atomic_numbers_jax, masses, n_steps=n_md_steps
            )
            print(f"  Energy drift: {cuda_dynamics['energy_drift']:.4f} kcal/mol")
            print(f"  Energy fluctuation (std): {cuda_dynamics['energy_fluctuation']:.4f} kcal/mol")
            print(f"  Mean temperature: {cuda_dynamics['mean_temperature']:.1f} ± {cuda_dynamics['std_temperature']:.1f} K")
            print()

            system_results['dynamics'] = {
                'jax': jax_dynamics,
                'cuda': cuda_dynamics,
            }
        else:
            print("3. DYNAMICS QUALITY")
            print("-" * 80)
            print("  Skipped for large system (>1000 atoms)")
            print()

        results[system_name] = system_results

    # Save results
    output_file = Path("benchmark_implicit_solvent_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print()

    # Print summary
    print("SUMMARY:")
    print("-" * 80)
    for system_name, system_results in results.items():
        if 'performance' in system_results:
            perf = system_results['performance']
            print(f"{system_name.upper()} ({perf['natoms']} atoms):")
            print(f"  JAX:  {perf['jax']['mean']*1000:.2f} ms/eval")
            print(f"  CUDA: {perf['cuda']['mean']*1000:.2f} ms/eval")
            print(f"  Speedup: {perf['speedup']:.2f}x")
            print()

if __name__ == "__main__":
    import sys
    sys.exit(main())
