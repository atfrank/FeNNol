#!/usr/bin/env python3
"""Test script to verify backside attack restraint optimization and write frequency control."""

import jax
import jax.numpy as jnp
import numpy as np
import time
from src.fennol.md.restraints import backside_attack_restraint_force, _backside_attack_core_vectorized

def create_test_coordinates():
    """Create test coordinates for SN2 reaction: Nu- + CH3Cl -> CH3Nu + Cl-"""
    # Simple test system: Fluoride attacking methyl chloride
    # F-...CH3-Cl geometry
    n_atoms = 5
    coordinates = jnp.zeros((n_atoms, 3))
    
    # Atom 0: F- (nucleophile) - starts at 4.0 Å from C
    coordinates = coordinates.at[0].set(jnp.array([-4.0, 0.0, 0.0]))
    
    # Atom 1: C (carbon) - at origin
    coordinates = coordinates.at[1].set(jnp.array([0.0, 0.0, 0.0]))
    
    # Atom 2: Cl (leaving group) - 1.8 Å from C
    coordinates = coordinates.at[2].set(jnp.array([1.8, 0.0, 0.0]))
    
    # Atoms 3-4: H atoms on carbon (tetrahedral geometry)
    coordinates = coordinates.at[3].set(jnp.array([0.0, 1.0, 0.0]))
    coordinates = coordinates.at[4].set(jnp.array([0.0, -0.5, 0.866]))
    
    return coordinates

def benchmark_backside_attack():
    """Benchmark the backside attack restraint performance."""
    print("=== Backside Attack Restraint Performance Test ===\n")
    
    # Create test system
    coordinates = create_test_coordinates()
    nucleophile = 0  # F-
    carbon = 1       # C
    leaving_group = 2 # Cl
    
    # Test parameters
    target_angle = jnp.pi  # 180 degrees
    angle_force_constant = 0.1
    target_distance = 2.5
    distance_force_constant = 0.05
    
    # Number of iterations for benchmarking
    n_iterations = 1000
    
    print(f"Test system: F- attacking CH3Cl")
    print(f"Number of iterations: {n_iterations}")
    print(f"Target angle: {target_angle * 180 / jnp.pi:.1f} degrees")
    print(f"Target Nu-C distance: {target_distance} Å\n")
    
    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    _ = backside_attack_restraint_force(
        coordinates, nucleophile, carbon, leaving_group,
        target_angle, angle_force_constant, target_distance, distance_force_constant
    )
    
    # Time the optimized version
    print("\nBenchmarking optimized implementation...")
    start_time = time.time()
    
    for _ in range(n_iterations):
        energy, forces = backside_attack_restraint_force(
            coordinates, nucleophile, carbon, leaving_group,
            target_angle, angle_force_constant, target_distance, distance_force_constant
        )
    
    optimized_time = time.time() - start_time
    
    print(f"Optimized time: {optimized_time:.3f} seconds")
    print(f"Time per iteration: {optimized_time/n_iterations*1000:.3f} ms")
    print(f"Energy: {energy:.6f}")
    print(f"Max force magnitude: {jnp.max(jnp.linalg.norm(forces, axis=1)):.6f}")
    
    # Test with different configurations
    print("\n=== Testing different configurations ===")
    
    # Test without distance restraint
    print("\n1. Angle restraint only:")
    energy, forces = backside_attack_restraint_force(
        coordinates, nucleophile, carbon, leaving_group,
        target_angle, angle_force_constant, None, None
    )
    print(f"   Energy: {energy:.6f}")
    
    # Test with inversion prevention disabled
    print("\n2. With inversion prevention disabled:")
    energy, forces = backside_attack_restraint_force(
        coordinates, nucleophile, carbon, leaving_group,
        target_angle, angle_force_constant, target_distance, distance_force_constant,
        prevent_inversion=False
    )
    print(f"   Energy: {energy:.6f}")
    
    # Test with different angles
    print("\n3. Testing different angles:")
    for angle_deg in [120, 150, 180]:
        angle_rad = angle_deg * jnp.pi / 180
        energy, _ = backside_attack_restraint_force(
            coordinates, nucleophile, carbon, leaving_group,
            angle_rad, angle_force_constant, None, None
        )
        print(f"   Angle {angle_deg}°: Energy = {energy:.6f}")

def test_write_frequency():
    """Test the write frequency control feature."""
    print("\n\n=== Write Frequency Control Test ===\n")
    
    # Create a test input file with write_frequency parameter
    test_input = """
calculation:
  type: md
  steps: 100
  
restraints:
  backside_sn2:
    type: backside_attack
    nucleophile: 0
    carbon: 1
    leaving_group: 2
    target_angle: 180
    angle_force_constant: 0.1
    target_distance: [4.0, 3.5, 3.0, 2.5, 2.0]
    distance_force_constant: 0.05
    write_realtime_pmf: true
    write_frequency: 10  # Write every 10 steps instead of every step
    output_dir: test_output
"""
    
    print("Test configuration:")
    print("- Time-varying target distance: [4.0, 3.5, 3.0, 2.5, 2.0] Å")
    print("- Write frequency: every 10 steps")
    print("- Total steps: 100")
    print("\nWith write_frequency=10, we expect 11 data points (steps 0, 10, 20, ..., 100)")
    print("instead of 101 data points (every step)")
    
    # Save test input
    with open('/home/aaron/ATX/software/ATF-FeNNol/v2/test_backside_write_freq.fnl', 'w') as f:
        f.write(test_input)
    
    print("\nTest input file created: test_backside_write_freq.fnl")
    print("To run the test, use: fennol md test_backside_write_freq.fnl")

if __name__ == "__main__":
    # Run performance benchmark
    benchmark_backside_attack()
    
    # Test write frequency feature
    test_write_frequency()
    
    print("\n=== Summary ===")
    print("1. Backside attack restraint has been optimized with:")
    print("   - JIT compilation for core calculations")
    print("   - Vectorized operations instead of Python loops")
    print("   - Analytical derivatives for angle forces")
    print("\n2. Added write_frequency parameter to control output frequency:")
    print("   - Default: write every step (write_frequency=1)")
    print("   - Can be set to any integer to reduce output file size")
    print("   - Example: write_frequency=100 writes every 100 steps")