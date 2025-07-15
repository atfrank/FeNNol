#!/usr/bin/env python3
"""
Command-line interface for transition state optimization using FeNNol
"""

import sys
import os
import io
import argparse
import time
from pathlib import Path

import jax
import numpy as np

from ..utils.input_parser import parse_input
from .initial import load_model, load_system_data, initialize_preprocessing
from .transition_state import find_transition_state


def main():
    """Main entry point for TS optimization CLI"""
    
    # Set up unbuffered output
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="fennol_ts",
        description="Find transition states using FeNNol neural network potentials"
    )
    parser.add_argument("input_file", type=Path, help="Input parameter file")
    parser.add_argument("--method", choices=["quasi_newton", "dimer", "sn2"], 
                       help="TS optimization method (overrides input file)")
    parser.add_argument("--max-iterations", type=int,
                       help="Maximum number of optimization iterations")
    parser.add_argument("--force-tolerance", type=float,
                       help="Force convergence tolerance")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"],
                       help="Device to run on (overrides input file)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--no-multimodel", action="store_true",
                       help="Disable multi-model XYZ trajectory writing")
    parser.add_argument("--no-pdb", action="store_true",
                       help="Disable PDB trajectory writing")
    
    args = parser.parse_args()
    
    # Parse input file
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
        
    simulation_parameters = parse_input(args.input_file)
    
    # Override parameters from command line
    if args.method:
        simulation_parameters["ts_method"] = args.method
    if args.max_iterations:
        simulation_parameters["min_max_iterations"] = args.max_iterations
    if args.force_tolerance:
        simulation_parameters["min_force_tolerance"] = args.force_tolerance
    if args.device:
        simulation_parameters["device"] = args.device
    if args.no_multimodel:
        simulation_parameters["write_multimodel_trajectory"] = False
    if args.no_pdb:
        simulation_parameters["write_pdb_trajectory"] = False
        
    # Ensure TS mode is enabled
    simulation_parameters["transition_state"] = True
    simulation_parameters["ts_only"] = True
    
    # Set up device
    device = simulation_parameters.get("device", "cpu").lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda") or device.startswith("gpu"):
        if ":" in device:
            num = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = num
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "gpu"
    
    _device = jax.devices(device)[0]
    jax.config.update("jax_default_device", _device)
    
    # Set precision
    enable_x64 = simulation_parameters.get("double_precision", False)
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"
    
    # Set matrix multiplication precision
    matmul_precision = simulation_parameters.get("matmul_prec", "highest").lower()
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    
    if args.verbose:
        print(f"# Device: {device}")
        print(f"# Precision: {fprec}")
        print(f"# Matrix multiplication precision: {matmul_precision}")
    
    # Run TS optimization
    try:
        run_ts_optimization(simulation_parameters, fprec, args.verbose)
    except Exception as e:
        print(f"Error during TS optimization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_ts_optimization(simulation_parameters, fprec, verbose=False):
    """Run the transition state optimization"""
    
    start_time = time.time()
    
    if verbose:
        print("# Loading model...")
    
    # Load model
    model = load_model(simulation_parameters)
    
    if verbose:
        print(f"# Model loaded: {model.__class__.__name__}")
        print(f"# Energy unit: {model.energy_unit}")
    
    # Load system data
    system_data, conformation = load_system_data(simulation_parameters, fprec)
    nat = system_data["nat"]
    
    if verbose:
        print(f"# System loaded: {nat} atoms")
        print(f"# System name: {system_data['name']}")
    
    # Initialize preprocessing
    preproc_state, conformation = initialize_preprocessing(
        simulation_parameters, model, conformation, system_data
    )
    
    if verbose:
        print("# Preprocessing initialized")
    
    # Print optimization settings
    ts_method = simulation_parameters.get("ts_method", "quasi_newton")
    max_iterations = simulation_parameters.get("min_max_iterations", 500)
    force_tolerance = simulation_parameters.get("min_force_tolerance", 1e-4)
    
    print("#" + "=" * 60)
    print("# TRANSITION STATE OPTIMIZATION")
    print("#" + "=" * 60)
    print(f"# Method: {ts_method}")
    print(f"# Number of atoms: {nat}")
    print(f"# Maximum iterations: {max_iterations}")
    print(f"# Force tolerance: {force_tolerance}")
    print(f"# Energy unit: {model.energy_unit}")
    print("#" + "=" * 60)
    
    # Run TS optimization
    ts_result = find_transition_state(
        model, system_data, conformation, simulation_parameters, fprec
    )
    
    # Print final results
    total_time = time.time() - start_time
    
    print("#" + "=" * 60)
    print("# OPTIMIZATION COMPLETE")
    print("#" + "=" * 60)
    print(f"# Converged: {ts_result.get('converged', False)}")
    print(f"# Final energy: {ts_result['energy']:.8f}")
    print(f"# Max force: {np.max(np.abs(ts_result['forces'])):.8f}")
    print(f"# RMS force: {np.sqrt(np.mean(ts_result['forces']**2)):.8f}")
    
    if 'n_negative_eigenvalues' in ts_result:
        print(f"# Number of negative eigenvalues: {ts_result['n_negative_eigenvalues']}")
    if 'lowest_eigenvalue' in ts_result and ts_result['lowest_eigenvalue'] is not None:
        print(f"# Lowest eigenvalue: {ts_result['lowest_eigenvalue']:.6f}")
    if 'lowest_curvature' in ts_result:
        print(f"# Lowest curvature: {ts_result['lowest_curvature']:.6f}")
        
    print(f"# Total time: {total_time:.2f} seconds")
    print("#" + "=" * 60)
    
    # Save final structure
    output_name = system_data["name"]
    
    # Write final TS structure
    from ..utils.io import write_xyz_frame
    with open(f"{output_name}.ts.xyz", "w") as f:
        properties = {
            "energy": ts_result["energy"],
            "converged": ts_result.get("converged", False)
        }
        write_xyz_frame(
            f,
            system_data["symbols"],
            ts_result["coordinates"],
            properties=properties
        )
    
    print(f"# Final TS structure saved to: {output_name}.ts.xyz")
    
    # List additional output files
    additional_files = []
    if simulation_parameters.get("write_multimodel_trajectory", True):
        additional_files.append(f"{output_name}.multimodel.xyz")
    if simulation_parameters.get("write_pdb_trajectory", True):
        additional_files.append(f"{output_name}.traj.pdb")
    
    # Check for best TS guess files
    best_ts_files = [f"{output_name}.best_ts_guess.xyz", f"{output_name}.best_ts_guess.pdb"]
    for file in best_ts_files:
        if os.path.exists(file):
            additional_files.append(file)
    
    if additional_files:
        print("# Additional output files:")
        for file in additional_files:
            print(f"#   {file}")
    
    if not ts_result.get('converged', False):
        print("# Warning: Optimization did not converge!")
        return False
        
    return True


if __name__ == "__main__":
    main()