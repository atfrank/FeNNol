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


def build_parameters_from_cli(args):
    """Build simulation parameters dictionary from CLI arguments"""
    
    # Determine output prefix from XYZ filename if not provided
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = Path(args.xyz).stem
    
    # Base parameters
    params = {
        # Model and system
        "model_file": args.model,
        "xyz_input/file": args.xyz,
        "xyz_input/indexed": False,
        "xyz_input/has_comment_line": True,
        
        # TS optimization
        "transition_state": True,
        "ts_only": True,
        "ts_method": args.method,
        
        # General optimization
        "min_max_iterations": args.max_iterations,
        "min_force_tolerance": args.force_tolerance,
        "min_print_freq": args.print_freq,
        "output_prefix": output_prefix,
        
        # Device and precision
        "device": args.device or "cpu",
        "enable_x64": args.double_precision,
        "matmul_prec": args.matmul_precision,
        
        # Output options
        "write_multimodel_trajectory": not args.no_multimodel,
        "write_pdb_trajectory": not args.no_pdb,
        "traj_format": "xyz"
    }
    
    # Method-specific parameters
    if args.method == "quasi_newton":
        params.update({
            "ts_trust_radius": args.ts_trust_radius,
            "ts_max_uphill_steps": args.ts_max_uphill_steps,
            "ts_initial_hessian_scale": args.ts_initial_hessian_scale
        })
    
    elif args.method == "dimer":
        params.update({
            "dimer_separation": args.dimer_separation,
            "dimer_rotation_tolerance": args.dimer_rotation_tolerance,
            "dimer_max_rotations": args.dimer_max_rotations
        })
    
    elif args.method == "sn2":
        # Validate SN2 parameters
        if not args.sn2_nu_index or not args.sn2_c_index or not args.sn2_lg_index:
            print("Error: SN2 method requires --sn2-nu-index, --sn2-c-index, and --sn2-lg-index")
            sys.exit(1)
        
        params.update({
            "sn2_nu_index": args.sn2_nu_index,
            "sn2_c_index": args.sn2_c_index,
            "sn2_lg_index": args.sn2_lg_index,
            "sn2_target_nu_c_distance": args.sn2_target_nu_c_distance,
            "sn2_target_c_lg_distance": args.sn2_target_c_lg_distance
        })
    
    return params


def override_parameters_from_cli(simulation_parameters, args):
    """Override simulation parameters with CLI arguments"""
    
    # Override basic parameters
    if args.method:
        simulation_parameters["ts_method"] = args.method
    if args.max_iterations != 500:  # Only override if not default
        simulation_parameters["min_max_iterations"] = args.max_iterations
    if args.force_tolerance != 1e-4:  # Only override if not default
        simulation_parameters["min_force_tolerance"] = args.force_tolerance
    if args.device:
        simulation_parameters["device"] = args.device
    if args.double_precision:
        simulation_parameters["enable_x64"] = True
    if args.matmul_precision != "highest":
        simulation_parameters["matmul_prec"] = args.matmul_precision
    if args.output_prefix:
        simulation_parameters["output_prefix"] = args.output_prefix
    if args.print_freq != 1:
        simulation_parameters["min_print_freq"] = args.print_freq
    
    # Override output options
    if args.no_multimodel:
        simulation_parameters["write_multimodel_trajectory"] = False
    if args.no_pdb:
        simulation_parameters["write_pdb_trajectory"] = False
    
    # Override method-specific parameters
    method = args.method or simulation_parameters.get("ts_method", "quasi_newton")
    
    if method == "quasi_newton":
        if args.ts_trust_radius != 0.3:
            simulation_parameters["ts_trust_radius"] = args.ts_trust_radius
        if args.ts_max_uphill_steps != 5:
            simulation_parameters["ts_max_uphill_steps"] = args.ts_max_uphill_steps
        if args.ts_initial_hessian_scale != 0.1:
            simulation_parameters["ts_initial_hessian_scale"] = args.ts_initial_hessian_scale
    
    elif method == "dimer":
        if args.dimer_separation != 0.01:
            simulation_parameters["dimer_separation"] = args.dimer_separation
        if args.dimer_rotation_tolerance != 0.1:
            simulation_parameters["dimer_rotation_tolerance"] = args.dimer_rotation_tolerance
        if args.dimer_max_rotations != 10:
            simulation_parameters["dimer_max_rotations"] = args.dimer_max_rotations
    
    elif method == "sn2":
        if args.sn2_nu_index:
            simulation_parameters["sn2_nu_index"] = args.sn2_nu_index
        if args.sn2_c_index:
            simulation_parameters["sn2_c_index"] = args.sn2_c_index
        if args.sn2_lg_index:
            simulation_parameters["sn2_lg_index"] = args.sn2_lg_index
        if args.sn2_target_nu_c_distance != 2.2:
            simulation_parameters["sn2_target_nu_c_distance"] = args.sn2_target_nu_c_distance
        if args.sn2_target_c_lg_distance != 2.2:
            simulation_parameters["sn2_target_c_lg_distance"] = args.sn2_target_c_lg_distance


def main():
    """Main entry point for TS optimization CLI"""
    
    # Set up unbuffered output
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    
    # Pre-parse to check for device setting before importing JAX
    import argparse
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--device", choices=["cpu", "gpu", "cuda"])
    pre_args, _ = pre_parser.parse_known_args()
    
    # Set JAX platform early if CPU is specified
    if pre_args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="fennol_ts",
        description="Find transition states using FeNNol neural network potentials"
    )
    parser.add_argument("input_file", type=Path, nargs='?', help="Input parameter file (optional if using CLI-only mode)")
    
    # Required arguments for CLI-only mode
    parser.add_argument("--xyz", type=str, help="XYZ structure file")
    parser.add_argument("--model", type=str, help="Model file path")
    
    # TS method selection
    parser.add_argument("--method", choices=["quasi_newton", "dimer", "sn2"], 
                       help="TS optimization method")
    
    # General optimization parameters
    parser.add_argument("--max-iterations", type=int, default=500,
                       help="Maximum number of optimization iterations (default: 500)")
    parser.add_argument("--force-tolerance", type=float, default=1e-4,
                       help="Force convergence tolerance (default: 1e-4)")
    parser.add_argument("--output-prefix", type=str,
                       help="Output file prefix (default: input filename)")
    
    # Device and precision
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"],
                       help="Device to run on")
    parser.add_argument("--double-precision", action="store_true",
                       help="Use double precision (float64)")
    parser.add_argument("--matmul-precision", choices=["highest", "high", "float32"],
                       default="highest", help="Matrix multiplication precision")
    
    # Quasi-Newton specific parameters
    parser.add_argument("--ts-trust-radius", type=float, default=0.3,
                       help="Trust radius for TS optimization (default: 0.3)")
    parser.add_argument("--ts-max-uphill-steps", type=int, default=5,
                       help="Maximum uphill steps allowed (default: 5)")
    parser.add_argument("--ts-initial-hessian-scale", type=float, default=0.1,
                       help="Initial Hessian scale factor (default: 0.1)")
    
    # Dimer method specific parameters
    parser.add_argument("--dimer-separation", type=float, default=0.01,
                       help="Dimer separation distance (default: 0.01)")
    parser.add_argument("--dimer-rotation-tolerance", type=float, default=0.1,
                       help="Dimer rotation tolerance (default: 0.1)")
    parser.add_argument("--dimer-max-rotations", type=int, default=10,
                       help="Maximum dimer rotations (default: 10)")
    
    # SN2 method specific parameters
    parser.add_argument("--sn2-nu-index", type=int,
                       help="Nucleophile atom index (1-based)")
    parser.add_argument("--sn2-c-index", type=int,
                       help="Carbon atom index (1-based)")
    parser.add_argument("--sn2-lg-index", type=int,
                       help="Leaving group atom index (1-based)")
    parser.add_argument("--sn2-target-nu-c-distance", type=float, default=2.2,
                       help="Target Nu-C distance (default: 2.2)")
    parser.add_argument("--sn2-target-c-lg-distance", type=float, default=2.2,
                       help="Target C-LG distance (default: 2.2)")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--no-multimodel", action="store_true",
                       help="Disable multi-model XYZ trajectory writing")
    parser.add_argument("--no-pdb", action="store_true",
                       help="Disable PDB trajectory writing")
    parser.add_argument("--print-freq", type=int, default=1,
                       help="Print frequency for optimization steps (default: 1)")
    
    args = parser.parse_args()
    
    # Determine if we're in CLI-only mode
    cli_only_mode = args.input_file is None
    
    if cli_only_mode:
        # Validate required arguments for CLI-only mode
        if not args.xyz:
            print("Error: --xyz is required when not using an input file")
            sys.exit(1)
        if not args.model:
            print("Error: --model is required when not using an input file")
            sys.exit(1)
        if not args.method:
            print("Error: --method is required when not using an input file")
            sys.exit(1)
        
        # Check if XYZ file exists
        if not Path(args.xyz).exists():
            print(f"Error: XYZ file '{args.xyz}' not found")
            sys.exit(1)
        
        # Check if model file exists
        if not Path(args.model).exists():
            print(f"Error: Model file '{args.model}' not found")
            sys.exit(1)
        
        # Build simulation parameters from CLI arguments
        simulation_parameters = build_parameters_from_cli(args)
        
    else:
        # Parse input file
        if not args.input_file.exists():
            print(f"Error: Input file '{args.input_file}' not found")
            sys.exit(1)
            
        simulation_parameters = parse_input(args.input_file)
        
        # Override parameters from command line
        override_parameters_from_cli(simulation_parameters, args)
        
    # Ensure TS mode is enabled
    simulation_parameters["transition_state"] = True
    simulation_parameters["ts_only"] = True
    
    # Set up device
    device = simulation_parameters.get("device", "cpu").lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
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