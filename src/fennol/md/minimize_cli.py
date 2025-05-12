#!/usr/bin/env python
import sys, os, io
import argparse
import time
from pathlib import Path
import math

import numpy as np
import jax

from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .initial import load_model, load_system_data, initialize_preprocessing
from .minimize import minimize_system

def main():
    """Command-line interface for energy minimization"""
    # Configure stdout for unbuffered output
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(prog="fennol_minimize", 
                                    description="Energy minimization for molecular systems")
    parser.add_argument("param_file", type=Path, help="Parameter file")
    parser.add_argument("--method", type=str, choices=["sd", "cg", "lbfgs", "simple_sd", "auto"], default=None,
                        help="Minimization method: sd (steepest descent), cg (conjugate gradient), lbfgs, simple_sd (robust SD), auto")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Maximum number of iterations")
    parser.add_argument("--ftol", type=float, default=None,
                        help="Force tolerance for convergence")
    parser.add_argument("--etol", type=float, default=None,
                        help="Energy tolerance for convergence")
    parser.add_argument("--print-freq", type=int, default=None,
                        help="Print frequency for iterations")
    parser.add_argument("--double", action="store_true",
                        help="Use double precision (float64)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu"],
                        help="Device to run on: cpu or gpu")
    parser.add_argument("--robust", action="store_true", 
                        help="Use robust methods for complex systems")
    args = parser.parse_args()
    
    # Load simulation parameters
    simulation_parameters = parse_input(args.param_file)
    
    # Override parameters from command line if provided
    if args.method is not None:
        simulation_parameters["min_method"] = args.method
    if args.max_iter is not None:
        simulation_parameters["min_max_iterations"] = args.max_iter
    if args.ftol is not None:
        simulation_parameters["min_force_tolerance"] = args.ftol
    if args.etol is not None:
        simulation_parameters["min_energy_tolerance"] = args.etol
    if args.print_freq is not None:
        simulation_parameters["min_print_freq"] = args.print_freq
    if args.device is not None:
        simulation_parameters["device"] = args.device
    
    # Handle robust mode
    if args.robust:
        simulation_parameters["min_method"] = "simple_sd"
        print("# Using robust minimization mode with simple_sd algorithm")
        
    # Auto-detect biomolecular systems
    system_name = simulation_parameters.get("coordinates", "").lower()
    if ("rna" in system_name or "dna" in system_name or "protein" in system_name or "ast" in system_name) and not args.method:
        simulation_parameters["min_method"] = "simple_sd"
        print(f"# Auto-detected biomolecular system: {system_name}")
        print("# Using simple_sd minimizer for better robustness")
    
    # If method is auto, we'll let dynamic.py decide based on system size
    if simulation_parameters.get("min_method", "") == "auto":
        print("# Using auto method selection based on system size")
        
    # Always set minimize and minimize_only to True
    simulation_parameters["minimize"] = True
    simulation_parameters["minimize_only"] = True
    
    # Set higher print frequency for complex minimization
    if simulation_parameters.get("min_method", "") == "simple_sd" and "min_print_freq" not in simulation_parameters:
        simulation_parameters["min_print_freq"] = 1  # More frequent updates for simple_sd
    
    # Set device
    device: str = simulation_parameters.get("device", "cpu").lower()
    if device == "cpu":
        device = "cpu"
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
    enable_x64 = simulation_parameters.get("double_precision", False) or args.double
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"

    # Set matmul precision
    matmul_precision = simulation_parameters.get("matmul_prec", "highest").lower()
    assert matmul_precision in [
        "default",
        "high",
        "highest",
    ], "matmul_prec must be one of 'default','high','highest'"
    if matmul_precision != "highest":
        print(f"# Setting matmul precision to '{matmul_precision}'")
    if matmul_precision == "default" and fprec == "float32":
        print(
            "# Warning: default matmul precision involves float16 operations which may lead to large numerical errors on energy and pressure estimations ! It is recommended to set matmul_prec to 'high' or 'highest'."
        )
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    
    # Run minimization directly
    tstart = time.time()
    
    # Initialize model and system
    model = load_model(simulation_parameters)
    system_data, conformation = load_system_data(simulation_parameters, fprec)
    preproc_state, conformation = initialize_preprocessing(
        simulation_parameters, model, conformation, system_data
    )
    
    # Run minimization
    result = minimize_system(model, system_data, conformation, simulation_parameters, fprec)
    
    # Print final summary
    print("\n# Minimization Summary:")
    print(f"# Total wall time: {time.time() - tstart:.2f} seconds")
    print(f"# Final energy: {result['energy']:.8f}")
    print(f"# Maximum force: {jnp.max(jnp.abs(result['forces'])):.8f}")
    print(f"# RMS force: {jnp.sqrt(jnp.mean(result['forces']**2)):.8f}")
    
    # Return minimized coordinates in the final result
    return result


if __name__ == "__main__":
    main()