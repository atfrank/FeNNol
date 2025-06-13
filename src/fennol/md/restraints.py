"""
Fixed restraints implementation for FeNNol.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Callable, Tuple, List, Optional
from .colvars import colvar_distance, colvar_angle, colvar_dihedral, colvar_sphere_distance
from ..utils.io import read_pdb, parse_atom_selection, calculate_center_of_mass
from ..utils.periodic_table import ATOMIC_MASSES, PERIODIC_TABLE_REV_IDX

# Global variables to track restraint state
current_force_constants = {}
current_simulation_step = 0  # Will be incremented by apply_restraints
restraint_debug_enabled = False  # Will be set based on input parameters
initial_coordinates_store = None  # Store initial coordinates for adaptive restraints

# PMF estimation variables
pmf_data = {}  # Store PMF data for each restraint
pmf_enabled = False  # Will be set based on input parameters

# Function to access or update the current step
def get_current_step():
    global current_simulation_step
    return current_simulation_step

def increment_step():
    global current_simulation_step
    current_simulation_step += 1
    return current_simulation_step

def initialize_pmf_data(restraint_name: str, windows: List[float], colvar_type: str, 
                        write_realtime: bool = False, output_dir: str = ".",
                        force_constants: List[float] = None, simulation_id: str = None):
    """Initialize PMF data collection for a restraint."""
    global pmf_data
    import time
    import os
    
    # Generate unique simulation ID if not provided
    if simulation_id is None:
        simulation_id = f"sim_{int(time.time())}_{os.getpid()}"
    
    pmf_data[restraint_name] = {
        'windows': windows,  # Target values for each window
        'colvar_type': colvar_type,  # Type of collective variable
        'samples': {i: [] for i in range(len(windows))},  # Samples for each window
        'current_window': 0,  # Current window index
        'equilibration_steps': 0,  # Steps to equilibrate before sampling
        'sampling_started': False,
        'write_realtime': write_realtime,
        'output_dir': output_dir,
        # New metadata for multi-simulation analysis
        'simulation_id': simulation_id,
        'force_constants': force_constants or [1.0] * len(windows),
        'window_start_times': [None] * len(windows),
        'window_end_times': [None] * len(windows),
        'window_start_steps': [None] * len(windows),
        'window_end_steps': [None] * len(windows),
        'initialization_time': time.time(),
        'window_statistics': {i: {'mean': None, 'std': None, 'min': None, 'max': None, 'count': 0} 
                             for i in range(len(windows))}
    }
    
    # Initialize real-time output file if enabled
    if write_realtime:
        filename = f"{output_dir}/pmf_{restraint_name}_realtime.dat"
        with open(filename, 'w') as f:
            f.write(f"# PMF Real-time Data for restraint: {restraint_name}\n")
            f.write(f"# Simulation ID: {simulation_id}\n")
            f.write(f"# Collective variable: {colvar_type}\n")
            f.write(f"# Number of windows: {len(windows)}\n")
            f.write(f"# Window values: {windows}\n")
            f.write(f"# Force constants: {pmf_data[restraint_name]['force_constants']}\n")
            f.write(f"# Initialization time: {time.ctime(pmf_data[restraint_name]['initialization_time'])}\n")
            f.write(f"# Columns: Step Window_Index {colvar_type}_Value Force_Constant\n")
            f.write(f"#{'Step':>7s} {'Win':>3s} {colvar_type.capitalize():>12s} {'Force_K':>10s}\n")
        print(f"# PMF real-time output: {filename}")
    
    # Print initialization message
    print(f"\n# ========== PMF INITIALIZATION ==========")
    print(f"# Restraint: {restraint_name}")
    print(f"# Collective variable: {colvar_type}")
    print(f"# Number of windows: {len(windows)}")
    print(f"# Window values: {windows}")
    print(f"# PMF estimation: ACTIVE")
    if write_realtime:
        print(f"# Real-time output: ENABLED")
    print(f"# ==========================================\n")

def update_pmf_sample(restraint_name: str, colvar_value: float, window_idx: int, 
                      write_realtime: bool = False, output_dir: str = ".", force_constant: float = None):
    """Add a sample to the PMF data for the current window."""
    global pmf_data, current_simulation_step
    import time
    import numpy as np
    
    if restraint_name in pmf_data and pmf_data[restraint_name]['sampling_started']:
        # Add sample to data
        # Ensure window_idx is an integer key
        window_idx = int(window_idx)
        if window_idx not in pmf_data[restraint_name]['samples']:
            pmf_data[restraint_name]['samples'][window_idx] = []
        pmf_data[restraint_name]['samples'][window_idx].append(colvar_value)
        
        # Track window timing
        if pmf_data[restraint_name]['window_start_times'][window_idx] is None:
            pmf_data[restraint_name]['window_start_times'][window_idx] = time.time()
            pmf_data[restraint_name]['window_start_steps'][window_idx] = current_simulation_step
        
        # Update window statistics
        samples = pmf_data[restraint_name]['samples'][window_idx]
        stats = pmf_data[restraint_name]['window_statistics'][window_idx]
        samples_array = np.array(samples)
        stats['count'] = len(samples)
        stats['mean'] = np.mean(samples_array)
        stats['std'] = np.std(samples_array) if len(samples) > 1 else 0.0
        stats['min'] = np.min(samples_array)
        stats['max'] = np.max(samples_array)
        
        # Use force constant from PMF data if not provided
        if force_constant is None:
            force_constant = pmf_data[restraint_name]['force_constants'][window_idx]
        
        # Write to real-time file if enabled
        if write_realtime:
            filename = f"{output_dir}/pmf_{restraint_name}_realtime.dat"
            with open(filename, 'a') as f:
                f.write(f"{current_simulation_step:8d} {window_idx:3d} {colvar_value:12.6f} {force_constant:10.3f}\n")

def print_pmf_progress(restraint_name: str, current_step: int, window_idx: int, 
                      equilibration_steps: int, update_steps: int, colvar_value: float):
    """Print PMF sampling progress information."""
    global pmf_data, restraint_debug_enabled
    
    if not restraint_debug_enabled or restraint_name not in pmf_data:
        return
    
    data = pmf_data[restraint_name]
    steps_in_window = current_step % update_steps
    is_equilibrating = steps_in_window < equilibration_steps
    
    # Print at key points: start of window, start of sampling, periodically during sampling
    print_progress = False
    if steps_in_window == 0:  # New window
        print_progress = True
        print(f"\n# PMF [{restraint_name}] === ENTERING WINDOW {window_idx + 1}/{len(data['windows'])} ===")
        print(f"#   Target {data['colvar_type']}: {data['windows'][window_idx]:.3f}")
        print(f"#   Equilibration: {equilibration_steps} steps, Sampling: {update_steps - equilibration_steps} steps")
    elif steps_in_window == equilibration_steps:  # Start sampling
        print_progress = True
        print(f"# PMF [{restraint_name}] === STARTING SAMPLING for window {window_idx + 1} ===")
    elif not is_equilibrating and steps_in_window % 500 == 0:  # Periodic progress
        print_progress = True
    
    if print_progress:
        if is_equilibrating:
            progress = (steps_in_window / equilibration_steps) * 100
            print(f"# PMF [{restraint_name}] Window {window_idx + 1}: EQUILIBRATING {progress:.1f}% | {data['colvar_type']}={colvar_value:.3f}")
        else:
            samples_collected = len(data['samples'][window_idx])
            sampling_steps = steps_in_window - equilibration_steps
            total_sampling_steps = update_steps - equilibration_steps
            progress = (sampling_steps / total_sampling_steps) * 100
            
            # Calculate statistics for current window
            if samples_collected > 0:
                import numpy as np
                samples = np.array(data['samples'][window_idx])
                mean_val = np.mean(samples)
                std_val = np.std(samples) if len(samples) > 1 else 0.0
                print(f"# PMF [{restraint_name}] Window {window_idx + 1}: SAMPLING {progress:.1f}% | " + 
                      f"Samples={samples_collected} | Current={colvar_value:.3f} | " +
                      f"Mean={mean_val:.3f}±{std_val:.3f}")

def print_pmf_window_summary(restraint_name: str, window_idx: int):
    """Print summary statistics when leaving a window."""
    global pmf_data, restraint_debug_enabled
    
    if not restraint_debug_enabled or restraint_name not in pmf_data:
        return
        
    data = pmf_data[restraint_name]
    samples = data['samples'][window_idx]
    
    if len(samples) > 0:
        import numpy as np
        samples_array = np.array(samples)
        print(f"# PMF [{restraint_name}] === WINDOW {window_idx + 1} COMPLETE ===")
        print(f"#   Target: {data['windows'][window_idx]:.3f}")
        print(f"#   Samples collected: {len(samples)}")
        print(f"#   Mean ± StdDev: {np.mean(samples_array):.3f} ± {np.std(samples_array):.3f}")
        print(f"#   Range: [{np.min(samples_array):.3f}, {np.max(samples_array):.3f}]")
        
        # Calculate instantaneous PMF if we have previous window
        if window_idx > 0 and len(data['samples'][window_idx - 1]) > 0:
            prev_mean = np.mean(data['samples'][window_idx - 1])
            curr_mean = np.mean(samples_array)
            delta_pmf = -0.001987 * 300 * np.log(len(samples) / len(data['samples'][window_idx - 1]))
            print(f"#   Approx. ΔPMF from previous: {delta_pmf:.2f} kcal/mol")

def calculate_pmf(restraint_name: str, temperature: float = 300.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate PMF using WHAM (Weighted Histogram Analysis Method) for umbrella sampling.
    Returns (window_centers, pmf_values, uncertainties) in kcal/mol.
    """
    import numpy as np
    from scipy import optimize
    
    if restraint_name not in pmf_data:
        return [], [], []
    
    data = pmf_data[restraint_name]
    windows = data['windows']
    samples = data['samples']
    force_constants = data.get('force_constants', [1.0] * len(windows))
    
    # Skip empty windows
    valid_data = []
    
    # Debug: Check samples structure
    if restraint_debug_enabled:
        print(f"# DEBUG: Samples dictionary type: {type(samples)}")
        print(f"# DEBUG: Samples keys: {list(samples.keys())}")
        print(f"# DEBUG: First few key types: {[type(k) for k in list(samples.keys())[:3]]}")
    
    for i, (window, k) in enumerate(zip(windows, force_constants)):
        # Try to get samples with error handling
        try:
            window_samples = samples.get(i, samples.get(str(i), []))
            if restraint_debug_enabled and i < 3:
                print(f"# DEBUG: Window {i} - samples found: {len(window_samples)}")
        except Exception as e:
            if restraint_debug_enabled:
                print(f"# DEBUG: Error getting samples for window {i}: {e}")
            window_samples = []
            
        if len(window_samples) > 0:
            valid_data.append({
                'index': i,
                'center': window,
                'k': k,
                'samples': np.array(window_samples)
            })
    
    if len(valid_data) == 0:
        return [], [], []
    
    # WHAM implementation
    kB_T = 0.001987 * temperature  # kcal/mol
    n_windows = len(valid_data)
    
    # Debug: Print window information
    if restraint_debug_enabled:
        print(f"\n# WHAM DEBUG for {restraint_name}:")
        print(f"# Temperature: {temperature} K, kB_T: {kB_T:.6f} kcal/mol")
        print(f"# Number of valid windows: {n_windows}")
        for i, win_data in enumerate(valid_data):
            print(f"# Window {i}: center={win_data['center']:.3f}, k={win_data['k']:.3f}, n_samples={len(win_data['samples'])}")
    
    # Collect all samples
    all_samples = []
    sample_to_window = []
    for i, win_data in enumerate(valid_data):
        all_samples.extend(win_data['samples'])
        sample_to_window.extend([i] * len(win_data['samples']))
    
    all_samples = np.array(all_samples)
    sample_to_window = np.array(sample_to_window)
    n_samples = len(all_samples)
    
    # Calculate bias energies for all samples in all windows
    # U_i(x) = 0.5 * k_i * (x - x_i)^2
    bias_energies = np.zeros((n_samples, n_windows))
    for j, win_data in enumerate(valid_data):
        bias_energies[:, j] = 0.5 * win_data['k'] * (all_samples - win_data['center'])**2
    
    # Initialize free energies
    f_i = np.zeros(n_windows)
    
    # WHAM iteration
    max_iter = 1000
    tolerance = 1e-6
    
    for iteration in range(max_iter):
        f_i_old = f_i.copy()
        
        # Calculate denominator for each sample
        # denominator = sum_j N_j * exp(-beta * (U_j(x) - f_j))
        denominators = np.zeros(n_samples)
        for x_idx in range(n_samples):
            denom = 0.0
            for j in range(n_windows):
                N_j = len(valid_data[j]['samples'])
                denom += N_j * np.exp(-bias_energies[x_idx, j] / kB_T + f_i[j] / kB_T)
            denominators[x_idx] = denom
        
        # Update free energies
        # f_i = -kT * ln(sum_x 1/denominator * exp(-beta * U_i(x)))
        for i in range(n_windows):
            # Sum over samples from window i
            window_mask = sample_to_window == i
            window_samples_idx = np.where(window_mask)[0]
            
            if len(window_samples_idx) > 0:
                sum_val = 0.0
                for idx in window_samples_idx:
                    sum_val += 1.0 / denominators[idx] * np.exp(-bias_energies[idx, i] / kB_T)
                f_i[i] = -kB_T * np.log(sum_val)
        
        # Center free energies (set minimum to 0)
        f_i -= np.min(f_i)
        
        # Check convergence
        max_diff = np.max(np.abs(f_i - f_i_old))
        if restraint_debug_enabled and iteration % 100 == 0:
            print(f"# WHAM iteration {iteration}: max_diff={max_diff:.6e}, f_i={f_i}")
        if max_diff < tolerance:
            if restraint_debug_enabled:
                print(f"# WHAM converged after {iteration} iterations")
            break
    
    # Calculate PMF at window centers
    window_centers = [d['center'] for d in valid_data]
    pmf_values = f_i.tolist()
    
    # Calculate uncertainties using bootstrap
    n_bootstrap = 100
    bootstrap_pmfs = []
    
    for _ in range(n_bootstrap):
        # Resample data for each window
        bootstrap_samples = {}
        for i, win_data in enumerate(valid_data):
            n_win_samples = len(win_data['samples'])
            resample_idx = np.random.choice(n_win_samples, n_win_samples, replace=True)
            bootstrap_samples[i] = win_data['samples'][resample_idx]
        
        # Collect all bootstrap samples
        all_boot_samples = []
        boot_sample_to_window = []
        for i, win_data in enumerate(valid_data):
            all_boot_samples.extend(bootstrap_samples[i])
            boot_sample_to_window.extend([i] * len(bootstrap_samples[i]))
        
        all_boot_samples = np.array(all_boot_samples)
        boot_sample_to_window = np.array(boot_sample_to_window)
        
        # Calculate bias energies for bootstrap samples
        boot_bias_energies = np.zeros((len(all_boot_samples), n_windows))
        for j, win_data in enumerate(valid_data):
            boot_bias_energies[:, j] = 0.5 * win_data['k'] * (all_boot_samples - win_data['center'])**2
        
        # Run WHAM on bootstrap sample
        boot_f_i = np.zeros(n_windows)
        for iteration in range(100):  # Fewer iterations for bootstrap
            boot_f_i_old = boot_f_i.copy()
            
            # Calculate denominators
            boot_denominators = np.zeros(len(all_boot_samples))
            for x_idx in range(len(all_boot_samples)):
                denom = 0.0
                for j in range(n_windows):
                    N_j = len(bootstrap_samples[j])
                    denom += N_j * np.exp(-boot_bias_energies[x_idx, j] / kB_T + boot_f_i[j] / kB_T)
                boot_denominators[x_idx] = denom
            
            # Update free energies
            for i in range(n_windows):
                window_mask = boot_sample_to_window == i
                window_samples_idx = np.where(window_mask)[0]
                
                if len(window_samples_idx) > 0:
                    sum_val = 0.0
                    for idx in window_samples_idx:
                        sum_val += 1.0 / boot_denominators[idx] * np.exp(-boot_bias_energies[idx, i] / kB_T)
                    boot_f_i[i] = -kB_T * np.log(sum_val)
            
            # Center free energies
            boot_f_i -= np.min(boot_f_i)
            
            # Check convergence
            if np.max(np.abs(boot_f_i - boot_f_i_old)) < tolerance * 10:
                break
        
        bootstrap_pmfs.append(boot_f_i)
    
    # Calculate standard errors
    bootstrap_pmfs = np.array(bootstrap_pmfs)
    uncertainties = np.std(bootstrap_pmfs, axis=0).tolist()
    
    # Sort by window center
    sorted_indices = np.argsort(window_centers)
    window_centers = [window_centers[i] for i in sorted_indices]
    pmf_values = [pmf_values[i] for i in sorted_indices]
    uncertainties = [uncertainties[i] for i in sorted_indices]
    
    return window_centers, pmf_values, uncertainties

def write_pmf_output_deprecated(filename: str = "pmf_output.dat", temperature: float = 300.0):
    """
    Write PMF data to output file for all restraints with PMF estimation enabled.
    Enhanced version with metadata for multi-simulation analysis.
    
    Args:
        filename: Output filename
        temperature: Temperature in Kelvin for PMF calculation
    """
    global pmf_data
    import time
    import json
    
    if not pmf_data:
        return
        
    # Debug: Print what's in pmf_data
    if restraint_debug_enabled:
        print("\n# DEBUG: write_pmf_output - pmf_data contents:")
        for rname, data in pmf_data.items():
            print(f"#   Restraint: {rname}")
            print(f"#   Windows: {len(data['windows'])}")
            print(f"#   Samples keys: {list(data['samples'].keys())}")
            sample_counts = [len(data['samples'].get(i, [])) for i in range(len(data['windows']))]
            print(f"#   Sample counts: {sample_counts}")
            print(f"#   Total samples: {sum(sample_counts)}")
    
    # Print summary for final windows and mark end times
    for restraint_name, data in pmf_data.items():
        last_window_idx = len(data['windows']) - 1
        if len(data['samples'][last_window_idx]) > 0:
            print_pmf_window_summary(restraint_name, last_window_idx)
            # Mark end time for final window
            if data['window_end_times'][last_window_idx] is None:
                data['window_end_times'][last_window_idx] = time.time()
                data['window_end_steps'][last_window_idx] = current_simulation_step
    
    with open(filename, 'w') as f:
        f.write("# PMF Estimation Results - Enhanced for Multi-Simulation Analysis\n")
        f.write(f"# Temperature: {temperature} K\n")
        f.write(f"# Output generated: {time.ctime()}\n")
        f.write("#\n")
        
        for restraint_name, data in pmf_data.items():
            windows, pmf_values, uncertainties = calculate_pmf(restraint_name, temperature)
            
            if not windows:
                continue
            
            # Write simulation metadata
            f.write(f"\n# ===== RESTRAINT: {restraint_name} =====\n")
            f.write(f"# Simulation ID: {data['simulation_id']}\n")
            f.write(f"# Collective variable: {data['colvar_type']}\n")
            f.write(f"# Initialization time: {time.ctime(data['initialization_time'])}\n")
            f.write(f"# Total simulation duration: {time.time() - data['initialization_time']:.2f} seconds\n")
            f.write("#\n")
            
            # Write window configuration
            f.write(f"# Window configuration:\n")
            f.write(f"# Number of windows: {len(data['windows'])}\n")
            f.write(f"# Window centers: {data['windows']}\n")
            f.write(f"# Force constants: {data['force_constants']}\n")
            f.write("#\n")
            
            # Write detailed window statistics
            f.write(f"# Window statistics:\n")
            f.write(f"# {'Win':>3s} {'Target':>10s} {'Force_K':>10s} {'Samples':>8s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'Start_Step':>12s} {'End_Step':>12s}\n")
            for i, (window_val, force_k) in enumerate(zip(data['windows'], data['force_constants'])):
                stats = data['window_statistics'][i]
                start_step = data['window_start_steps'][i] if data['window_start_steps'][i] is not None else "N/A"
                end_step = data['window_end_steps'][i] if data['window_end_steps'][i] is not None else "N/A"
                f.write(f"# {i:3d} {window_val:10.6f} {force_k:10.3f} {stats['count']:8d} "
                       f"{stats['mean'] if stats['mean'] is not None else 0.0:10.6f} "
                       f"{stats['std'] if stats['std'] is not None else 0.0:10.6f} "
                       f"{stats['min'] if stats['min'] is not None else 0.0:10.6f} "
                       f"{stats['max'] if stats['max'] is not None else 0.0:10.6f} "
                       f"{str(start_step):>12s} {str(end_step):>12s}\n")
            f.write("#\n")
            
            # Write PMF data
            f.write(f"# PMF Results:\n")
            f.write(f"# {'Window_Center':>15s} {'PMF(kcal/mol)':>15s} {'Samples':>8s} {'Uncertainty':>12s}\n")
            
            for i, (window, pmf, uncertainty) in enumerate(zip(windows, pmf_values, uncertainties)):
                # Find corresponding window index
                window_idx = None
                for j, w in enumerate(data['windows']):
                    if abs(w - window) < 1e-6:
                        window_idx = j
                        break
                
                if window_idx is not None:
                    n_samples = len(data['samples'][window_idx])
                else:
                    n_samples = 0
                
                f.write(f"{window:15.6f} {pmf:15.6f} {n_samples:8d} {uncertainty:12.6f}\n")
            
            f.write("#\n")
            
            # Write machine-readable metadata as JSON comment
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_json_serializable(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj
            
            metadata = {
                'restraint_name': restraint_name,
                'simulation_id': data['simulation_id'],
                'colvar_type': data['colvar_type'],
                'windows': data['windows'],
                'force_constants': data['force_constants'],
                'window_statistics': convert_to_json_serializable(data['window_statistics']),
                'window_start_steps': data['window_start_steps'],
                'window_end_steps': data['window_end_steps'],
                'initialization_time': data['initialization_time'],
                'temperature': temperature,
                'total_samples': sum(len(samples) for samples in data['samples'].values())
            }
            f.write(f"# METADATA_JSON: {json.dumps(metadata)}\n")
            f.write("#\n")
    
    print(f"# Enhanced PMF data written to {filename}")
    print(f"# File includes simulation metadata for multi-simulation analysis")

def print_pmf_status(step: int = None):
    """Print current PMF estimation status for all active restraints."""
    global pmf_data, current_simulation_step
    
    if not pmf_data:
        return
    
    if step is None:
        step = current_simulation_step
    
    print("\n# ========== PMF ESTIMATION STATUS ==========")
    print(f"# Current step: {step}")
    
    for restraint_name, data in pmf_data.items():
        total_windows = len(data['windows'])
        windows_with_samples = sum(1 for samples in data['samples'].values() if len(samples) > 0)
        total_samples = sum(len(samples) for samples in data['samples'].values())
        
        print(f"\n# Restraint: {restraint_name}")
        print(f"#   Collective variable: {data['colvar_type']}")
        print(f"#   Windows completed: {windows_with_samples}/{total_windows}")
        print(f"#   Total samples collected: {total_samples}")
        
        # Show current window status
        for window_idx, window_value in enumerate(data['windows']):
            n_samples = len(data['samples'][window_idx])
            if n_samples > 0:
                import numpy as np
                samples_array = np.array(data['samples'][window_idx])
                mean_val = np.mean(samples_array)
                std_val = np.std(samples_array) if n_samples > 1 else 0.0
                status = f"{n_samples} samples, mean={mean_val:.3f}±{std_val:.3f}"
            else:
                status = "pending"
            print(f"#     Window {window_idx + 1} (target={window_value:.3f}): {status}")
    
    print("# ==========================================\n")

def print_pmf_final_summary(temperature: float = 300.0):
    """Print final PMF summary including calculated PMF values."""
    global pmf_data
    
    if not pmf_data:
        return
    
    print("\n# ========== FINAL PMF RESULTS ==========")
    print(f"# Temperature: {temperature} K")
    
    for restraint_name, data in pmf_data.items():
        windows, pmf_values, uncertainties = calculate_pmf(restraint_name, temperature)
        
        if not windows:
            print(f"\n# Restraint: {restraint_name} - NO DATA")
            continue
        
        print(f"\n# Restraint: {restraint_name}")
        print(f"# Collective variable: {data['colvar_type']}")
        print(f"# Total samples: {sum(len(samples) for samples in data['samples'].values())}")
        print("#")
        print("# Window   Target    Mean±Std     Samples   PMF(kcal/mol)  Uncertainty")
        print("# -------  --------  -----------  --------  -------------  -----------")
        
        for i, (window, pmf, uncertainty) in enumerate(zip(windows, pmf_values, uncertainties)):
            if i < len(data['samples']) and len(data['samples'][i]) > 0:
                import numpy as np
                samples = np.array(data['samples'][i])
                mean_val = np.mean(samples)
                std_val = np.std(samples) if len(samples) > 1 else 0.0
                n_samples = len(samples)
                print(f"# {i+1:^7d}  {window:^8.3f}  {mean_val:>5.3f}±{std_val:<4.3f}  {n_samples:^8d}  {pmf:>13.3f}  ±{uncertainty:>10.3f}")
        
        # Find and report barrier
        if len(pmf_values) > 1:
            import numpy as np
            pmf_array = np.array(pmf_values)
            barrier = np.max(pmf_array) - np.min(pmf_array)
            barrier_idx = np.argmax(pmf_array)
            print(f"#")
            print(f"# PMF barrier: {barrier:.2f} kcal/mol at window {barrier_idx + 1} (target={windows[barrier_idx]:.3f})")
    
    print("# =======================================\n")

def ensure_proper_coordinates(coordinates):
    """Ensure coordinates have the correct shape [natoms, 3]"""
    if coordinates.ndim == 1:
        # If we have flattened coordinates, reshape them
        natoms = coordinates.shape[0] // 3
        return coordinates.reshape(natoms, 3)
    return coordinates

def harmonic_restraint(value: float, target: float, force_constant: float) -> Tuple[float, float]:
    """
    Calculate the harmonic restraint energy and force.
    
    Args:
        value: Current value of the collective variable
        target: Target value for the restraint
        force_constant: Force constant for the harmonic restraint (k)
    
    Returns:
        Tuple containing (energy, force)
        Energy = 0.5 * k * (value - target)^2
        Force = k * (value - target)
    """
    diff = value - target
    energy = 0.5 * force_constant * diff**2
    force = force_constant * diff
    return energy, force

def flat_bottom_restraint(value: float, target: float, force_constant: float, 
                          tolerance: float) -> Tuple[float, float]:
    """
    Calculate a flat-bottom restraint energy and force. 
    No energy penalty within tolerance, harmonic outside.
    
    Args:
        value: Current value of the collective variable
        target: Target value for the restraint
        force_constant: Force constant for the harmonic restraint (k)
        tolerance: Width of the flat region (no energy penalty within target ± tolerance)
    
    Returns:
        Tuple containing (energy, force)
    """
    # Calculate deviation from target
    diff = value - target
    abs_diff = jnp.abs(diff)
    
    # Calculate violation (how far outside the flat region)
    # If within tolerance, violation is 0
    # If outside tolerance, violation is the amount beyond tolerance
    violation = jnp.maximum(0.0, abs_diff - tolerance)
    
    # Energy is 0.5*k*violation^2 (harmonic)
    energy = 0.5 * force_constant * violation**2
    
    # Force is k*violation*sign(diff)
    # Only non-zero when outside the tolerance
    sign = jnp.sign(diff)
    force = force_constant * violation * sign
    
    return energy, force

def one_sided_harmonic_restraint(value: float, target: float, force_constant: float, 
                                side: str = "lower") -> Tuple[float, float]:
    """
    Calculate a one-sided harmonic restraint energy and force.
    Forces go to zero when the value is on the non-restrained side of the target.
    
    Args:
        value: Current value of the collective variable
        target: Target value for the restraint
        force_constant: Force constant for the harmonic restraint (k)
        side: Which side to apply the restraint on ("lower" or "upper")
            - "lower": Applies force when value < target (pulls up to target)
            - "upper": Applies force when value > target (pulls down to target)
    
    Returns:
        Tuple containing (energy, force)
    """
    # Calculate deviation from target
    diff = value - target
    
    if side == "lower":
        # Apply restraint only when value < target (pull up to target)
        violation = jnp.minimum(0.0, diff)  # Will be negative when below target, 0 when above
        
        # Energy is 0.5*k*violation^2 (harmonic)
        energy = 0.5 * force_constant * violation**2
        
        # Force is -k*violation (positive when below target, pulling up)
        # Zero when above target
        force = -force_constant * violation
        
    elif side == "upper":
        # Apply restraint only when value > target (pull down to target)
        violation = jnp.maximum(0.0, diff)  # Will be positive when above target, 0 when below
        
        # Energy is 0.5*k*violation^2 (harmonic)
        energy = 0.5 * force_constant * violation**2
        
        # Force is -k*violation (negative when above target, pulling down)
        # Zero when below target
        force = -force_constant * violation
        
    else:
        raise ValueError(f"Invalid side parameter: {side}. Must be 'lower' or 'upper'.")
    
    return energy, force

def distance_restraint_force(coordinates: jnp.ndarray, atom1: int, atom2: int, 
                            target: float, force_constant: float, 
                            restraint_function: Callable = harmonic_restraint) -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for a distance restraint.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        atom1, atom2: Indices of atoms for distance measurement
        target: Target distance
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Check coordinates shape and reshape if needed
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Get atom positions
    pos1 = coordinates[atom1]
    pos2 = coordinates[atom2]
    
    # Calculate the vector between atoms and its length
    r_vec = pos1 - pos2
    r = jnp.linalg.norm(r_vec)
    
    # Unit vector in the direction of r_vec
    # Avoid division by zero with a safe normalization
    r_safe = jnp.maximum(r, 1e-10)  # Ensure r is not zero
    unit_vec = r_vec / r_safe
    
    # Calculate restraint energy and force magnitude
    energy, force_magnitude = restraint_function(r, target, force_constant)
    
    # Forces on the atoms (equal and opposite)
    forces = jnp.zeros_like(coordinates)
    force_vec = force_magnitude * unit_vec
    
    # Update forces for the two atoms
    forces = forces.at[atom1].set(-force_vec)
    forces = forces.at[atom2].set(force_vec)
    
    return energy, forces

def angle_restraint_force(coordinates: jnp.ndarray, atom1: int, atom2: int, atom3: int,
                         target: float, force_constant: float,
                         restraint_function: Callable = harmonic_restraint) -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for an angle restraint.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        atom1, atom2, atom3: Indices of atoms defining the angle (atom2 is the vertex)
        target: Target angle in radians
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Check coordinates shape and reshape if needed
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Get atom positions
    pos1 = coordinates[atom1]
    pos2 = coordinates[atom2]
    pos3 = coordinates[atom3]
    
    # Calculate vectors
    v1 = pos1 - pos2
    v2 = pos3 - pos2
    
    # Normalize vectors
    v1_norm = jnp.linalg.norm(v1)
    v2_norm = jnp.linalg.norm(v2)
    v1_unit = v1 / jnp.maximum(v1_norm, 1e-10)
    v2_unit = v2 / jnp.maximum(v2_norm, 1e-10)
    
    # Calculate angle
    cos_angle = jnp.dot(v1_unit, v2_unit)
    # Clamp to avoid numerical issues at extreme values
    cos_angle = jnp.clip(cos_angle, -0.99999999, 0.99999999)
    angle = jnp.arccos(cos_angle)
    
    # Calculate restraint energy and force magnitude
    energy, force_magnitude = restraint_function(angle, target, force_constant)
    
    # Forces calculation based on derivative of arccos
    sin_angle = jnp.sin(angle)
    
    # Avoid division by zero
    sin_angle = jnp.where(sin_angle < 1e-8, 1e-8, sin_angle)
    
    # Components perpendicular to the vectors
    perp1 = v2_unit - cos_angle * v1_unit
    perp2 = v1_unit - cos_angle * v2_unit
    
    # Calculate forces on each atom
    forces = jnp.zeros_like(coordinates)
    
    # Force magnitudes
    f1_mag = -force_magnitude / (v1_norm * sin_angle)
    f3_mag = -force_magnitude / (v2_norm * sin_angle)
    
    # Force vectors
    f1 = f1_mag * perp1
    f3 = f3_mag * perp2
    f2 = -(f1 + f3)  # Total force must be zero
    
    # Update forces for the three atoms
    forces = forces.at[atom1].set(f1)
    forces = forces.at[atom2].set(f2)
    forces = forces.at[atom3].set(f3)
    
    return energy, forces

def dihedral_restraint_force(coordinates: jnp.ndarray, atom1: int, atom2: int, atom3: int, atom4: int,
                            target: float, force_constant: float,
                            restraint_function: Callable = harmonic_restraint) -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for a dihedral angle restraint.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        atom1, atom2, atom3, atom4: Indices of atoms defining the dihedral
        target: Target dihedral angle in radians
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Check coordinates shape and reshape if needed
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Get the dihedral angle using colvar_dihedral
    dihedral = colvar_dihedral(coordinates, atom1, atom2, atom3, atom4)
    
    # Calculate restraint energy and force magnitude
    energy, force_magnitude = restraint_function(dihedral, target, force_constant)
    
    # For dihedral forces, use automatic differentiation with JAX
    # Define a function that calculates just the dihedral angle
    def calc_dihedral(coords):
        proper_coords = ensure_proper_coordinates(coords)
        return colvar_dihedral(proper_coords, atom1, atom2, atom3, atom4)
    
    # Use JAX to compute the gradient
    grad_fn = jax.grad(calc_dihedral)
    gradients = grad_fn(coordinates)
    
    # Scale gradients by force magnitude to get forces
    forces = -force_magnitude * gradients
    
    return energy, forces

@partial(jax.jit, static_argnames=['mode'])
def _spherical_restraint_vectorized(atom_positions: jnp.ndarray, center: jnp.ndarray, 
                                   radius: float, force_constant: float, mode: str) -> Tuple[float, jnp.ndarray]:
    """Vectorized spherical restraint calculation (JIT compiled)."""
    # Calculate vectors from center and distances
    vectors_from_center = atom_positions - center[None, :]
    distances = jnp.linalg.norm(vectors_from_center, axis=1)
    
    if mode == "outside":
        # Keep atoms inside sphere - apply force when r > radius
        violations = jnp.maximum(0.0, distances - radius)
    else:  # mode == "inside"
        # Keep atoms outside sphere - apply force when r < radius
        violations = jnp.maximum(0.0, radius - distances)
    
    # Early exit if no violations (common case for well-behaved systems)
    total_violation = jnp.sum(violations)
    
    def compute_forces_and_energy():
        # Calculate unit vectors (avoid division by zero)
        safe_distances = jnp.maximum(distances, 1e-10)
        unit_vectors = vectors_from_center / safe_distances[:, None]
        
        # Vectorized harmonic restraint calculation
        # Energy = 0.5 * k * violation^2, Force = k * violation
        energies = 0.5 * force_constant * violations**2
        force_magnitudes = force_constant * violations
        
        # Calculate force vectors
        if mode == "outside":
            # Force points inward (toward center)
            force_vectors = -force_magnitudes[:, None] * unit_vectors
        else:
            # Force points outward (away from center)
            force_vectors = force_magnitudes[:, None] * unit_vectors
        
        # Only include non-zero violations
        active_mask = violations > 0
        energies = jnp.where(active_mask, energies, 0.0)
        force_vectors = jnp.where(active_mask[:, None], force_vectors, 0.0)
        
        return jnp.sum(energies), force_vectors
    
    def no_violations():
        return 0.0, jnp.zeros_like(atom_positions)
    
    # Conditional computation - only compute forces if there are violations
    total_energy, force_vectors = jax.lax.cond(
        total_violation > 0.0,
        lambda: compute_forces_and_energy(),
        lambda: no_violations()
    )
    
    return total_energy, force_vectors


def spherical_boundary_restraint_force(coordinates: jnp.ndarray, atom_indices: jnp.ndarray,
                                      center: jnp.ndarray, radius: float, force_constant: float,
                                      restraint_function: Callable = harmonic_restraint,
                                      mode: str = "outside") -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for a spherical boundary restraint.
    
    This restraint applies forces to keep atoms within (or outside) a spherical region.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        atom_indices: Indices of atoms to apply restraint to (array or "all")
        center: Center of the sphere [x, y, z]
        radius: Target radius of the sphere
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        mode: "outside" to keep atoms inside, "inside" to keep atoms outside
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Check coordinates shape and reshape if needed
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Convert center to jnp array
    center = jnp.array(center)
    
    # Handle atom indices
    if isinstance(atom_indices, str) and atom_indices == "all":
        atom_indices = jnp.arange(len(coordinates))
    else:
        atom_indices = jnp.array(atom_indices)
    
    # Get positions of constrained atoms
    atom_positions = coordinates[atom_indices]
    
    # Use fast vectorized implementation for harmonic restraints (most common case)
    if restraint_function == harmonic_restraint:
        total_energy, force_vectors = _spherical_restraint_vectorized(
            atom_positions, center, radius, force_constant, mode
        )
        
        # Scatter forces back to full coordinate array
        forces = jnp.zeros_like(coordinates)
        forces = forces.at[atom_indices].add(force_vectors)
        
        return total_energy, forces
    
    # Fallback to flexible implementation for custom restraint functions
    else:
        return _spherical_boundary_restraint_force_flexible(
            coordinates, atom_indices, center, radius, force_constant, restraint_function, mode
        )


def _spherical_boundary_restraint_force_flexible(coordinates: jnp.ndarray, atom_indices: jnp.ndarray,
                                                center: jnp.ndarray, radius: float, force_constant: float,
                                                restraint_function: Callable, mode: str) -> Tuple[float, jnp.ndarray]:
    """Flexible implementation for non-harmonic restraint functions."""
    # Get positions of constrained atoms
    atom_positions = coordinates[atom_indices]
    
    # Calculate distances from center for each atom
    vectors_from_center = atom_positions - center[None, :]
    distances = jnp.linalg.norm(vectors_from_center, axis=1)
    
    # Calculate unit vectors (avoid division by zero)
    safe_distances = jnp.maximum(distances, 1e-10)
    unit_vectors = vectors_from_center / safe_distances[:, None]
    
    # Initialize forces
    forces = jnp.zeros_like(coordinates)
    total_energy = 0.0
    
    # Apply restraint based on mode
    if mode == "outside":
        # Keep atoms inside sphere - apply force when r > radius
        violations = distances - radius
        # Use one-sided restraint - only apply when outside
        for i, (violation, dist, unit_vec) in enumerate(zip(violations, distances, unit_vectors)):
            if violation > 0:  # Atom is outside sphere
                energy, force_magnitude = restraint_function(dist, radius, force_constant)
                # Force points inward (toward center)
                force_vec = -force_magnitude * unit_vec
                atom_idx = atom_indices[i]
                forces = forces.at[atom_idx].add(force_vec)
                total_energy += energy
    else:  # mode == "inside"
        # Keep atoms outside sphere - apply force when r < radius
        violations = radius - distances
        # Use one-sided restraint - only apply when inside
        for i, (violation, dist, unit_vec) in enumerate(zip(violations, distances, unit_vectors)):
            if violation > 0:  # Atom is inside sphere
                energy, force_magnitude = restraint_function(dist, radius, force_constant)
                # Force points outward (away from center)
                force_vec = force_magnitude * unit_vec
                atom_idx = atom_indices[i]
                forces = forces.at[atom_idx].add(force_vec)
                total_energy += energy
    
    return total_energy, forces


def calculate_rmsd(coords1: jnp.ndarray, coords2: jnp.ndarray, 
                   atom_indices: Optional[jnp.ndarray] = None) -> float:
    """
    Calculate RMSD between two sets of coordinates.
    
    Args:
        coords1: First set of coordinates [n_atoms, 3]
        coords2: Second set of coordinates [n_atoms, 3]
        atom_indices: Optional indices of atoms to include in RMSD calculation
        
    Returns:
        RMSD value in Angstroms
    """
    if atom_indices is not None:
        coords1 = coords1[atom_indices]
        coords2 = coords2[atom_indices]
    
    # Calculate squared deviations
    squared_deviations = jnp.sum((coords1 - coords2)**2, axis=1)
    
    # Calculate RMSD
    rmsd = jnp.sqrt(jnp.mean(squared_deviations))
    
    return rmsd


@partial(jax.jit, static_argnames=['mode'])
def _rmsd_restraint_vectorized(coordinates: jnp.ndarray, reference: jnp.ndarray,
                               atom_indices: jnp.ndarray, target_rmsd: float,
                               force_constant: float, mode: str = "harmonic") -> Tuple[float, jnp.ndarray]:
    """Optimized vectorized RMSD restraint calculation."""
    # Get selected atoms
    selected_coords = coordinates[atom_indices]
    selected_ref = reference[atom_indices]
    
    # Calculate displacement vectors for each atom
    displacements = selected_coords - selected_ref
    
    # Calculate per-atom squared deviations
    squared_deviations = jnp.sum(displacements**2, axis=1)
    
    # Calculate current RMSD
    n_atoms = len(atom_indices)
    mean_squared_deviation = jnp.mean(squared_deviations)
    current_rmsd = jnp.sqrt(mean_squared_deviation)
    
    # Calculate RMSD deviation from target
    rmsd_diff = current_rmsd - target_rmsd
    
    if mode == "harmonic":
        # Harmonic restraint: E = 0.5 * k * (RMSD - target)^2
        energy = 0.5 * force_constant * rmsd_diff**2
        
        # Calculate gradient of RMSD with respect to coordinates
        # d(RMSD)/dx_i = (1/RMSD) * (1/N) * (x_i - x_i_ref)
        # where N is the number of atoms
        
        # Avoid division by zero
        safe_rmsd = jnp.maximum(current_rmsd, 1e-10)
        
        # Force = -k * (RMSD - target) * d(RMSD)/dx_i
        force_prefactor = -force_constant * rmsd_diff / (safe_rmsd * n_atoms)
        force_vectors = force_prefactor * displacements
        
    elif mode == "flat_bottom":
        # Flat-bottom restraint: only apply force if RMSD > target
        def apply_restraint():
            en = 0.5 * force_constant * rmsd_diff**2
            safe_rmsd = jnp.maximum(current_rmsd, 1e-10)
            force_prefactor = -force_constant * rmsd_diff / (safe_rmsd * n_atoms)
            fvec = force_prefactor * displacements
            return en, fvec
        
        def no_restraint():
            return 0.0, jnp.zeros_like(displacements)
        
        energy, force_vectors = jax.lax.cond(
            rmsd_diff > 0.0,
            apply_restraint,
            no_restraint
        )
    
    # Create full force array
    forces = jnp.zeros_like(coordinates)
    forces = forces.at[atom_indices].set(force_vectors)
    
    return energy, forces


def rmsd_restraint_force(coordinates: jnp.ndarray, reference_coordinates: jnp.ndarray,
                        atom_indices: jnp.ndarray, target_rmsd: float, force_constant: float,
                        restraint_function: Callable = harmonic_restraint,
                        mode: str = "harmonic") -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for an RMSD restraint.
    
    This restraint applies forces to maintain a target RMSD from a reference structure.
    
    Args:
        coordinates: Current atomic coordinates [n_atoms, 3]
        reference_coordinates: Reference atomic coordinates [n_atoms, 3]
        atom_indices: Indices of atoms to include in RMSD calculation (array or "all")
        target_rmsd: Target RMSD value in Angstroms
        force_constant: Force constant for the restraint
        restraint_function: Function to calculate restraint energy and force
        mode: "harmonic" for standard restraint, "flat_bottom" for one-sided
        
    Returns:
        Tuple containing (energy, forces array)
    """
    # Ensure coordinates are in the right shape
    coordinates = ensure_proper_coordinates(coordinates)
    reference_coordinates = ensure_proper_coordinates(reference_coordinates)
    
    # Handle atom indices
    if isinstance(atom_indices, str) and atom_indices == "all":
        atom_indices = jnp.arange(len(coordinates))
    else:
        atom_indices = jnp.array(atom_indices)
    
    # Use optimized implementation for standard modes
    if mode in ["harmonic", "flat_bottom"]:
        return _rmsd_restraint_vectorized(
            coordinates, reference_coordinates, atom_indices,
            target_rmsd, force_constant, mode
        )
    
    # Fallback for custom restraint functions
    else:
        return _rmsd_restraint_force_flexible(
            coordinates, reference_coordinates, atom_indices,
            target_rmsd, force_constant, restraint_function
        )


def _rmsd_restraint_force_flexible(coordinates: jnp.ndarray, reference_coordinates: jnp.ndarray,
                                   atom_indices: jnp.ndarray, target_rmsd: float, force_constant: float,
                                   restraint_function: Callable) -> Tuple[float, jnp.ndarray]:
    """Flexible implementation for custom restraint functions."""
    # Calculate current RMSD
    current_rmsd = calculate_rmsd(coordinates, reference_coordinates, atom_indices)
    
    # Calculate energy using the restraint function
    energy, force_magnitude = restraint_function(current_rmsd, target_rmsd, force_constant)
    
    # Calculate forces
    forces = jnp.zeros_like(coordinates)
    
    if abs(force_magnitude) > 1e-10:  # Only calculate if significant
        # Get selected atoms
        selected_coords = coordinates[atom_indices]
        selected_ref = reference_coordinates[atom_indices]
        
        # Calculate displacement vectors
        displacements = selected_coords - selected_ref
        
        # Calculate gradient of RMSD
        n_atoms = len(atom_indices)
        safe_rmsd = jnp.maximum(current_rmsd, 1e-10)
        
        # Force = -force_magnitude * d(RMSD)/dx_i
        force_prefactor = -force_magnitude / (safe_rmsd * n_atoms)
        force_vectors = force_prefactor * displacements
        
        # Apply forces to selected atoms
        forces = forces.at[atom_indices].set(force_vectors)
    
    return energy, forces


def find_leaving_group(coordinates: jnp.ndarray, carbon: int, nucleophile: int, 
                      cutoff: float = 2.0) -> int:
    """
    Find the most likely leaving group atom by finding the atom bonded to carbon
    that is furthest from the nucleophile.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        carbon: Index of the carbon atom
        nucleophile: Index of the nucleophile atom
        cutoff: Maximum distance to consider as bonded (in Angstroms)
        
    Returns:
        Index of the most likely leaving group atom
    """
    coordinates = ensure_proper_coordinates(coordinates)
    carbon_pos = coordinates[carbon]
    nucleophile_pos = coordinates[nucleophile]
    
    # Find all atoms within cutoff distance of carbon (bonded atoms)
    distances_to_carbon = jnp.linalg.norm(coordinates - carbon_pos, axis=1)
    bonded_mask = (distances_to_carbon < cutoff) & (jnp.arange(len(coordinates)) != carbon)
    bonded_indices = jnp.where(bonded_mask)[0]
    
    if len(bonded_indices) == 0:
        raise ValueError(f"No atoms found bonded to carbon {carbon} within {cutoff} Angstroms")
    
    # Calculate distances from nucleophile to each bonded atom
    distances_to_nucleophile = jnp.linalg.norm(coordinates[bonded_indices] - nucleophile_pos, axis=1)
    
    # Return the bonded atom furthest from nucleophile
    furthest_idx = jnp.argmax(distances_to_nucleophile)
    return bonded_indices[furthest_idx]

def adaptive_sn2_restraint_force(coordinates: jnp.ndarray, nucleophile: int, carbon: int, 
                               leaving_group: Optional[int], simulation_steps: int,
                               initial_coordinates: Optional[jnp.ndarray] = None,
                               current_step: Optional[int] = None) -> Tuple[float, jnp.ndarray]:
    """
    Adaptive SN2 restraint that automatically configures parameters based on the system.
    
    This restraint automatically determines:
    - Initial and target angles based on molecular geometry
    - Appropriate force constants based on molecular distances  
    - Smooth trajectory path to avoid system distortion
    - Adaptive distance restraints to guide the reaction
    
    Args:
        coordinates: Current atomic coordinates [n_atoms, 3]
        nucleophile: Index of the nucleophile atom
        carbon: Index of the carbon being attacked
        leaving_group: Index of the leaving group atom (if None, will auto-detect)
        simulation_steps: Total number of simulation steps
        initial_coordinates: Initial system coordinates for reference (if None, uses current)
        current_step: Current simulation step
        
    Returns:
        Tuple containing (energy, forces array)
    """
    global current_simulation_step, initial_coordinates_store
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Auto-detect leaving group if not provided
    if leaving_group is None:
        leaving_group = find_leaving_group(coordinates, carbon, nucleophile)
    
    # Use initial coordinates if provided, otherwise use stored or current
    if initial_coordinates is None:
        if initial_coordinates_store is None:
            # First time called - store current coordinates as initial
            initial_coordinates_store = coordinates.copy()
        initial_coordinates = initial_coordinates_store
    else:
        initial_coordinates = ensure_proper_coordinates(initial_coordinates)
    
    # Get initial geometry
    initial_angle = colvar_angle(initial_coordinates, nucleophile, carbon, leaving_group)
    initial_nu_c_dist = colvar_distance(initial_coordinates, nucleophile, carbon)
    initial_c_lg_dist = colvar_distance(initial_coordinates, carbon, leaving_group)
    
    # Determine reaction phases based on total steps
    phase1_steps = int(simulation_steps * 0.2)  # 20% - initial approach
    phase2_steps = int(simulation_steps * 0.5)  # 50% - angle adjustment  
    phase3_steps = int(simulation_steps * 0.3)  # 30% - final reaction
    
    # Determine current phase
    if current_step is None:
        current_step = current_simulation_step
    
    if current_step < phase1_steps:
        phase = 1
        phase_progress = current_step / phase1_steps
    elif current_step < phase1_steps + phase2_steps:
        phase = 2
        phase_progress = (current_step - phase1_steps) / phase2_steps
    else:
        phase = 3
        phase_progress = (current_step - phase1_steps - phase2_steps) / phase3_steps
    
    # Phase-dependent parameters
    if phase == 1:
        # Phase 1: Gentle approach while maintaining initial angle
        target_angle = initial_angle
        angle_fc = 0.05  # Soft constraint to maintain angle
        
        # Gradually decrease Nu-C distance
        target_nu_c_dist = initial_nu_c_dist - (initial_nu_c_dist - 4.0) * phase_progress * 0.5
        dist_fc = 0.02
        
    elif phase == 2:
        # Phase 2: Gradually adjust angle to 180 degrees
        angle_diff = jnp.pi - initial_angle
        target_angle = initial_angle + angle_diff * phase_progress
        
        # Increase angle force constant as we progress
        angle_fc = 0.05 + 0.15 * phase_progress
        
        # Continue Nu-C approach
        target_nu_c_dist = 4.0 - 1.5 * phase_progress
        dist_fc = 0.05 + 0.05 * phase_progress
        
    else:  # phase == 3
        # Phase 3: Strong enforcement of SN2 geometry
        target_angle = jnp.pi  # 180 degrees
        angle_fc = 0.2 + 0.3 * phase_progress
        
        # Final approach for reaction
        target_nu_c_dist = 2.5 - 0.8 * phase_progress
        dist_fc = 0.1 + 0.1 * phase_progress
    
    # Calculate restraint energies and forces
    # Angle restraint with flat-bottom to allow some flexibility
    angle_energy, angle_forces = angle_restraint_force(
        coordinates, nucleophile, carbon, leaving_group,
        target_angle, angle_fc, 
        partial(flat_bottom_restraint, tolerance=0.1)  # ~5.7 degrees tolerance
    )
    
    # Distance restraint for Nu-C
    dist_energy, dist_forces = distance_restraint_force(
        coordinates, nucleophile, carbon,
        target_nu_c_dist, dist_fc,
        partial(flat_bottom_restraint, tolerance=0.2)  # 0.2 Å tolerance
    )
    
    # Add adaptive leaving group restraint in phase 3
    lg_energy = 0.0
    lg_forces = jnp.zeros_like(coordinates)
    
    if phase >= 2:
        # Encourage C-LG bond elongation
        current_c_lg_dist = colvar_distance(coordinates, carbon, leaving_group)
        target_c_lg_dist = initial_c_lg_dist + 0.5 * (phase - 1)  # Gradually increase
        lg_fc = 0.02 * (phase - 1)  # Start soft, increase in phase 3
        
        lg_energy, lg_forces = distance_restraint_force(
            coordinates, carbon, leaving_group,
            target_c_lg_dist, lg_fc,
            partial(one_sided_harmonic_restraint, side="lower")  # Only push away, don't pull back
        )
    
    # Inversion prevention - stronger in later phases
    inversion_energy = 0.0
    inversion_forces = jnp.zeros_like(coordinates)
    
    if phase >= 2:
        # Calculate distances
        pos_nu = coordinates[nucleophile]
        pos_c = coordinates[carbon]
        pos_lg = coordinates[leaving_group]
        
        dist_nu_c = jnp.linalg.norm(pos_nu - pos_c)
        dist_nu_lg = jnp.linalg.norm(pos_nu - pos_lg)
        
        # Penalty if Nu-LG < Nu-C (wrong ordering)
        violation = jnp.maximum(0.0, dist_nu_c - dist_nu_lg + 0.5)  # 0.5 Å buffer
        penalty_fc = angle_fc * (1.0 + phase - 1)  # Increase with phase
        
        inversion_energy = 0.5 * penalty_fc * violation**2
        
        # Forces to maintain proper ordering
        if violation > 0:
            nu_lg_vec = pos_nu - pos_lg
            nu_lg_unit = nu_lg_vec / jnp.maximum(dist_nu_lg, 1e-10)
            force_nu_from_lg = penalty_fc * violation * nu_lg_unit
            
            inversion_forces = inversion_forces.at[nucleophile].add(force_nu_from_lg)
            inversion_forces = inversion_forces.at[leaving_group].add(-force_nu_from_lg)
    
    # Total energy and forces
    total_energy = angle_energy + dist_energy + lg_energy + inversion_energy
    total_forces = angle_forces + dist_forces + lg_forces + inversion_forces
    
    # Debug output for monitoring
    if restraint_debug_enabled and (current_step % 1000 == 0 or current_step < 10):
        import math
        current_angle = colvar_angle(coordinates, nucleophile, carbon, leaving_group)
        current_nu_c = colvar_distance(coordinates, nucleophile, carbon)
        current_c_lg = colvar_distance(coordinates, carbon, leaving_group)
        
        print(f"# ADAPTIVE SN2 DEBUG step={current_step} phase={phase} progress={phase_progress:.2f}")
        print(f"#   Angle: {current_angle*180/math.pi:.1f}° (target={target_angle*180/math.pi:.1f}°) fc={angle_fc:.3f}")
        print(f"#   Nu-C: {current_nu_c:.2f}Å (target={target_nu_c_dist:.2f}Å) fc={dist_fc:.3f}")
        print(f"#   C-LG: {current_c_lg:.2f}Å", end="")
        if phase >= 2:
            print(f" (target={target_c_lg_dist:.2f}Å) fc={lg_fc:.3f}")
        else:
            print()
        print(f"#   Total E={total_energy:.4f} (angle={angle_energy:.4f}, dist={dist_energy:.4f}, lg={lg_energy:.4f}, inv={inversion_energy:.4f})")
    
    return total_energy, total_forces


@partial(jax.jit, static_argnames=['prevent_inversion'])
def _backside_attack_core_vectorized(coordinates: jnp.ndarray, nucleophile: int, carbon: int,
                                    leaving_group: int, target_angle: float, angle_force_constant: float,
                                    target_distance: Optional[float], distance_force_constant: Optional[float],
                                    prevent_inversion: bool, inversion_penalty_factor: float) -> Tuple[float, jnp.ndarray]:
    """Optimized vectorized core computation for backside attack restraint."""
    # Get atom positions
    pos_nu = coordinates[nucleophile]
    pos_c = coordinates[carbon]
    pos_lg = coordinates[leaving_group]
    
    # Calculate all distances at once
    nu_c_vec = pos_nu - pos_c
    c_lg_vec = pos_c - pos_lg
    nu_lg_vec = pos_nu - pos_lg
    
    dist_nu_c = jnp.linalg.norm(nu_c_vec)
    dist_c_lg = jnp.linalg.norm(c_lg_vec)
    dist_nu_lg = jnp.linalg.norm(nu_lg_vec)
    
    # Calculate angle using dot product (more efficient than colvar_angle)
    # cos(angle) = (Nu-C) · (C-LG) / (|Nu-C| * |C-LG|)
    safe_dist_nu_c = jnp.maximum(dist_nu_c, 1e-10)
    safe_dist_c_lg = jnp.maximum(dist_c_lg, 1e-10)
    
    dot_product = jnp.dot(nu_c_vec, c_lg_vec)
    cos_angle = dot_product / (safe_dist_nu_c * safe_dist_c_lg)
    cos_angle = jnp.clip(cos_angle, -0.9999999, 0.9999999)
    current_angle = jnp.arccos(cos_angle)
    
    # Initialize total energy and forces
    total_energy = 0.0
    forces = jnp.zeros_like(coordinates)
    
    # Angle restraint energy (harmonic)
    angle_diff = current_angle - target_angle
    angle_energy = 0.5 * angle_force_constant * angle_diff**2
    total_energy += angle_energy
    
    # Angle forces using analytical derivatives
    sin_angle = jnp.sin(current_angle)
    sin_angle_safe = jnp.maximum(jnp.abs(sin_angle), 1e-10) * jnp.sign(sin_angle)
    
    dangle_force_magnitude = angle_force_constant * angle_diff / sin_angle_safe
    
    # Derivatives of cos(angle) with respect to positions
    inv_dist_nu_c = 1.0 / safe_dist_nu_c
    inv_dist_c_lg = 1.0 / safe_dist_c_lg
    
    # Force contributions from angle restraint
    f_nu_from_angle = dangle_force_magnitude * (
        c_lg_vec * inv_dist_nu_c * inv_dist_c_lg -
        nu_c_vec * dot_product * inv_dist_nu_c**3 * inv_dist_c_lg
    )
    
    f_lg_from_angle = dangle_force_magnitude * (
        nu_c_vec * inv_dist_nu_c * inv_dist_c_lg -
        c_lg_vec * dot_product * inv_dist_nu_c * inv_dist_c_lg**3
    )
    
    f_c_from_angle = -(f_nu_from_angle + f_lg_from_angle)
    
    forces = forces.at[nucleophile].add(f_nu_from_angle)
    forces = forces.at[carbon].add(f_c_from_angle)
    forces = forces.at[leaving_group].add(f_lg_from_angle)
    
    # Distance restraint (if specified)
    if target_distance is not None and distance_force_constant is not None:
        dist_diff = dist_nu_c - target_distance
        dist_energy = 0.5 * distance_force_constant * dist_diff**2
        total_energy += dist_energy
        
        # Distance force
        nu_c_unit = nu_c_vec / safe_dist_nu_c
        dist_force_magnitude = distance_force_constant * dist_diff
        force_nu_from_dist = -dist_force_magnitude * nu_c_unit
        
        forces = forces.at[nucleophile].add(force_nu_from_dist)
        forces = forces.at[carbon].add(-force_nu_from_dist)
    
    # Inversion prevention
    if prevent_inversion:
        violation = jnp.maximum(0.0, dist_nu_c - dist_nu_lg)
        penalty_fc = angle_force_constant * inversion_penalty_factor
        
        inversion_energy = 0.5 * penalty_fc * violation**2
        total_energy += inversion_energy
        
        # Only apply forces if there's a violation
        violation_force_magnitude = penalty_fc * violation
        
        # Force to increase Nu-LG distance
        nu_lg_unit = nu_lg_vec / jnp.maximum(dist_nu_lg, 1e-10)
        force_nu_from_lg = violation_force_magnitude * nu_lg_unit
        
        # Force to decrease Nu-C distance
        nu_c_unit = nu_c_vec / safe_dist_nu_c
        force_nu_to_c = -violation_force_magnitude * nu_c_unit * 0.5
        
        forces = forces.at[nucleophile].add(force_nu_from_lg + force_nu_to_c)
        forces = forces.at[leaving_group].add(-force_nu_from_lg)
        forces = forces.at[carbon].add(-force_nu_to_c)
    
    return total_energy, forces


def backside_attack_restraint_force(coordinates: jnp.ndarray, nucleophile: int, carbon: int, 
                                  leaving_group: Optional[int], target_angle: float, 
                                  angle_force_constant: float, target_distance: Optional[float] = None,
                                  distance_force_constant: Optional[float] = None,
                                  restraint_function: Callable = harmonic_restraint,
                                  prevent_inversion: bool = True,
                                  inversion_penalty_factor: float = 2.0,
                                  target_angle_list: Optional[List[float]] = None,
                                  angle_update_steps: Optional[int] = None,
                                  current_step: Optional[int] = None) -> Tuple[float, jnp.ndarray]:
    """
    Calculate the force and energy for a backside attack restraint.
    
    This restraint encourages the nucleophile to approach the carbon from the opposite 
    side of the leaving group, maintaining a linear geometry (180 degree angle).
    Optionally also constrains the nucleophile-carbon distance.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        nucleophile: Index of the nucleophile atom
        carbon: Index of the carbon being attacked
        leaving_group: Index of the leaving group atom (if None, will auto-detect)
        target_angle: Target angle in radians (typically π for 180°)
        angle_force_constant: Force constant for the angle restraint
        target_distance: Optional target nucleophile-carbon distance (in Angstroms)
        distance_force_constant: Optional force constant for distance restraint
        restraint_function: Function to calculate restraint energy and force
        prevent_inversion: Whether to add penalties to prevent Nu-LG-C arrangement
        inversion_penalty_factor: Factor to multiply force constants for penalty terms
        target_angle_list: Optional list of target angles for time-varying behavior
        angle_update_steps: Number of steps between angle updates
        current_step: Current simulation step (for time-varying angle)
        
    Returns:
        Tuple containing (energy, forces array)
    """
    global restraint_debug_enabled, current_simulation_step
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Auto-detect leaving group if not provided
    if leaving_group is None:
        leaving_group = find_leaving_group(coordinates, carbon, nucleophile)
    
    # Handle time-varying target angle if specified
    actual_target_angle = target_angle
    if target_angle_list is not None and angle_update_steps is not None and current_step is not None:
        actual_target_angle = time_varying_force_constant(
            target_angle_list, angle_update_steps, current_step,
            name="backside_angle_target"
        )
    
    # Use optimized implementation for harmonic restraints (most common case)
    if restraint_function == harmonic_restraint:
        total_energy, total_forces = _backside_attack_core_vectorized(
            coordinates, nucleophile, carbon, leaving_group,
            actual_target_angle, angle_force_constant,
            target_distance, distance_force_constant,
            prevent_inversion, inversion_penalty_factor
        )
    else:
        # Fallback to original implementation for custom restraint functions
        total_energy, total_forces = _backside_attack_restraint_force_flexible(
            coordinates, nucleophile, carbon, leaving_group,
            actual_target_angle, angle_force_constant,
            target_distance, distance_force_constant,
            restraint_function, prevent_inversion, inversion_penalty_factor
        )
    
    # Debug output if enabled
    if restraint_debug_enabled and (current_simulation_step % 10 == 0 or current_simulation_step <= 5):
        import math
        # Calculate current values for debug (using colvar functions for simplicity)
        angle = colvar_angle(coordinates, nucleophile, carbon, leaving_group)
        angle_deg = angle * 180.0 / math.pi
        actual_target_angle_deg = actual_target_angle * 180.0 / math.pi
        print(f"# RESTRAINT DEBUG [backside_attack] step={current_simulation_step}: angle={angle_deg:.2f}° (target={actual_target_angle_deg:.2f}°), angle_fc={angle_force_constant:.6f}, total_E={total_energy:.6f}")
        
        if target_distance is not None and distance_force_constant is not None:
            distance = colvar_distance(coordinates, nucleophile, carbon)
            print(f"#   distance={distance:.3f}Å (target={target_distance:.3f}Å), dist_fc={distance_force_constant:.6f}")
    
    return total_energy, total_forces


def _backside_attack_restraint_force_flexible(coordinates: jnp.ndarray, nucleophile: int, carbon: int,
                                            leaving_group: int, target_angle: float, 
                                            angle_force_constant: float, target_distance: Optional[float],
                                            distance_force_constant: Optional[float],
                                            restraint_function: Callable, prevent_inversion: bool,
                                            inversion_penalty_factor: float) -> Tuple[float, jnp.ndarray]:
    """Flexible implementation for non-harmonic restraint functions."""
    # Calculate angle component (always present)
    angle_energy, angle_forces = angle_restraint_force(
        coordinates, nucleophile, carbon, leaving_group, 
        target_angle, angle_force_constant, restraint_function
    )
    
    total_energy = angle_energy
    total_forces = angle_forces
    
    # Add distance component if specified
    if target_distance is not None and distance_force_constant is not None:
        distance_energy, distance_forces = distance_restraint_force(
            coordinates, nucleophile, carbon,
            target_distance, distance_force_constant, restraint_function
        )
        total_energy += distance_energy
        total_forces += distance_forces
    
    # Add inversion prevention if enabled
    if prevent_inversion:
        # Get atom positions
        pos_nu = coordinates[nucleophile]
        pos_c = coordinates[carbon]
        pos_lg = coordinates[leaving_group]
        
        # Calculate distances
        dist_nu_c = jnp.linalg.norm(pos_nu - pos_c)
        dist_nu_lg = jnp.linalg.norm(pos_nu - pos_lg)
        
        # Add a penalty if Nu-LG distance is smaller than Nu-C distance
        violation = jnp.maximum(0.0, dist_nu_c - dist_nu_lg)
        penalty_force_constant = angle_force_constant * inversion_penalty_factor
        
        # Energy penalty: harmonic penalty on the violation
        inversion_energy = 0.5 * penalty_force_constant * violation**2
        
        # Calculate forces for the inversion penalty
        nu_lg_vec = pos_nu - pos_lg
        nu_lg_unit = nu_lg_vec / jnp.maximum(dist_nu_lg, 1e-10)
        force_nu_from_lg = penalty_force_constant * violation * nu_lg_unit
        
        nu_c_vec = pos_nu - pos_c
        nu_c_unit = nu_c_vec / jnp.maximum(dist_nu_c, 1e-10)
        force_nu_to_c = -penalty_force_constant * violation * nu_c_unit * 0.5
        
        # Apply forces
        inversion_forces = jnp.zeros_like(coordinates)
        inversion_forces = inversion_forces.at[nucleophile].add(force_nu_from_lg + force_nu_to_c)
        inversion_forces = inversion_forces.at[leaving_group].add(-force_nu_from_lg)
        inversion_forces = inversion_forces.at[carbon].add(-force_nu_to_c)
        
        total_energy += inversion_energy
        total_forces += inversion_forces
    
    return total_energy, total_forces

def time_varying_force_constant(force_constants: List[float], update_steps: int, current_step: int,
                             debug: bool = False, name: str = "", mode: str = "interpolate") -> float:
    """
    Calculate the force constant for the current step based on a list of force constants
    that change over time at evenly spaced intervals.
    
    Args:
        force_constants: List of force constant values to interpolate between
        update_steps: Number of steps between force constant updates
        current_step: Current simulation step
        debug: Whether to print debug information
        name: Name of the restraint (for debugging)
        mode: "interpolate" for smooth linear interpolation, "discrete" for step changes
        
    Returns:
        Current force constant value
    """
    # Use our global step counter instead of the parameter
    # This ensures consistency with apply_restraints
    global current_simulation_step
    true_step = current_simulation_step  # Use global step that's incremented in apply_restraints
    
    if len(force_constants) == 1:
        return force_constants[0]
    
    if mode == "discrete":
        # Discrete mode: jump to new value at each update interval
        segment_idx = min(true_step // update_steps, len(force_constants) - 1)
        current_k = force_constants[segment_idx]
    else:
        # Interpolate mode (default): smooth linear interpolation
        # Determine which segment we're in and how far through it
        num_segments = len(force_constants) - 1
        total_transition_steps = num_segments * update_steps
        
        # Handle case where we've gone past the last force constant
        if true_step >= total_transition_steps:
            return force_constants[-1]
        
        # Calculate which segment we're in and progress through that segment
        segment_idx = min(true_step // update_steps, num_segments - 1)
        progress = (true_step % update_steps) / update_steps
        
        # Linearly interpolate between adjacent force constants
        start_k = force_constants[segment_idx]
        end_k = force_constants[segment_idx + 1]
        current_k = start_k + progress * (end_k - start_k)
    
    # Store this in the global tracking dictionary
    global current_force_constants
    if 'current_force_constants' not in globals():
        current_force_constants = {}
    current_force_constants[name] = current_k
    
    return current_k

def setup_restraints(restraint_definitions: Dict[str, Any], nsteps: Optional[int] = None,
                    initial_coordinates: Optional[jnp.ndarray] = None, 
                    pdb_data: Optional[Dict] = None,
                    symbols: Optional[List[str]] = None) -> Tuple[Dict[str, Callable], List[Callable], Dict[str, Any]]:
    """
    Set up restraint calculators based on the restraint definitions in the input file.
    
    Args:
        restraint_definitions: Dictionary containing restraint definitions from input file
        nsteps: Total number of simulation steps (used to auto-calculate update_steps)
        initial_coordinates: Initial system coordinates (for auto-center calculation)
        pdb_data: Optional PDB data for atom selection
        symbols: Optional element symbols for mass calculation
        
    Returns:
        Tuple containing (restraint_energies, restraint_forces, restraint_metadata):
        - restraint_energies: Dict mapping restraint names to energy calculator functions
        - restraint_forces: List of force calculator functions for all restraints
        - restraint_metadata: Dict containing metadata for time-varying restraints
    """
    restraint_energies = {}
    restraint_forces = []
    restraint_metadata = {
        "time_varying": False,
        "restraints": {}
    }
    
    for restraint_name, restraint_def in restraint_definitions.items():
        restraint_type = restraint_def.get("type", "distance")
        restraint_style = restraint_def.get("style", "harmonic")
        tolerance = float(restraint_def.get("tolerance", 0.0))
        side = restraint_def.get("side", "lower")  # For one-sided restraints
        
        # Handle target - defer parsing until we know the restraint type
        # since backside_attack can have a list of targets
        target_raw = restraint_def.get("target", 0.0)
        
        # Handle force constant - can be a single value or a list for time-varying
        force_constant_raw = restraint_def.get("force_constant", 10.0)
        
        # Check if we have a list of force constants (for time-varying)
        time_varying = False
        interpolation_mode = "interpolate"  # Default mode
        if isinstance(force_constant_raw, list):
            force_constants = [float(k) for k in force_constant_raw]
            
            # Calculate update_steps: if nsteps provided and update_steps not specified, 
            # divide total steps by number of windows
            if "update_steps" not in restraint_def and nsteps is not None:
                n_windows = len(force_constants)
                update_steps = int(nsteps / n_windows)
                print(f"# Auto-calculated update_steps for '{restraint_name}': {update_steps} ({nsteps} steps / {n_windows} windows)")
            else:
                update_steps = int(restraint_def.get("update_steps", 1000))
            
            force_constant = force_constants[0]  # Initial value
            time_varying = True
            
            # Get interpolation mode (default is "interpolate" for backward compatibility)
            interpolation_mode = restraint_def.get("interpolation_mode", "interpolate")
            if interpolation_mode not in ["interpolate", "discrete"]:
                raise ValueError(f"Invalid interpolation_mode: {interpolation_mode}. Must be 'interpolate' or 'discrete'.")
            
            # Check if PMF estimation is requested
            estimate_pmf = restraint_def.get("estimate_pmf", False)
            equilibration_ratio = float(restraint_def.get("equilibration_ratio", 0.2))  # 20% equilibration by default
            
            # Print PMF configuration info if enabled
            if estimate_pmf and interpolation_mode == "discrete":
                equilibration_steps = int(update_steps * equilibration_ratio)
                sampling_steps = update_steps - equilibration_steps
                print(f"\n# PMF CONFIGURATION for restraint '{restraint_name}':")
                print(f"#   Interpolation mode: {interpolation_mode}")
                print(f"#   Estimate PMF: {estimate_pmf} (Note: PMF calculation moved to post-processing)")
                print(f"#   Update steps: {update_steps}")
                print(f"#   Equilibration: {equilibration_steps} steps ({equilibration_ratio*100:.0f}%)")
                print(f"#   Sampling: {sampling_steps} steps ({(1-equilibration_ratio)*100:.0f}%)")
                print(f"#   Windows to scan: {len(force_constants)}")
                print(f"#   Force constant values: {force_constants}")
            elif estimate_pmf and interpolation_mode != "discrete":
                print(f"\n# PMF WARNING for restraint '{restraint_name}':")
                print(f"#   estimate_pmf=yes but interpolation_mode='{interpolation_mode}'")
                print(f"#   PMF requires interpolation_mode='discrete' - PMF will be DISABLED")
            
            # Store metadata for this time-varying restraint
            restraint_metadata["time_varying"] = True
            restraint_metadata["restraints"][restraint_name] = {
                "force_constants": force_constants,
                "update_steps": update_steps,
                "interpolation_mode": interpolation_mode,
                "estimate_pmf": estimate_pmf and interpolation_mode == "discrete",
                "equilibration_ratio": equilibration_ratio,
                "current_idx": 0
            }
        else:
            force_constant = float(force_constant_raw)
        
        # Select the appropriate restraint function
        if restraint_style == "harmonic":
            restraint_function = harmonic_restraint
        elif restraint_style == "flat_bottom":
            restraint_function = partial(flat_bottom_restraint, tolerance=tolerance)
        elif restraint_style == "one_sided":
            restraint_function = partial(one_sided_harmonic_restraint, side=side)
        else:
            raise ValueError(f"Unknown restraint style: {restraint_style}")
            
        # Configure the appropriate force calculator based on restraint type
        if restraint_type == "distance":
            # For distance restraints, target is always a single float
            target = float(target_raw)
            atom1 = int(restraint_def["atom1"])
            atom2 = int(restraint_def["atom2"])
            
            if time_varying:
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, a1=atom1, a2=atom2, t=target, 
                               fc_list=force_constants, upd_steps=update_steps, rname=restraint_name,
                               interp_mode=interpolation_mode,
                               rf=restraint_function, rm=restraint_metadata):
                    coordinates = ensure_proper_coordinates(coordinates)
                    distance = colvar_distance(coordinates, a1, a2)
                    # Get current force constant based on step
                    fc = time_varying_force_constant(fc_list, upd_steps, step, mode=interp_mode)
                    energy, _ = rf(distance, t, fc)
                    # Store current force constant value for reporting
                    if rm["time_varying"] and rname in rm["restraints"]:
                        rm["restraints"][rname]["current_fc"] = fc
                        
                        # PMF sampling if enabled
                        if rm["restraints"][rname].get("estimate_pmf", False) and interp_mode == "discrete":
                            # Use global step counter for consistency
                            global current_simulation_step
                            true_step = current_simulation_step
                            window_idx = min(true_step // upd_steps, len(dist_fc_list) - 1)
                            equilibration_steps = int(upd_steps * rm["restraints"][rname]["equilibration_ratio"])
                            
                            # PMF calculation moved to post-processing
                            # Initialize marker to prevent re-initialization
                            if rname not in pmf_data:
                                pmf_data[rname] = {'initialized': True}
                            
                            # PMF sampling moved to post-processing
                    return energy
                
                # Create time-varying force calculator
                def time_varying_force_calc(coordinates, step, a1=atom1, a2=atom2, t=target,
                                          fc_list=force_constants, upd_steps=update_steps, 
                                          interp_mode=interpolation_mode,
                                          rf=restraint_function, rname=restraint_name):
                    # Get current force constant based on step
                    # Get the force constant from our reliable function
                    fc = time_varying_force_constant(fc_list, upd_steps, step, name=rname, mode=interp_mode)
                    current_force_constants[rname] = fc
                    
                    # Use the standard force calculator with the current force constant
                    return distance_restraint_force(coordinates, a1, a2, t, fc, rf)
                
                force_calc = time_varying_force_calc
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, a1=atom1, a2=atom2, t=target, fc=force_constant, rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    distance = colvar_distance(coordinates, a1, a2)
                    energy, _ = rf(distance, t, fc)
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    distance_restraint_force, 
                    atom1=atom1, 
                    atom2=atom2, 
                    target=target, 
                    force_constant=force_constant,
                    restraint_function=restraint_function
                )
            
        elif restraint_type == "angle":
            # For angle restraints, target is always a single float
            target = float(target_raw)
            atom1 = int(restraint_def["atom1"])
            atom2 = int(restraint_def["atom2"])
            atom3 = int(restraint_def["atom3"])
            
            if time_varying:
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, t=target, 
                               fc_list=force_constants, upd_steps=update_steps, rname=restraint_name,
                               interp_mode=interpolation_mode,
                               rf=restraint_function, rm=restraint_metadata):
                    coordinates = ensure_proper_coordinates(coordinates)
                    angle = colvar_angle(coordinates, a1, a2, a3)
                    # Get current force constant based on step
                    fc = time_varying_force_constant(fc_list, upd_steps, step, mode=interp_mode)
                    energy, _ = rf(angle, t, fc)
                    # Store current force constant value for reporting
                    if rm["time_varying"] and rname in rm["restraints"]:
                        rm["restraints"][rname]["current_fc"] = fc
                    return energy
                
                # Create time-varying force calculator
                def time_varying_force_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, t=target,
                                          fc_list=force_constants, upd_steps=update_steps, 
                                          interp_mode=interpolation_mode,
                                          rf=restraint_function, rname=restraint_name):
                    # Get current force constant based on step
                    # Get the force constant from our reliable function
                    fc = time_varying_force_constant(fc_list, upd_steps, step, name=rname, mode=interp_mode)
                    current_force_constants[rname] = fc
                    
                    # Use the standard force calculator with the current force constant
                    return angle_restraint_force(coordinates, a1, a2, a3, t, fc, rf)
                
                force_calc = time_varying_force_calc
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, a1=atom1, a2=atom2, a3=atom3, t=target, fc=force_constant, rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    angle = colvar_angle(coordinates, a1, a2, a3)
                    energy, _ = rf(angle, t, fc)
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    angle_restraint_force, 
                    atom1=atom1, 
                    atom2=atom2, 
                    atom3=atom3, 
                    target=target, 
                    force_constant=force_constant,
                    restraint_function=restraint_function
                )
            
        elif restraint_type == "dihedral":
            # For dihedral restraints, target is always a single float
            target = float(target_raw)
            atom1 = int(restraint_def["atom1"])
            atom2 = int(restraint_def["atom2"])
            atom3 = int(restraint_def["atom3"])
            atom4 = int(restraint_def["atom4"])
            
            if time_varying:
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, a4=atom4, t=target, 
                               fc_list=force_constants, upd_steps=update_steps, rname=restraint_name,
                               interp_mode=interpolation_mode,
                               rf=restraint_function, rm=restraint_metadata):
                    coordinates = ensure_proper_coordinates(coordinates)
                    dihedral = colvar_dihedral(coordinates, a1, a2, a3, a4)
                    # Get current force constant based on step
                    fc = time_varying_force_constant(fc_list, upd_steps, step, mode=interp_mode)
                    energy, _ = rf(dihedral, t, fc)
                    # Store current force constant value for reporting
                    if rm["time_varying"] and rname in rm["restraints"]:
                        rm["restraints"][rname]["current_fc"] = fc
                    return energy
                
                # Create time-varying force calculator
                def time_varying_force_calc(coordinates, step, a1=atom1, a2=atom2, a3=atom3, a4=atom4, t=target,
                                          fc_list=force_constants, upd_steps=update_steps, 
                                          interp_mode=interpolation_mode,
                                          rf=restraint_function, rname=restraint_name):
                    # Get current force constant based on step
                    # Get the force constant from our reliable function
                    fc = time_varying_force_constant(fc_list, upd_steps, step, name=rname, mode=interp_mode)
                    current_force_constants[rname] = fc
                    
                    # Use the standard force calculator with the current force constant
                    return dihedral_restraint_force(coordinates, a1, a2, a3, a4, t, fc, rf)
                
                force_calc = time_varying_force_calc
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, a1=atom1, a2=atom2, a3=atom3, a4=atom4, t=target, fc=force_constant, rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    dihedral = colvar_dihedral(coordinates, a1, a2, a3, a4)
                    energy, _ = rf(dihedral, t, fc)
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    dihedral_restraint_force, 
                    atom1=atom1, 
                    atom2=atom2, 
                    atom3=atom3, 
                    atom4=atom4, 
                    target=target, 
                    force_constant=force_constant,
                    restraint_function=restraint_function
                )
        
        elif restraint_type == "adaptive_sn2":
            # New adaptive SN2 restraint type
            nucleophile = int(restraint_def["nucleophile"])
            carbon = int(restraint_def["carbon"])
            leaving_group = restraint_def.get("leaving_group")  # Optional - can be None
            if leaving_group is not None:
                leaving_group = int(leaving_group)
            
            # Get total simulation steps - required for adaptive restraint
            simulation_steps = int(restraint_def["simulation_steps"])
            
            # Create energy calculator for adaptive SN2
            def energy_calc(coordinates, step, nu=nucleophile, c=carbon, lg=leaving_group,
                           sim_steps=simulation_steps, rname=restraint_name):
                coordinates = ensure_proper_coordinates(coordinates)
                energy, _ = adaptive_sn2_restraint_force(coordinates, nu, c, lg, sim_steps, 
                                                       current_step=step)
                return energy
            
            # Create force calculator for adaptive SN2
            def force_calc(coordinates, step, nu=nucleophile, c=carbon, lg=leaving_group,
                          sim_steps=simulation_steps):
                return adaptive_sn2_restraint_force(coordinates, nu, c, lg, sim_steps,
                                                  current_step=step)
            
            # Mark as time-varying since it changes with simulation progress
            restraint_metadata["time_varying"] = True
            restraint_metadata["restraints"][restraint_name] = {
                "type": "adaptive_sn2",
                "simulation_steps": simulation_steps
            }
            
        elif restraint_type == "backside_attack":
            nucleophile = int(restraint_def["nucleophile"])
            carbon = int(restraint_def["carbon"])
            leaving_group = restraint_def.get("leaving_group")  # Optional - can be None
            if leaving_group is not None:
                leaving_group = int(leaving_group)
            
            # Handle target angle - can be in degrees or radians, or a list for time-varying
            # Use target_raw which was already extracted above
            target_angle_raw = target_raw
            target_angle_time_varying = False
            target_angles = None
            
            # Check if we have a list of target angles (for time-varying)
            if isinstance(target_angle_raw, list):
                # Convert each angle in the list
                target_angles = []
                for angle in target_angle_raw:
                    if isinstance(angle, (int, float)) and abs(angle) > 2 * jnp.pi:
                        # Convert from degrees to radians
                        target_angles.append(float(angle) * jnp.pi / 180.0)
                    else:
                        # Already in radians
                        target_angles.append(float(angle))
                target_angle = target_angles[0]  # Initial value
                target_angle_time_varying = True
            else:
                # Single value
                if isinstance(target_angle_raw, (int, float)) and abs(target_angle_raw) > 2 * jnp.pi:
                    # Convert from degrees to radians
                    target_angle = float(target_angle_raw) * jnp.pi / 180.0
                else:
                    # Already in radians
                    target_angle = float(target_angle_raw)
            
            # New parameters for composite restraint
            target_distance_raw = restraint_def.get("target_distance")  # Optional distance target
            target_distance_time_varying = False
            target_distances = None
            
            # Check if we have a list of target distances (for time-varying)
            if target_distance_raw is not None:
                if isinstance(target_distance_raw, list):
                    # Time-varying target distance
                    target_distances = [float(d) for d in target_distance_raw]
                    target_distance = target_distances[0]  # Initial value
                    target_distance_time_varying = True
                else:
                    # Single value
                    target_distance = float(target_distance_raw)
            else:
                target_distance = None
            
            # Handle force constants - can be separate for angle and distance
            angle_force_constant_raw = restraint_def.get("angle_force_constant", force_constant_raw)
            distance_force_constant_raw = restraint_def.get("distance_force_constant")
            
            # Inversion prevention parameters
            prevent_inversion = restraint_def.get("prevent_inversion", True)
            inversion_penalty_factor = float(restraint_def.get("inversion_penalty_factor", 2.0))
            
            # Write frequency for reaction coordinate data (default: every step)
            write_frequency = int(restraint_def.get("write_frequency", 1))
            
            # Handle time-varying for angle force constant
            angle_time_varying = False
            if isinstance(angle_force_constant_raw, list):
                angle_force_constants = [float(k) for k in angle_force_constant_raw]
                angle_force_constant = angle_force_constants[0]
                angle_time_varying = True
            else:
                angle_force_constant = float(angle_force_constant_raw)
            
            # Handle time-varying for distance force constant
            distance_time_varying = False
            distance_force_constant = None
            distance_force_constants = None
            if distance_force_constant_raw is not None:
                if isinstance(distance_force_constant_raw, list):
                    distance_force_constants = [float(k) for k in distance_force_constant_raw]
                    distance_force_constant = distance_force_constants[0]
                    distance_time_varying = True
                else:
                    distance_force_constant = float(distance_force_constant_raw)
            
            # If any component is time-varying, the whole restraint is time-varying
            restraint_time_varying = angle_time_varying or distance_time_varying or target_angle_time_varying or target_distance_time_varying or time_varying
            
            if restraint_time_varying:
                # Calculate update_steps for backside attack restraint
                # Determine the number of windows based on which parameter is varying
                n_windows = 1
                if target_distances is not None:
                    n_windows = len(target_distances)
                elif angle_force_constants is not None:
                    n_windows = len(angle_force_constants)
                elif distance_force_constants is not None:
                    n_windows = len(distance_force_constants)
                elif target_angles is not None:
                    n_windows = len(target_angles)
                
                # Auto-calculate update_steps if not provided
                if "update_steps" not in restraint_def and nsteps is not None and n_windows > 1:
                    update_steps = int(nsteps / n_windows)
                    print(f"# Auto-calculated update_steps for '{restraint_name}': {update_steps} ({nsteps} steps / {n_windows} windows)")
                else:
                    update_steps = int(restraint_def.get("update_steps", 1000))
                # Get interpolation mode (default is "interpolate" for backward compatibility)
                interpolation_mode = restraint_def.get("interpolation_mode", "interpolate")
                if interpolation_mode not in ["interpolate", "discrete"]:
                    raise ValueError(f"Invalid interpolation_mode: {interpolation_mode}. Must be 'interpolate' or 'discrete'.")
                
                # Check if PMF estimation is requested
                estimate_pmf = restraint_def.get("estimate_pmf", False)
                equilibration_ratio = float(restraint_def.get("equilibration_ratio", 0.2))  # 20% equilibration by default
                
                # Print PMF configuration info for backside_attack if enabled
                if estimate_pmf and interpolation_mode == "discrete":
                    equilibration_steps = int(update_steps * equilibration_ratio)
                    sampling_steps = update_steps - equilibration_steps
                    print(f"\n# PMF CONFIGURATION for restraint '{restraint_name}':")
                    print(f"#   Type: backside_attack")
                    print(f"#   Interpolation mode: {interpolation_mode}")
                    print(f"#   Estimate PMF: {estimate_pmf} (Note: PMF calculation moved to post-processing)")
                    print(f"#   Update steps: {update_steps}")
                    print(f"#   Equilibration: {equilibration_steps} steps ({equilibration_ratio*100:.0f}%)")
                    print(f"#   Sampling: {sampling_steps} steps ({(1-equilibration_ratio)*100:.0f}%)")
                    
                    # Show which parameter is being scanned
                    if target_distance_time_varying:
                        print(f"#   Scanning: target_distance")
                        print(f"#   Windows: {len(target_distances)}")
                        print(f"#   Distance values: {target_distances}")
                    elif target_angle_time_varying:
                        print(f"#   Scanning: target_angle")
                        print(f"#   Windows: {len(target_angles)}")
                        angle_degrees = [float(a) * 180.0 / 3.14159 for a in target_angles]
                        print(f"#   Angle values (degrees): {angle_degrees}")
                    elif angle_time_varying:
                        print(f"#   Scanning: angle_force_constant")
                        print(f"#   Windows: {len(angle_force_constants)}")
                        print(f"#   Force constant values: {angle_force_constants}")
                    elif distance_time_varying:
                        print(f"#   Scanning: distance_force_constant")
                        print(f"#   Windows: {len(distance_force_constants)}")
                        print(f"#   Force constant values: {distance_force_constants}")
                    
                    # Check for real-time output
                    write_realtime = restraint_def.get("write_realtime_pmf", False)
                    if write_realtime:
                        print(f"#   Real-time output: ENABLED")
                        
                elif estimate_pmf and interpolation_mode != "discrete":
                    print(f"\n# PMF WARNING for restraint '{restraint_name}':")
                    print(f"#   estimate_pmf=yes but interpolation_mode='{interpolation_mode}'")
                    print(f"#   PMF requires interpolation_mode='discrete' - PMF will be DISABLED")
                
                # Store metadata for time-varying restraints
                if not restraint_metadata["time_varying"]:
                    restraint_metadata["time_varying"] = True
                
                # Check for real-time output
                write_realtime_pmf = restraint_def.get("write_realtime_pmf", False)
                
                restraint_metadata["restraints"][restraint_name] = {
                    "angle_force_constants": angle_force_constants if angle_time_varying else [angle_force_constant],
                    "distance_force_constants": distance_force_constants if distance_time_varying else ([distance_force_constant] if distance_force_constant is not None else None),
                    "target_angles": target_angles if target_angle_time_varying else [target_angle],
                    "target_distances": target_distances if target_distance_time_varying else ([target_distance] if target_distance is not None else None),
                    "update_steps": update_steps,
                    "interpolation_mode": interpolation_mode,
                    "estimate_pmf": estimate_pmf and interpolation_mode == "discrete",
                    "equilibration_ratio": equilibration_ratio,
                    "write_realtime_pmf": write_realtime_pmf,
                    "output_dir": ".",
                    "current_idx": 0
                }
                
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, nucleophile=nucleophile, carbon=carbon, 
                               leaving_group=leaving_group, t_angle=target_angle, t_dist=target_distance,
                               angle_fc_list=angle_force_constants if angle_time_varying else [angle_force_constant],
                               dist_fc_list=distance_force_constants if distance_time_varying else ([distance_force_constant] if distance_force_constant is not None else None),
                               target_angle_list=target_angles if target_angle_time_varying else [target_angle],
                               target_distance_list=target_distances if target_distance_time_varying else ([target_distance] if target_distance is not None else None),
                               angle_time_var=target_angle_time_varying,
                               distance_time_var=target_distance_time_varying,
                               upd_steps=update_steps, rname=restraint_name,
                               interp_mode=interpolation_mode,
                               rf=restraint_function, rm=restraint_metadata):
                    # Declare all globals at the top
                    global current_simulation_step, restraint_debug_enabled, pmf_data
                    
                    coordinates = ensure_proper_coordinates(coordinates)
                    
                    # Auto-detect leaving group if not provided
                    if leaving_group is None:
                        lg = find_leaving_group(coordinates, carbon, nucleophile)
                    else:
                        lg = leaving_group
                    
                    # Calculate angle energy with potentially time-varying target
                    angle = colvar_angle(coordinates, nucleophile, carbon, lg)
                    angle_fc = time_varying_force_constant(angle_fc_list, upd_steps, step, mode=interp_mode) if angle_time_varying else angle_fc_list[0]
                    
                    # Get current target angle if time-varying
                    current_target_angle = time_varying_force_constant(target_angle_list, upd_steps, step, name=f"{rname}_target_angle", mode=interp_mode) if angle_time_var else t_angle
                    
                    angle_energy, _ = rf(angle, current_target_angle, angle_fc)
                    
                    total_energy = angle_energy
                    
                    # Add distance energy if specified
                    if target_distance_list is not None and dist_fc_list is not None:
                        distance = colvar_distance(coordinates, nucleophile, carbon)
                        dist_fc = time_varying_force_constant(dist_fc_list, upd_steps, step, mode=interp_mode) if distance_time_varying else dist_fc_list[0]
                        # Get current target distance if time-varying
                        current_target_distance = time_varying_force_constant(target_distance_list, upd_steps, step, name=f"{rname}_target_distance", mode=interp_mode) if distance_time_var else target_distance_list[0]
                        dist_energy, _ = rf(distance, current_target_distance, dist_fc)
                        total_energy += dist_energy
                    
                    # Write reaction coordinate data if requested
                    write_realtime = rm["restraints"][rname].get("write_realtime_pmf", False)
                    
                    # Debug: Print if this is being called
                    if step == 0:
                        print(f"# DEBUG: backside_attack energy_calc called for {rname}, write_realtime={write_realtime}")
                        if write_realtime:
                            print(f"#   Will write to reaction_coords_{rname}.dat")
                    
                    if write_realtime:
                        # Initialize output file on first call
                        if rname not in pmf_data:
                            pmf_data[rname] = {'initialized': True}
                            output_dir = rm["restraints"][rname].get("output_dir", ".")
                            filename = f"{output_dir}/reaction_coords_{rname}.dat"
                            with open(filename, 'w') as f:
                                f.write(f"# Reaction coordinate data for restraint: {rname}\n")
                                f.write(f"# Nucleophile: {nucleophile}, Carbon: {carbon}, Leaving group: {lg}\n")
                                f.write(f"# Number of windows: {len(target_distance_list) if target_distance_list else 1}\n")
                                f.write(f"# Target distances: {target_distance_list}\n")
                                f.write(f"# Columns: Step Window_Index Nu-C_Distance(Ang) Nu-C-LG_Angle(deg) C-LG_Distance(Ang) Target_Distance Force_Constant\n")
                                f.write(f"#{'Step':>8s} {'Win':>3s} {'Nu-C_Dist':>12s} {'Angle':>12s} {'C-LG_Dist':>12s} {'Target':>10s} {'Force_K':>10s}\n")
                            print(f"# Reaction coordinate output: {filename}")
                        
                        # Calculate distances
                        nu_c_distance = colvar_distance(coordinates, nucleophile, carbon)
                        clg_distance = colvar_distance(coordinates, carbon, lg)
                        
                        # Get current window index
                        true_step = current_simulation_step
                        window_idx = min(true_step // upd_steps, len(target_distance_list) - 1) if target_distance_list else 0
                        
                        # Get current target distance and force constant
                        if target_distance_list is not None and len(target_distance_list) > 0:
                            current_target = target_distance_list[window_idx] if window_idx < len(target_distance_list) else target_distance_list[-1]
                            current_fc = dist_fc_list[window_idx] if dist_fc_list and window_idx < len(dist_fc_list) else (dist_fc if dist_fc is not None else 0.0)
                        else:
                            current_target = 0.0
                            current_fc = 0.0
                        
                        # Convert angle to degrees
                        import math
                        angle_deg = angle * 180.0 / math.pi
                        
                        # Write to file
                        output_dir = rm["restraints"][rname].get("output_dir", ".")
                        filename = f"{output_dir}/reaction_coords_{rname}.dat"
                        with open(filename, 'a') as f:
                            f.write(f"{true_step:8d} {window_idx:3d} {nu_c_distance:12.6f} {angle_deg:12.6f} {clg_distance:12.6f} {current_target:10.6f} {current_fc:10.3f}\n")
                    
                    # Debug output if enabled
                    if restraint_debug_enabled and (step % 10 == 0 or step < 5):
                        import math
                        angle_deg = angle * 180.0 / math.pi
                        current_target_angle_deg = current_target_angle * 180.0 / math.pi
                        print(f"# RESTRAINT DEBUG [{rname}] step={step}: angle={angle_deg:.2f}° (target={current_target_angle_deg:.2f}°), angle_fc={angle_fc:.2f}, angle_E={angle_energy:.4f}", end="")
                        if target_distance_list is not None and dist_fc_list is not None:
                            print(f", dist={distance:.3f}Å (target={current_target_distance:.3f}Å), dist_fc={dist_fc:.2f}, dist_E={dist_energy:.4f}", end="")
                        print(f", total_E={total_energy:.4f}")
                    
                    # Store current force constant values for reporting
                    if rm["time_varying"] and rname in rm["restraints"]:
                        rm["restraints"][rname]["current_angle_fc"] = angle_fc
                        if t_dist is not None and dist_fc_list is not None:
                            rm["restraints"][rname]["current_distance_fc"] = dist_fc if distance_time_varying else dist_fc_list[0]
                        
                        # PMF sampling if enabled
                        estimate_pmf_flag = rm["restraints"][rname].get("estimate_pmf", False)
                        has_target_distance_list = target_distance_list is not None
                        
                        
                        if estimate_pmf_flag and interp_mode == "discrete" and has_target_distance_list:
                            # Use global step counter for consistency
                            true_step = current_simulation_step
                            window_idx = min(true_step // upd_steps, len(target_distance_list) - 1)
                            equilibration_steps = int(upd_steps * rm["restraints"][rname]["equilibration_ratio"])
                            
                            # PMF calculation moved to post-processing
                            # Reaction coordinate writing is handled in the energy_calc function
                    
                    return total_energy
                
                # Create time-varying force calculator
                def time_varying_force_calc(coordinates, step, nucleophile=nucleophile, carbon=carbon,
                                          leaving_group=leaving_group, t_angle=target_angle, t_dist=target_distance,
                                          angle_fc_list=angle_force_constants if angle_time_varying else [angle_force_constant],
                                          dist_fc_list=distance_force_constants if distance_time_varying else ([distance_force_constant] if distance_force_constant is not None else None),
                                          target_angle_list=target_angles if target_angle_time_varying else [target_angle],
                                          target_distance_list=target_distances if target_distance_time_varying else ([target_distance] if target_distance is not None else None),
                                          angle_time_var=target_angle_time_varying,
                                          distance_time_var=target_distance_time_varying,
                                          upd_steps=update_steps, interp_mode=interpolation_mode,
                                          rf=restraint_function, rname=restraint_name, rm=restraint_metadata):
                    # Get current force constants based on step
                    angle_fc = time_varying_force_constant(angle_fc_list, upd_steps, step, name=f"{rname}_angle", mode=interp_mode) if angle_time_varying else angle_fc_list[0]
                    dist_fc = None
                    if t_dist is not None and dist_fc_list is not None:
                        dist_fc = time_varying_force_constant(dist_fc_list, upd_steps, step, name=f"{rname}_distance", mode=interp_mode) if distance_time_varying else dist_fc_list[0]
                    
                    current_force_constants[f"{rname}_angle"] = angle_fc
                    if dist_fc is not None:
                        current_force_constants[f"{rname}_distance"] = dist_fc
                    
                    # Get current target angle if time-varying
                    current_target_angle = time_varying_force_constant(target_angle_list, upd_steps, step, name=f"{rname}_target_angle", mode=interp_mode) if angle_time_var else t_angle
                    
                    # Get current target distance if time-varying
                    current_target_distance = None
                    if target_distance_list is not None:
                        current_target_distance = time_varying_force_constant(target_distance_list, upd_steps, step, name=f"{rname}_target_distance", mode=interp_mode) if distance_time_var else target_distance_list[0]
                    
                    # Write reaction coordinate data if requested
                    global restraint_debug_enabled, current_simulation_step, pmf_data
                    true_step = current_simulation_step
                    
                    write_realtime = rm["restraints"][rname].get("write_realtime_pmf", False)
                    
                    # Debug: Print if this is being called
                    if step == 0:
                        print(f"# DEBUG: time_varying_force_calc called for {rname}, write_realtime={write_realtime}")
                        if write_realtime:
                            print(f"#   Will write to reaction_coords_{rname}.dat")
                    
                    if write_realtime and target_distance_list is not None:
                        # Get write frequency from restraint config
                        write_freq = rm["restraints"][rname].get("write_frequency", 1)
                        
                        # Initialize output file on first call
                        if rname not in pmf_data:
                            pmf_data[rname] = {'initialized': True}
                            output_dir = rm["restraints"][rname].get("output_dir", ".")
                            filename = f"{output_dir}/reaction_coords_{rname}.dat"
                            
                            # Auto-detect leaving group if not provided
                            lg = leaving_group if leaving_group is not None else find_leaving_group(coordinates, carbon, nucleophile)
                            
                            with open(filename, 'w') as f:
                                f.write(f"# Reaction coordinate data for restraint: {rname}\n")
                                f.write(f"# Nucleophile: {nucleophile}, Carbon: {carbon}, Leaving group: {lg}\n")
                                f.write(f"# Number of windows: {len(target_distance_list)}\n")
                                f.write(f"# Target distances: {target_distance_list}\n")
                                f.write(f"# Write frequency: every {write_freq} steps\n")
                                f.write(f"# Columns: Step Window_Index Nu-C_Distance(Ang) Nu-C-LG_Angle(deg) C-LG_Distance(Ang) Target_Distance Force_Constant\n")
                                f.write(f"#{'Step':>8s} {'Win':>3s} {'Nu-C_Dist':>12s} {'Angle':>12s} {'C-LG_Dist':>12s} {'Target':>10s} {'Force_K':>10s}\n")
                            print(f"# Reaction coordinate output: {filename} (writing every {write_freq} steps)")
                        
                        # Only write data if current step is divisible by write_frequency
                        if true_step % write_freq == 0:
                            # Calculate all distances and angle
                            from .colvars import colvar_distance, colvar_angle
                            
                            # Auto-detect leaving group if needed
                            lg = leaving_group if leaving_group is not None else find_leaving_group(coordinates, carbon, nucleophile)
                            
                            nu_c_distance = colvar_distance(coordinates, nucleophile, carbon)
                            clg_distance = colvar_distance(coordinates, carbon, lg)
                            angle = colvar_angle(coordinates, nucleophile, carbon, lg)
                            
                            # Get current window index
                            window_idx = min(true_step // upd_steps, len(target_distance_list) - 1)
                            
                            # Get current target distance and force constant
                            current_target = target_distance_list[window_idx] if window_idx < len(target_distance_list) else target_distance_list[-1]
                            current_fc = dist_fc_list[window_idx] if dist_fc_list and window_idx < len(dist_fc_list) else (dist_fc if dist_fc is not None else 0.0)
                            
                            # Convert angle to degrees
                            import math
                            angle_deg = angle * 180.0 / math.pi
                            
                            # Write to file
                            output_dir = rm["restraints"][rname].get("output_dir", ".")
                            filename = f"{output_dir}/reaction_coords_{rname}.dat"
                            with open(filename, 'a') as f:
                                f.write(f"{true_step:8d} {window_idx:3d} {nu_c_distance:12.6f} {angle_deg:12.6f} {clg_distance:12.6f} {current_target:10.6f} {current_fc:10.3f}\n")
                    
                    # Use the backside attack force calculator with the current force constants and target angle
                    return backside_attack_restraint_force(coordinates, nucleophile, carbon, leaving_group, 
                                                         current_target_angle, angle_fc, current_target_distance, dist_fc, rf,
                                                         prevent_inversion=prevent_inversion,
                                                         inversion_penalty_factor=inversion_penalty_factor,
                                                         target_angle_list=target_angle_list if angle_time_var else None,
                                                         angle_update_steps=upd_steps if angle_time_var else None,
                                                         current_step=step if angle_time_var else None)
                
                force_calc = time_varying_force_calc
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, nucleophile=nucleophile, carbon=carbon, 
                               leaving_group=leaving_group, t_angle=target_angle, t_dist=target_distance,
                               angle_fc=angle_force_constant, dist_fc=distance_force_constant, rf=restraint_function,
                               rname=restraint_name):
                    coordinates = ensure_proper_coordinates(coordinates)
                    
                    # Auto-detect leaving group if not provided
                    if leaving_group is None:
                        lg = find_leaving_group(coordinates, carbon, nucleophile)
                    else:
                        lg = leaving_group
                    
                    # Calculate angle energy
                    angle = colvar_angle(coordinates, nucleophile, carbon, lg)
                    angle_energy, _ = rf(angle, t_angle, angle_fc)
                    
                    total_energy = angle_energy
                    
                    # Add distance energy if specified
                    if t_dist is not None and dist_fc is not None:
                        distance = colvar_distance(coordinates, nucleophile, carbon)
                        dist_energy, _ = rf(distance, t_dist, dist_fc)
                        total_energy += dist_energy
                    
                    # Debug output if enabled
                    global restraint_debug_enabled, current_simulation_step
                    if restraint_debug_enabled and (current_simulation_step % 10 == 0 or current_simulation_step < 5):
                        import math
                        angle_deg = angle * 180.0 / math.pi
                        print(f"# RESTRAINT DEBUG [{rname}] step={current_simulation_step}: angle={angle_deg:.2f}° (target={t_angle*180/math.pi:.2f}°), angle_fc={angle_fc:.2f}, angle_E={angle_energy:.4f}", end="")
                        if t_dist is not None and dist_fc is not None:
                            print(f", dist={distance:.3f}Å (target={t_dist:.3f}Å), dist_fc={dist_fc:.2f}, dist_E={dist_energy:.4f}", end="")
                        print(f", total_E={total_energy:.4f}")
                    
                    return total_energy
                
                # Create standard force calculator
                force_calc = partial(
                    backside_attack_restraint_force, 
                    nucleophile=nucleophile, 
                    carbon=carbon, 
                    leaving_group=leaving_group, 
                    target_angle=target_angle,
                    angle_force_constant=angle_force_constant,
                    target_distance=target_distance,
                    distance_force_constant=distance_force_constant,
                    restraint_function=restraint_function,
                    prevent_inversion=prevent_inversion,
                    inversion_penalty_factor=inversion_penalty_factor
                )
            
        elif restraint_type == "spherical_boundary":
            # Spherical boundary restraint
            radius = float(restraint_def.get("radius", 10.0))
            mode = restraint_def.get("mode", "outside")  # "outside" keeps atoms inside, "inside" keeps atoms outside
            
            # Handle PDB reference file if specified
            pdb_file = restraint_def.get("pdb_file")
            if pdb_file and pdb_data is None:
                # Read PDB file if not already provided
                print(f"# Reading PDB file for restraint '{restraint_name}': {pdb_file}")
                pdb_data = read_pdb(pdb_file)
            
            # Handle atom selection
            atom_indices_raw = restraint_def.get("atoms", "all")
            atom_selection = restraint_def.get("atom_selection")
            
            if atom_selection is not None:
                # Use advanced selection criteria
                if pdb_data is None:
                    raise ValueError(f"PDB data required for atom_selection in restraint '{restraint_name}'")
                
                # Parse selection criteria
                coords_for_selection = initial_coordinates if initial_coordinates is not None else pdb_data.get('coordinates')
                selected_indices = parse_atom_selection(atom_selection, pdb_data, coords_for_selection)
                atom_indices = jnp.array(selected_indices)
                print(f"# Restraint '{restraint_name}': Selected {len(selected_indices)} atoms using selection criteria")
                
            elif atom_indices_raw == "all":
                atom_indices = "all"
            elif isinstance(atom_indices_raw, list):
                atom_indices = jnp.array([int(idx) for idx in atom_indices_raw])
            else:
                raise ValueError(f"Invalid atoms specification for spherical_boundary: {atom_indices_raw}")
            
            # Handle center - can be specified, auto-calculated, or default to origin
            center_raw = restraint_def.get("center")
            if center_raw is None or center_raw == "auto":
                # Auto-calculate center from selected atoms
                if initial_coordinates is None:
                    raise ValueError(f"Initial coordinates required for auto-center in restraint '{restraint_name}'")
                
                # Get indices for center calculation
                if isinstance(atom_indices, str) and atom_indices == "all":
                    center_indices = list(range(len(initial_coordinates)))
                else:
                    center_indices = atom_indices.tolist()
                
                # Calculate center of mass
                if symbols is not None:
                    # Use actual atomic masses
                    import numpy as np
                    masses = np.array([ATOMIC_MASSES[PERIODIC_TABLE_REV_IDX.get(sym, 0)] for sym in symbols])
                else:
                    masses = None
                
                center_np = calculate_center_of_mass(jnp.asarray(initial_coordinates), center_indices, masses)
                center = jnp.array(center_np)
                print(f"# Restraint '{restraint_name}': Auto-calculated center at {center}")
            else:
                # Use specified center
                center = jnp.array([float(x) for x in center_raw])
            
            # Handle time-varying radius if specified
            radius_time_varying = False
            radius_list = None
            if isinstance(restraint_def.get("radius"), list):
                radius_list = [float(r) for r in restraint_def["radius"]]
                radius = radius_list[0]
                radius_time_varying = True
                time_varying = True
            
            if time_varying:
                # Auto-calculate update_steps if needed
                if radius_time_varying and "update_steps" not in restraint_def and nsteps is not None:
                    n_windows = len(radius_list)
                    update_steps = int(nsteps / n_windows)
                    print(f"# Auto-calculated update_steps for '{restraint_name}': {update_steps}")
                else:
                    update_steps = int(restraint_def.get("update_steps", 1000))
                
                # Store metadata
                restraint_metadata["time_varying"] = True
                restraint_metadata["restraints"][restraint_name] = {
                    "force_constants": force_constants if isinstance(force_constant_raw, list) else [force_constant],
                    "radii": radius_list if radius_time_varying else [radius],
                    "update_steps": update_steps,
                    "interpolation_mode": interpolation_mode
                }
                
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, atoms=atom_indices, c=center,
                               r_list=radius_list if radius_time_varying else [radius],
                               fc_list=force_constants if isinstance(force_constant_raw, list) else [force_constant],
                               upd_steps=update_steps, rname=restraint_name,
                               interp_mode=interpolation_mode, m=mode,
                               rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    
                    # Get current radius and force constant
                    if radius_time_varying:
                        current_radius = time_varying_force_constant(r_list, upd_steps, step, 
                                                                    name=f"{rname}_radius", mode=interp_mode)
                    else:
                        current_radius = r_list[0]
                    
                    fc = time_varying_force_constant(fc_list, upd_steps, step, 
                                                   name=rname, mode=interp_mode)
                    
                    # Calculate energy using spherical boundary force function
                    energy, _ = spherical_boundary_restraint_force(
                        coordinates, atoms, c, current_radius, fc, rf, m
                    )
                    return energy
                
                # Create time-varying force calculator
                def force_calc(coordinates, step, atoms=atom_indices, c=center,
                             r_list=radius_list if radius_time_varying else [radius],
                             fc_list=force_constants if isinstance(force_constant_raw, list) else [force_constant],
                             upd_steps=update_steps, rname=restraint_name,
                             interp_mode=interpolation_mode, m=mode,
                             rf=restraint_function):
                    # Get current radius and force constant
                    if radius_time_varying:
                        current_radius = time_varying_force_constant(r_list, upd_steps, step,
                                                                    name=f"{rname}_radius", mode=interp_mode)
                    else:
                        current_radius = r_list[0]
                    
                    fc = time_varying_force_constant(fc_list, upd_steps, step,
                                                   name=rname, mode=interp_mode)
                    
                    return spherical_boundary_restraint_force(
                        coordinates, atoms, c, current_radius, fc, rf, m
                    )
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, atoms=atom_indices, c=center, r=radius,
                               fc=force_constant, rf=restraint_function, m=mode):
                    coordinates = ensure_proper_coordinates(coordinates)
                    energy, _ = spherical_boundary_restraint_force(
                        coordinates, atoms, c, r, fc, rf, m
                    )
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    spherical_boundary_restraint_force,
                    atom_indices=atom_indices,
                    center=center,
                    radius=radius,
                    force_constant=force_constant,
                    restraint_function=restraint_function,
                    mode=mode
                )
        
        elif restraint_type == "rmsd":
            # RMSD restraint implementation
            target_rmsd_raw = restraint_def.get("target_rmsd", 1.0)
            mode = restraint_def.get("mode", "harmonic")  # "harmonic" or "flat_bottom"
            
            # Handle reference structure - required parameter
            reference_file = restraint_def.get("reference_file")
            reference_coords_raw = restraint_def.get("reference_coordinates")
            
            if reference_file is not None:
                # Read reference structure from PDB file
                print(f"# Reading reference structure for RMSD restraint '{restraint_name}': {reference_file}")
                ref_pdb_data = read_pdb(reference_file)
                reference_coordinates = jnp.array(ref_pdb_data["coordinates"])
                
                # Store for atom selection if needed
                if pdb_data is None:
                    pdb_data = ref_pdb_data
                    
            elif reference_coords_raw is not None:
                # Use provided coordinates directly
                reference_coordinates = jnp.array(reference_coords_raw)
            elif initial_coordinates is not None:
                # Use initial coordinates as reference
                reference_coordinates = jnp.array(initial_coordinates)
                print(f"# RMSD restraint '{restraint_name}': Using initial coordinates as reference")
            else:
                raise ValueError(f"RMSD restraint '{restraint_name}' requires reference_file, reference_coordinates, or initial_coordinates")
            
            # Handle atom selection - similar to spherical_boundary
            atom_indices_raw = restraint_def.get("atoms", "all")
            atom_selection = restraint_def.get("atom_selection")
            
            if atom_selection is not None:
                # Use advanced selection criteria
                if pdb_data is None:
                    raise ValueError(f"PDB data required for atom_selection in RMSD restraint '{restraint_name}'")
                
                # Parse selection criteria
                coords_for_selection = initial_coordinates if initial_coordinates is not None else reference_coordinates
                selected_indices = parse_atom_selection(atom_selection, pdb_data, coords_for_selection)
                atom_indices = jnp.array(selected_indices)
                print(f"# RMSD restraint '{restraint_name}': Selected {len(selected_indices)} atoms using selection criteria")
                
            elif atom_indices_raw == "all":
                atom_indices = "all"
            elif isinstance(atom_indices_raw, list):
                atom_indices = jnp.array([int(idx) for idx in atom_indices_raw])
            else:
                raise ValueError(f"Invalid atoms specification for RMSD restraint: {atom_indices_raw}")
            
            # Handle time-varying target RMSD and/or force constant
            target_rmsd_time_varying = False
            target_rmsd_list = None
            
            if isinstance(target_rmsd_raw, list):
                target_rmsd_list = [float(r) for r in target_rmsd_raw]
                target_rmsd = target_rmsd_list[0]
                target_rmsd_time_varying = True
                time_varying = True
            else:
                target_rmsd = float(target_rmsd_raw)
            
            if time_varying:
                # Auto-calculate update_steps if needed
                if target_rmsd_time_varying and "update_steps" not in restraint_def and nsteps is not None:
                    n_windows = len(target_rmsd_list)
                    update_steps = int(nsteps / n_windows)
                    print(f"# Auto-calculated update_steps for RMSD restraint '{restraint_name}': {update_steps}")
                else:
                    update_steps = int(restraint_def.get("update_steps", 1000))
                
                # Store metadata
                restraint_metadata["time_varying"] = True
                restraint_metadata["restraints"][restraint_name] = {
                    "force_constants": force_constants if isinstance(force_constant_raw, list) else [force_constant],
                    "target_rmsds": target_rmsd_list if target_rmsd_time_varying else [target_rmsd],
                    "update_steps": update_steps,
                    "interpolation_mode": interpolation_mode
                }
                
                # Create time-varying energy calculator
                def energy_calc(coordinates, step, ref_coords=reference_coordinates, atoms=atom_indices,
                               rmsd_list=target_rmsd_list if target_rmsd_time_varying else [target_rmsd],
                               fc_list=force_constants if isinstance(force_constant_raw, list) else [force_constant],
                               upd_steps=update_steps, rname=restraint_name,
                               interp_mode=interpolation_mode, m=mode,
                               rf=restraint_function):
                    coordinates = ensure_proper_coordinates(coordinates)
                    
                    # Get current target RMSD and force constant
                    if target_rmsd_time_varying:
                        current_target_rmsd = time_varying_force_constant(rmsd_list, upd_steps, step,
                                                                        name=f"{rname}_target_rmsd", mode=interp_mode)
                    else:
                        current_target_rmsd = rmsd_list[0]
                    
                    fc = time_varying_force_constant(fc_list, upd_steps, step,
                                                   name=rname, mode=interp_mode)
                    
                    # Calculate energy using RMSD restraint force function
                    energy, _ = rmsd_restraint_force(
                        coordinates, ref_coords, atoms, current_target_rmsd, fc, rf, m
                    )
                    return energy
                
                # Create time-varying force calculator
                def force_calc(coordinates, step, ref_coords=reference_coordinates, atoms=atom_indices,
                             rmsd_list=target_rmsd_list if target_rmsd_time_varying else [target_rmsd],
                             fc_list=force_constants if isinstance(force_constant_raw, list) else [force_constant],
                             upd_steps=update_steps, rname=restraint_name,
                             interp_mode=interpolation_mode, m=mode,
                             rf=restraint_function):
                    # Get current target RMSD and force constant
                    if target_rmsd_time_varying:
                        current_target_rmsd = time_varying_force_constant(rmsd_list, upd_steps, step,
                                                                        name=f"{rname}_target_rmsd", mode=interp_mode)
                    else:
                        current_target_rmsd = rmsd_list[0]
                    
                    fc = time_varying_force_constant(fc_list, upd_steps, step,
                                                   name=rname, mode=interp_mode)
                    
                    return rmsd_restraint_force(
                        coordinates, ref_coords, atoms, current_target_rmsd, fc, rf, m
                    )
            else:
                # Create standard energy calculator
                def energy_calc(coordinates, ref_coords=reference_coordinates, atoms=atom_indices,
                               target_rmsd=target_rmsd, fc=force_constant, rf=restraint_function, m=mode):
                    coordinates = ensure_proper_coordinates(coordinates)
                    energy, _ = rmsd_restraint_force(
                        coordinates, ref_coords, atoms, target_rmsd, fc, rf, m
                    )
                    return energy
                
                # Create standard force calculator
                force_calc = partial(
                    rmsd_restraint_force,
                    reference_coordinates=reference_coordinates,
                    atom_indices=atom_indices,
                    target_rmsd=target_rmsd,
                    force_constant=force_constant,
                    restraint_function=restraint_function,
                    mode=mode
                )
            
        else:
            raise ValueError(f"Unknown restraint type: {restraint_type}")
        
        # Store the calculators
        restraint_energies[restraint_name] = energy_calc
        restraint_forces.append(force_calc)
    
    return restraint_energies, restraint_forces, restraint_metadata

def apply_restraints(coordinates: jnp.ndarray, restraint_forces: List[Callable], 
                  step: int = 0) -> Tuple[float, jnp.ndarray]:
    """
    Apply all restraints to calculate total restraint energy and forces.
    
    Args:
        coordinates: Atomic coordinates [n_atoms, 3]
        restraint_forces: List of force calculator functions for all restraints
        step: Current simulation step (needed for time-varying restraints)
        
    Returns:
        Tuple containing (total_energy, total_forces)
    """
    # Ensure coordinates are in the right shape
    coordinates = ensure_proper_coordinates(coordinates)
    
    # Use our global step counter instead of the parameter
    # Increment global counter - this guarantees progression regardless of what step is passed in
    global current_simulation_step, restraint_debug_enabled
    current_simulation_step += 1
    true_step = current_simulation_step
    
    # Debug to verify restraints are being called
    if restraint_debug_enabled and true_step <= 2:
        print(f"# DEBUG: apply_restraints called at step {true_step}, processing {len(restraint_forces)} restraints")
    
    
    # Pre-allocate a single forces array to reuse for all restraints
    total_energy = 0.0
    total_forces = jnp.zeros_like(coordinates)
    
    # Calculate and accumulate energy and forces from all restraints
    for i, force_calc in enumerate(restraint_forces):
        # Check if this is a time-varying restraint force calculator
        # We can't check __code__ on partial objects, so we need a safer approach
        try:
            # Try to call with step parameter - this will work for time-varying functions
            # Use our global step counter instead of the parameter
            energy, forces = force_calc(coordinates, step=true_step)
        except TypeError:
            # If it fails, call without step parameter for standard force calculators
            energy, forces = force_calc(coordinates)
            
        total_energy += energy
        total_forces += forces
        
    return total_energy, total_forces