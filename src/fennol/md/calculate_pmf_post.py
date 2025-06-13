#!/usr/bin/env python3
"""
Post-processing script to calculate PMF from reaction coordinate data using WHAM.
"""

import numpy as np
import argparse
from typing import List, Tuple, Dict
import sys

def parse_reaction_coords(filename: str) -> Dict:
    """Parse reaction coordinate file and return data organized by windows."""
    window_data = {}
    target_distances = []
    force_constants = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('# Target distances:'):
                # Parse target distances from header
                dist_str = line.split(':')[1].strip()[1:-1]  # Remove brackets
                target_distances = [float(x) for x in dist_str.split(',')]
                continue
                
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 7:
                step = int(parts[0])
                window_idx = int(parts[1])
                nu_c_dist = float(parts[2])
                angle = float(parts[3])
                c_lg_dist = float(parts[4])
                target = float(parts[5])
                force_k = float(parts[6])
                
                if window_idx not in window_data:
                    window_data[window_idx] = {
                        'distances': [],
                        'angles': [],
                        'c_lg_distances': [],
                        'target': target,
                        'force_k': force_k
                    }
                
                window_data[window_idx]['distances'].append(nu_c_dist)
                window_data[window_idx]['angles'].append(angle)
                window_data[window_idx]['c_lg_distances'].append(c_lg_dist)
    
    return window_data, target_distances


def calculate_pmf_wham(window_data: Dict, temperature: float = 300.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate PMF using WHAM (Weighted Histogram Analysis Method).
    Returns (window_centers, pmf_values, uncertainties) in kcal/mol.
    """
    kB_T = 0.001987 * temperature  # kcal/mol
    
    # Collect data for WHAM
    valid_windows = []
    all_samples = []
    sample_to_window = []
    
    for window_idx in sorted(window_data.keys()):
        data = window_data[window_idx]
        if len(data['distances']) > 0:
            valid_windows.append({
                'index': window_idx,
                'center': data['target'],
                'k': data['force_k'],
                'samples': np.array(data['distances'])
            })
            all_samples.extend(data['distances'])
            sample_to_window.extend([len(valid_windows)-1] * len(data['distances']))
    
    if len(valid_windows) == 0:
        return [], [], []
    
    all_samples = np.array(all_samples)
    sample_to_window = np.array(sample_to_window)
    n_windows = len(valid_windows)
    n_samples = len(all_samples)
    
    # Calculate bias energies for all samples in all windows
    # U_i(x) = 0.5 * k_i * (x - x_i)^2
    bias_energies = np.zeros((n_samples, n_windows))
    for j, win_data in enumerate(valid_windows):
        bias_energies[:, j] = 0.5 * win_data['k'] * (all_samples - win_data['center'])**2
    
    # Initialize free energies
    f_i = np.zeros(n_windows)
    
    # WHAM iteration
    max_iter = 1000
    tolerance = 1e-6
    
    print(f"Running WHAM with {n_windows} windows and {n_samples} total samples...")
    
    for iteration in range(max_iter):
        f_i_old = f_i.copy()
        
        # Calculate denominators
        denominators = np.zeros(n_samples)
        for x_idx in range(n_samples):
            denom = 0.0
            for j in range(n_windows):
                N_j = len(valid_windows[j]['samples'])
                denom += N_j * np.exp(-bias_energies[x_idx, j] / kB_T + f_i[j] / kB_T)
            denominators[x_idx] = denom
        
        # Update free energies
        for i in range(n_windows):
            window_mask = sample_to_window == i
            window_samples_idx = np.where(window_mask)[0]
            
            if len(window_samples_idx) > 0:
                sum_val = 0.0
                for idx in window_samples_idx:
                    sum_val += 1.0 / denominators[idx] * np.exp(-bias_energies[idx, i] / kB_T)
                f_i[i] = -kB_T * np.log(sum_val)
        
        # Center free energies
        f_i -= np.min(f_i)
        
        # Check convergence
        max_diff = np.max(np.abs(f_i - f_i_old))
        if max_diff < tolerance:
            print(f"WHAM converged after {iteration} iterations")
            break
    
    # Calculate uncertainties using bootstrap
    n_bootstrap = 100
    bootstrap_pmfs = []
    
    print(f"Calculating uncertainties with {n_bootstrap} bootstrap samples...")
    
    for b in range(n_bootstrap):
        # Resample data
        boot_valid_windows = []
        boot_all_samples = []
        boot_sample_to_window = []
        
        for i, win_data in enumerate(valid_windows):
            n_win_samples = len(win_data['samples'])
            resample_idx = np.random.choice(n_win_samples, n_win_samples, replace=True)
            resampled = win_data['samples'][resample_idx]
            
            boot_valid_windows.append({
                'samples': resampled,
                'k': win_data['k'],
                'center': win_data['center']
            })
            boot_all_samples.extend(resampled)
            boot_sample_to_window.extend([i] * n_win_samples)
        
        boot_all_samples = np.array(boot_all_samples)
        boot_sample_to_window = np.array(boot_sample_to_window)
        
        # Calculate bias energies for bootstrap
        boot_bias_energies = np.zeros((len(boot_all_samples), n_windows))
        for j, win_data in enumerate(boot_valid_windows):
            boot_bias_energies[:, j] = 0.5 * win_data['k'] * (boot_all_samples - win_data['center'])**2
        
        # Run WHAM on bootstrap
        boot_f_i = np.zeros(n_windows)
        for _ in range(100):  # Fewer iterations for bootstrap
            boot_f_i_old = boot_f_i.copy()
            
            # Calculate denominators
            boot_denominators = np.zeros(len(boot_all_samples))
            for x_idx in range(len(boot_all_samples)):
                denom = 0.0
                for j in range(n_windows):
                    N_j = len(boot_valid_windows[j]['samples'])
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
    uncertainties = np.std(bootstrap_pmfs, axis=0)
    
    # Get window centers and sort
    window_centers = [d['center'] for d in valid_windows]
    sorted_indices = np.argsort(window_centers)
    
    window_centers = [window_centers[i] for i in sorted_indices]
    pmf_values = [f_i[i] for i in sorted_indices]
    uncertainties = [uncertainties[i] for i in sorted_indices]
    
    return window_centers, pmf_values, uncertainties


def main():
    parser = argparse.ArgumentParser(description='Calculate PMF from reaction coordinate data')
    parser.add_argument('input', help='Reaction coordinate file (reaction_coords_*.dat)')
    parser.add_argument('-T', '--temperature', type=float, default=300.0,
                        help='Temperature in Kelvin (default: 300.0)')
    parser.add_argument('-o', '--output', default='pmf_calculated.dat',
                        help='Output PMF file (default: pmf_calculated.dat)')
    
    args = parser.parse_args()
    
    print(f"Reading reaction coordinate data from {args.input}...")
    window_data, target_distances = parse_reaction_coords(args.input)
    
    print(f"Found {len(window_data)} windows with data")
    for win_idx in sorted(window_data.keys()):
        data = window_data[win_idx]
        print(f"  Window {win_idx}: {len(data['distances'])} samples, "
              f"target={data['target']:.3f}, k={data['force_k']:.3f}")
    
    # Calculate PMF
    window_centers, pmf_values, uncertainties = calculate_pmf_wham(window_data, args.temperature)
    
    # Write output
    with open(args.output, 'w') as f:
        f.write(f"# PMF calculated from {args.input}\n")
        f.write(f"# Temperature: {args.temperature} K\n")
        f.write(f"# Number of windows: {len(window_centers)}\n")
        f.write("#\n")
        f.write("# Window_Center(Ang)  PMF(kcal/mol)  Uncertainty(kcal/mol)\n")
        
        for center, pmf, unc in zip(window_centers, pmf_values, uncertainties):
            f.write(f"{center:18.6f} {pmf:15.6f} {unc:20.6f}\n")
    
    print(f"\nPMF results written to {args.output}")
    
    # Print summary
    print("\nPMF Summary:")
    print("Window   Target    PMF(kcal/mol)  Uncertainty")
    print("------  --------  -------------  -----------")
    for i, (center, pmf, unc) in enumerate(zip(window_centers, pmf_values, uncertainties)):
        print(f"{i+1:6d}  {center:8.3f}  {pmf:13.3f}  {unc:11.3f}")
    
    # Find barrier
    if len(pmf_values) > 1:
        barrier = max(pmf_values) - min(pmf_values)
        max_idx = pmf_values.index(max(pmf_values))
        print(f"\nBarrier height: {barrier:.2f} kcal/mol at {window_centers[max_idx]:.3f} Ang")


if __name__ == '__main__':
    main()