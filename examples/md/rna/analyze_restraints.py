#!/usr/bin/env python
"""
Script to analyze the results of a restraint test simulation.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_colvars_file(filename):
    """Read a colvars file and return the data."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split()
    
    # Parse data
    data = []
    for line in lines[1:]:
        if line.strip():
            data.append([float(x) for x in line.strip().split()])
    
    # Convert to numpy array
    data = np.array(data)
    
    # Create a dict with columns
    result = {}
    for i, col in enumerate(header):
        if i < data.shape[1]:
            result[col] = data[:, i]
    
    return result

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <colvars_file>")
        sys.exit(1)
    
    colvars_file = sys.argv[1]
    data = read_colvars_file(colvars_file)
    
    # Print the column names
    print(f"Columns in the file: {list(data.keys())}")
    
    # Create time array if available
    if "time[ps]" in data:
        time = data["time[ps]"]
    else:
        time = np.arange(len(next(iter(data.values()))))
    
    # Plot all data columns except time
    plt.figure(figsize=(12, 8))
    
    for name, values in data.items():
        if name != "time[ps]":
            plt.plot(time, values, label=name)
    
    plt.xlabel('Time (ps)')
    plt.ylabel('Value')
    plt.title('Colvar Values vs Time')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_file = Path(colvars_file).with_suffix('.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Calculate statistics
    print("\nStatistics:")
    for name, values in data.items():
        if name != "time[ps]":
            print(f"{name}:")
            print(f"  Mean: {np.mean(values):.6f}")
            print(f"  Std:  {np.std(values):.6f}")
            print(f"  Min:  {np.min(values):.6f}")
            print(f"  Max:  {np.max(values):.6f}")
            print(f"  Range: {np.max(values) - np.min(values):.6f}")
            print()

if __name__ == "__main__":
    main()