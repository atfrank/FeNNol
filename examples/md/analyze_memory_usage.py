"""
Memory usage analysis for restraint simulation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import re
import psutil
import time
import subprocess
from pathlib import Path

def run_with_memory_tracking(command, output_file="memory_usage.txt"):
    """Run a command and track memory usage over time."""
    # Start the command as a subprocess
    process = subprocess.Popen(command, shell=True)
    
    # Get the process ID
    pid = process.pid
    
    # Open file to save memory usage data
    with open(output_file, 'w') as f:
        f.write("# Time(s) Memory(MB)\n")
        
        # Track memory usage until process finishes
        start_time = time.time()
        while process.poll() is None:  # While process is still running
            try:
                # Get memory info
                p = psutil.Process(pid)
                memory_mb = p.memory_info().rss / 1024 / 1024
                
                # Calculate elapsed time
                current_time = time.time() - start_time
                
                # Write data
                f.write(f"{current_time:.2f} {memory_mb:.2f}\n")
                f.flush()  # Ensure data is written to file
                
                # Wait before next reading
                time.sleep(0.5)
                
            except psutil.NoSuchProcess:
                # Process might have ended between poll and trying to get memory
                break
    
    # Wait for process to complete if not already done
    process.wait()
    
    return process.returncode

def plot_memory_usage(memory_file="memory_usage.txt", output_file="memory_usage.png"):
    """Plot memory usage from the data file."""
    # Load data
    data = np.loadtxt(memory_file)
    time_vals = data[:, 0]
    memory_vals = data[:, 1]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_vals, memory_vals)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage During Restraint Simulation")
    plt.grid(True)
    
    # Add some statistics
    max_memory = np.max(memory_vals)
    avg_memory = np.mean(memory_vals)
    plt.axhline(y=max_memory, color='r', linestyle='--', label=f"Max: {max_memory:.1f} MB")
    plt.axhline(y=avg_memory, color='g', linestyle='--', label=f"Avg: {avg_memory:.1f} MB")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return max_memory, avg_memory

def main():
    """Run simulation with memory tracking and plot results."""
    # Define the command to run
    command = "python -m fennol.md.dynamic examples/md/test_restraint_memory.fnl"
    
    print(f"Running command: {command}")
    print("Tracking memory usage...")
    
    # Run with memory tracking
    return_code = run_with_memory_tracking(command)
    
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
        return
    
    # Plot the memory usage
    max_memory, avg_memory = plot_memory_usage()
    
    print(f"Simulation completed. Memory usage:")
    print(f"  Maximum: {max_memory:.1f} MB")
    print(f"  Average: {avg_memory:.1f} MB")
    print(f"Plot saved to memory_usage.png")

if __name__ == "__main__":
    main()