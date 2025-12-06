#!/usr/bin/env python3
"""
CUDA Performance Benchmark Script for FeNNol MD

This script benchmarks the performance of CUDA-accelerated molecular dynamics
compared to CPU-only execution on various system sizes.
"""

import subprocess
import time
import os
import sys
import argparse
from pathlib import Path
import re

# System configurations to benchmark
SYSTEMS = {
    'watertiny': {'atoms': 81, 'desc': '27 water molecules'},
    'watersmall': {'atoms': 648, 'desc': '216 water molecules'},
    'waterbox': {'atoms': 1500, 'desc': '500 water molecules'},
    'waterbig': {'atoms': 4800, 'desc': '1600 water molecules'},
    'waterhuge': {'atoms': 12000, 'desc': '4000 water molecules'},
}


def create_benchmark_input(system_name, device, nsteps=1000):
    """Create a benchmark input file for the specified system and device."""
    examples_dir = Path(__file__).parent / 'examples' / 'md'
    system_dir = examples_dir / system_name

    if not system_dir.exists():
        raise ValueError(f"System directory not found: {system_dir}")

    # Read original input
    original_input = system_dir / 'input.fnl'
    if not original_input.exists():
        raise ValueError(f"Input file not found: {original_input}")

    with open(original_input, 'r') as f:
        lines = f.readlines()

    # Modify for benchmark
    modified_lines = []
    for line in lines:
        # Set device
        if line.strip().startswith('device'):
            modified_lines.append(f'device {device}\n')
        # Set nsteps for benchmark
        elif line.strip().startswith('nsteps'):
            modified_lines.append(f'nsteps = {nsteps}\n')
        # Enable timing output
        elif line.strip().startswith('print_timings'):
            modified_lines.append('print_timings yes\n')
        # Disable trajectory output for benchmark
        elif line.strip().startswith('tdump'):
            modified_lines.append(f'#tdump[ps] = 1000.  # Disabled for benchmark\n')
        else:
            modified_lines.append(line)

    # Write benchmark input
    benchmark_dir = Path('benchmark_runs') / system_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Copy model file
    model_file = examples_dir / 'ani2x.fnx'
    if model_file.exists():
        import shutil
        shutil.copy(model_file, benchmark_dir / 'ani2x.fnx')
        # Fix model path in input
        for i, line in enumerate(modified_lines):
            if line.strip().startswith('model_file'):
                modified_lines[i] = 'model_file ani2x.fnx\n'

    benchmark_input = benchmark_dir / f'input_{device.replace(":", "_")}.fnl'
    with open(benchmark_input, 'w') as f:
        f.writelines(modified_lines)

    # Copy xyz file if needed
    xyz_file = system_dir / f'{system_name}.xyz'
    if xyz_file.exists():
        import shutil
        shutil.copy(xyz_file, benchmark_dir / f'{system_name}.xyz')

    return benchmark_input


def run_benchmark(input_file, timeout=600):
    """Run a single benchmark and extract timing information."""
    start_time = time.time()

    try:
        # Run fennol_md
        result = subprocess.run(
            ['fennol_md', str(input_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=input_file.parent
        )

        elapsed_time = time.time() - start_time

        # Extract performance metrics from output
        output = result.stdout + result.stderr

        # Look for timing information
        timings = {}
        for line in output.split('\n'):
            # Extract average time per step
            if 'Average time per step' in line or 'ms/step' in line:
                match = re.search(r'([\d.]+)\s*ms', line)
                if match:
                    timings['ms_per_step'] = float(match.group(1))

            # Extract total simulation time
            if 'Total simulation time' in line or 'Simulation completed' in line:
                match = re.search(r'([\d.]+)\s*s', line)
                if match:
                    timings['total_sim_time'] = float(match.group(1))

        # Calculate steps per second
        if 'ms_per_step' in timings:
            timings['steps_per_sec'] = 1000.0 / timings['ms_per_step']

        return {
            'success': result.returncode == 0,
            'wall_time': elapsed_time,
            'timings': timings,
            'output': output[:2000],  # First 2000 chars for debugging
            'returncode': result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'wall_time': timeout,
            'timings': {},
            'output': 'TIMEOUT',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'wall_time': time.time() - start_time,
            'timings': {},
            'output': str(e),
            'returncode': -2
        }


def print_results_table(results):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*100)
    print("CUDA Performance Benchmark Results")
    print("="*100)
    print(f"{'System':<15} {'Atoms':>7} {'Device':>10} {'Wall Time':>12} {'ms/step':>10} {'Steps/s':>10} {'Speedup':>10}")
    print("-"*100)

    speedups = {}
    for system_name in results:
        if system_name not in SYSTEMS:
            continue

        atoms = SYSTEMS[system_name]['atoms']

        # Get CPU and CUDA results
        cpu_result = results[system_name].get('cpu')
        cuda_result = results[system_name].get('cuda:0')

        if cpu_result and cpu_result['success']:
            wall_time = f"{cpu_result['wall_time']:.2f}s"
            ms_per_step = cpu_result['timings'].get('ms_per_step', 'N/A')
            steps_per_sec = cpu_result['timings'].get('steps_per_sec', 'N/A')

            ms_str = f"{ms_per_step:.3f}" if isinstance(ms_per_step, float) else ms_per_step
            sps_str = f"{steps_per_sec:.2f}" if isinstance(steps_per_sec, float) else steps_per_sec

            print(f"{system_name:<15} {atoms:>7} {'CPU':>10} {wall_time:>12} {ms_str:>10} {sps_str:>10} {'-':>10}")

        if cuda_result and cuda_result['success']:
            wall_time = f"{cuda_result['wall_time']:.2f}s"
            ms_per_step = cuda_result['timings'].get('ms_per_step', 'N/A')
            steps_per_sec = cuda_result['timings'].get('steps_per_sec', 'N/A')

            ms_str = f"{ms_per_step:.3f}" if isinstance(ms_per_step, float) else ms_per_step
            sps_str = f"{steps_per_sec:.2f}" if isinstance(steps_per_sec, float) else steps_per_sec

            # Calculate speedup
            speedup = 'N/A'
            if cpu_result and cpu_result['success'] and isinstance(ms_per_step, float):
                cpu_ms = cpu_result['timings'].get('ms_per_step')
                if isinstance(cpu_ms, float):
                    speedup = f"{cpu_ms / ms_per_step:.2f}x"
                    speedups[system_name] = cpu_ms / ms_per_step

            print(f"{system_name:<15} {atoms:>7} {'CUDA':>10} {wall_time:>12} {ms_str:>10} {sps_str:>10} {speedup:>10}")

        if system_name in results:
            print()  # Blank line between systems

    print("="*100)

    # Print summary
    if speedups:
        avg_speedup = sum(speedups.values()) / len(speedups)
        print(f"\nAverage CUDA Speedup: {avg_speedup:.2f}x")
        print(f"Best Speedup: {max(speedups.values()):.2f}x ({max(speedups, key=speedups.get)})")
        print(f"Worst Speedup: {min(speedups.values()):.2f}x ({min(speedups, key=speedups.get)})")


def main():
    parser = argparse.ArgumentParser(description='Benchmark FeNNol CUDA performance')
    parser.add_argument('--systems', nargs='+', choices=list(SYSTEMS.keys()) + ['all'],
                       default=['watersmall', 'waterbox'],
                       help='Systems to benchmark (default: watersmall waterbox)')
    parser.add_argument('--nsteps', type=int, default=1000,
                       help='Number of MD steps (default: 1000)')
    parser.add_argument('--devices', nargs='+', default=['cpu', 'cuda:0'],
                       help='Devices to test (default: cpu cuda:0)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per benchmark in seconds (default: 600)')

    args = parser.parse_args()

    # Expand 'all' to all systems
    if 'all' in args.systems:
        systems_to_test = list(SYSTEMS.keys())
    else:
        systems_to_test = args.systems

    print(f"Benchmarking FeNNol CUDA Performance")
    print(f"Systems: {', '.join(systems_to_test)}")
    print(f"Devices: {', '.join(args.devices)}")
    print(f"Steps: {args.nsteps}")
    print(f"Timeout: {args.timeout}s per run")
    print()

    results = {}

    for system_name in systems_to_test:
        print(f"\n{'='*80}")
        print(f"Benchmarking: {system_name} ({SYSTEMS[system_name]['desc']})")
        print(f"{'='*80}")

        results[system_name] = {}

        for device in args.devices:
            print(f"\nRunning on {device}...")

            # Create benchmark input
            try:
                input_file = create_benchmark_input(system_name, device, args.nsteps)
                print(f"Created input: {input_file}")

                # Run benchmark
                result = run_benchmark(input_file, timeout=args.timeout)
                results[system_name][device] = result

                if result['success']:
                    print(f"✓ Completed in {result['wall_time']:.2f}s")
                    if 'ms_per_step' in result['timings']:
                        print(f"  {result['timings']['ms_per_step']:.3f} ms/step")
                    if 'steps_per_sec' in result['timings']:
                        print(f"  {result['timings']['steps_per_sec']:.2f} steps/s")
                else:
                    print(f"✗ Failed (return code: {result['returncode']})")
                    print(f"  Output: {result['output'][:500]}")

            except Exception as e:
                print(f"✗ Error: {e}")
                results[system_name][device] = {
                    'success': False,
                    'wall_time': 0,
                    'timings': {},
                    'output': str(e),
                    'returncode': -3
                }

    # Print results table
    print_results_table(results)

    # Save results to file
    import json
    results_file = Path('benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
