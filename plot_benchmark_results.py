#!/usr/bin/env python
"""
Plot benchmark results for JAX vs CUDA implicit solvent
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open('benchmark_implicit_solvent_results.json') as f:
    results = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Implicit Solvent Benchmark: JAX vs CUDA', fontsize=16, fontweight='bold')

# 1. Performance comparison (time)
ax = axes[0, 0]
systems = list(results.keys())
jax_times = [results[s]['performance']['jax']['mean'] * 1000 for s in systems]
cuda_times = [results[s]['performance']['cuda']['mean'] * 1000 for s in systems]
natoms = [results[s]['performance']['natoms'] for s in systems]

x = np.arange(len(systems))
width = 0.35

bars1 = ax.bar(x - width/2, jax_times, width, label='JAX', color='#2E86AB')
bars2 = ax.bar(x + width/2, cuda_times, width, label='CUDA', color='#A23B72')

ax.set_ylabel('Time per Evaluation (ms)', fontweight='bold')
ax.set_title('Performance Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{s.upper()}\n({natoms[i]} atoms)" for i, s in enumerate(systems)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 2. Speedup
ax = axes[0, 1]
speedups = [results[s]['performance']['speedup'] for s in systems]
colors = ['green' if s > 1 else 'red' for s in speedups]

bars = ax.bar(systems, speedups, color=colors, alpha=0.7)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (1.0x)')
ax.set_ylabel('Speedup (CUDA vs JAX)', fontweight='bold')
ax.set_title('CUDA Speedup Factor', fontweight='bold')
ax.set_xticks(range(len(systems)))
ax.set_xticklabels([s.upper() for s in systems])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, speedup) in enumerate(zip(bars, speedups)):
    label = f'{speedup:.2f}x'
    color = 'green' if speedup > 1 else 'red'
    ax.text(i, speedup + 0.1, label, ha='center', va='bottom',
            fontweight='bold', color=color, fontsize=10)

# 3. Throughput comparison
ax = axes[0, 2]
jax_throughput = [results[s]['performance']['natoms'] / results[s]['performance']['jax']['mean']
                  for s in systems]
cuda_throughput = [results[s]['performance']['natoms'] / results[s]['performance']['cuda']['mean']
                   for s in systems]

bars1 = ax.bar(x - width/2, jax_throughput, width, label='JAX', color='#2E86AB')
bars2 = ax.bar(x + width/2, cuda_throughput, width, label='CUDA', color='#A23B72')

ax.set_ylabel('Throughput (atoms/second)', fontweight='bold')
ax.set_title('Throughput Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.upper() for s in systems])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. Energy accuracy
ax = axes[1, 0]
energy_jax = [results[s]['accuracy']['energy_jax'] for s in systems]
energy_cuda = [results[s]['accuracy']['energy_cuda'] for s in systems]

bars1 = ax.bar(x - width/2, energy_jax, width, label='JAX', color='#2E86AB')
bars2 = ax.bar(x + width/2, energy_cuda, width, label='CUDA', color='#A23B72')

ax.set_ylabel('Solvation Energy (kcal/mol)', fontweight='bold')
ax.set_title('Energy Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s.upper() for s in systems])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 5. Energy difference
ax = axes[1, 1]
energy_rel_diff = [results[s]['accuracy']['energy_rel_diff'] * 100 for s in systems]

bars = ax.bar(systems, energy_rel_diff, color=['orange', 'green'], alpha=0.7)
ax.set_ylabel('Relative Energy Difference (%)', fontweight='bold')
ax.set_title('Energy Accuracy (JAX vs CUDA)', fontweight='bold')
ax.set_xticks(range(len(systems)))
ax.set_xticklabels([s.upper() for s in systems])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, diff) in enumerate(zip(bars, energy_rel_diff)):
    ax.text(i, diff + 0.5, f'{diff:.1f}%', ha='center', va='bottom', fontsize=10)

# 6. MD dynamics comparison (water only)
ax = axes[1, 2]
if 'dynamics' in results.get('water', {}):
    jax_dyn = results['water']['dynamics']['jax']
    cuda_dyn = results['water']['dynamics']['cuda']

    steps = list(range(len(jax_dyn['energies'])))
    ax.plot(steps, jax_dyn['energies'], label='JAX', color='#2E86AB', linewidth=2)
    ax.plot(steps, cuda_dyn['energies'], label='CUDA', color='#A23B72', linewidth=2)

    ax.set_xlabel('MD Step', fontweight='bold')
    ax.set_ylabel('Total Energy (kcal/mol)', fontweight='bold')
    ax.set_title('MD Energy Conservation (Water)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add drift annotations
    jax_drift = jax_dyn['energy_drift']
    cuda_drift = cuda_dyn['energy_drift']
    ax.text(0.05, 0.95, f'JAX drift: {jax_drift:.1f} kcal/mol\nCUDA drift: {cuda_drift:.1f} kcal/mol',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
else:
    ax.text(0.5, 0.5, 'No dynamics data available',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('MD Energy Conservation', fontweight='bold')

plt.tight_layout()

# Save figure
output_file = Path('implicit_solvent_benchmark.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved benchmark plot to: {output_file}")

plt.show()
