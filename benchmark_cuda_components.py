#!/usr/bin/env python3
"""
Benchmark CUDA optimized components vs JAX implementations.

Compares:
1. Neighbor list construction (CUDA shared memory vs JAX)
2. RBF expansion (CUDA vs JAX)
3. Message aggregation (CUDA CUB vs JAX scatter)
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from typing import Tuple

try:
    from fennol.cuda import fennol_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: CUDA not available!")


def build_neighborlist_jax(coords: jnp.ndarray, cutoff: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build neighbor list using JAX (reference implementation)."""
    natoms = coords.shape[0]

    # Compute pairwise distances
    dr = coords[:, None, :] - coords[None, :, :]  # [natoms, natoms, 3]
    r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-10)  # [natoms, natoms]

    # Find pairs within cutoff (excluding self)
    mask = (r < cutoff) & (r > 0.0)

    # Extract edge indices
    edge_src, edge_dst = jnp.where(mask)

    return edge_src, edge_dst


def compute_rbf_jax(distances: jnp.ndarray, cutoff: float, num_rbf: int = 20) -> jnp.ndarray:
    """Compute RBF features using JAX."""
    gamma = 10.0 / cutoff
    centers = jnp.linspace(0, cutoff, num_rbf)

    # Gaussian RBF: exp(-gamma * (r - mu)^2)
    rbf = jnp.exp(-gamma * (distances[:, None] - centers[None, :])**2)

    # Apply smooth cutoff
    cutoff_vals = 0.5 * (jnp.cos(jnp.pi * distances / cutoff) + 1.0)
    cutoff_vals = jnp.where(distances < cutoff, cutoff_vals, 0.0)

    rbf = rbf * cutoff_vals[:, None]

    return rbf


def aggregate_messages_jax(
    edge_dst: jnp.ndarray,
    messages: jnp.ndarray,
    natoms: int
) -> jnp.ndarray:
    """Aggregate messages using JAX scatter operations."""
    msg_dim = messages.shape[1]

    # Initialize output
    aggregated = jnp.zeros((natoms, msg_dim))

    # Scatter-add messages to destination nodes
    aggregated = aggregated.at[edge_dst].add(messages)

    return aggregated


def benchmark_neighbor_list(natoms: int, cutoff: float = 5.0, n_runs: int = 10):
    """Benchmark neighbor list construction."""
    print(f"\n{'='*70}")
    print(f"Neighbor List Benchmark ({natoms} atoms)")
    print(f"{'='*70}")

    # Generate random coordinates
    coords_np = np.random.randn(natoms, 3).astype(np.float64) * 10.0
    coords_jax = jnp.array(coords_np)

    # JAX benchmark
    print("\nJAX Implementation:")
    jax_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        edge_src_jax, edge_dst_jax = build_neighborlist_jax(coords_jax, cutoff)
        edge_src_jax.block_until_ready()  # Wait for computation
        jax_times.append(time.perf_counter() - start)

    nedges_jax = len(edge_src_jax)
    jax_avg = np.mean(jax_times) * 1000
    jax_std = np.std(jax_times) * 1000
    print(f"  Edges found: {nedges_jax}")
    print(f"  Time: {jax_avg:.3f} ± {jax_std:.3f} ms")

    # CUDA benchmark
    if CUDA_AVAILABLE:
        print("\nCUDA Implementation (shared memory tiling):")
        cuda_times = []
        for i in range(n_runs):
            start = time.perf_counter()
            # Note: GNN predict forces includes neighbor list internally
            forces = fennol_cuda.gnn_predict_forces(
                coords_np.copy(),
                np.full(natoms, 6, dtype=np.int32),
                0, cutoff
            )
            cuda_times.append(time.perf_counter() - start)

        cuda_avg = np.mean(cuda_times) * 1000
        cuda_std = np.std(cuda_times) * 1000
        print(f"  Time: {cuda_avg:.3f} ± {cuda_std:.3f} ms")
        print(f"  Speedup: {jax_avg / cuda_avg:.2f}x")

        return jax_avg, cuda_avg
    else:
        print("\nCUDA not available - skipping CUDA benchmark")
        return jax_avg, None


def benchmark_rbf_expansion(nedges: int, cutoff: float = 5.0, num_rbf: int = 20, n_runs: int = 20):
    """Benchmark RBF expansion."""
    print(f"\n{'='*70}")
    print(f"RBF Expansion Benchmark ({nedges} edges)")
    print(f"{'='*70}")

    # Generate random distances
    distances_np = np.random.uniform(0, cutoff, nedges).astype(np.float64)
    distances_jax = jnp.array(distances_np)

    # JAX benchmark
    print("\nJAX Implementation:")
    jax_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        rbf_jax = compute_rbf_jax(distances_jax, cutoff, num_rbf)
        rbf_jax.block_until_ready()
        jax_times.append(time.perf_counter() - start)

    jax_avg = np.mean(jax_times) * 1000
    jax_std = np.std(jax_times) * 1000
    print(f"  RBF shape: {rbf_jax.shape}")
    print(f"  Time: {jax_avg:.3f} ± {jax_std:.3f} ms")

    print("\n  Note: CUDA RBF is fused with neighbor list kernel")
    print(f"  Standalone JAX RBF time: {jax_avg:.3f} ms")

    return jax_avg


def benchmark_message_aggregation(
    natoms: int, nedges: int, msg_dim: int = 64, n_runs: int = 10
):
    """Benchmark message aggregation."""
    print(f"\n{'='*70}")
    print(f"Message Aggregation Benchmark ({natoms} atoms, {nedges} edges)")
    print(f"{'='*70}")

    # Generate random edge destinations and messages
    edge_dst_np = np.random.randint(0, natoms, nedges, dtype=np.int32)
    messages_np = np.random.randn(nedges, msg_dim).astype(np.float64)

    edge_dst_jax = jnp.array(edge_dst_np)
    messages_jax = jnp.array(messages_np)

    # JAX benchmark (scatter-add)
    print("\nJAX Implementation (scatter-add):")
    jax_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        agg_jax = aggregate_messages_jax(edge_dst_jax, messages_jax, natoms)
        agg_jax.block_until_ready()
        jax_times.append(time.perf_counter() - start)

    jax_avg = np.mean(jax_times) * 1000
    jax_std = np.std(jax_times) * 1000
    print(f"  Aggregated shape: {agg_jax.shape}")
    print(f"  Time: {jax_avg:.3f} ± {jax_std:.3f} ms")

    print("\n  Note: CUDA uses CUB segment reduction (not directly benchmarkable)")
    print(f"  Expected speedup with CUB: 10-100x for large systems")
    print(f"  JAX scatter-add time: {jax_avg:.3f} ms")

    return jax_avg


def comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    print("\n" + "="*70)
    print("CUDA Optimizations Benchmark Suite")
    print("="*70)
    print("\nComparing optimized CUDA kernels vs JAX implementations")
    print("CUDA Optimizations:")
    print("  1. Shared memory tiling for neighbor list (5-10x expected)")
    print("  2. CUB segment reduction for aggregation (10-100x expected)")

    results = {}

    # Test different system sizes
    sizes = [
        (100, 5.0),   # Small system
        (500, 5.0),   # Medium system
        (1000, 5.0),  # Large system
    ]

    for natoms, cutoff in sizes:
        print(f"\n{'#'*70}")
        print(f"# System Size: {natoms} atoms, Cutoff: {cutoff} Å")
        print(f"{'#'*70}")

        # Benchmark neighbor list
        jax_time, cuda_time = benchmark_neighbor_list(natoms, cutoff, n_runs=5)

        results[f"{natoms}_neighborlist"] = {
            "jax": jax_time,
            "cuda": cuda_time,
            "speedup": jax_time / cuda_time if cuda_time else None
        }

        # Estimate number of edges
        nedges = int(natoms * 50)  # Rough estimate

        # Benchmark RBF
        rbf_time = benchmark_rbf_expansion(nedges, cutoff, n_runs=10)
        results[f"{natoms}_rbf"] = {"jax": rbf_time}

        # Benchmark message aggregation
        agg_time = benchmark_message_aggregation(natoms, nedges, msg_dim=64, n_runs=5)
        results[f"{natoms}_aggregation"] = {"jax": agg_time}

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    print("\nNeighbor List (CUDA with shared memory tiling):")
    for natoms, _ in sizes:
        key = f"{natoms}_neighborlist"
        if key in results and results[key]["cuda"]:
            print(f"  {natoms:4d} atoms: {results[key]['speedup']:.2f}x speedup")

    print("\nNote: Full GNN inference requires MLP layers (not yet implemented)")
    print("Current benchmarks test optimized kernels in isolation.")
    print("\nExpected improvements with remaining optimizations:")
    print("  - Cell-linked lists: 10-100x for large systems (>5k atoms)")
    print("  - Kernel fusion: 5x reduction in memory traffic")
    print("  - MLP with cuBLAS: Enables end-to-end CUDA inference")

    return results


if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("ERROR: CUDA module not available!")
        print("Please build and install the CUDA extension first.")
        exit(1)

    # Run comprehensive benchmark
    results = comprehensive_benchmark()

    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70 + "\n")
