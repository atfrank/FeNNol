#!/usr/bin/env python3
"""
Test cuBLAS-based MLP layers for GNN.

Tests:
1. Dense layer with SiLU activation
2. Dense layer with ReLU activation
3. Dense layer with no activation
4. Correctness vs NumPy reference
5. Performance benchmark
"""

import numpy as np
import time

try:
    from fennol.cuda import fennol_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("ERROR: CUDA not available!")
    exit(1)


def silu_numpy(x):
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x / (1 + np.exp(-x))


def relu_numpy(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)


def dense_layer_numpy(x, w, b, activation='silu'):
    """Reference dense layer implementation."""
    y = x @ w + b  # Broadcasting bias

    if activation == 'silu':
        y = silu_numpy(y)
    elif activation == 'relu':
        y = relu_numpy(y)
    elif activation == 'none':
        pass  # No activation
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return y


def test_dense_layer_correctness():
    """Test dense layer correctness against NumPy reference."""
    print("\n" + "="*70)
    print("Dense Layer Correctness Test")
    print("="*70)

    # Test parameters
    batch = 100
    in_dim = 64
    out_dim = 128

    # Random inputs
    np.random.seed(42)
    x = np.random.randn(batch, in_dim).astype(np.float64)
    w = np.random.randn(in_dim, out_dim).astype(np.float64)
    b = np.random.randn(out_dim).astype(np.float64)

    # Initialize cuBLAS
    fennol_cuda.init_cublas()

    # Test each activation function
    for activation in ['silu', 'relu', 'none']:
        print(f"\nTesting {activation} activation:")

        # NumPy reference
        y_ref = dense_layer_numpy(x, w, b, activation)

        # CUDA implementation
        y_cuda = fennol_cuda.dense_layer(x, w, b, activation)

        # Check shapes
        assert y_cuda.shape == y_ref.shape, f"Shape mismatch: {y_cuda.shape} vs {y_ref.shape}"

        # Check values
        max_error = np.abs(y_cuda - y_ref).max()
        mean_error = np.abs(y_cuda - y_ref).mean()
        rel_error = np.abs((y_cuda - y_ref) / (np.abs(y_ref) + 1e-8)).mean()

        print(f"  Shape: {y_cuda.shape}")
        print(f"  Max error: {max_error:.2e}")
        print(f"  Mean error: {mean_error:.2e}")
        print(f"  Relative error: {rel_error:.2e}")

        # Tolerance for numerical differences (cuBLAS vs NumPy)
        assert max_error < 1e-8, f"Max error too large: {max_error}"
        assert mean_error < 1e-10, f"Mean error too large: {mean_error}"

        print(f"  ✅ {activation} activation passed!")

    fennol_cuda.cleanup_cublas()
    print("\n✅ All correctness tests passed!")


def test_dense_layer_performance():
    """Benchmark dense layer performance."""
    print("\n" + "="*70)
    print("Dense Layer Performance Benchmark")
    print("="*70)

    # Test configurations
    configs = [
        (100, 64, 128, "Small (GNN edge features)"),
        (1000, 128, 128, "Medium (GNN nodes)"),
        (10000, 128, 64, "Large (Many edges)"),
    ]

    # Initialize cuBLAS
    fennol_cuda.init_cublas()

    print("\nBenchmarking cuBLAS dense layers:")
    print(f"{'Config':<30} {'Shape':<20} {'Time (ms)':<15} {'Throughput'}")
    print("-" * 75)

    for batch, in_dim, out_dim, name in configs:
        # Random inputs
        x = np.random.randn(batch, in_dim).astype(np.float64)
        w = np.random.randn(in_dim, out_dim).astype(np.float64)
        b = np.random.randn(out_dim).astype(np.float64)

        # Warm-up
        _ = fennol_cuda.dense_layer(x, w, b, 'silu')

        # Benchmark
        n_runs = 100
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            y = fennol_cuda.dense_layer(x, w, b, 'silu')
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000

        # Compute FLOPs
        # Dense layer: batch * in_dim * out_dim (matmul) + batch * out_dim (bias)
        flops = 2 * batch * in_dim * out_dim + batch * out_dim
        gflops = (flops / 1e9) / (avg_time / 1000)

        shape_str = f"[{batch}, {in_dim}]×[{in_dim}, {out_dim}]"
        time_str = f"{avg_time:.3f} ± {std_time:.3f}"
        throughput_str = f"{gflops:.2f} GFLOPS"

        print(f"{name:<30} {shape_str:<20} {time_str:<15} {throughput_str}")

    fennol_cuda.cleanup_cublas()
    print("\n✅ Performance benchmark complete!")


def test_mlp_multi_layer():
    """Test multi-layer MLP (manual composition for now)."""
    print("\n" + "="*70)
    print("Multi-Layer MLP Test")
    print("="*70)

    # 3-layer MLP: 64 -> 128 -> 128 -> 64
    batch = 100
    layer_dims = [64, 128, 128, 64]

    # Random inputs and weights
    np.random.seed(123)
    x = np.random.randn(batch, layer_dims[0]).astype(np.float64)

    weights = [
        np.random.randn(layer_dims[i], layer_dims[i+1]).astype(np.float64)
        for i in range(len(layer_dims) - 1)
    ]
    biases = [
        np.random.randn(layer_dims[i+1]).astype(np.float64)
        for i in range(len(layer_dims) - 1)
    ]

    print(f"\nMLP Architecture: {' -> '.join(map(str, layer_dims))}")

    # Initialize cuBLAS
    fennol_cuda.init_cublas()

    # Forward pass through each layer
    y = x
    for i, (w, b) in enumerate(zip(weights, biases)):
        activation = 'silu' if i < len(weights) - 1 else 'none'  # No activation on last layer
        y = fennol_cuda.dense_layer(y, w, b, activation)
        print(f"  Layer {i+1}: {y.shape}, activation={activation}")

    # Reference NumPy implementation
    y_ref = x
    for i, (w, b) in enumerate(zip(weights, biases)):
        activation = 'silu' if i < len(weights) - 1 else 'none'
        y_ref = dense_layer_numpy(y_ref, w, b, activation)

    # Check correctness
    max_error = np.abs(y - y_ref).max()
    print(f"\nMax error vs NumPy: {max_error:.2e}")

    assert max_error < 1e-6, f"Error too large: {max_error}"

    fennol_cuda.cleanup_cublas()
    print("\n✅ Multi-layer MLP test passed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("cuBLAS MLP Layers Test Suite")
    print("="*70)
    print("\nTesting:")
    print("  1. Dense layer correctness (SiLU, ReLU, None)")
    print("  2. Performance benchmarks")
    print("  3. Multi-layer MLP composition")

    test_dense_layer_correctness()
    test_dense_layer_performance()
    test_mlp_multi_layer()

    print("\n" + "="*70)
    print("All MLP tests passed!")
    print("="*70)
    print("\nNext steps:")
    print("  - Integrate MLP layers into GNN pipeline")
    print("  - Add model weight loading from JAX parameters")
    print("  - Implement end-to-end GNN inference")
    print("="*70 + "\n")
