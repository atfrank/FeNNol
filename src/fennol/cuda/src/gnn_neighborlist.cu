#include "gnn_solvent.cuh"
#include "common.cuh"
#include <cmath>

namespace fennol {
namespace cuda {
namespace gnn {

/**
 * Kernel to build neighbor list for GNN.
 *
 * Each thread processes one atom and finds all neighbors within cutoff.
 */
__global__ void build_neighborlist_kernel(
    int natoms,
    const double* __restrict__ coords,
    double cutoff,
    int* __restrict__ edge_src,
    int* __restrict__ edge_dst,
    int* __restrict__ edge_count,
    int max_edges_per_atom
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    double xi = coords[i * 3 + 0];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];

    double cutoff_sq = cutoff * cutoff;
    int count = 0;

    // Find all neighbors within cutoff
    for (int j = 0; j < natoms; j++) {
        if (i == j) continue;  // Skip self

        double xj = coords[j * 3 + 0];
        double yj = coords[j * 3 + 1];
        double zj = coords[j * 3 + 2];

        double dx = xi - xj;
        double dy = yi - yj;
        double dz = zi - zj;
        double r_sq = dx * dx + dy * dy + dz * dz;

        if (r_sq < cutoff_sq) {
            if (count < max_edges_per_atom) {
                // Compute global edge index
                int edge_idx = i * max_edges_per_atom + count;
                edge_src[edge_idx] = i;
                edge_dst[edge_idx] = j;
                count++;
            }
        }
    }

    // Store count for this atom
    edge_count[i] = count;
}

/**
 * Host function to build neighbor list.
 */
void build_neighborlist(
    int natoms,
    const double* coords,
    double cutoff,
    int* edge_src,
    int* edge_dst,
    int* num_edges,
    int max_edges
) {
    // Estimate max edges per atom (typically ~50-100 for 5 Å cutoff)
    int max_edges_per_atom = 100;

    // Temporary storage for per-atom edge counts
    int* d_edge_count;
    CUDA_CHECK(cudaMalloc(&d_edge_count, natoms * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_edge_count, 0, natoms * sizeof(int)));

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    build_neighborlist_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        coords,
        cutoff,
        edge_src,
        edge_dst,
        d_edge_count,
        max_edges_per_atom
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy edge counts to host and compute total
    int* h_edge_count = new int[natoms];
    CUDA_CHECK(cudaMemcpy(h_edge_count, d_edge_count, natoms * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int total_edges = 0;
    for (int i = 0; i < natoms; i++) {
        total_edges += h_edge_count[i];
    }

    *num_edges = total_edges;

    // Cleanup
    delete[] h_edge_count;
    CUDA_CHECK(cudaFree(d_edge_count));

    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Kernel to compute RBF expansion of distances.
 *
 * Uses Gaussian RBF: φ_k(r) = exp(-γ(r - μ_k)²)
 */
__global__ void compute_rbf_kernel(
    int nedges,
    const double* __restrict__ distances,
    double cutoff,
    int num_rbf,
    double* __restrict__ rbf_features
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= nedges) return;

    double r = distances[e];

    // RBF parameters
    double gamma = 10.0 / cutoff;
    double delta_mu = cutoff / (num_rbf - 1);

    // Compute RBF for each basis function
    for (int k = 0; k < num_rbf; k++) {
        double mu = k * delta_mu;
        double rbf_val = exp(-gamma * (r - mu) * (r - mu));

        // Apply cutoff function: 0.5 * (cos(πr/r_cut) + 1)
        double cutoff_val = 0.5 * (cos(M_PI * r / cutoff) + 1.0);
        if (r >= cutoff) cutoff_val = 0.0;

        rbf_features[e * num_rbf + k] = rbf_val * cutoff_val;
    }
}

/**
 * Host function to compute RBF features.
 */
void compute_rbf_features(
    int nedges,
    const double* distances,
    double cutoff,
    int num_rbf,
    double* rbf_features
) {
    int threads_per_block = 256;
    int num_blocks = (nedges + threads_per_block - 1) / threads_per_block;

    compute_rbf_kernel<<<num_blocks, threads_per_block>>>(
        nedges,
        distances,
        cutoff,
        num_rbf,
        rbf_features
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gnn
} // namespace cuda
} // namespace fennol
