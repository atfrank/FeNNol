/**
 * Neighbor List Implementation
 *
 * Implements Verlet neighbor list for GB implicit solvent calculations.
 */

#include "neighborlist.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace fennol {
namespace cuda {
namespace neighborlist {

/**
 * Build neighbor list kernel (simple O(N²) version).
 *
 * Each thread processes one atom i and finds all neighbors within build_cutoff.
 */
__global__ void build_neighborlist_kernel(
    int natoms,
    const double* __restrict__ coords,
    float build_cutoff,
    int max_neighbors,
    int* __restrict__ neighbor_atoms,
    int* __restrict__ neighbor_counts,
    int* __restrict__ neighbor_offsets,
    int* __restrict__ overflow_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    // Load coordinates for atom i
    double xi = coords[i * 3 + 0];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];

    float build_cutoff_sq = build_cutoff * build_cutoff;
    int count = 0;
    int offset = i * max_neighbors;  // Pre-allocated space for this atom
    bool overflowed = false;

    // Find all neighbors within build_cutoff
    for (int j = 0; j < natoms; j++) {
        if (i == j) continue;  // Skip self

        // Compute distance
        double dx = xi - coords[j * 3 + 0];
        double dy = yi - coords[j * 3 + 1];
        double dz = zi - coords[j * 3 + 2];
        double r_sq = dx * dx + dy * dy + dz * dz;

        // Check if within build cutoff
        if (r_sq < build_cutoff_sq) {
            if (count < max_neighbors) {
                // Store neighbor index
                neighbor_atoms[offset + count] = j;
                count++;
            } else {
                // Buffer overflow - set flag
                overflowed = true;
            }
        }
    }

    // Store results
    neighbor_counts[i] = count;
    neighbor_offsets[i] = offset;

    // Atomic increment if overflowed (for statistics)
    if (overflowed) {
        atomicAdd(overflow_count, 1);
    }
}

/**
 * Check if rebuild needed using parallel reduction.
 *
 * Computes max displacement using shared memory reduction.
 */
__global__ void check_rebuild_kernel(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ build_coords,
    float* __restrict__ max_displacement
) {
    __shared__ float s_max_disp[256];  // Shared memory for reduction

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Compute displacement for this atom
    float disp = 0.0f;
    if (i < natoms) {
        double dx = coords[i * 3 + 0] - build_coords[i * 3 + 0];
        double dy = coords[i * 3 + 1] - build_coords[i * 3 + 1];
        double dz = coords[i * 3 + 2] - build_coords[i * 3 + 2];
        disp = sqrt(dx * dx + dy * dy + dz * dz);
    }

    s_max_disp[tid] = disp;
    __syncthreads();

    // Parallel reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_disp[tid] = fmaxf(s_max_disp[tid], s_max_disp[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        atomicMaxFloat(max_displacement, s_max_disp[0]);
    }
}

/**
 * Atomic max for float (helper function).
 *
 * CUDA doesn't provide atomicMax for float, so we implement it.
 */
__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

/**
 * Simple coordinate copy kernel.
 */
__global__ void copy_coords_kernel(
    int natoms,
    const double* __restrict__ coords,
    double* __restrict__ build_coords
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    build_coords[i * 3 + 0] = coords[i * 3 + 0];
    build_coords[i * 3 + 1] = coords[i * 3 + 1];
    build_coords[i * 3 + 2] = coords[i * 3 + 2];
}

} // namespace neighborlist
} // namespace cuda
} // namespace fennol
