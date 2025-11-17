#include "implicit_solvent.cuh"
#include "common.cuh"
#include <cmath>

namespace fennol {
namespace cuda {
namespace implicit_solvent {

// Coulomb constant in kcal·Å·mol⁻¹·e⁻²
constexpr double COULOMB_CONST = 332.0636;

/**
 * Device function to compute GB function f_GB and its derivative
 *
 * f_GB(r, R_i, R_j) = sqrt(r² + R_i*R_j * exp(-r²/(4*R_i*R_j)))
 */
__device__ void compute_f_gb_and_deriv(
    double r,
    double R_i,
    double R_j,
    double& f_gb,
    double& df_gb_dr
) {
    double R_product = R_i * R_j;
    double r_sq = r * r;

    // Exponential term
    double exp_arg = -r_sq / (4.0 * R_product);
    double exp_term = exp(exp_arg);

    // f_GB = sqrt(r² + R_i*R_j * exp(-r²/(4*R_i*R_j)))
    double arg = r_sq + R_product * exp_term;
    f_gb = sqrt(arg);

    // Derivative: df_GB/dr = (2r - R_i*R_j*exp_term * r/(2*R_i*R_j)) / (2*f_GB)
    //                      = r * (1 - exp_term/2) / f_GB
    df_gb_dr = r * (1.0 - 0.5 * exp_term) / f_gb;
}

/**
 * Kernel to compute GB electrostatic energy and forces
 *
 * Each thread computes contribution for one atom pair (i, j) with i < j
 * Uses atomic operations to accumulate forces
 */
__global__ void compute_gb_pairwise_kernel(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ charges,
    const double* __restrict__ born_radii,
    double dielectric,
    double cutoff,
    double* __restrict__ energy,
    double* __restrict__ forces
) {
    // Thread ID for pair (i, j)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of unique pairs
    int total_pairs = natoms * (natoms - 1) / 2;

    if (tid >= total_pairs) return;

    // Map linear thread ID to (i, j) pair with i < j
    // Using inverse triangular number formula
    int i = (int)((sqrt(1.0 + 8.0 * tid) - 1.0) / 2.0);
    int j = tid - i * (i + 1) / 2;
    j = j + i + 1;

    if (j >= natoms) return;

    // Load coordinates
    double xi = coords[i * 3 + 0];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];

    double xj = coords[j * 3 + 0];
    double yj = coords[j * 3 + 1];
    double zj = coords[j * 3 + 2];

    // Compute distance
    double dx = xi - xj;
    double dy = yi - yj;
    double dz = zi - zj;
    double r_sq = dx * dx + dy * dy + dz * dz;
    double r = sqrt(r_sq);

    // Apply cutoff
    if (r > cutoff) return;

    // Load charges and Born radii
    double qi = charges[i];
    double qj = charges[j];
    double R_i = born_radii[i];
    double R_j = born_radii[j];

    // Compute f_GB and its derivative
    double f_gb, df_gb_dr;
    compute_f_gb_and_deriv(r, R_i, R_j, f_gb, df_gb_dr);

    // GB factor: -0.5 * (1 - 1/ε) * COULOMB
    double gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST;

    // Pairwise energy: E_ij = gb_factor * q_i * q_j / f_GB
    double E_pair = gb_factor * qi * qj / f_gb;

    // Accumulate energy (using atomic add to avoid race conditions)
    atomicAdd(energy, E_pair);

    // Force magnitude: F = -dE/dr = gb_factor * q_i * q_j * df_GB/dr / f_GB²
    double force_mag = gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb);

    // Force vector (on atom i, opposite on atom j)
    double fx = force_mag * dx / r;
    double fy = force_mag * dy / r;
    double fz = force_mag * dz / r;

    // Accumulate forces using atomic operations
    atomicAdd(&forces[i * 3 + 0], fx);
    atomicAdd(&forces[i * 3 + 1], fy);
    atomicAdd(&forces[i * 3 + 2], fz);

    atomicAdd(&forces[j * 3 + 0], -fx);
    atomicAdd(&forces[j * 3 + 1], -fy);
    atomicAdd(&forces[j * 3 + 2], -fz);
}

/**
 * Kernel to compute Born self-energy (self-solvation term)
 */
__global__ void compute_born_self_energy_kernel(
    int natoms,
    const double* __restrict__ charges,
    const double* __restrict__ born_radii,
    double dielectric,
    double* __restrict__ energy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    double qi = charges[i];
    double R_i = born_radii[i];

    // GB factor
    double gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST;

    // Self-energy: E_self = gb_factor * q_i² / R_i
    double E_self = gb_factor * qi * qi / R_i;

    // Accumulate to total energy
    atomicAdd(energy, E_self);
}

/**
 * Host function to compute GB electrostatic energy and forces
 */
void compute_gb_energy_forces(
    int natoms,
    const double* coords,
    const double* charges,
    const double* born_radii,
    double dielectric,
    double cutoff,
    double* energy,
    double* forces
) {
    // Initialize energy and forces to zero
    CUDA_CHECK(cudaMemset(energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(forces, 0, natoms * 3 * sizeof(double)));

    // Launch configuration for pairwise interactions
    int total_pairs = natoms * (natoms - 1) / 2;
    int threads_per_block = 256;
    int num_blocks = (total_pairs + threads_per_block - 1) / threads_per_block;

    // Compute pairwise GB energy and forces
    compute_gb_pairwise_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        coords,
        charges,
        born_radii,
        dielectric,
        cutoff,
        energy,
        forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute Born self-energy
    num_blocks = (natoms + threads_per_block - 1) / threads_per_block;
    compute_born_self_energy_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        charges,
        born_radii,
        dielectric,
        energy
    );
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure completion
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace implicit_solvent
} // namespace cuda
} // namespace fennol
