#include "implicit_solvent.cuh"
#include "common.cuh"
#include <cmath>

namespace fennol {
namespace cuda {
namespace implicit_solvent {

/**
 * Device function to compute pairwise descreening integral
 *
 * This implements the integral I(r_ij, ρ_i, ρ_j) used in the OBC model
 * for computing Born radii.
 */
__device__ double descreening_integral(
    double r,
    double rho_i,
    double rho_j
) {
    // Handle self-interaction and very close atoms
    if (r < 0.001) {
        return 0.0;
    }

    double upper_limit = rho_i + rho_j;
    double lower_limit = fabs(rho_i - rho_j);

    double integral;

    if (r < lower_limit) {
        // Complete overlap
        integral = 0.5 * (1.0 / (lower_limit * lower_limit) -
                         1.0 / (upper_limit * upper_limit));
    } else if (r < upper_limit) {
        // Partial overlap
        integral = 0.5 * (1.0 / (r * r) -
                         1.0 / (upper_limit * upper_limit));
    } else {
        // No overlap
        integral = 0.0;
    }

    return integral * rho_i;
}

/**
 * Kernel to compute pairwise descreening contributions
 *
 * Each thread computes descreening for one atom from all other atoms
 */
__global__ void compute_descreening_kernel(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ intrinsic_radii,
    double cutoff,
    double* __restrict__ psi_sum
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    // Load atom i data
    double xi = coords[i * 3 + 0];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];
    double rho_i = intrinsic_radii[i];

    double psi = 0.0;
    double cutoff_sq = cutoff * cutoff;

    // Loop over all atoms j
    for (int j = 0; j < natoms; j++) {
        if (i == j) continue;

        // Compute distance
        double dx = xi - coords[j * 3 + 0];
        double dy = yi - coords[j * 3 + 1];
        double dz = zi - coords[j * 3 + 2];
        double r_sq = dx * dx + dy * dy + dz * dz;

        // Apply cutoff
        if (r_sq > cutoff_sq) continue;

        double r = sqrt(r_sq);
        double rho_j = intrinsic_radii[j];

        // Accumulate descreening integral
        psi += descreening_integral(r, rho_i, rho_j);
    }

    psi_sum[i] = psi;
}

/**
 * Kernel to compute Born radii from descreening integrals (OBC formula)
 */
__global__ void compute_born_radii_kernel(
    int natoms,
    const double* __restrict__ intrinsic_radii,
    const double* __restrict__ b_params,
    const double* __restrict__ c_params,
    const double* __restrict__ psi_sum,
    double* __restrict__ born_radii
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    double rho = intrinsic_radii[i];
    double psi = psi_sum[i];
    double b = b_params[i];
    double c = c_params[i];

    // OBC formula: 1/R_i = 1/ρ_i - tanh(ψ - b*ψ² + c*ψ³) / ρ_i
    double psi_2 = psi * psi;
    double psi_3 = psi_2 * psi;
    double tanh_arg = psi - b * psi_2 + c * psi_3;
    double tanh_val = tanh(tanh_arg);

    double R_inv = 1.0 / rho - tanh_val / rho;

    // Ensure Born radius is at least as large as intrinsic radius
    double R = 1.0 / R_inv;
    if (R < rho) {
        R = rho;
    }

    born_radii[i] = R;
}

/**
 * Host function to compute Born radii using OBC model
 */
void compute_born_radii_obc(
    int natoms,
    const double* coords,
    const double* intrinsic_radii,
    const double* b_params,
    const double* c_params,
    double cutoff,
    double* born_radii
) {
    // Allocate temporary storage for psi_sum
    double* d_psi_sum;
    CUDA_CHECK(cudaMalloc(&d_psi_sum, natoms * sizeof(double)));

    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Step 1: Compute descreening integrals
    compute_descreening_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        coords,
        intrinsic_radii,
        cutoff,
        d_psi_sum
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Compute Born radii from descreening
    compute_born_radii_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        intrinsic_radii,
        b_params,
        c_params,
        d_psi_sum,
        born_radii
    );
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary memory
    CUDA_CHECK(cudaFree(d_psi_sum));
}

} // namespace implicit_solvent
} // namespace cuda
} // namespace fennol
