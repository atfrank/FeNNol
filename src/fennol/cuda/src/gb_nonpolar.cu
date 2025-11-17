#include "implicit_solvent.cuh"
#include "common.cuh"
#include <cmath>

namespace fennol {
namespace cuda {
namespace implicit_solvent {

/**
 * Kernel to compute non-polar (surface area) energy
 *
 * Simple approximation: SA_i ≈ 4π(R_i + r_probe)²
 * More sophisticated implementations could use LCPO or numerical SASA
 */
__global__ void compute_nonpolar_energy_kernel(
    int natoms,
    const double* __restrict__ born_radii,
    const double* __restrict__ gamma_params,
    double probe_radius,
    double* __restrict__ energy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    double R_i = born_radii[i];
    double gamma_i = gamma_params[i];

    // Effective radius including probe
    double R_eff = R_i + probe_radius;

    // Surface area: 4π R²
    double SA_i = 4.0 * M_PI * R_eff * R_eff;

    // Non-polar energy: E_np = γ * SA
    double E_np = gamma_i * SA_i;

    // Accumulate to total energy
    atomicAddDouble(energy, E_np);
}

/**
 * Kernel to compute non-polar forces
 *
 * For now, forces are set to zero since the surface area term
 * is a small correction and computing dSA/dr is complex.
 *
 * A full implementation would compute:
 * F_i = -γ_i * dSA_i/dr_i
 */
__global__ void compute_nonpolar_forces_kernel(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ born_radii,
    const double* __restrict__ gamma_params,
    double probe_radius,
    double* __restrict__ forces
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    // For simplified implementation, forces from surface area are neglected
    // The Born radii already account for most of the solvation effects
    // A full implementation would compute numerical derivatives of SASA

    // forces[i*3+0] += 0.0; // fx
    // forces[i*3+1] += 0.0; // fy
    // forces[i*3+2] += 0.0; // fz

    // Note: forces array is already initialized to zero in the calling function
}

/**
 * Host function to compute non-polar (surface area) energy and forces
 */
void compute_nonpolar_sasa(
    int natoms,
    const double* coords,
    const double* born_radii,
    const double* gamma_params,
    double probe_radius,
    double* energy,
    double* forces
) {
    // Initialize energy to zero
    CUDA_CHECK(cudaMemset(energy, 0, sizeof(double)));

    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Compute non-polar energy
    compute_nonpolar_energy_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        born_radii,
        gamma_params,
        probe_radius,
        energy
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute non-polar forces (currently zero)
    compute_nonpolar_forces_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        coords,
        born_radii,
        gamma_params,
        probe_radius,
        forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure completion
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace implicit_solvent
} // namespace cuda
} // namespace fennol
