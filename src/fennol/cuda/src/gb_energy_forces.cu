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
    //                      = (2r - r*exp_term/2) / (2*f_GB)
    //                      = r * (1 - exp_term/4) / f_GB
    df_gb_dr = r * (1.0 - 0.25 * exp_term) / f_gb;
}

/**
 * Device function to compute GB function f_GB and its derivative (FLOAT version)
 *
 * Mixed precision optimization: Use FP32 for intermediate calculations
 * while keeping Born radii in FP64 for accuracy.
 *
 * f_GB(r, R_i, R_j) = sqrt(r² + R_i*R_j * exp(-r²/(4*R_i*R_j)))
 */
__device__ void compute_f_gb_and_deriv_f(
    float r,
    double R_i,     // Keep Born radii in double!
    double R_j,
    float& f_gb,
    float& df_gb_dr
) {
    float R_product = (float)(R_i * R_j);
    float r_sq = r * r;

    // Exponential term (FP32)
    float exp_arg = -r_sq / (4.0f * R_product);
    float exp_term = expf(exp_arg);

    // f_GB = sqrt(r² + R_i*R_j * exp(-r²/(4*R_i*R_j)))
    float arg = r_sq + R_product * exp_term;
    f_gb = sqrtf(arg);

    // Derivative: df_GB/dr = r * (1 - exp_term/4) / f_GB
    df_gb_dr = r * (1.0f - 0.25f * exp_term) / f_gb;
}

/**
 * BASIC kernel to compute GB electrostatic energy and forces (UNOPTIMIZED)
 *
 * Each thread computes contribution for one atom pair (i, j) with i < j
 * Uses atomic operations to accumulate forces - HIGH ATOMIC CONTENTION!
 *
 * NOTE: This is kept for reference. Use compute_gb_pairwise_kernel_tiled() for 5-10x speedup.
 */
__global__ void compute_gb_pairwise_kernel_basic(
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
    atomicAddDouble(energy, E_pair);

    // Force magnitude: F = -dE/dr = gb_factor * q_i * q_j * df_GB/dr / f_GB²
    double force_mag = gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb);

    // Force vector (on atom i, opposite on atom j)
    double fx = force_mag * dx / r;
    double fy = force_mag * dy / r;
    double fz = force_mag * dz / r;

    // Accumulate forces using atomic operations (HIGH CONTENTION - SLOW!)
    atomicAddDouble(&forces[i * 3 + 0], fx);
    atomicAddDouble(&forces[i * 3 + 1], fy);
    atomicAddDouble(&forces[i * 3 + 2], fz);

    atomicAddDouble(&forces[j * 3 + 0], -fx);
    atomicAddDouble(&forces[j * 3 + 1], -fy);
    atomicAddDouble(&forces[j * 3 + 2], -fz);
}

/**
 * OPTIMIZED kernel using shared memory tiling
 *
 * Each thread handles one atom i and computes its interactions with all atoms j
 * using tiled shared memory access. This avoids atomic contention and enables
 * coalesced memory access for 5-10x speedup.
 *
 * Strategy:
 * - Each thread handles one atom i
 * - Process atoms j in tiles loaded to shared memory (COALESCED)
 * - Accumulate local energy/forces (no atomics needed per thread)
 * - Only one atomic add per thread for total energy
 */
__global__ void compute_gb_pairwise_kernel_tiled(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ charges,
    const double* __restrict__ born_radii,
    double dielectric,
    double cutoff,
    double* __restrict__ energy,
    double* __restrict__ forces
) {
    // Shared memory for atom tile
    extern __shared__ double s_data[];
    double* s_coords = s_data;                      // [TILE_SIZE * 3]
    double* s_charges = &s_data[blockDim.x * 3];   // [TILE_SIZE]
    double* s_radii = &s_data[blockDim.x * 4];     // [TILE_SIZE]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load atom i data (register variables)
    double xi, yi, zi, qi, R_i;
    if (i < natoms) {
        xi = coords[i * 3 + 0];
        yi = coords[i * 3 + 1];
        zi = coords[i * 3 + 2];
        qi = charges[i];
        R_i = born_radii[i];
    }

    // Accumulators for atom i
    double E_i = 0.0;
    double fx_i = 0.0;
    double fy_i = 0.0;
    double fz_i = 0.0;

    double cutoff_sq = cutoff * cutoff;
    double gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST;

    // Number of tiles
    int num_tiles = (natoms + blockDim.x - 1) / blockDim.x;

    // Process atoms in tiles
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * blockDim.x;
        int load_idx = tile_start + tid;

        // Load tile into shared memory (COALESCED)
        if (load_idx < natoms) {
            s_coords[tid * 3 + 0] = coords[load_idx * 3 + 0];
            s_coords[tid * 3 + 1] = coords[load_idx * 3 + 1];
            s_coords[tid * 3 + 2] = coords[load_idx * 3 + 2];
            s_charges[tid] = charges[load_idx];
            s_radii[tid] = born_radii[load_idx];
        }
        __syncthreads();

        // Compute interactions with this tile
        if (i < natoms) {
            int tile_size = min((int)blockDim.x, natoms - tile_start);

            for (int t = 0; t < tile_size; t++) {
                int j = tile_start + t;

                // Skip self-interaction
                if (i == j) continue;

                // Load atom j data from shared memory (FAST!)
                double xj = s_coords[t * 3 + 0];
                double yj = s_coords[t * 3 + 1];
                double zj = s_coords[t * 3 + 2];
                double qj = s_charges[t];
                double R_j = s_radii[t];

                // Compute distance
                double dx = xi - xj;
                double dy = yi - yj;
                double dz = zi - zj;
                double r_sq = dx * dx + dy * dy + dz * dz;

                // Apply cutoff
                if (r_sq > cutoff_sq) continue;

                double r = sqrt(r_sq);

                // Compute f_GB and derivative
                double f_gb, df_gb_dr;
                compute_f_gb_and_deriv(r, R_i, R_j, f_gb, df_gb_dr);

                // Pairwise energy contribution (count each pair once, so multiply by 0.5)
                double E_pair = 0.5 * gb_factor * qi * qj / f_gb;
                E_i += E_pair;

                // Force magnitude (NO 0.5 factor here!)
                // Each thread computes full force for atom i from all neighbors j
                // The pair (i,j) is visited twice: once by thread i, once by thread j
                // Each gets the correct force magnitude without needing to divide by 2
                double force_mag = gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb);

                // Force components (displacement vector is i - j)
                double fx = force_mag * dx / r;
                double fy = force_mag * dy / r;
                double fz = force_mag * dz / r;

                // Accumulate forces on atom i only
                // Thread i processes (i,j) and adds force to i
                // Thread j processes (j,i) and adds force to j
                // This gives Newton's 3rd law automatically with the 0.5 factor
                fx_i += fx;
                fy_i += fy;
                fz_i += fz;
            }
        }
        __syncthreads();
    }

    // Write results (no atomics needed for forces since each thread writes its own atom)
    if (i < natoms) {
        forces[i * 3 + 0] = fx_i;
        forces[i * 3 + 1] = fy_i;
        forces[i * 3 + 2] = fz_i;

        // Accumulate total energy (only one atomic per thread)
        atomicAddDouble(energy, E_i);
    }
}

/**
 * NEIGHBOR LIST VERSION - Optimized GB pairwise kernel using precomputed neighbor list
 *
 * This kernel uses a precomputed Verlet neighbor list to skip pairs beyond cutoff.
 * Expected speedup: 2.5-5× over tiled version.
 *
 * Strategy:
 * - Each thread handles one atom i
 * - Loop over neighbors from precomputed list (not all atoms!)
 * - Cutoff check still needed (build_cutoff > cutoff due to skin)
 * - No shared memory tiling needed (neighbor list already filtered)
 */
__global__ void compute_gb_pairwise_kernel_neighborlist(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ charges,
    const double* __restrict__ born_radii,
    double dielectric,
    double cutoff,
    const int* __restrict__ neighbor_atoms,
    const int* __restrict__ neighbor_counts,
    const int* __restrict__ neighbor_offsets,
    double* __restrict__ energy,
    double* __restrict__ forces
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    // Load atom i data
    double xi = coords[i * 3 + 0];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];
    double qi = charges[i];
    double R_i = born_radii[i];

    // Accumulators
    double E_i = 0.0;
    double fx_i = 0.0;
    double fy_i = 0.0;
    double fz_i = 0.0;

    double cutoff_sq = cutoff * cutoff;
    double gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST;

    // Get neighbor list info for atom i
    int offset = neighbor_offsets[i];
    int num_neighbors = neighbor_counts[i];

    // Loop over neighbors ONLY (not all atoms!)
    for (int n = 0; n < num_neighbors; n++) {
        int j = neighbor_atoms[offset + n];

        // Load atom j data
        double xj = coords[j * 3 + 0];
        double yj = coords[j * 3 + 1];
        double zj = coords[j * 3 + 2];
        double qj = charges[j];
        double R_j = born_radii[j];

        // Compute distance
        double dx = xi - xj;
        double dy = yi - yj;
        double dz = zi - zj;
        double r_sq = dx * dx + dy * dy + dz * dz;

        // Cutoff check (needed because neighbor list uses build_cutoff > cutoff)
        if (r_sq > cutoff_sq) continue;

        double r = sqrt(r_sq);

        // Compute f_GB and derivative
        double f_gb, df_gb_dr;
        compute_f_gb_and_deriv(r, R_i, R_j, f_gb, df_gb_dr);

        // Pairwise energy contribution (count each pair once, so multiply by 0.5)
        double E_pair = 0.5 * gb_factor * qi * qj / f_gb;
        E_i += E_pair;

        // Force magnitude
        double force_mag = gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb);

        // Force components
        double fx = force_mag * dx / r;
        double fy = force_mag * dy / r;
        double fz = force_mag * dz / r;

        // Accumulate forces on atom i
        fx_i += fx;
        fy_i += fy;
        fz_i += fz;
    }

    // Write results
    forces[i * 3 + 0] = fx_i;
    forces[i * 3 + 1] = fy_i;
    forces[i * 3 + 2] = fz_i;

    // Accumulate total energy
    atomicAddDouble(energy, E_i);
}

/**
 * MIXED PRECISION neighbor list GB pairwise energy/forces kernel
 *
 * Uses FP32 for intermediate calculations (coords, distances, f_GB, force magnitudes)
 * while keeping FP64 for critical values (Born radii, energy/force accumulation).
 *
 * Expected speedup: 1.5-2× from reduced memory bandwidth
 */
__global__ void compute_gb_pairwise_kernel_neighborlist_mixed(
    int natoms,
    const double* __restrict__ coords,           // Input still FP64
    const double* __restrict__ charges,          // Input still FP64
    const double* __restrict__ born_radii,       // Keep Born radii in FP64!
    double dielectric,
    double cutoff,
    const int* __restrict__ neighbor_atoms,
    const int* __restrict__ neighbor_counts,
    const int* __restrict__ neighbor_offsets,
    double* __restrict__ energy,
    double* __restrict__ forces
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    // Load atom i data and convert to FP32 for intermediate calculations
    float xi = (float)coords[i * 3 + 0];
    float yi = (float)coords[i * 3 + 1];
    float zi = (float)coords[i * 3 + 2];
    float qi = (float)charges[i];
    double R_i = born_radii[i];  // Keep Born radii in FP64!

    // Accumulators (keep in FP64 for precision)
    double E_i = 0.0;
    double fx_i = 0.0;
    double fy_i = 0.0;
    double fz_i = 0.0;

    float cutoff_sq = (float)(cutoff * cutoff);
    float gb_factor = (float)(-0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST);

    // Get neighbor list info for atom i
    int offset = neighbor_offsets[i];
    int num_neighbors = neighbor_counts[i];

    // Loop over neighbors ONLY (not all atoms!)
    for (int n = 0; n < num_neighbors; n++) {
        int j = neighbor_atoms[offset + n];

        // Load atom j data and convert to FP32
        float xj = (float)coords[j * 3 + 0];
        float yj = (float)coords[j * 3 + 1];
        float zj = (float)coords[j * 3 + 2];
        float qj = (float)charges[j];
        double R_j = born_radii[j];  // Keep Born radii in FP64!

        // Compute distance in FP32
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float r_sq = dx * dx + dy * dy + dz * dz;

        // Cutoff check
        if (r_sq > cutoff_sq) continue;

        float r = sqrtf(r_sq);

        // Compute f_GB and derivative in FP32
        float f_gb, df_gb_dr;
        compute_f_gb_and_deriv_f(r, R_i, R_j, f_gb, df_gb_dr);

        // Pairwise energy contribution (FP32, then cast to FP64 for accumulation)
        float E_pair = 0.5f * gb_factor * qi * qj / f_gb;
        E_i += (double)E_pair;

        // Force magnitude (FP32)
        float force_mag = gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb);

        // Force components (FP32, then cast to FP64 for accumulation)
        float fx = force_mag * dx / r;
        float fy = force_mag * dy / r;
        float fz = force_mag * dz / r;

        fx_i += (double)fx;
        fy_i += (double)fy;
        fz_i += (double)fz;
    }

    // Write results (FP64)
    forces[i * 3 + 0] = fx_i;
    forces[i * 3 + 1] = fy_i;
    forces[i * 3 + 2] = fz_i;

    // Accumulate total energy (FP64)
    atomicAddDouble(energy, E_i);
}

/**
 * PHASE 3C: FUSED kernel to compute GB pairwise forces AND dE/dR in single pass
 *
 * This kernel fuses compute_gb_pairwise_kernel_neighborlist_mixed() and compute_dE_dR_neighborlist_mixed()
 * into a single kernel to:
 *   1. Eliminate one kernel launch overhead
 *   2. Reuse computed values (r, f_GB, df_gb_dr)
 *   3. Reduce memory traffic (single neighbor list traversal)
 *
 * Expected speedup: ~1.10× over running both kernels separately
 *
 * Computes:
 *   - GB pairwise energy and forces (output: energy, forces)
 *   - dE/dR derivatives (output: dE_dR)
 *
 * Both use the same:
 *   - Neighbor list traversal
 *   - Mixed precision strategy (FP32 intermediates, FP64 critical values)
 *   - Cutoff distance
 */
__global__ void compute_gb_and_dE_dR_fused_mixed_kernel(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ charges,
    const double* __restrict__ born_radii,
    double dielectric,
    double cutoff,
    const int* __restrict__ neighbor_atoms,
    const int* __restrict__ neighbor_counts,
    const int* __restrict__ neighbor_offsets,
    double* __restrict__ energy,      // Output: GB pairwise energy
    double* __restrict__ forces,      // Output: GB pairwise forces [natoms*3]
    double* __restrict__ dE_dR        // Output: dE/dR for each atom [natoms]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    // Load atom i data and convert to FP32 for intermediate calculations
    float xi = (float)coords[i * 3 + 0];
    float yi = (float)coords[i * 3 + 1];
    float zi = (float)coords[i * 3 + 2];
    float qi = (float)charges[i];
    double R_i = born_radii[i];  // Keep Born radii in FP64!

    // Accumulators for GB forces (keep in FP64 for precision)
    double E_i = 0.0;
    double fx_i = 0.0;
    double fy_i = 0.0;
    double fz_i = 0.0;

    // Accumulator for dE/dR (FP64)
    const double COULOMB_CONST = 332.0636;
    double gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST;

    // Initialize dE/dR with self-energy contribution (FP64)
    // dE_self/dR_i = gb_factor * (-q_i² / R_i²)
    double dE_dRi = gb_factor * (-(double)qi * qi / (R_i * R_i));

    float cutoff_sq = (float)(cutoff * cutoff);
    float gb_factor_f = (float)gb_factor;

    // Get neighbor list info for atom i
    int offset = neighbor_offsets[i];
    int num_neighbors = neighbor_counts[i];

    // Loop over neighbors ONLY (single pass for both computations!)
    for (int n = 0; n < num_neighbors; n++) {
        int j = neighbor_atoms[offset + n];

        // Load atom j data and convert to FP32
        float xj = (float)coords[j * 3 + 0];
        float yj = (float)coords[j * 3 + 1];
        float zj = (float)coords[j * 3 + 2];
        float qj = (float)charges[j];
        double R_j = born_radii[j];  // Keep Born radii in FP64!

        // Compute distance in FP32 (SHARED COMPUTATION)
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float r_sq = dx * dx + dy * dy + dz * dz;

        // Cutoff check (SHARED COMPUTATION)
        if (r_sq > cutoff_sq) continue;

        float r = sqrtf(r_sq);

        // Compute f_GB and BOTH derivatives in FP32 (KEY REUSE!)
        // We need both df_gb/dr (for forces) and df_gb/dR_i (for dE/dR)
        float f_gb, df_gb_dr, df_gb_dRi;

        // Compute f_GB and df_gb/dr for forces
        compute_f_gb_and_deriv_f(r, R_i, R_j, f_gb, df_gb_dr);

        // Compute df_gb/dR_i for dE/dR (reuse f_gb!)
        float R_i_f = (float)R_i;
        float R_j_f = (float)R_j;
        float R_product = R_i_f * R_j_f;
        float r_sq_local = r * r;

        if (R_product >= 1e-12f) {
            float exp_arg = -r_sq_local / (4.0f * R_product);
            float exp_term = expf(exp_arg);
            float r_sq_term = r_sq_local / (4.0f * R_product);
            float dRiRj_exp_dRi = R_j_f * exp_term * (1.0f + r_sq_term);
            df_gb_dRi = 0.5f / f_gb * dRiRj_exp_dRi;
        } else {
            df_gb_dRi = 0.0f;
        }

        // ===== GB PAIRWISE ENERGY AND FORCES =====

        // Pairwise energy contribution (FP32, then cast to FP64 for accumulation)
        float E_pair = 0.5f * gb_factor_f * qi * qj / f_gb;
        E_i += (double)E_pair;

        // Force magnitude (FP32)
        float force_mag = gb_factor_f * qi * qj * df_gb_dr / (f_gb * f_gb);

        // Force components (FP32, then cast to FP64 for accumulation)
        float fx = force_mag * dx / r;
        float fy = force_mag * dy / r;
        float fz = force_mag * dz / r;

        fx_i += (double)fx;
        fy_i += (double)fy;
        fz_i += (double)fz;

        // ===== dE/dR COMPUTATION =====

        // Pairwise contribution to ∂E/∂Rᵢ (FP32 intermediate, then cast to FP64)
        // ∂E/∂R_i += gb_factor * q_i*q_j * (-1/f_GB²) * (∂f_GB/∂R_i)
        float contrib_dE_dR = gb_factor_f * qi * qj * (-1.0f / (f_gb * f_gb)) * df_gb_dRi;
        dE_dRi += (double)contrib_dE_dR;
    }

    // Write results (FP64)
    forces[i * 3 + 0] = fx_i;
    forces[i * 3 + 1] = fy_i;
    forces[i * 3 + 2] = fz_i;

    // Write dE/dR (FP64)
    dE_dR[i] = dE_dRi;

    // Accumulate total energy (FP64)
    atomicAddDouble(energy, E_i);
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
    atomicAddDouble(energy, E_self);
}

/**
 * Host function to compute GB electrostatic energy and forces (OPTIMIZED)
 *
 * Uses shared memory tiling for 5-10x speedup over basic version.
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

    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Shared memory size for tiled kernel
    // Need: coords[TILE_SIZE * 3] + charges[TILE_SIZE] + radii[TILE_SIZE]
    int shared_mem_size = threads_per_block * 5 * sizeof(double);  // 3 coords + 1 charge + 1 radius

    // Compute pairwise GB energy and forces (OPTIMIZED with shared memory tiling)
    compute_gb_pairwise_kernel_tiled<<<num_blocks, threads_per_block, shared_mem_size>>>(
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

/**
 * Host function to compute GB electrostatic energy and forces (BASIC version)
 *
 * This is kept for comparison and testing purposes.
 * Use compute_gb_energy_forces() for production (5-10x faster).
 */
void compute_gb_energy_forces_basic(
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

    // Compute pairwise GB energy and forces (BASIC unoptimized version)
    compute_gb_pairwise_kernel_basic<<<num_blocks, threads_per_block>>>(
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

/**
 * NEIGHBOR LIST VERSION - Host function to compute GB energy/forces with neighbor list
 *
 * Uses precomputed Verlet neighbor list for 2.5-5× speedup.
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3] (device)
 * @param charges Partial charges [natoms] (device)
 * @param born_radii Born radii [natoms] (device)
 * @param dielectric Solvent dielectric constant
 * @param cutoff Cutoff distance
 * @param neighbor_atoms Neighbor list atoms array (device)
 * @param neighbor_counts Number of neighbors per atom (device)
 * @param neighbor_offsets Start index for each atom (device)
 * @param energy Output: total GB energy (device)
 * @param forces Output: forces [natoms, 3] (device)
 */
void compute_gb_energy_forces_neighborlist(
    int natoms,
    const double* coords,
    const double* charges,
    const double* born_radii,
    double dielectric,
    double cutoff,
    const int* neighbor_atoms,
    const int* neighbor_counts,
    const int* neighbor_offsets,
    double* energy,
    double* forces
) {
    // Initialize energy and forces to zero
    CUDA_CHECK(cudaMemset(energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(forces, 0, natoms * 3 * sizeof(double)));

    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Compute pairwise GB energy and forces using NEIGHBOR LIST
    compute_gb_pairwise_kernel_neighborlist<<<num_blocks, threads_per_block>>>(
        natoms,
        coords,
        charges,
        born_radii,
        dielectric,
        cutoff,
        neighbor_atoms,
        neighbor_counts,
        neighbor_offsets,
        energy,
        forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute Born self-energy (same as before)
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

/**
 * MIXED PRECISION NEIGHBOR LIST VERSION - Compute GB energy and forces
 *
 * Uses FP32 for intermediate calculations (coords, distances, f_GB, force magnitudes)
 * while keeping FP64 for critical values (Born radii, energy/force accumulation).
 *
 * Expected speedup: 1.5-2× from reduced memory bandwidth
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3] (device, FP64)
 * @param charges Partial atomic charges [natoms] (device, FP64)
 * @param born_radii Effective Born radii [natoms] (device, FP64)
 * @param dielectric Solvent dielectric constant
 * @param cutoff Cutoff distance
 * @param neighbor_atoms Neighbor list atoms array (device)
 * @param neighbor_counts Number of neighbors per atom (device)
 * @param neighbor_offsets Start index for each atom (device)
 * @param energy Output: GB electrostatic energy [1] (device, FP64)
 * @param forces Output: forces on atoms [natoms, 3] (device, FP64)
 */
void compute_gb_energy_forces_neighborlist_mixed(
    int natoms,
    const double* coords,
    const double* charges,
    const double* born_radii,
    double dielectric,
    double cutoff,
    const int* neighbor_atoms,
    const int* neighbor_counts,
    const int* neighbor_offsets,
    double* energy,
    double* forces
) {
    // Initialize energy and forces to zero
    CUDA_CHECK(cudaMemset(energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(forces, 0, natoms * 3 * sizeof(double)));

    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Compute pairwise GB energy and forces using MIXED PRECISION NEIGHBOR LIST
    compute_gb_pairwise_kernel_neighborlist_mixed<<<num_blocks, threads_per_block>>>(
        natoms,
        coords,
        charges,
        born_radii,
        dielectric,
        cutoff,
        neighbor_atoms,
        neighbor_counts,
        neighbor_offsets,
        energy,
        forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute Born self-energy (same as before, FP64)
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

/**
 * PHASE 3C: FUSED version - Compute GB energy/forces AND dE/dR in single kernel
 *
 * This is the fused version that combines:
 *   - compute_gb_pairwise_kernel_neighborlist_mixed()
 *   - compute_dE_dR_neighborlist_mixed()
 *
 * Benefits:
 *   1. Eliminates one kernel launch overhead (~1.05× gain)
 *   2. Reuses computed values (r, f_GB, derivatives)
 *   3. Single neighbor list traversal
 *
 * Expected speedup: ~1.10× over running both kernels separately
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms*3] (device, FP64)
 * @param charges Partial atomic charges [natoms] (device, FP64)
 * @param born_radii Effective Born radii [natoms] (device, FP64)
 * @param dielectric Solvent dielectric constant
 * @param cutoff Cutoff distance
 * @param neighbor_atoms Neighbor list atoms array (device)
 * @param neighbor_counts Number of neighbors per atom (device)
 * @param neighbor_offsets Start index for each atom (device)
 * @param energy Output: GB electrostatic energy [1] (device, FP64)
 * @param forces Output: forces on atoms [natoms*3] (device, FP64)
 * @param dE_dR Output: dE/dR for each atom [natoms] (device, FP64)
 */
void compute_gb_and_dE_dR_fused_mixed(
    int natoms,
    const double* coords,
    const double* charges,
    const double* born_radii,
    double dielectric,
    double cutoff,
    const int* neighbor_atoms,
    const int* neighbor_counts,
    const int* neighbor_offsets,
    double* energy,
    double* forces,
    double* dE_dR
) {
    // Initialize energy and forces to zero
    CUDA_CHECK(cudaMemset(energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(forces, 0, natoms * 3 * sizeof(double)));

    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Compute GB pairwise energy/forces AND dE/dR in SINGLE FUSED KERNEL
    compute_gb_and_dE_dR_fused_mixed_kernel<<<num_blocks, threads_per_block>>>(
        natoms,
        coords,
        charges,
        born_radii,
        dielectric,
        cutoff,
        neighbor_atoms,
        neighbor_counts,
        neighbor_offsets,
        energy,
        forces,
        dE_dR
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute Born self-energy
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
