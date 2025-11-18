#include "implicit_solvent.cuh"
#include "common.cuh"
#include <cmath>

namespace fennol {
namespace cuda {
namespace implicit_solvent {

/**
 * Compute derivative of OBC descreening integral with respect to distance.
 *
 * This is ∂I(r, ρᵢ, ρⱼ)/∂r, needed for Born radii force derivatives.
 */
__device__ double descreening_integral_derivative(
    double r,
    double rho_i,
    double rho_j
) {
    if (r < 0.001) {
        return 0.0;
    }

    double upper_limit = rho_i + rho_j;
    double lower_limit = fabs(rho_i - rho_j);

    // HCT (Hawkins-Cramer-Truhlar) descreening derivative formula
    // This matches OpenMM's implementation in gbsaObc.cc

    if (r < lower_limit) {
        // Complete overlap region - derivative is zero
        return 0.0;
    } else if (r < upper_limit) {
        // Partial overlap region - use HCT derivative formula from OpenMM
        // This matches ReferenceObc.cpp in OpenMM

        double s_j = rho_j;  // Scaled radius of atom j

        // Compute lower bound: l_ij = 1/max(ρᵢ, |r - sⱼ|)
        double abs_diff = fabs(r - s_j);
        double lower_bound = (rho_i > abs_diff) ? rho_i : abs_diff;
        double l_ij = 1.0 / lower_bound;

        // Compute upper bound: u_ij = 1/(r + sⱼ)
        double u_ij = 1.0 / (r + s_j);

        double l_ij2 = l_ij * l_ij;
        double u_ij2 = u_ij * u_ij;
        double s_j2 = s_j * s_j;
        double r_inv = 1.0 / r;
        double r2_inv = r_inv * r_inv;

        // HCT derivative formula from OpenMM:
        // t3 = 0.125*(1 + s_j²/r²)*(l_ij² - u_ij²) + 0.25*log(u_ij/l_ij)/r²
        // ∂ψ/∂r = t3 / r
        //
        // Note: OpenMM's comment says "dL/dr & dU/dr are zero (this can be shown analytically)"
        // This is because l_ij and u_ij are clamped by the max() operation

        double t3 = 0.125 * (1.0 + s_j2 * r2_inv) * (l_ij2 - u_ij2)
                  + 0.25 * log(u_ij / l_ij) * r2_inv;

        double deriv = t3 * r_inv;

        return deriv;
    } else {
        // No overlap
        return 0.0;
    }
}

/**
 * Compute derivative of OBC Born radius with respect to descreening sum.
 *
 * For OBC: 1/Rᵢ = 1/ρᵢ - tanh(ψ - b*ψ² + c*ψ³) / ρᵢ
 *
 * This returns ∂(1/R)/∂ψ (derivative of INVERSE Born radius), matching OpenMM's obcChain.
 *
 * ∂(1/R)/∂ψ = -sech²(ψ - b*ψ² + c*ψ³) * (1 - 2b*ψ + 3c*ψ²) / ρᵢ
 */
__device__ double born_radius_derivative_wrt_psi(
    double R_i,
    double rho_i,
    double psi,
    double b,
    double c
) {
    // Argument to tanh
    double psi_2 = psi * psi;
    double psi_3 = psi_2 * psi;
    double tanh_arg = psi - b * psi_2 + c * psi_3;

    // Derivative of tanh argument w.r.t. psi
    double dtanh_arg_dpsi = 1.0 - 2.0 * b * psi + 3.0 * c * psi_2;

    // sech²(x) = 1 - tanh²(x)
    double tanh_val = tanh(tanh_arg);
    double sech_squared = 1.0 - tanh_val * tanh_val;

    // ∂(1/R)/∂ψ = -sech²(...) * d(...)/dψ / ρᵢ
    // Note the negative sign! tanh is increasing, so 1/R decreases as ψ increases
    double d_invR_dpsi = -sech_squared * dtanh_arg_dpsi / rho_i;

    return d_invR_dpsi;
}

/**
 * Compute the GB effective interaction function f_GB and its derivatives.
 *
 * f_GB = sqrt(r² + Rᵢ*Rⱼ*exp(-r²/(4*Rᵢ*Rⱼ)))
 *
 * Returns:
 * - f_GB value
 * - ∂f_GB/∂Rᵢ (if df_dRi is not NULL)
 * - ∂f_GB/∂Rⱼ (if df_dRj is not NULL)
 */
__device__ void compute_f_gb_and_derivatives(
    double r,
    double R_i,
    double R_j,
    double* f_gb,
    double* df_dRi,
    double* df_dRj
) {
    double r_sq = r * r;
    double RiRj = R_i * R_j;

    // Avoid division by zero
    if (RiRj < 1e-12) {
        *f_gb = r;
        if (df_dRi) *df_dRi = 0.0;
        if (df_dRj) *df_dRj = 0.0;
        return;
    }

    double exp_arg = -r_sq / (4.0 * RiRj);
    double exp_val = exp(exp_arg);
    double RiRj_exp = RiRj * exp_val;

    // f_GB = sqrt(r² + Rᵢ*Rⱼ*exp(-r²/(4*Rᵢ*Rⱼ)))
    double f_gb_sq = r_sq + RiRj_exp;
    *f_gb = sqrt(f_gb_sq);

    // Compute derivatives if requested
    if (df_dRi || df_dRj) {
        // Common term for both derivatives
        // ∂(Rᵢ*Rⱼ*exp(...))/∂Rᵢ = Rⱼ*exp(...) * (1 + r²/(4*Rᵢ*Rⱼ))
        //                        = Rⱼ*exp(...) + r²*exp(...)/(4*Rᵢ)
        double exp_term = exp_val;
        double r_sq_term = r_sq / (4.0 * RiRj);

        // ∂f_GB/∂Rᵢ = 0.5 / f_GB * ∂(r² + Rᵢ*Rⱼ*exp(...))/∂Rᵢ
        //           = 0.5 / f_GB * Rⱼ*exp(...) * (1 + r²/(4*Rᵢ*Rⱼ))
        //           = 0.5 / f_GB * (Rⱼ*exp + r²*exp/(4*Rᵢ))
        if (df_dRi) {
            double dRiRj_exp_dRi = R_j * exp_term * (1.0 + r_sq_term);
            *df_dRi = 0.5 / (*f_gb) * dRiRj_exp_dRi;
        }

        // By symmetry, ∂f_GB/∂Rⱼ has the same form with i and j swapped
        if (df_dRj) {
            double dRiRj_exp_dRj = R_i * exp_term * (1.0 + r_sq_term);
            *df_dRj = 0.5 / (*f_gb) * dRiRj_exp_dRj;
        }
    }
}

/**
 * Kernel to compute ∂E/∂Rᵢ for all atoms.
 *
 * This computes the complete energy derivative with respect to Born radius:
 *   ∂E/∂Rᵢ = self-energy term + sum of pairwise terms
 *
 * Where:
 *   Self: -qᵢ²/Rᵢ²
 *   Pair: Σⱼ qᵢqⱼ * (-1/f_GB²) * (∂f_GB/∂Rᵢ)
 */
__global__ void compute_dE_dR(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ charges,
    const double* __restrict__ born_radii,
    double dielectric,
    double cutoff,
    double* __restrict__ dE_dR  // Output: ∂E/∂Rᵢ for each atom
) {
    extern __shared__ double s_data_dE[];
    double* s_coords_dE = s_data_dE;
    double* s_charges_dE = &s_data_dE[blockDim.x * 3];
    double* s_born_radii_dE = &s_data_dE[blockDim.x * 4];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    const double COULOMB_CONST = 332.0636;
    double gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST;

    double xi, yi, zi, qi, R_i;
    if (i < natoms) {
        xi = coords[i * 3 + 0];
        yi = coords[i * 3 + 1];
        zi = coords[i * 3 + 2];
        qi = charges[i];
        R_i = born_radii[i];
    }

    // Initialize ∂E/∂Rᵢ with self-energy contribution
    // Self-energy: E_self = gb_factor * q_i² / R_i
    // So: ∂E_self/∂R_i = gb_factor * q_i² * (-1/R_i²) = -gb_factor * q_i² / R_i²
    //
    // We will add ALL pairwise contributions below
    double dE_dRi = 0.0;
    double dE_dRi_self = 0.0;
    if (i < natoms) {
        dE_dRi_self = gb_factor * (-qi * qi / (R_i * R_i));
        dE_dRi = dE_dRi_self;

        // DEBUG: Print computed value for atoms 0 and 1
        if (i == 0 || i == 1) {
            printf("GPU[compute_dE_dR]: atom %d: dE_self/dRi = %.8f\n", i, dE_dRi_self);
            printf("GPU[compute_dE_dR]: atom %d: qi=%.6f, R_i=%.6f, gb_factor=%.6f\n", i, qi, R_i, gb_factor);
        }
    }

    double cutoff_sq = cutoff * cutoff;
    int num_tiles = (natoms + blockDim.x - 1) / blockDim.x;

    // Accumulate pairwise contributions to ∂E/∂R_i
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * blockDim.x;
        int load_idx = tile_start + tid;

        if (load_idx < natoms) {
            s_coords_dE[tid * 3 + 0] = coords[load_idx * 3 + 0];
            s_coords_dE[tid * 3 + 1] = coords[load_idx * 3 + 1];
            s_coords_dE[tid * 3 + 2] = coords[load_idx * 3 + 2];
            s_charges_dE[tid] = charges[load_idx];
            s_born_radii_dE[tid] = born_radii[load_idx];
        }
        __syncthreads();

        if (i < natoms) {
            int tile_size = min((int)blockDim.x, natoms - tile_start);

            for (int t = 0; t < tile_size; t++) {
                int j = tile_start + t;
                if (i == j) continue;

                double xj = s_coords_dE[t * 3 + 0];
                double yj = s_coords_dE[t * 3 + 1];
                double zj = s_coords_dE[t * 3 + 2];
                double qj = s_charges_dE[t];
                double R_j = s_born_radii_dE[t];

                double dx = xi - xj;
                double dy = yi - yj;
                double dz = zi - zj;
                double r_sq = dx * dx + dy * dy + dz * dz;

                if (r_sq > cutoff_sq) continue;

                double r = sqrt(r_sq);

                // Compute f_GB and ∂f_GB/∂Rᵢ
                double f_gb, df_gb_dRi;
                compute_f_gb_and_derivatives(r, R_i, R_j, &f_gb, &df_gb_dRi, NULL);

                // Pairwise contribution to ∂E/∂Rᵢ
                // NOTE: No factor of 0.5 here! Even though each pair (i,j) is visited twice
                // (once by thread i, once by thread j), the derivatives ∂E/∂R_i and ∂E/∂R_j
                // are DIFFERENT, so we need the full contribution for each.
                dE_dRi += gb_factor * qi * qj * (-1.0 / (f_gb * f_gb)) * df_gb_dRi;
            }
        }
        __syncthreads();
    }

    if (i < natoms) {
        dE_dR[i] = dE_dRi;

        // DEBUG: Print final value being written for atoms 0 and 1
        if (i == 0 || i == 1) {
            printf("GPU[compute_dE_dR]: atom %d: TOTAL dE_dR[%d] = %.8f (self=%.8f, pairwise=%.8f)\n",
                   i, i, dE_dRi, dE_dRi_self, dE_dRi - dE_dRi_self);
        }
    }
}

/**
 * REDUCTION kernel to convert ∂E/∂R to ∂E/∂ψ (OpenMM multi-pass approach).
 *
 * This kernel implements the same operation as OpenMM's reduceBornForce:
 *   ∂E/∂ψᵢ = (∂E/∂Rᵢ) × Rᵢ² × obcChain
 *
 * Where obcChain = ∂Rᵢ/∂ψᵢ is the OBC chain rule derivative.
 *
 * This separates the chain rule into two steps:
 * 1. This kernel: ∂E/∂R → ∂E/∂ψ
 * 2. Force kernel: ∂E/∂ψ × ∂ψ/∂r → F
 */
__global__ void reduce_born_force(
    int natoms,
    const double* __restrict__ dE_dR,           // Input: ∂E/∂Rᵢ
    const double* __restrict__ born_radii,      // Input: Rᵢ
    const double* __restrict__ intrinsic_radii, // Input: ρᵢ
    const double* __restrict__ b_params,        // Input: b (OBC parameter)
    const double* __restrict__ c_params,        // Input: c (OBC parameter)
    const double* __restrict__ psi_sum,         // Input: ψᵢ
    double* __restrict__ dE_dpsi                // Output: ∂E/∂ψᵢ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= natoms) return;

    // Load data for atom i
    double dE_dR_i = dE_dR[i];
    double R_i = born_radii[i];
    double rho_i = intrinsic_radii[i];
    double psi_i = psi_sum[i];
    double b_i = b_params[i];
    double c_i = c_params[i];

    // Compute obcChain = ∂(1/R_i)/∂ψᵢ using OBC formula (matches OpenMM)
    double obcChain = born_radius_derivative_wrt_psi(R_i, rho_i, psi_i, b_i, c_i);

    // Convert: ∂E/∂ψᵢ = (∂E/∂Rᵢ) × (∂R_i/∂ψᵢ)
    //                 = (∂E/∂Rᵢ) × (-R_i²) × (∂(1/R_i)/∂ψᵢ)
    //                 = (∂E/∂Rᵢ) × (-R_i²) × obcChain
    //
    // Trying POSITIVE sign to match OpenMM (they use: force *= R²×obcChain)
    double dE_dpsi_i = dE_dR_i * R_i * R_i * obcChain;

    // Write result
    dE_dpsi[i] = dE_dpsi_i;

    // DEBUG: Print for first two atoms
    if (i == 0 || i == 1) {
        printf("GPU[reduce_born_force]: atom %d: dE_dR=%.8f, R=%.8f, obcChain=%.8f, dE_dpsi=%.8f\n",
               i, dE_dR_i, R_i, obcChain, dE_dpsi_i);
    }
}

/**
 * OPTIMIZED kernel to compute Born radii derivative forces using shared memory tiling.
 *
 * This computes the force contribution from Born radii derivatives:
 *   F_born = -∑ᵢ (∂E/∂Rᵢ) * (∂Rᵢ/∂r)
 *
 * This is the MISSING term in the previous implementation!
 *
 * Strategy:
 * - Each thread handles one atom i
 * - Use pre-computed ∂E/∂Rᵢ from dE_dR array
 * - For each atom j in tiles:
 *   - Compute ∂ψᵢ/∂rᵢⱼ (descreening derivative)
 *   - Compute ∂Rᵢ/∂ψᵢ (Born radius derivative)
 *   - Accumulate force: (∂E/∂Rᵢ) * (∂Rᵢ/∂ψᵢ) * (∂ψᵢ/∂rᵢⱼ) * (rᵢⱼ/|rᵢⱼ|)
 *
 * Uses modern CUDA features:
 * - Shared memory tiling for coalesced access
 * - Warp shuffle for efficient reductions (future)
 * - Minimize atomic operations
 */
__global__ void compute_born_radii_forces_tiled(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ charges,
    const double* __restrict__ born_radii,
    const double* __restrict__ intrinsic_radii,
    const double* __restrict__ b_params,
    const double* __restrict__ c_params,
    const double* __restrict__ psi_sum,  // Descreening sum for each atom
    const double* __restrict__ dE_dR,     // Pre-computed ∂E/∂Rᵢ for each atom
    double dielectric,
    double cutoff,
    double* __restrict__ born_forces  // Output: Born radii force contribution [natoms, 3]
) {
    // Shared memory for atom tile
    extern __shared__ double s_data[];
    double* s_coords = s_data;                       // [TILE_SIZE * 3]
    double* s_charges = &s_data[blockDim.x * 3];    // [TILE_SIZE]
    double* s_born_radii = &s_data[blockDim.x * 4]; // [TILE_SIZE]
    double* s_intrinsic_radii = &s_data[blockDim.x * 5];  // [TILE_SIZE]
    double* s_b_params = &s_data[blockDim.x * 6];   // [TILE_SIZE]
    double* s_c_params = &s_data[blockDim.x * 7];   // [TILE_SIZE]
    double* s_psi_sum = &s_data[blockDim.x * 8];    // [TILE_SIZE]
    double* s_dE_dR = &s_data[blockDim.x * 9];      // [TILE_SIZE]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load atom i data
    double xi, yi, zi, qi, R_i, rho_i, b_i, c_i, psi_i, dE_dR_i;
    if (i < natoms) {
        xi = coords[i * 3 + 0];
        yi = coords[i * 3 + 1];
        zi = coords[i * 3 + 2];
        qi = charges[i];
        R_i = born_radii[i];
        rho_i = intrinsic_radii[i];
        b_i = b_params[i];
        c_i = c_params[i];
        psi_i = psi_sum[i];
        dE_dR_i = dE_dR[i];  // Pre-computed self-energy ∂E/∂Rᵢ only

        // DEBUG: Print loaded value for atom 0
        if (i == 0) {
            printf("GPU[force_kernel]: atom 0: LOADED dE_dR_i = %.8f (self + all pairwise)\n", dE_dR_i);
            printf("GPU[force_kernel]: atom 0: qi=%.6f, R_i=%.6f, rho_i=%.6f, psi_i=%.6f\n", qi, R_i, rho_i, psi_i);
        }
    }

    // Accumulators for force on atom i from Born radii derivatives
    double fx_born_i = 0.0;
    double fy_born_i = 0.0;
    double fz_born_i = 0.0;

    double cutoff_sq = cutoff * cutoff;

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
            s_born_radii[tid] = born_radii[load_idx];
            s_intrinsic_radii[tid] = intrinsic_radii[load_idx];
            s_b_params[tid] = b_params[load_idx];
            s_c_params[tid] = c_params[load_idx];
            s_psi_sum[tid] = psi_sum[load_idx];
            s_dE_dR[tid] = dE_dR[load_idx];
        }
        __syncthreads();

        // Compute Born radii force contributions from this tile
        if (i < natoms) {
            int tile_size = min((int)blockDim.x, natoms - tile_start);

            for (int t = 0; t < tile_size; t++) {
                int j = tile_start + t;

                // Skip self-interaction
                if (i == j) continue;

                // Load atom j data from shared memory
                double xj = s_coords[t * 3 + 0];
                double yj = s_coords[t * 3 + 1];
                double zj = s_coords[t * 3 + 2];
                double R_j = s_born_radii[t];
                double rho_j = s_intrinsic_radii[t];
                double b_j = s_b_params[t];
                double c_j = s_c_params[t];
                double psi_j = s_psi_sum[t];
                double dE_dR_j = s_dE_dR[t];  // Pre-computed ∂E/∂Rⱼ

                // Compute distance
                double dx = xi - xj;
                double dy = yi - yj;
                double dz = zi - zj;
                double r_sq = dx * dx + dy * dy + dz * dz;

                // Apply cutoff
                if (r_sq > cutoff_sq) continue;

                double r = sqrt(r_sq);
                double r_inv = 1.0 / r;

                // CONTRIBUTION 1: Force from R_i changing
                // Compute ∂ψᵢ/∂rᵢⱼ (how i's descreening changes with distance)
                double dpsi_i_dr = descreening_integral_derivative(r, rho_i, rho_j);
                double force_mag_Ri = 0.0;

                if (fabs(dpsi_i_dr) > 1e-12) {
                    // Compute ∂Rᵢ/∂ψᵢ
                    double dR_i_dpsi = born_radius_derivative_wrt_psi(R_i, rho_i, psi_i, b_i, c_i);

                    // Chain rule: ∂Rᵢ/∂rᵢⱼ
                    double dR_i_dr = dR_i_dpsi * dpsi_i_dr;

                    // Force magnitude from R_i chain
                    force_mag_Ri = -dE_dR_i * dR_i_dr;
                }

                // CONTRIBUTION 2: Force from R_j changing
                // Compute ∂ψⱼ/∂rᵢⱼ (how j's descreening changes with distance)
                double dpsi_j_dr = descreening_integral_derivative(r, rho_j, rho_i);
                double force_mag_Rj = 0.0;

                if (fabs(dpsi_j_dr) > 1e-12) {
                    // Compute ∂Rⱼ/∂ψⱼ
                    double dR_j_dpsi = born_radius_derivative_wrt_psi(R_j, rho_j, psi_j, b_j, c_j);

                    // Chain rule: ∂Rⱼ/∂rᵢⱼ
                    double dR_j_dr = dR_j_dpsi * dpsi_j_dr;

                    // Force magnitude from R_j chain
                    force_mag_Rj = -dE_dR_j * dR_j_dr;
                }

                // TOTAL Born force magnitude (both chains)
                double force_mag_total = force_mag_Ri + force_mag_Rj;

                // DEBUG: Print for first pair (0,1)
                if (i == 0 && j == 1) {
                    double dR_i_dpsi_debug = born_radius_derivative_wrt_psi(R_i, rho_i, psi_i, b_i, c_i);
                    double dR_j_dpsi_debug = born_radius_derivative_wrt_psi(R_j, rho_j, psi_j, b_j, c_j);
                    double dR_i_dr_debug = dR_i_dpsi_debug * dpsi_i_dr;
                    double dR_j_dr_debug = dR_j_dpsi_debug * dpsi_j_dr;

                    printf("GPU[force_kernel]: pair (0,1): dE_dR_i=%.8f, dE_dR_j=%.8f\n", dE_dR_i, dE_dR_j);
                    printf("GPU[force_kernel]: pair (0,1): dpsi_i_dr=%.8f, dpsi_j_dr=%.8f\n", dpsi_i_dr, dpsi_j_dr);
                    printf("GPU[force_kernel]: pair (0,1): dR_i_dpsi=%.8f, dR_j_dpsi=%.8f\n", dR_i_dpsi_debug, dR_j_dpsi_debug);
                    printf("GPU[force_kernel]: pair (0,1): dR_i_dr=%.8f, dR_j_dr=%.8f\n", dR_i_dr_debug, dR_j_dr_debug);
                    printf("GPU[force_kernel]: pair (0,1): force_mag_Ri=%.8f, force_mag_Rj=%.8f\n", force_mag_Ri, force_mag_Rj);
                    printf("GPU[force_kernel]: pair (0,1): force_mag_total=%.8f\n", force_mag_total);
                }

                // Accumulate force on atom i
                // R_i contribution: original sign (worked for 2-atom test)
                // R_j contribution: OPPOSITE sign (different chain rule direction)
                fx_born_i += force_mag_Ri * dx * r_inv;
                fy_born_i += force_mag_Ri * dy * r_inv;
                fz_born_i += force_mag_Ri * dz * r_inv;

                fx_born_i -= force_mag_Rj * dx * r_inv;
                fy_born_i -= force_mag_Rj * dy * r_inv;
                fz_born_i -= force_mag_Rj * dz * r_inv;
            }
        }
        __syncthreads();
    }

    // Write Born radii force contribution for atom i
    // No atomics needed - each thread writes to its own unique location
    if (i < natoms) {
        born_forces[i * 3 + 0] = fx_born_i;
        born_forces[i * 3 + 1] = fy_born_i;
        born_forces[i * 3 + 2] = fz_born_i;

        // DEBUG: Print final force for atom 0
        if (i == 0) {
            printf("GPU[force_kernel]: atom 0: FINAL born_forces = [%.8f, %.8f, %.8f]\n",
                   fx_born_i, fy_born_i, fz_born_i);
        }
    }
}

/**
 * SIMPLIFIED kernel to apply Born radius forces using ∂E/∂ψ (OpenMM multi-pass approach).
 *
 * This is the final step in OpenMM's multi-pass architecture:
 *   F_born = -Σⱼ (∂E/∂ψᵢ) × (∂ψᵢ/∂rᵢⱼ) × (rᵢ - rⱼ)/r
 *
 * Input: ∂E/∂ψᵢ (already includes R² × obcChain from reduce_born_force)
 * Output: Born radius derivative forces
 *
 * This is MUCH simpler than the previous kernel because we don't need to:
 * - Compute ∂Rᵢ/∂ψᵢ (already done in reduce_born_force)
 * - Handle R_j contributions separately (already in ∂E/∂ψ via dE/dR calculation)
 */
__global__ void apply_born_forces_tiled(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ intrinsic_radii,
    const double* __restrict__ dE_dpsi,  // Input: ∂E/∂ψᵢ for each atom
    double cutoff,
    double* __restrict__ born_forces  // Output: Born radii force contribution [natoms, 3]
) {
    // Shared memory for tile (only need coords and radii and dE_dpsi)
    extern __shared__ double s_data[];
    double* s_coords = s_data;                    // [TILE_SIZE * 3]
    double* s_intrinsic_radii = &s_data[blockDim.x * 3];  // [TILE_SIZE]
    double* s_dE_dpsi = &s_data[blockDim.x * 4];          // [TILE_SIZE]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load atom i data
    double xi, yi, zi, rho_i, dE_dpsi_i;
    if (i < natoms) {
        xi = coords[i * 3 + 0];
        yi = coords[i * 3 + 1];
        zi = coords[i * 3 + 2];
        rho_i = intrinsic_radii[i];
        dE_dpsi_i = dE_dpsi[i];

        // DEBUG: Print loaded value for atom 0
        if (i == 0) {
            printf("GPU[apply_born_forces]: atom 0: LOADED dE_dpsi_i = %.8f\n", dE_dpsi_i);
            printf("GPU[apply_born_forces]: atom 0: rho_i=%.6f\n", rho_i);
        }
    }

    // Accumulators for force on atom i
    double fx_born_i = 0.0;
    double fy_born_i = 0.0;
    double fz_born_i = 0.0;

    double cutoff_sq = cutoff * cutoff;

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
            s_intrinsic_radii[tid] = intrinsic_radii[load_idx];
            s_dE_dpsi[tid] = dE_dpsi[load_idx];
        }
        __syncthreads();

        // Compute Born radii force contributions from this tile
        if (i < natoms) {
            int tile_size = min((int)blockDim.x, natoms - tile_start);

            for (int t = 0; t < tile_size; t++) {
                int j = tile_start + t;

                // Skip self-interaction
                if (i == j) continue;

                // Load atom j data from shared memory
                double xj = s_coords[t * 3 + 0];
                double yj = s_coords[t * 3 + 1];
                double zj = s_coords[t * 3 + 2];
                double rho_j = s_intrinsic_radii[t];

                // Compute displacement vector (j - i), matching OpenMM's getDeltaR
                // OpenMM uses: getDeltaR(atomI, atomJ) = atomJ - atomI
                double dx = xj - xi;
                double dy = yj - yi;
                double dz = zj - zi;
                double r_sq = dx * dx + dy * dy + dz * dz;

                // Apply cutoff
                if (r_sq > cutoff_sq) continue;

                double r = sqrt(r_sq);
                double r_inv = 1.0 / r;

                // Compute ∂ψᵢ/∂rᵢⱼ (derivative of descreening integral for atom i)
                // This tells us how ψᵢ changes when distance r changes
                double dpsi_i_dr = descreening_integral_derivative(r, rho_i, rho_j);

                // Force magnitude from ψᵢ changing (matches OpenMM's approach):
                // de = (∂E/∂ψᵢ) × (∂ψᵢ/∂r) / r
                //
                // OpenMM computes: de = bornForces[i] * t3 * r_inv
                // where bornForces[i] already contains ∂E/∂ψᵢ and t3/r = ∂ψᵢ/∂r
                //
                // This force is applied to BOTH atoms (Newton's 3rd law):
                // - Subtract from atom i: forces[i] -= de × (rⱼ - rᵢ)
                // - Add to atom j: forces[j] += de × (rⱼ - rᵢ)
                double de = dE_dpsi_i * dpsi_i_dr * r_inv;

                // Displacement vector components (already computed as dx, dy, dz = rᵢ - rⱼ)
                double force_x = de * dx;
                double force_y = de * dy;
                double force_z = de * dz;

                // DEBUG: Print for first pair (0,1)
                if (i == 0 && j == 1) {
                    printf("GPU[apply_born_forces]: pair (0,1): dE_dpsi_i=%.8f\n", dE_dpsi_i);
                    printf("GPU[apply_born_forces]: pair (0,1): dpsi_i_dr=%.8f\n", dpsi_i_dr);
                    printf("GPU[apply_born_forces]: pair (0,1): de=%.8f\n", de);
                    printf("GPU[apply_born_forces]: pair (0,1): force=%.8f, %.8f, %.8f\n", force_x, force_y, force_z);
                }

                // Accumulate force on atom i (subtract)
                fx_born_i -= force_x;
                fy_born_i -= force_y;
                fz_born_i -= force_z;

                // Apply force to atom j (add) using atomicAdd since j is in shared memory tile
                // Note: This creates equal and opposite forces on the two atoms
                atomicAdd(&born_forces[j * 3 + 0], force_x);
                atomicAdd(&born_forces[j * 3 + 1], force_y);
                atomicAdd(&born_forces[j * 3 + 2], force_z);
            }
        }
        __syncthreads();
    }

    // Write Born radii force contribution for atom i using atomicAdd
    // (since other threads may have added forces to this atom via Newton's 3rd law)
    if (i < natoms) {
        atomicAdd(&born_forces[i * 3 + 0], fx_born_i);
        atomicAdd(&born_forces[i * 3 + 1], fy_born_i);
        atomicAdd(&born_forces[i * 3 + 2], fz_born_i);

        // DEBUG: Print final force for atom 0
        if (i == 0) {
            printf("GPU[apply_born_forces]: atom 0: FINAL born_forces = [%.8f, %.8f, %.8f]\n",
                   fx_born_i, fy_born_i, fz_born_i);
        }
    }
}

/**
 * Kernel to add two arrays: a += b
 */
__global__ void add_arrays_kernel(int n, double* a, const double* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

/**
 * Host function to compute complete GB forces including Born radii derivatives.
 *
 * This adds the missing Born radii derivative term to make forces accurate!
 */
void compute_gb_forces_complete(
    int natoms,
    const double* coords,
    const double* charges,
    const double* born_radii,
    const double* intrinsic_radii,
    const double* b_params,
    const double* c_params,
    const double* psi_sum,
    double dielectric,
    double cutoff,
    double* energy,
    double* forces  // Will contain: direct forces + Born radii forces
) {
    // Step 1: Compute direct pairwise forces (already implemented)
    compute_gb_energy_forces(
        natoms,
        coords,
        charges,
        born_radii,
        dielectric,
        cutoff,
        energy,
        forces
    );

    // Step 2: Pre-compute ∂E/∂Rᵢ for all atoms
    double* dE_dR;
    CUDA_CHECK(cudaMalloc(&dE_dR, natoms * sizeof(double)));

    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Shared memory for compute_dE_dR: coords[256*3] + charges[256] + born_radii[256]
    // Total: 256 * (3 + 1 + 1) = 256 * 5 doubles
    int shared_mem_size_dE_dR = threads_per_block * 5 * sizeof(double);

    compute_dE_dR<<<num_blocks, threads_per_block, shared_mem_size_dE_dR>>>(
        natoms,
        coords,
        charges,
        born_radii,
        dielectric,
        cutoff,
        dE_dR
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 3: Compute Born radii derivative forces using pre-computed dE_dR
    double* born_forces;
    CUDA_CHECK(cudaMalloc(&born_forces, natoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(born_forces, 0, natoms * 3 * sizeof(double)));

    // Shared memory for force kernel:
    // coords[256*3] + charges[256] + born_radii[256] + intrinsic_radii[256] + b_params[256] + c_params[256] + psi_sum[256] + dE_dR[256]
    // Total: 256 * (3 + 1 + 1 + 1 + 1 + 1 + 1 + 1) = 256 * 10 doubles
    int shared_mem_size = threads_per_block * 10 * sizeof(double);

    compute_born_radii_forces_tiled<<<num_blocks, threads_per_block, shared_mem_size>>>(
        natoms,
        coords,
        charges,
        born_radii,
        intrinsic_radii,
        b_params,
        c_params,
        psi_sum,
        dE_dR,  // Pass pre-computed ∂E/∂Rᵢ
        dielectric,
        cutoff,
        born_forces
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 4: Add Born radii forces to direct forces on GPU
    // F_total = F_direct + F_born
    int total_elements = natoms * 3;
    int blocks_add = (total_elements + 255) / 256;

    add_arrays_kernel<<<blocks_add, 256>>>(total_elements, forces, born_forces);
    CUDA_CHECK(cudaGetLastError());

    // Clean up
    CUDA_CHECK(cudaFree(dE_dR));
    CUDA_CHECK(cudaFree(born_forces));
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Host function to compute ∂E/∂R for each atom (OpenMM multi-pass step 2).
 */
void compute_dE_dR_host(
    int natoms,
    const double* coords,
    const double* charges,
    const double* born_radii,
    double dielectric,
    double cutoff,
    double* dE_dR  // Output
) {
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Shared memory: coords[256*3] + charges[256] + born_radii[256]
    int shared_mem_size = threads_per_block * 5 * sizeof(double);

    compute_dE_dR<<<num_blocks, threads_per_block, shared_mem_size>>>(
        natoms,
        coords,
        charges,
        born_radii,
        dielectric,
        cutoff,
        dE_dR
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Host function to convert ∂E/∂R to ∂E/∂ψ (OpenMM multi-pass step 3).
 */
void reduce_born_force_host(
    int natoms,
    const double* dE_dR,
    const double* born_radii,
    const double* intrinsic_radii,
    const double* b_params,
    const double* c_params,
    const double* psi_sum,
    double* dE_dpsi  // Output
) {
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    reduce_born_force<<<num_blocks, threads_per_block>>>(
        natoms, dE_dR, born_radii, intrinsic_radii, b_params, c_params, psi_sum, dE_dpsi
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Host function to apply Born radius forces using ∂E/∂ψ (OpenMM multi-pass step 4).
 */
void apply_born_forces_host(
    int natoms,
    const double* coords,
    const double* intrinsic_radii,
    const double* dE_dpsi,
    double cutoff,
    double* born_forces  // Output
) {
    int threads_per_block = 256;
    int num_blocks = (natoms + threads_per_block - 1) / threads_per_block;

    // Shared memory size: 5 doubles per thread (coords[3] + radius + dE_dpsi)
    int shared_mem_size = threads_per_block * 5 * sizeof(double);

    apply_born_forces_tiled<<<num_blocks, threads_per_block, shared_mem_size>>>(
        natoms, coords, intrinsic_radii, dE_dpsi, cutoff, born_forces
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace implicit_solvent
} // namespace cuda
} // namespace fennol
