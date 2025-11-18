#ifndef IMPLICIT_SOLVENT_CUH
#define IMPLICIT_SOLVENT_CUH

#include <cuda_runtime.h>

namespace fennol {
namespace cuda {
namespace implicit_solvent {

/**
 * Compute Born radii using OBC (Onufriev-Bashford-Case) model
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param intrinsic_radii Intrinsic atomic radii (ρ) [natoms]
 * @param b_params OBC b parameters [natoms]
 * @param c_params OBC c parameters [natoms]
 * @param cutoff Cutoff distance for interactions
 * @param born_radii Output: effective Born radii [natoms]
 */
void compute_born_radii_obc(
    int natoms,
    const double* coords,
    const double* intrinsic_radii,
    const double* b_params,
    const double* c_params,
    double cutoff,
    double* born_radii
);

/**
 * Compute Born radii and return descreening sum (needed for accurate forces)
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param intrinsic_radii Intrinsic atomic radii (ρ) [natoms]
 * @param b_params OBC b parameters [natoms]
 * @param c_params OBC c parameters [natoms]
 * @param cutoff Cutoff distance for interactions
 * @param born_radii Output: effective Born radii [natoms]
 * @param psi_sum Output: descreening sum [natoms]
 */
void compute_born_radii_obc_with_psi(
    int natoms,
    const double* coords,
    const double* intrinsic_radii,
    const double* b_params,
    const double* c_params,
    double cutoff,
    double* born_radii,
    double* psi_sum
);

/**
 * Compute GB electrostatic energy and forces (INCOMPLETE - missing Born radii derivatives)
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param charges Partial atomic charges [natoms]
 * @param born_radii Effective Born radii [natoms]
 * @param dielectric Solvent dielectric constant
 * @param cutoff Cutoff distance
 * @param energy Output: GB electrostatic energy [1]
 * @param forces Output: forces on atoms [natoms, 3] (INCOMPLETE!)
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
);

/**
 * Compute COMPLETE GB electrostatic energy and forces including Born radii derivatives
 *
 * This is the CORRECT version that includes all force terms:
 *   F = F_direct + F_born_radii_derivatives
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param charges Partial atomic charges [natoms]
 * @param born_radii Effective Born radii [natoms]
 * @param intrinsic_radii Intrinsic atomic radii [natoms]
 * @param b_params OBC b parameters [natoms]
 * @param c_params OBC c parameters [natoms]
 * @param psi_sum Descreening sum [natoms]
 * @param dielectric Solvent dielectric constant
 * @param cutoff Cutoff distance
 * @param energy Output: GB electrostatic energy [1]
 * @param forces Output: COMPLETE forces on atoms [natoms, 3]
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
    double* forces
);

/**
 * Compute non-polar (surface area) energy and forces
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param born_radii Effective Born radii [natoms]
 * @param gamma_params Surface tension coefficients [natoms]
 * @param probe_radius Solvent probe radius
 * @param energy Output: non-polar energy [1]
 * @param forces Output: forces on atoms [natoms, 3]
 */
void compute_nonpolar_sasa(
    int natoms,
    const double* coords,
    const double* born_radii,
    const double* gamma_params,
    double probe_radius,
    double* energy,
    double* forces
);

/**
 * Compute ∂E/∂R for each atom (OpenMM multi-pass approach, step 2)
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param charges Partial atomic charges [natoms]
 * @param born_radii Born radii [natoms]
 * @param dielectric Solvent dielectric constant
 * @param cutoff Cutoff distance
 * @param dE_dR Output: ∂E/∂Rᵢ for each atom [natoms]
 */
void compute_dE_dR_host(
    int natoms,
    const double* coords,
    const double* charges,
    const double* born_radii,
    double dielectric,
    double cutoff,
    double* dE_dR
);

/**
 * Convert ∂E/∂R to ∂E/∂ψ (OpenMM multi-pass approach, step 3)
 *
 * @param natoms Number of atoms
 * @param dE_dR Input: ∂E/∂Rᵢ for each atom [natoms]
 * @param born_radii Born radii [natoms]
 * @param intrinsic_radii Intrinsic radii [natoms]
 * @param b_params OBC b parameters [natoms]
 * @param c_params OBC c parameters [natoms]
 * @param psi_sum Descreening sum [natoms]
 * @param dE_dpsi Output: ∂E/∂ψᵢ for each atom [natoms]
 */
void reduce_born_force_host(
    int natoms,
    const double* dE_dR,
    const double* born_radii,
    const double* intrinsic_radii,
    const double* b_params,
    const double* c_params,
    const double* psi_sum,
    double* dE_dpsi
);

/**
 * Apply Born radius forces using ∂E/∂ψ (OpenMM multi-pass approach, step 4)
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param intrinsic_radii Intrinsic radii [natoms]
 * @param dE_dpsi ∂E/∂ψᵢ for each atom [natoms]
 * @param cutoff Cutoff distance
 * @param born_forces Output: Born radius derivative forces [natoms, 3]
 */
void apply_born_forces_host(
    int natoms,
    const double* coords,
    const double* intrinsic_radii,
    const double* dE_dpsi,
    double cutoff,
    double* born_forces
);

} // namespace implicit_solvent
} // namespace cuda
} // namespace fennol

#endif // IMPLICIT_SOLVENT_CUH
