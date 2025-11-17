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
 * Compute GB electrostatic energy and forces
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param charges Partial atomic charges [natoms]
 * @param born_radii Effective Born radii [natoms]
 * @param dielectric Solvent dielectric constant
 * @param cutoff Cutoff distance
 * @param energy Output: GB electrostatic energy [1]
 * @param forces Output: forces on atoms [natoms, 3]
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

} // namespace implicit_solvent
} // namespace cuda
} // namespace fennol

#endif // IMPLICIT_SOLVENT_CUH
