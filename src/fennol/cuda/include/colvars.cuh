#ifndef FENNOL_CUDA_COLVARS_CUH
#define FENNOL_CUDA_COLVARS_CUH

#include "common.cuh"

namespace fennol {
namespace cuda {

/**
 * Compute distance collective variable
 * CV = ||r_j - r_i||
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_i - first atom index
 * @param atom_j - second atom index
 * @param value [out] - distance value
 * @param gradient [natoms, 3, out] - gradient of CV w.r.t. coordinates
 */
void colvar_distance(
    const double* coordinates,
    int atom_i,
    int atom_j,
    int natoms,
    double* value,
    double* gradient
);

/**
 * Compute angle collective variable
 * CV = angle(r_i - r_j, r_k - r_j)
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_i - first atom index
 * @param atom_j - central atom index
 * @param atom_k - third atom index
 * @param natoms - number of atoms
 * @param value [out] - angle value (radians)
 * @param gradient [natoms, 3, out] - gradient of CV w.r.t. coordinates
 */
void colvar_angle(
    const double* coordinates,
    int atom_i,
    int atom_j,
    int atom_k,
    int natoms,
    double* value,
    double* gradient
);

/**
 * Compute dihedral collective variable
 * CV = dihedral(r_i, r_j, r_k, r_l)
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_i - first atom index
 * @param atom_j - second atom index
 * @param atom_k - third atom index
 * @param atom_l - fourth atom index
 * @param natoms - number of atoms
 * @param value [out] - dihedral value (radians)
 * @param gradient [natoms, 3, out] - gradient of CV w.r.t. coordinates
 */
void colvar_dihedral(
    const double* coordinates,
    int atom_i,
    int atom_j,
    int atom_k,
    int atom_l,
    int natoms,
    double* value,
    double* gradient
);

/**
 * Compute center of mass
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param masses [natoms] - atomic masses
 * @param atom_indices [n_selected] - indices of atoms to include
 * @param n_selected - number of selected atoms
 * @param com [3, out] - center of mass
 */
void compute_center_of_mass(
    const double* coordinates,
    const double* masses,
    const int* atom_indices,
    int n_selected,
    double* com
);

/**
 * Compute RMSD between two structures
 * Performs optimal alignment (Kabsch algorithm) and returns RMSD
 *
 * @param coordinates [natoms, 3] - current coordinates
 * @param reference [natoms, 3] - reference coordinates
 * @param masses [natoms] - atomic masses (for weighted RMSD)
 * @param atom_indices [n_selected] - indices of atoms to include
 * @param n_selected - number of selected atoms
 * @param use_mass_weighting - whether to use mass-weighted RMSD
 * @param rmsd [out] - RMSD value
 * @param gradient [natoms, 3, out] - gradient of RMSD w.r.t. coordinates
 */
void colvar_rmsd(
    const double* coordinates,
    const double* reference,
    const double* masses,
    const int* atom_indices,
    int n_selected,
    int natoms,
    bool use_mass_weighting,
    double* rmsd,
    double* gradient
);

} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_COLVARS_CUH
