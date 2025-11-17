#ifndef FENNOL_CUDA_RESTRAINTS_CUH
#define FENNOL_CUDA_RESTRAINTS_CUH

#include "common.cuh"

namespace fennol {
namespace cuda {

/**
 * Harmonic distance restraint
 * E = 0.5 * k * (r - r0)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_indices [nrestraints, 2] - pairs of atom indices
 * @param target_distances [nrestraints] - target distances
 * @param force_constants [nrestraints] - force constants
 * @param natoms - number of atoms
 * @param nrestraints - number of restraints
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void harmonic_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
);

/**
 * One-sided lower distance restraint
 * E = 0.5 * k * max(0, r0 - r)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_indices [nrestraints, 2] - pairs of atom indices
 * @param target_distances [nrestraints] - minimum allowed distances
 * @param force_constants [nrestraints] - force constants
 * @param natoms - number of atoms
 * @param nrestraints - number of restraints
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void lower_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
);

/**
 * One-sided upper distance restraint
 * E = 0.5 * k * max(0, r - r0)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_indices [nrestraints, 2] - pairs of atom indices
 * @param target_distances [nrestraints] - maximum allowed distances
 * @param force_constants [nrestraints] - force constants
 * @param natoms - number of atoms
 * @param nrestraints - number of restraints
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void upper_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
);

/**
 * Flat-bottom distance restraint
 * E = 0.5 * k * max(0, |r - r0| - tolerance)^2
 * No energy penalty within target ± tolerance, harmonic outside
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_indices [nrestraints, 2] - pairs of atom indices
 * @param target_distances [nrestraints] - target distances
 * @param force_constants [nrestraints] - force constants
 * @param tolerances [nrestraints] - tolerance (half-width of flat region)
 * @param natoms - number of atoms
 * @param nrestraints - number of restraints
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void flat_bottom_distance_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_distances,
    const double* force_constants,
    const double* tolerances,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
);

/**
 * Harmonic angle restraint
 * E = 0.5 * k * (theta - theta0)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_indices [nrestraints, 3] - triplets of atom indices
 * @param target_angles [nrestraints] - target angles (radians)
 * @param force_constants [nrestraints] - force constants
 * @param natoms - number of atoms
 * @param nrestraints - number of restraints
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void harmonic_angle_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_angles,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
);

/**
 * Harmonic dihedral restraint
 * E = 0.5 * k * (phi - phi0)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_indices [nrestraints, 4] - quartets of atom indices
 * @param target_dihedrals [nrestraints] - target dihedrals (radians)
 * @param force_constants [nrestraints] - force constants
 * @param natoms - number of atoms
 * @param nrestraints - number of restraints
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void harmonic_dihedral_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* target_dihedrals,
    const double* force_constants,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
);

/**
 * Spherical boundary restraint
 * E = 0.5 * k * max(0, r - r0)^2 for each atom
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_indices [nrestraints] - atom indices to restrain
 * @param center [3] - center of sphere
 * @param radius - radius of sphere
 * @param force_constant - force constant
 * @param natoms - number of atoms
 * @param nrestraints - number of restrained atoms
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void spherical_boundary_restraint(
    const double* coordinates,
    const int* atom_indices,
    const double* center,
    double radius,
    double force_constant,
    int natoms,
    int nrestraints,
    double* energy,
    double* forces
);

/**
 * RMSD restraint
 * E = 0.5 * k * (RMSD - RMSD0)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param reference [natoms, 3] - reference coordinates
 * @param atom_indices [nrestraints] - atom indices to include in RMSD
 * @param masses [natoms] - atomic masses (for mass-weighted RMSD)
 * @param target_rmsd - target RMSD
 * @param force_constant - force constant
 * @param natoms - number of atoms
 * @param nrestraints - number of atoms in RMSD calculation
 * @param use_mass_weighting - whether to use mass-weighted RMSD
 * @param energy [out] - total restraint energy
 * @param forces [natoms, 3, out] - restraint forces (accumulated)
 */
void rmsd_restraint(
    const double* coordinates,
    const double* reference,
    const int* atom_indices,
    const double* masses,
    double target_rmsd,
    double force_constant,
    int natoms,
    int nrestraints,
    bool use_mass_weighting,
    double* energy,
    double* forces
);

} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_RESTRAINTS_CUH
