#ifndef FENNOL_CUDA_PHYSICS_CUH
#define FENNOL_CUDA_PHYSICS_CUH

#include "common.cuh"

namespace fennol {
namespace cuda {

/**
 * Lennard-Jones (12-6) pairwise interaction
 * E = 4*epsilon * ((sigma/r)^12 - (sigma/r)^6)
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_pairs [npairs, 2] - pairs of atom indices
 * @param epsilons [npairs] - LJ epsilon parameters
 * @param sigmas [npairs] - LJ sigma parameters
 * @param natoms - number of atoms
 * @param npairs - number of pairs
 * @param energy [out] - total LJ energy
 * @param forces [natoms, 3, out] - LJ forces (accumulated)
 */
void lennard_jones_pairwise(
    const double* coordinates,
    const int* atom_pairs,
    const double* epsilons,
    const double* sigmas,
    int natoms,
    int npairs,
    double* energy,
    double* forces
);

/**
 * Coulomb electrostatics (direct, no periodicity)
 * E = k_e * q_i * q_j / r_ij
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param charges [natoms] - atomic charges
 * @param atom_pairs [npairs, 2] - pairs of atom indices
 * @param natoms - number of atoms
 * @param npairs - number of pairs
 * @param coulomb_constant - Coulomb constant (k_e)
 * @param energy [out] - total electrostatic energy
 * @param forces [natoms, 3, out] - electrostatic forces (accumulated)
 */
void coulomb_direct(
    const double* coordinates,
    const double* charges,
    const int* atom_pairs,
    int natoms,
    int npairs,
    double coulomb_constant,
    double* energy,
    double* forces
);

/**
 * ZBL (Ziegler-Biersack-Littmark) repulsion potential
 * Used for short-range nuclear repulsion
 *
 * E_ZBL = (Z_i * Z_j * e^2 / r) * phi(r / a)
 * where phi is the ZBL screening function
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atomic_numbers [natoms] - atomic numbers (Z)
 * @param atom_pairs [npairs, 2] - pairs of atom indices
 * @param natoms - number of atoms
 * @param npairs - number of pairs
 * @param cutoff - cutoff distance for ZBL
 * @param energy [out] - total ZBL energy
 * @param forces [natoms, 3, out] - ZBL forces (accumulated)
 */
void zbl_repulsion(
    const double* coordinates,
    const int* atomic_numbers,
    const int* atom_pairs,
    int natoms,
    int npairs,
    double cutoff,
    double* energy,
    double* forces
);

/**
 * NLH (Nordlund-Lehtola-Hobler) repulsion potential
 * Element-pair-specific repulsion potential with improved accuracy
 *
 * Based on: K. Nordlund, S. Lehtola, G. Hobler
 * "Repulsive interatomic potentials calculated at three levels of theory"
 * Physical Review A 111, 032818 (2025)
 *
 * E_NLH = (Z_i * Z_j * e^2 / r) * sum_k(c_k * exp(-alpha_k * r_scaled))
 * where coefficients are pair-specific
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atomic_numbers [natoms] - atomic numbers (Z)
 * @param atom_pairs [npairs, 2] - pairs of atom indices
 * @param pair_coefficients [npairs, 6] - NLH coefficients (a1,b1,a2,b2,a3,b3) per pair
 * @param natoms - number of atoms
 * @param npairs - number of pairs
 * @param cutoff - cutoff distance for NLH
 * @param energy [out] - total NLH energy
 * @param forces [natoms, 3, out] - NLH forces (accumulated)
 */
void nlh_repulsion(
    const double* coordinates,
    const int* atomic_numbers,
    const int* atom_pairs,
    const double* pair_coefficients,
    int natoms,
    int npairs,
    double cutoff,
    double* energy,
    double* forces
);

/**
 * Dispersion (Van der Waals) interaction using C6 coefficients
 * E = -C6 / r^6
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param atom_pairs [npairs, 2] - pairs of atom indices
 * @param c6_coefficients [npairs] - C6 dispersion coefficients
 * @param natoms - number of atoms
 * @param npairs - number of pairs
 * @param energy [out] - total dispersion energy
 * @param forces [natoms, 3, out] - dispersion forces (accumulated)
 */
void dispersion_c6(
    const double* coordinates,
    const int* atom_pairs,
    const double* c6_coefficients,
    int natoms,
    int npairs,
    double* energy,
    double* forces
);

/**
 * Harmonic bond potential
 * E = 0.5 * k * (r - r0)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param bond_indices [nbonds, 2] - pairs of bonded atoms
 * @param equilibrium_lengths [nbonds] - equilibrium bond lengths
 * @param force_constants [nbonds] - bond force constants
 * @param natoms - number of atoms
 * @param nbonds - number of bonds
 * @param energy [out] - total bond energy
 * @param forces [natoms, 3, out] - bond forces (accumulated)
 */
void harmonic_bonds(
    const double* coordinates,
    const int* bond_indices,
    const double* equilibrium_lengths,
    const double* force_constants,
    int natoms,
    int nbonds,
    double* energy,
    double* forces
);

/**
 * Harmonic angle potential
 * E = 0.5 * k * (theta - theta0)^2
 *
 * @param coordinates [natoms, 3] - atomic coordinates
 * @param angle_indices [nangles, 3] - triplets of atoms forming angles
 * @param equilibrium_angles [nangles] - equilibrium angles (radians)
 * @param force_constants [nangles] - angle force constants
 * @param natoms - number of atoms
 * @param nangles - number of angles
 * @param energy [out] - total angle energy
 * @param forces [natoms, 3, out] - angle forces (accumulated)
 */
void harmonic_angles(
    const double* coordinates,
    const int* angle_indices,
    const double* equilibrium_angles,
    const double* force_constants,
    int natoms,
    int nangles,
    double* energy,
    double* forces
);

} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_PHYSICS_CUH
