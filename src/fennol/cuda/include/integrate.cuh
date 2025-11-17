#ifndef FENNOL_CUDA_INTEGRATE_CUH
#define FENNOL_CUDA_INTEGRATE_CUH

#include "common.cuh"

namespace fennol {
namespace cuda {

/**
 * Velocity Verlet Integration - Step A (first half)
 * Updates positions and half-step velocities
 *
 * v_half = v + (dt/2) * f/m
 * x_new = x + dt * v_half
 *
 * @param coordinates [natoms, 3] - atomic coordinates (in/out)
 * @param velocities [natoms, 3] - atomic velocities (in/out)
 * @param forces [natoms, 3] - atomic forces
 * @param masses [natoms] - atomic masses
 * @param dt - timestep
 * @param natoms - number of atoms
 */
void velocity_verlet_step_a(
    double* coordinates,
    double* velocities,
    const double* forces,
    const double* masses,
    double dt,
    int natoms
);

/**
 * Velocity Verlet Integration - Step B (second half)
 * Completes velocity update and calculates kinetic energy
 *
 * v_new = v_half + (dt/2) * f/m
 *
 * @param velocities [natoms, 3] - atomic velocities (in/out)
 * @param forces [natoms, 3] - atomic forces
 * @param masses [natoms] - atomic masses
 * @param dt - timestep
 * @param natoms - number of atoms
 * @param kinetic_energy [out] - computed kinetic energy
 * @param kinetic_tensor [9, out] - computed kinetic energy tensor (row-major)
 */
void velocity_verlet_step_b(
    double* velocities,
    const double* forces,
    const double* masses,
    double dt,
    int natoms,
    double* kinetic_energy,
    double* kinetic_tensor
);

/**
 * Scale velocities (for thermostats)
 *
 * v_new = v * scale
 *
 * @param velocities [natoms, 3] - atomic velocities (in/out)
 * @param scale - scaling factor
 * @param natoms - number of atoms
 */
void scale_velocities(
    double* velocities,
    double scale,
    int natoms
);

/**
 * Compute kinetic energy and tensor
 *
 * @param velocities [natoms, 3] - atomic velocities
 * @param masses [natoms] - atomic masses
 * @param natoms - number of atoms
 * @param kinetic_energy [out] - computed kinetic energy
 * @param kinetic_tensor [9, out] - computed kinetic energy tensor (row-major)
 */
void compute_kinetic_energy(
    const double* velocities,
    const double* masses,
    int natoms,
    double* kinetic_energy,
    double* kinetic_tensor
);

} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_INTEGRATE_CUH
