#ifndef FENNOL_CUDA_THERMOSTATS_CUH
#define FENNOL_CUDA_THERMOSTATS_CUH

#include "common.cuh"

namespace fennol {
namespace cuda {

/**
 * Velocity rescaling thermostat
 * Rescales velocities to achieve target temperature
 *
 * v_new = v * sqrt(T_target / T_current)
 *
 * @param velocities [natoms, 3] - atomic velocities (in/out)
 * @param masses [natoms] - atomic masses
 * @param natoms - number of atoms
 * @param target_temperature - target temperature (K)
 * @param current_temperature [out] - current temperature before rescaling
 */
void velocity_rescale_thermostat(
    double* velocities,
    const double* masses,
    int natoms,
    double target_temperature,
    double* current_temperature
);

/**
 * Berendsen thermostat
 * Weakly couples system to heat bath
 *
 * v_new = v * sqrt(1 + dt/tau * (T_target/T_current - 1))
 *
 * @param velocities [natoms, 3] - atomic velocities (in/out)
 * @param masses [natoms] - atomic masses
 * @param natoms - number of atoms
 * @param target_temperature - target temperature (K)
 * @param coupling_time - coupling time constant tau
 * @param dt - timestep
 * @param current_temperature [out] - current temperature before rescaling
 */
void berendsen_thermostat(
    double* velocities,
    const double* masses,
    int natoms,
    double target_temperature,
    double coupling_time,
    double dt,
    double* current_temperature
);

/**
 * Compute current temperature from velocities
 *
 * T = (2 * KE) / (k_B * N_dof)
 * where KE = 0.5 * sum(m * v^2)
 *
 * @param velocities [natoms, 3] - atomic velocities
 * @param masses [natoms] - atomic masses
 * @param natoms - number of atoms
 * @param ndof - number of degrees of freedom (typically 3*natoms - 6)
 * @param temperature [out] - computed temperature (K)
 */
void compute_temperature(
    const double* velocities,
    const double* masses,
    int natoms,
    int ndof,
    double* temperature
);

/**
 * Andersen thermostat
 * Randomly reassigns velocities from Maxwell-Boltzmann distribution
 *
 * @param velocities [natoms, 3] - atomic velocities (in/out)
 * @param masses [natoms] - atomic masses
 * @param natoms - number of atoms
 * @param target_temperature - target temperature (K)
 * @param collision_frequency - frequency of random collisions
 * @param dt - timestep
 * @param random_state - random number generator state
 */
void andersen_thermostat(
    double* velocities,
    const double* masses,
    int natoms,
    double target_temperature,
    double collision_frequency,
    double dt,
    unsigned long long* random_state
);

} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_THERMOSTATS_CUH
