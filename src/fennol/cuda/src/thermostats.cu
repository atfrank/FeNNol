#include "../include/thermostats.cuh"
#include <cmath>

namespace fennol {
namespace cuda {

// Boltzmann constant in eV/K
constexpr double kB = 8.617333e-5;

// Kernel to compute kinetic energy and temperature
__global__ void compute_temperature_kernel(
    const double* velocities,
    const double* masses,
    int natoms,
    double* partial_energies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_energy = 0.0;

    if (idx < natoms) {
        double mass = masses[idx];
        Vec3 vel;
        vel.x = velocities[idx * 3 + 0];
        vel.y = velocities[idx * 3 + 1];
        vel.z = velocities[idx * 3 + 2];

        local_energy = 0.5 * mass * vel.norm_squared();
    }

    // Block-level reduction
    double block_energy = blockReduceSum(local_energy);
    if (threadIdx.x == 0) {
        partial_energies[blockIdx.x] = block_energy;
    }
}

void compute_temperature(
    const double* velocities,
    const double* masses,
    int natoms,
    int ndof,
    double* temperature
) {
    int nblocks = (natoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double* d_partial_energies;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nblocks * sizeof(double)));

    compute_temperature_kernel<<<nblocks, BLOCK_SIZE>>>(
        velocities, masses, natoms, d_partial_energies
    );
    CUDA_CHECK(cudaGetLastError());

    // Reduce partial energies on host (simpler for now)
    double* h_partial_energies = new double[nblocks];
    CUDA_CHECK(cudaMemcpy(h_partial_energies, d_partial_energies,
                          nblocks * sizeof(double), cudaMemcpyDeviceToHost));

    double total_ke = 0.0;
    for (int i = 0; i < nblocks; ++i) {
        total_ke += h_partial_energies[i];
    }

    // T = (2 * KE) / (k_B * N_dof)
    *temperature = (2.0 * total_ke) / (kB * ndof);

    delete[] h_partial_energies;
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Velocity rescaling kernel
__global__ void velocity_rescale_kernel(
    double* velocities,
    double scale_factor,
    int natoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms * 3) return;

    velocities[idx] *= scale_factor;
}

void velocity_rescale_thermostat(
    double* velocities,
    const double* masses,
    int natoms,
    double target_temperature,
    double* current_temperature
) {
    // Compute current temperature
    int ndof = 3 * natoms - 6; // Subtract COM motion
    compute_temperature(velocities, masses, natoms, ndof, current_temperature);

    if (*current_temperature < 1e-10) {
        *current_temperature = target_temperature;
        return; // Avoid division by zero
    }

    // Compute scale factor
    double scale_factor = sqrt(target_temperature / *current_temperature);

    // Rescale velocities
    int total_elements = natoms * 3;
    int nblocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    velocity_rescale_kernel<<<nblocks, BLOCK_SIZE>>>(
        velocities, scale_factor, natoms
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void berendsen_thermostat(
    double* velocities,
    const double* masses,
    int natoms,
    double target_temperature,
    double coupling_time,
    double dt,
    double* current_temperature
) {
    // Compute current temperature
    int ndof = 3 * natoms - 6;
    compute_temperature(velocities, masses, natoms, ndof, current_temperature);

    if (*current_temperature < 1e-10) {
        *current_temperature = target_temperature;
        return;
    }

    // Berendsen scale factor
    // lambda = sqrt(1 + dt/tau * (T_target/T_current - 1))
    double ratio = target_temperature / *current_temperature;
    double scale_factor = sqrt(1.0 + (dt / coupling_time) * (ratio - 1.0));

    // Rescale velocities
    int total_elements = natoms * 3;
    int nblocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    velocity_rescale_kernel<<<nblocks, BLOCK_SIZE>>>(
        velocities, scale_factor, natoms
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Andersen thermostat kernel
__global__ void andersen_kernel(
    double* velocities,
    const double* masses,
    int natoms,
    double target_temperature,
    double collision_probability,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms) return;

    // Simple linear congruential generator for random numbers
    unsigned long long state = seed + idx;
    state = state * 1103515245ULL + 12345ULL;
    double rand1 = (state & 0x7FFFFFFF) / (double)0x7FFFFFFF;

    // Check if this atom undergoes collision
    if (rand1 < collision_probability) {
        double mass = masses[idx];
        double sigma = sqrt(kB * target_temperature / mass);

        // Generate new velocities from Maxwell-Boltzmann
        // Using Box-Muller transform
        for (int d = 0; d < 3; ++d) {
            state = state * 1103515245ULL + 12345ULL;
            double u1 = (state & 0x7FFFFFFF) / (double)0x7FFFFFFF;
            state = state * 1103515245ULL + 12345ULL;
            double u2 = (state & 0x7FFFFFFF) / (double)0x7FFFFFFF;

            // Box-Muller transform
            double normal = sigma * sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
            velocities[idx * 3 + d] = normal;
        }
    }
}

void andersen_thermostat(
    double* velocities,
    const double* masses,
    int natoms,
    double target_temperature,
    double collision_frequency,
    double dt,
    unsigned long long* random_state
) {
    // Collision probability per atom
    double collision_probability = collision_frequency * dt;

    int nblocks = (natoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    andersen_kernel<<<nblocks, BLOCK_SIZE>>>(
        velocities, masses, natoms, target_temperature,
        collision_probability, *random_state
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update random state
    *random_state = (*random_state * 1103515245ULL + 12345ULL);
}

} // namespace cuda
} // namespace fennol
