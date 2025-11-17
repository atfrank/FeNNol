#include "../include/integrate.cuh"
#include <cmath>

namespace fennol {
namespace cuda {

// Kernel for velocity Verlet step A
__global__ void velocity_verlet_step_a_kernel(
    double* coordinates,
    double* velocities,
    const double* forces,
    const double* masses,
    double dt,
    double dt2,
    int natoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms) return;

    double mass = masses[idx];
    double dt2m = dt2 / mass;

    // Update velocities: v = v + (dt/2) * f / m
    for (int d = 0; d < 3; ++d) {
        int i = idx * 3 + d;
        velocities[i] += forces[i] * dt2m;
    }

    // Update positions: x = x + dt * v
    for (int d = 0; d < 3; ++d) {
        int i = idx * 3 + d;
        coordinates[i] += dt * velocities[i];
    }
}

void velocity_verlet_step_a(
    double* coordinates,
    double* velocities,
    const double* forces,
    const double* masses,
    double dt,
    int natoms
) {
    double dt2 = 0.5 * dt;
    int nblocks = (natoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    velocity_verlet_step_a_kernel<<<nblocks, BLOCK_SIZE>>>(
        coordinates, velocities, forces, masses, dt, dt2, natoms
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Kernel for velocity Verlet step B
__global__ void velocity_verlet_step_b_kernel(
    double* velocities,
    const double* forces,
    const double* masses,
    double dt2,
    int natoms,
    double* partial_energies,
    double* partial_tensors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread-local accumulation
    double local_energy = 0.0;
    double local_tensor[9] = {0.0};

    if (idx < natoms) {
        double mass = masses[idx];
        double dt2m = dt2 / mass;

        Vec3 vel;
        // Update velocities: v = v + (dt/2) * f / m
        for (int d = 0; d < 3; ++d) {
            int i = idx * 3 + d;
            velocities[i] += forces[i] * dt2m;
            if (d == 0) vel.x = velocities[i];
            else if (d == 1) vel.y = velocities[i];
            else vel.z = velocities[i];
        }

        // Compute kinetic energy contribution: 0.5 * m * v^2
        local_energy = 0.5 * mass * vel.norm_squared();

        // Compute kinetic tensor contribution: 0.5 * m * v_i * v_j
        double v[3] = {vel.x, vel.y, vel.z};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                local_tensor[i * 3 + j] = 0.5 * mass * v[i] * v[j];
            }
        }
    }

    // Block-level reduction for energy
    double block_energy = blockReduceSum(local_energy);
    if (threadIdx.x == 0) {
        partial_energies[blockIdx.x] = block_energy;
    }

    // Block-level reduction for tensor components
    for (int t = 0; t < 9; ++t) {
        double block_tensor_component = blockReduceSum(local_tensor[t]);
        if (threadIdx.x == 0) {
            partial_tensors[blockIdx.x * 9 + t] = block_tensor_component;
        }
    }
}

// Final reduction kernel
__global__ void final_reduction_kernel(
    const double* partial_energies,
    const double* partial_tensors,
    int nblocks,
    double* kinetic_energy,
    double* kinetic_tensor
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Reduce energies
        double total_energy = 0.0;
        for (int i = 0; i < nblocks; ++i) {
            total_energy += partial_energies[i];
        }
        *kinetic_energy = total_energy;

        // Reduce tensor components
        for (int t = 0; t < 9; ++t) {
            double total_tensor = 0.0;
            for (int i = 0; i < nblocks; ++i) {
                total_tensor += partial_tensors[i * 9 + t];
            }
            kinetic_tensor[t] = total_tensor;
        }
    }
}

void velocity_verlet_step_b(
    double* velocities,
    const double* forces,
    const double* masses,
    double dt,
    int natoms,
    double* kinetic_energy,
    double* kinetic_tensor
) {
    double dt2 = 0.5 * dt;
    int nblocks = (natoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate temporary storage for partial reductions
    double* d_partial_energies;
    double* d_partial_tensors;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nblocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial_tensors, nblocks * 9 * sizeof(double)));

    // Launch step B kernel
    velocity_verlet_step_b_kernel<<<nblocks, BLOCK_SIZE>>>(
        velocities, forces, masses, dt2, natoms, d_partial_energies, d_partial_tensors
    );
    CUDA_CHECK(cudaGetLastError());

    // Final reduction
    final_reduction_kernel<<<1, 1>>>(
        d_partial_energies, d_partial_tensors, nblocks, kinetic_energy, kinetic_tensor
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary storage
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaFree(d_partial_tensors));
}

// Kernel for scaling velocities
__global__ void scale_velocities_kernel(
    double* velocities,
    double scale,
    int natoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms * 3) return;

    velocities[idx] *= scale;
}

void scale_velocities(
    double* velocities,
    double scale,
    int natoms
) {
    int total_elements = natoms * 3;
    int nblocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    scale_velocities_kernel<<<nblocks, BLOCK_SIZE>>>(velocities, scale, natoms);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Kernel for computing kinetic energy
__global__ void compute_kinetic_energy_kernel(
    const double* velocities,
    const double* masses,
    int natoms,
    double* partial_energies,
    double* partial_tensors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread-local accumulation
    double local_energy = 0.0;
    double local_tensor[9] = {0.0};

    if (idx < natoms) {
        double mass = masses[idx];

        Vec3 vel;
        vel.x = velocities[idx * 3 + 0];
        vel.y = velocities[idx * 3 + 1];
        vel.z = velocities[idx * 3 + 2];

        // Compute kinetic energy contribution
        local_energy = 0.5 * mass * vel.norm_squared();

        // Compute kinetic tensor contribution
        double v[3] = {vel.x, vel.y, vel.z};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                local_tensor[i * 3 + j] = 0.5 * mass * v[i] * v[j];
            }
        }
    }

    // Block-level reduction
    double block_energy = blockReduceSum(local_energy);
    if (threadIdx.x == 0) {
        partial_energies[blockIdx.x] = block_energy;
    }

    for (int t = 0; t < 9; ++t) {
        double block_tensor_component = blockReduceSum(local_tensor[t]);
        if (threadIdx.x == 0) {
            partial_tensors[blockIdx.x * 9 + t] = block_tensor_component;
        }
    }
}

void compute_kinetic_energy(
    const double* velocities,
    const double* masses,
    int natoms,
    double* kinetic_energy,
    double* kinetic_tensor
) {
    int nblocks = (natoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate temporary storage
    double* d_partial_energies;
    double* d_partial_tensors;
    CUDA_CHECK(cudaMalloc(&d_partial_energies, nblocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial_tensors, nblocks * 9 * sizeof(double)));

    // Compute partial sums
    compute_kinetic_energy_kernel<<<nblocks, BLOCK_SIZE>>>(
        velocities, masses, natoms, d_partial_energies, d_partial_tensors
    );
    CUDA_CHECK(cudaGetLastError());

    // Final reduction
    final_reduction_kernel<<<1, 1>>>(
        d_partial_energies, d_partial_tensors, nblocks, kinetic_energy, kinetic_tensor
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary storage
    CUDA_CHECK(cudaFree(d_partial_energies));
    CUDA_CHECK(cudaFree(d_partial_tensors));
}

} // namespace cuda
} // namespace fennol
