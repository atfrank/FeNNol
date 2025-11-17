#include "../include/integrate.cuh"
#include <cmath>

namespace fennol {
namespace cuda {

// RAII wrapper for CUDA memory to prevent leaks on exceptions
template<typename T>
class CudaMemory {
    T* ptr = nullptr;
    size_t count = 0;

public:
    explicit CudaMemory(size_t n) : count(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
    }

    ~CudaMemory() {
        if (ptr) {
            cudaFree(ptr);  // Don't throw from destructor
        }
    }

    // Delete copy operations
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    T* get() { return ptr; }
};

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
    if (mass < 1e-10) return;  // Skip atoms with invalid mass
    double dt2m = dt2 / mass;

    // Update velocities: v = v + (dt/2) * f / m
    for (int d = 0; d < 3; ++d) {
        int i = idx * 3 + d;
        velocities[i] += forces[i] * dt2m;
    }

    // First half position update: x = x + (dt/2) * v
    // Note: Thermostat would be applied between these two half-steps
    // For now, we do both half-steps here (suitable for NVE)
    for (int d = 0; d < 3; ++d) {
        int i = idx * 3 + d;
        coordinates[i] += dt2 * velocities[i];
    }

    // Second half position update: x = x + (dt/2) * v
    for (int d = 0; d < 3; ++d) {
        int i = idx * 3 + d;
        coordinates[i] += dt2 * velocities[i];
    }
}

void velocity_verlet_step_a(
    double* coordinates,
    double* velocities,
    const double* forces,
    const double* masses,
    double dt,
    int natoms,
    cudaStream_t stream
) {
    double dt2 = 0.5 * dt;
    int nblocks = (natoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    velocity_verlet_step_a_kernel<<<nblocks, BLOCK_SIZE, 0, stream>>>(
        coordinates, velocities, forces, masses, dt, dt2, natoms
    );
    CUDA_CHECK(cudaGetLastError());

    // Only synchronize if using default stream (for backward compatibility)
    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
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
        // CRITICAL FIX: Cannot return early - must reach __syncthreads() in blockReduceSum()
        // Early return causes race condition and potential deadlock
        if (mass >= 1e-10) {  // Only process valid masses, but don't skip reduction
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
        // If mass < 1e-10, local_energy and local_tensor remain 0.0 (initialized above)
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
    double* kinetic_tensor,
    cudaStream_t stream
) {
    double dt2 = 0.5 * dt;
    int nblocks = (natoms + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate temporary storage for partial reductions using RAII
    // Prevents memory leaks if CUDA_CHECK throws exception
    CudaMemory<double> d_partial_energies(nblocks);
    CudaMemory<double> d_partial_tensors(nblocks * 9);

    // Launch step B kernel on stream
    velocity_verlet_step_b_kernel<<<nblocks, BLOCK_SIZE, 0, stream>>>(
        velocities, forces, masses, dt2, natoms, d_partial_energies.get(), d_partial_tensors.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Final reduction on stream
    final_reduction_kernel<<<1, 1, 0, stream>>>(
        d_partial_energies.get(), d_partial_tensors.get(), nblocks, kinetic_energy, kinetic_tensor
    );
    CUDA_CHECK(cudaGetLastError());

    // Synchronize before exiting (RAII destructors will free memory automatically)
    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    // Memory automatically freed by RAII destructors, even if exceptions occur
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

    // Allocate temporary storage using RAII
    // Prevents memory leaks if CUDA_CHECK throws exception
    CudaMemory<double> d_partial_energies(nblocks);
    CudaMemory<double> d_partial_tensors(nblocks * 9);

    // Compute partial sums
    compute_kinetic_energy_kernel<<<nblocks, BLOCK_SIZE>>>(
        velocities, masses, natoms, d_partial_energies.get(), d_partial_tensors.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Final reduction
    final_reduction_kernel<<<1, 1>>>(
        d_partial_energies.get(), d_partial_tensors.get(), nblocks, kinetic_energy, kinetic_tensor
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // Memory automatically freed by RAII destructors, even if exceptions occur
}

} // namespace cuda
} // namespace fennol
