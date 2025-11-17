#include "../include/multi_gpu.cuh"
#include "../include/integrate.cuh"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace fennol {
namespace cuda {

// Helper function to check CUDA errors for multi-GPU
#define CUDA_CHECK_MULTI(call, gpu_id) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("GPU ") + std::to_string(gpu_id) + \
                                   " CUDA error: " + cudaGetErrorString(error)); \
        } \
    } while(0)

int get_optimal_gpu_count(int natoms) {
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) return 0;

    // Rule of thumb: Use 1 GPU per 10K atoms, max available
    int optimal = std::max(1, natoms / 10000);
    return std::min(optimal, device_count);
}

bool check_peer_access(int gpu1, int gpu2) {
    int can_access;
    cudaDeviceCanAccessPeer(&can_access, gpu1, gpu2);
    return can_access != 0;
}

bool enable_peer_access(int ngpus) {
    bool all_enabled = true;

    for (int i = 0; i < ngpus; ++i) {
        cudaError_t set_err = cudaSetDevice(i);
        if (set_err != cudaSuccess) {
            all_enabled = false;
            continue;
        }

        for (int j = 0; j < ngpus; ++j) {
            if (i != j) {
                if (check_peer_access(i, j)) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        all_enabled = false;
                    }
                } else {
                    all_enabled = false;
                }
            }
        }
    }

    return all_enabled;
}

MultiGPUContext* initialize_multi_gpu(
    int natoms,
    const double* coordinates,
    const double* velocities,
    const double* masses,
    const double* box_size,
    double cutoff,
    int ngpus
) {
    // Auto-detect GPU count if not specified
    if (ngpus == 0) {
        ngpus = get_optimal_gpu_count(natoms);
        std::cout << "# Auto-detected " << ngpus << " GPUs for " << natoms << " atoms\n";
    }

    int device_count;
    cudaGetDeviceCount(&device_count);

    if (ngpus > device_count) {
        std::cerr << "# Warning: Requested " << ngpus << " GPUs but only "
                  << device_count << " available\n";
        ngpus = device_count;
    }

    if (ngpus == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Create context
    MultiGPUContext* ctx = new MultiGPUContext();
    ctx->ngpus = ngpus;
    ctx->natoms_global = natoms;
    ctx->cutoff = cutoff;
    ctx->streams = new cudaStream_t[ngpus];

    // Try to enable peer access
    ctx->use_peer_access = enable_peer_access(ngpus);
    if (ctx->use_peer_access) {
        std::cout << "# Peer-to-peer GPU access enabled\n";
    } else {
        std::cout << "# Using host memory for GPU communication\n";
    }

    // Create streams for each GPU
    for (int i = 0; i < ngpus; ++i) {
        CUDA_CHECK_MULTI(cudaSetDevice(i), i);
        cudaStreamCreate(&ctx->streams[i]);

        // Initialize domain with all pointers set to nullptr
        GPUDomain* domain = new GPUDomain();
        domain->gpu_id = i;
        domain->natoms_local = 0;
        domain->natoms_with_halo = 0;
        domain->local_atom_indices = nullptr;
        domain->halo_atom_indices = nullptr;
        domain->d_coordinates = nullptr;
        domain->d_velocities = nullptr;
        domain->d_forces = nullptr;
        domain->d_masses = nullptr;
        domain->d_halo_coords = nullptr;
        domain->d_send_buffer = nullptr;
        domain->d_recv_buffer = nullptr;
        ctx->domains.push_back(domain);
    }

    // Distribute atoms across GPUs
    distribute_atoms(ctx, coordinates, velocities, masses, box_size);

    std::cout << "# Multi-GPU initialized: " << ngpus << " GPUs, "
              << natoms << " atoms\n";

    return ctx;
}

void cleanup_multi_gpu(MultiGPUContext* ctx) {
    for (int i = 0; i < ctx->ngpus; ++i) {
        CUDA_CHECK_MULTI(cudaSetDevice(i), i);

        GPUDomain* domain = ctx->domains[i];

        // Synchronize to ensure all kernels are complete before freeing memory
        CUDA_CHECK_MULTI(cudaDeviceSynchronize(), i);

        // Free device memory
        if (domain->d_coordinates) cudaFree(domain->d_coordinates);
        if (domain->d_velocities) cudaFree(domain->d_velocities);
        if (domain->d_forces) cudaFree(domain->d_forces);
        if (domain->d_masses) cudaFree(domain->d_masses);
        if (domain->d_halo_coords) cudaFree(domain->d_halo_coords);
        if (domain->d_send_buffer) cudaFree(domain->d_send_buffer);
        if (domain->d_recv_buffer) cudaFree(domain->d_recv_buffer);

        // Free host memory
        if (domain->local_atom_indices) delete[] domain->local_atom_indices;
        if (domain->halo_atom_indices) delete[] domain->halo_atom_indices;

        // Destroy stream
        cudaStreamDestroy(ctx->streams[i]);

        delete domain;
    }

    delete[] ctx->streams;
    delete ctx;
}

void distribute_atoms(
    MultiGPUContext* ctx,
    const double* coordinates,
    const double* velocities,
    const double* masses,
    const double* box_size
) {
    int natoms = ctx->natoms_global;
    int ngpus = ctx->ngpus;

    // Simple 1D spatial decomposition along X-axis
    // For production, would use 3D decomposition
    double box_x = box_size[0];
    double domain_width = box_x / ngpus;

    for (int gpu = 0; gpu < ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];

        // Set domain bounds
        domain->bounds[0] = gpu * domain_width - ctx->cutoff;        // xmin
        domain->bounds[1] = (gpu + 1) * domain_width + ctx->cutoff;  // xmax
        domain->bounds[2] = -ctx->cutoff;                             // ymin
        domain->bounds[3] = box_size[1] + ctx->cutoff;                // ymax
        domain->bounds[4] = -ctx->cutoff;                             // zmin
        domain->bounds[5] = box_size[2] + ctx->cutoff;                // zmax

        // Count atoms in this domain
        std::vector<int> local_indices;
        for (int i = 0; i < natoms; ++i) {
            double x = coordinates[i * 3 + 0];
            double core_xmin = gpu * domain_width;
            double core_xmax = (gpu + 1) * domain_width;

            // Check if atom is in core domain (no halo)
            // Last GPU gets inclusive upper bound to handle x == box_x
            bool in_domain = (gpu == ngpus - 1) ?
                            (x >= core_xmin && x <= core_xmax) :
                            (x >= core_xmin && x < core_xmax);

            if (in_domain) {
                local_indices.push_back(i);
            }
        }

        domain->natoms_local = local_indices.size();

        // Allocate and copy local atom indices
        domain->local_atom_indices = new int[domain->natoms_local];
        std::copy(local_indices.begin(), local_indices.end(), domain->local_atom_indices);

        // Allocate device memory
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        CUDA_CHECK_MULTI(cudaMalloc(&domain->d_coordinates, domain->natoms_local * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&domain->d_velocities, domain->natoms_local * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&domain->d_forces, domain->natoms_local * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&domain->d_masses, domain->natoms_local * sizeof(double)), gpu);

        // Pack and copy data
        std::vector<double> local_coords(domain->natoms_local * 3);
        std::vector<double> local_vels(domain->natoms_local * 3);
        std::vector<double> local_masses(domain->natoms_local);

        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                local_coords[i * 3 + d] = coordinates[global_idx * 3 + d];
                local_vels[i * 3 + d] = velocities[global_idx * 3 + d];
            }
            local_masses[i] = masses[global_idx];
        }

        CUDA_CHECK_MULTI(cudaMemcpy(domain->d_coordinates, local_coords.data(),
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice), gpu);
        CUDA_CHECK_MULTI(cudaMemcpy(domain->d_velocities, local_vels.data(),
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice), gpu);
        CUDA_CHECK_MULTI(cudaMemcpy(domain->d_masses, local_masses.data(),
                                    domain->natoms_local * sizeof(double),
                                    cudaMemcpyHostToDevice), gpu);

        std::cout << "# GPU " << gpu << ": " << domain->natoms_local << " atoms\n";
    }
}

void exchange_halos(MultiGPUContext* ctx) {
    // Simplified halo exchange using host memory staging
    // For production, would use peer-to-peer or NCCL

    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        // Copy halo data from neighboring domains
        // In real implementation, would identify atoms in halo regions
        // and communicate only those atoms

        // For now, this is a placeholder
        // Full implementation would:
        // 1. Identify atoms near domain boundaries
        // 2. Pack into send buffer
        // 3. Copy to neighboring GPU (via peer access or host staging)
        // 4. Unpack into halo region
    }

    // Synchronize all GPUs
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);
        cudaStreamSynchronize(ctx->streams[gpu]);
    }
}

void multi_gpu_velocity_verlet_step_a(
    MultiGPUContext* ctx,
    const double* forces,
    double dt
) {
    // Store temporary device pointers for cleanup
    std::vector<double*> d_forces_locals(ctx->ngpus);

    // PHASE 1: Launch kernels on all GPUs asynchronously
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        // Pack forces for this domain
        std::vector<double> local_forces(domain->natoms_local * 3);
        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                local_forces[i * 3 + d] = forces[global_idx * 3 + d];
            }
        }

        // Allocate and copy forces to device asynchronously
        CUDA_CHECK_MULTI(cudaMalloc(&d_forces_locals[gpu], domain->natoms_local * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMemcpyAsync(d_forces_locals[gpu], local_forces.data(),
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice, ctx->streams[gpu]), gpu);

        // Execute step A on this GPU using stream (async)
        velocity_verlet_step_a(
            domain->d_coordinates,
            domain->d_velocities,
            d_forces_locals[gpu],
            domain->d_masses,
            dt,
            domain->natoms_local,
            ctx->streams[gpu]  // Use per-GPU stream for async execution
        );
    }

    // PHASE 2: Synchronize all GPUs
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);
        CUDA_CHECK_MULTI(cudaStreamSynchronize(ctx->streams[gpu]), gpu);
    }

    // PHASE 3: Cleanup temporary memory
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);
        cudaFree(d_forces_locals[gpu]);
    }

    // Exchange halo regions after position update
    exchange_halos(ctx);
}

void multi_gpu_velocity_verlet_step_b(
    MultiGPUContext* ctx,
    const double* forces,
    double dt,
    double* kinetic_energy,
    double* kinetic_tensor
) {
    // Store temporary device pointers for cleanup
    std::vector<double*> d_forces_locals(ctx->ngpus);
    std::vector<double*> d_kes(ctx->ngpus);
    std::vector<double*> d_ke_tensors(ctx->ngpus);

    // PHASE 1: Launch kernels on all GPUs asynchronously
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        // Pack forces
        std::vector<double> local_forces(domain->natoms_local * 3);
        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                local_forces[i * 3 + d] = forces[global_idx * 3 + d];
            }
        }

        // Allocate device memory
        CUDA_CHECK_MULTI(cudaMalloc(&d_forces_locals[gpu], domain->natoms_local * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&d_kes[gpu], sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&d_ke_tensors[gpu], 9 * sizeof(double)), gpu);

        // Copy forces to device asynchronously
        CUDA_CHECK_MULTI(cudaMemcpyAsync(d_forces_locals[gpu], local_forces.data(),
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice, ctx->streams[gpu]), gpu);

        // Execute step B on this GPU using stream (async)
        velocity_verlet_step_b(
            domain->d_velocities,
            d_forces_locals[gpu],
            domain->d_masses,
            dt,
            domain->natoms_local,
            d_kes[gpu],
            d_ke_tensors[gpu],
            ctx->streams[gpu]  // Use per-GPU stream for async execution
        );
    }

    // PHASE 2: Synchronize all GPUs
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);
        CUDA_CHECK_MULTI(cudaStreamSynchronize(ctx->streams[gpu]), gpu);
    }

    // PHASE 3: Gather results and accumulate
    double total_ke = 0.0;
    double total_tensor[9] = {0.0};

    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        double local_ke;
        double local_tensor[9];
        CUDA_CHECK_MULTI(cudaMemcpy(&local_ke, d_kes[gpu], sizeof(double), cudaMemcpyDeviceToHost), gpu);
        CUDA_CHECK_MULTI(cudaMemcpy(local_tensor, d_ke_tensors[gpu], 9 * sizeof(double), cudaMemcpyDeviceToHost), gpu);

        total_ke += local_ke;
        for (int i = 0; i < 9; ++i) {
            total_tensor[i] += local_tensor[i];
        }
    }

    // PHASE 4: Cleanup temporary memory
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);
        cudaFree(d_forces_locals[gpu]);
        cudaFree(d_kes[gpu]);
        cudaFree(d_ke_tensors[gpu]);
    }

    // Return total values
    *kinetic_energy = total_ke;
    for (int i = 0; i < 9; ++i) {
        kinetic_tensor[i] = total_tensor[i];
    }
}

void gather_coordinates(MultiGPUContext* ctx, double* coordinates) {
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        // Copy from device
        std::vector<double> local_coords(domain->natoms_local * 3);
        CUDA_CHECK_MULTI(cudaMemcpy(local_coords.data(), domain->d_coordinates,
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyDeviceToHost), gpu);

        // Scatter to global array
        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                coordinates[global_idx * 3 + d] = local_coords[i * 3 + d];
            }
        }
    }
}

void gather_velocities(MultiGPUContext* ctx, double* velocities) {
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        std::vector<double> local_vels(domain->natoms_local * 3);
        CUDA_CHECK_MULTI(cudaMemcpy(local_vels.data(), domain->d_velocities,
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyDeviceToHost), gpu);

        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                velocities[global_idx * 3 + d] = local_vels[i * 3 + d];
            }
        }
    }
}

void gather_forces(MultiGPUContext* ctx, double* forces) {
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        std::vector<double> local_forces(domain->natoms_local * 3);
        CUDA_CHECK_MULTI(cudaMemcpy(local_forces.data(), domain->d_forces,
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyDeviceToHost), gpu);

        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                forces[global_idx * 3 + d] = local_forces[i * 3 + d];
            }
        }
    }
}

} // namespace cuda
} // namespace fennol
