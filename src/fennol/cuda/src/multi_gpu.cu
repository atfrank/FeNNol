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
        cudaSetDevice(i);
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
        cudaSetDevice(i);
        cudaStreamCreate(&ctx->streams[i]);

        // Initialize domain
        GPUDomain* domain = new GPUDomain();
        domain->gpu_id = i;
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
        cudaSetDevice(i);

        GPUDomain* domain = ctx->domains[i];

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
            if (x >= core_xmin && x < core_xmax) {
                local_indices.push_back(i);
            }
        }

        domain->natoms_local = local_indices.size();

        // Allocate and copy local atom indices
        domain->local_atom_indices = new int[domain->natoms_local];
        std::copy(local_indices.begin(), local_indices.end(), domain->local_atom_indices);

        // Allocate device memory
        cudaSetDevice(gpu);

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
        cudaSetDevice(gpu);

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
        cudaSetDevice(gpu);
        cudaStreamSynchronize(ctx->streams[gpu]);
    }
}

void multi_gpu_velocity_verlet_step_a(
    MultiGPUContext* ctx,
    const double* forces,
    double dt
) {
    // Launch step A on all GPUs in parallel
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        cudaSetDevice(gpu);

        // Pack forces for this domain
        std::vector<double> local_forces(domain->natoms_local * 3);
        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                local_forces[i * 3 + d] = forces[global_idx * 3 + d];
            }
        }

        // Copy forces to device
        double* d_forces_local;
        CUDA_CHECK_MULTI(cudaMalloc(&d_forces_local, domain->natoms_local * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMemcpy(d_forces_local, local_forces.data(),
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice), gpu);

        // Execute step A on this GPU
        velocity_verlet_step_a(
            domain->d_coordinates,
            domain->d_velocities,
            d_forces_local,
            domain->d_masses,
            dt,
            domain->natoms_local
        );

        cudaFree(d_forces_local);
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
    double total_ke = 0.0;
    double total_tensor[9] = {0.0};

    // Launch step B on all GPUs in parallel
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        cudaSetDevice(gpu);

        // Pack forces
        std::vector<double> local_forces(domain->natoms_local * 3);
        for (int i = 0; i < domain->natoms_local; ++i) {
            int global_idx = domain->local_atom_indices[i];
            for (int d = 0; d < 3; ++d) {
                local_forces[i * 3 + d] = forces[global_idx * 3 + d];
            }
        }

        double* d_forces_local;
        CUDA_CHECK_MULTI(cudaMalloc(&d_forces_local, domain->natoms_local * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMemcpy(d_forces_local, local_forces.data(),
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyHostToDevice), gpu);

        // Allocate output
        double *d_ke, *d_ke_tensor;
        CUDA_CHECK_MULTI(cudaMalloc(&d_ke, sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&d_ke_tensor, 9 * sizeof(double)), gpu);

        // Execute step B
        velocity_verlet_step_b(
            domain->d_velocities,
            d_forces_local,
            domain->d_masses,
            dt,
            domain->natoms_local,
            d_ke,
            d_ke_tensor
        );

        // Copy results back and accumulate
        double local_ke;
        double local_tensor[9];
        CUDA_CHECK_MULTI(cudaMemcpy(&local_ke, d_ke, sizeof(double), cudaMemcpyDeviceToHost), gpu);
        CUDA_CHECK_MULTI(cudaMemcpy(local_tensor, d_ke_tensor, 9 * sizeof(double), cudaMemcpyDeviceToHost), gpu);

        total_ke += local_ke;
        for (int i = 0; i < 9; ++i) {
            total_tensor[i] += local_tensor[i];
        }

        cudaFree(d_forces_local);
        cudaFree(d_ke);
        cudaFree(d_ke_tensor);
    }

    *kinetic_energy = total_ke;
    for (int i = 0; i < 9; ++i) {
        kinetic_tensor[i] = total_tensor[i];
    }
}

void gather_coordinates(MultiGPUContext* ctx, double* coordinates) {
    for (int gpu = 0; gpu < ctx->ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        cudaSetDevice(gpu);

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
        cudaSetDevice(gpu);

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
        cudaSetDevice(gpu);

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
