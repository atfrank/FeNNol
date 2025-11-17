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

    // Set up neighbor information and allocate halo buffers
    for (int gpu = 0; gpu < ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        // Determine neighbors (1D decomposition: left and right)
        domain->neighbor_gpus.clear();
        domain->send_counts.clear();
        domain->recv_counts.clear();

        if (gpu > 0) {
            domain->neighbor_gpus.push_back(gpu - 1);  // Left neighbor
        }
        if (gpu < ngpus - 1) {
            domain->neighbor_gpus.push_back(gpu + 1);  // Right neighbor
        }

        // Estimate maximum halo atoms (conservative estimate)
        // Assume uniform density: atoms_per_unit = natoms / box_volume
        double box_volume = box_size[0] * box_size[1] * box_size[2];
        double density = natoms / box_volume;
        // Halo region volume: 2 * cutoff * box_y * box_z (left and right boundaries)
        double halo_volume = 2.0 * ctx->cutoff * box_size[1] * box_size[2];
        int max_halo_atoms = static_cast<int>(density * halo_volume * 1.5);  // 1.5x safety factor
        max_halo_atoms = std::max(max_halo_atoms, 100);  // Minimum buffer size

        // Allocate halo buffers on device
        // Each neighbor can send up to max_halo_atoms/2
        int buffer_size = max_halo_atoms / 2;
        CUDA_CHECK_MULTI(cudaMalloc(&domain->d_send_buffer, buffer_size * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&domain->d_recv_buffer, buffer_size * 3 * sizeof(double)), gpu);
        CUDA_CHECK_MULTI(cudaMalloc(&domain->d_halo_coords, max_halo_atoms * 3 * sizeof(double)), gpu);

        // Initialize send/recv counts for each neighbor
        for (size_t i = 0; i < domain->neighbor_gpus.size(); ++i) {
            domain->send_counts.push_back(0);
            domain->recv_counts.push_back(0);
        }
    }
}

void exchange_halos(MultiGPUContext* ctx) {
    // Halo exchange for 1D domain decomposition along X-axis
    // Each GPU exchanges boundary atoms with its neighbors

    int ngpus = ctx->ngpus;
    double cutoff = ctx->cutoff;

    // Storage for boundary atoms from each GPU
    std::vector<std::vector<double>> left_boundary_atoms(ngpus);   // Atoms to send to left neighbor
    std::vector<std::vector<double>> right_boundary_atoms(ngpus);  // Atoms to send to right neighbor

    // PHASE 1: Identify and extract boundary atoms from each GPU
    for (int gpu = 0; gpu < ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        // Copy coordinates from device to host
        std::vector<double> coords(domain->natoms_local * 3);
        CUDA_CHECK_MULTI(cudaMemcpy(coords.data(), domain->d_coordinates,
                                    domain->natoms_local * 3 * sizeof(double),
                                    cudaMemcpyDeviceToHost), gpu);

        // Compute domain boundaries (core domain without halo)
        double domain_width = domain->bounds[1] - domain->bounds[0] - 2.0 * cutoff;  // Remove halo margins
        double xmin = domain->bounds[0] + cutoff;  // Core domain start
        double xmax = domain->bounds[1] - cutoff;  // Core domain end

        // Identify atoms in left boundary (to send to left neighbor)
        if (gpu > 0) {
            for (int i = 0; i < domain->natoms_local; ++i) {
                double x = coords[i * 3 + 0];
                if (x >= xmin && x < xmin + cutoff) {
                    // Atom is in left boundary region
                    left_boundary_atoms[gpu].push_back(coords[i * 3 + 0]);
                    left_boundary_atoms[gpu].push_back(coords[i * 3 + 1]);
                    left_boundary_atoms[gpu].push_back(coords[i * 3 + 2]);
                }
            }
        }

        // Identify atoms in right boundary (to send to right neighbor)
        if (gpu < ngpus - 1) {
            for (int i = 0; i < domain->natoms_local; ++i) {
                double x = coords[i * 3 + 0];
                if (x > xmax - cutoff && x <= xmax) {
                    // Atom is in right boundary region
                    right_boundary_atoms[gpu].push_back(coords[i * 3 + 0]);
                    right_boundary_atoms[gpu].push_back(coords[i * 3 + 1]);
                    right_boundary_atoms[gpu].push_back(coords[i * 3 + 2]);
                }
            }
        }
    }

    // PHASE 2: Transfer boundary atoms to neighboring GPUs using host staging
    for (int gpu = 0; gpu < ngpus; ++gpu) {
        GPUDomain* domain = ctx->domains[gpu];
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);

        // Collect halo atoms for this GPU from its neighbors
        std::vector<double> halo_atoms;

        // Receive from left neighbor (their right boundary becomes our left halo)
        if (gpu > 0) {
            halo_atoms.insert(halo_atoms.end(),
                            right_boundary_atoms[gpu - 1].begin(),
                            right_boundary_atoms[gpu - 1].end());
        }

        // Receive from right neighbor (their left boundary becomes our right halo)
        if (gpu < ngpus - 1) {
            halo_atoms.insert(halo_atoms.end(),
                            left_boundary_atoms[gpu + 1].begin(),
                            left_boundary_atoms[gpu + 1].end());
        }

        // Update halo atom count
        domain->natoms_with_halo = domain->natoms_local + (halo_atoms.size() / 3);

        // Copy halo atoms to device
        if (!halo_atoms.empty()) {
            CUDA_CHECK_MULTI(cudaMemcpy(domain->d_halo_coords, halo_atoms.data(),
                                        halo_atoms.size() * sizeof(double),
                                        cudaMemcpyHostToDevice), gpu);
        }

        // Update send/recv counts for statistics
        int neighbor_idx = 0;
        if (gpu > 0) {
            int received_from_left = right_boundary_atoms[gpu - 1].size() / 3;
            domain->recv_counts[neighbor_idx] = received_from_left;
            neighbor_idx++;
        }
        if (gpu < ngpus - 1) {
            int received_from_right = left_boundary_atoms[gpu + 1].size() / 3;
            domain->recv_counts[neighbor_idx] = received_from_right;
        }

        neighbor_idx = 0;
        if (gpu > 0) {
            domain->send_counts[neighbor_idx] = left_boundary_atoms[gpu].size() / 3;
            neighbor_idx++;
        }
        if (gpu < ngpus - 1) {
            domain->send_counts[neighbor_idx] = right_boundary_atoms[gpu].size() / 3;
        }
    }

    // PHASE 3: Synchronize all GPUs
    for (int gpu = 0; gpu < ngpus; ++gpu) {
        CUDA_CHECK_MULTI(cudaSetDevice(gpu), gpu);
        CUDA_CHECK_MULTI(cudaStreamSynchronize(ctx->streams[gpu]), gpu);
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
