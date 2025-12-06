#ifndef FENNOL_CUDA_MULTI_GPU_CUH
#define FENNOL_CUDA_MULTI_GPU_CUH

#include "common.cuh"
#include <vector>

namespace fennol {
namespace cuda {

/**
 * Multi-GPU domain decomposition system
 *
 * Divides the simulation system across multiple GPUs for parallel execution.
 * Uses spatial domain decomposition with halo regions for communication.
 */

/**
 * GPU domain information
 * Each GPU handles a spatial subdomain with halo regions for neighbor communication
 */
struct GPUDomain {
    int gpu_id;                    // GPU device ID
    int natoms_local;              // Number of atoms in this domain (excluding halo)
    int natoms_with_halo;          // Total atoms including halo
    int max_halo_capacity;         // Maximum halo atoms that can be stored (for bounds checking)
    int* local_atom_indices;       // Indices of local atoms in global array
    int* halo_atom_indices;        // Indices of halo atoms

    // Spatial bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    double bounds[6];

    // Device pointers
    double* d_coordinates;         // Local coordinates
    double* d_velocities;          // Local velocities
    double* d_forces;              // Local forces
    double* d_masses;              // Local masses

    // Halo communication buffer
    double* d_halo_coords;         // Coordinates from neighboring domains

    // Neighbor information
    std::vector<int> neighbor_gpus;  // IDs of neighboring GPUs
    std::vector<int> send_counts;    // Number of atoms to send to each neighbor
    std::vector<int> recv_counts;    // Number of atoms to receive from each neighbor
};

/**
 * Multi-GPU context
 * Manages multiple GPUs for parallel MD simulation
 */
struct MultiGPUContext {
    int ngpus;                     // Number of GPUs
    std::vector<GPUDomain*> domains;  // Domain for each GPU
    cudaStream_t* streams;         // CUDA stream for each GPU

    // Global system information
    int natoms_global;             // Total number of atoms
    double cutoff;                 // Cutoff distance for halo region

    // Synchronization
    bool use_peer_access;          // Whether to use peer-to-peer GPU access
};

/**
 * Initialize multi-GPU context
 * Divides system across available GPUs
 *
 * @param natoms - total number of atoms
 * @param coordinates - global coordinates [natoms, 3]
 * @param velocities - global velocities [natoms, 3]
 * @param masses - atomic masses [natoms]
 * @param box_size - simulation box size [3]
 * @param cutoff - cutoff distance for halo region
 * @param ngpus - number of GPUs to use (0 = auto-detect)
 * @return Initialized multi-GPU context
 */
MultiGPUContext* initialize_multi_gpu(
    int natoms,
    const double* coordinates,
    const double* velocities,
    const double* masses,
    const double* box_size,
    double cutoff,
    int ngpus = 0
);

/**
 * Cleanup multi-GPU context
 *
 * @param ctx - multi-GPU context to cleanup
 */
void cleanup_multi_gpu(MultiGPUContext* ctx);

/**
 * Distribute atoms across GPUs using spatial domain decomposition
 *
 * @param ctx - multi-GPU context
 * @param coordinates - global coordinates [natoms, 3]
 * @param velocities - global velocities [natoms, 3]
 * @param masses - atomic masses [natoms]
 * @param box_size - simulation box size [3]
 */
void distribute_atoms(
    MultiGPUContext* ctx,
    const double* coordinates,
    const double* velocities,
    const double* masses,
    const double* box_size
);

/**
 * Exchange halo regions between neighboring GPUs
 * Communicates atom data in halo regions for force calculations
 *
 * @param ctx - multi-GPU context
 */
void exchange_halos(MultiGPUContext* ctx);

/**
 * Multi-GPU Velocity Verlet step A
 * Executes step A on all GPUs in parallel
 *
 * @param ctx - multi-GPU context
 * @param forces - global forces [natoms, 3]
 * @param dt - timestep
 */
void multi_gpu_velocity_verlet_step_a(
    MultiGPUContext* ctx,
    const double* forces,
    double dt
);

/**
 * Multi-GPU Velocity Verlet step B
 * Executes step B on all GPUs in parallel
 *
 * @param ctx - multi-GPU context
 * @param forces - global forces [natoms, 3]
 * @param dt - timestep
 * @param kinetic_energy [out] - total kinetic energy
 * @param kinetic_tensor [9, out] - kinetic energy tensor
 */
void multi_gpu_velocity_verlet_step_b(
    MultiGPUContext* ctx,
    const double* forces,
    double dt,
    double* kinetic_energy,
    double* kinetic_tensor
);

/**
 * Gather coordinates from all GPUs to host
 *
 * @param ctx - multi-GPU context
 * @param coordinates [natoms, 3, out] - gathered coordinates
 */
void gather_coordinates(
    MultiGPUContext* ctx,
    double* coordinates
);

/**
 * Gather velocities from all GPUs to host
 *
 * @param ctx - multi-GPU context
 * @param velocities [natoms, 3, out] - gathered velocities
 */
void gather_velocities(
    MultiGPUContext* ctx,
    double* velocities
);

/**
 * Gather forces from all GPUs to host
 *
 * @param ctx - multi-GPU context
 * @param forces [natoms, 3, out] - gathered forces
 */
void gather_forces(
    MultiGPUContext* ctx,
    double* forces
);

/**
 * Get optimal GPU count for system size
 *
 * @param natoms - number of atoms
 * @return recommended number of GPUs
 */
int get_optimal_gpu_count(int natoms);

/**
 * Check if peer-to-peer access is available between GPUs
 *
 * @param gpu1 - first GPU ID
 * @param gpu2 - second GPU ID
 * @return true if peer access available
 */
bool check_peer_access(int gpu1, int gpu2);

/**
 * Enable peer-to-peer access between all GPUs
 *
 * @param ngpus - number of GPUs
 * @return true if successful
 */
bool enable_peer_access(int ngpus);

} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_MULTI_GPU_CUH
