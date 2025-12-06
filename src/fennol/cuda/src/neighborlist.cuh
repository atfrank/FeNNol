/**
 * Neighbor List for GB Implicit Solvent
 *
 * Implements Verlet neighbor list to reduce O(N²) → O(N×M) complexity.
 *
 * Key features:
 * - Verlet buffer (skin): Build with cutoff+skin, rebuild when atoms move > skin/2
 * - Compact storage: Flat array with offsets (cache-friendly)
 * - Asymmetric: Store j for each i (simpler than symmetric storage)
 *
 * Performance:
 * - Current: Evaluate all 3.1M pairs for DHFR
 * - With neighbor list: Evaluate only ~105K pairs within cutoff
 * - Expected speedup: 2.5-5× (accounting for build overhead)
 */

#pragma once

#include <cuda_runtime.h>

namespace fennol {
namespace cuda {
namespace neighborlist {

/**
 * Neighbor list data structure (device-side)
 *
 * Memory layout:
 * - neighbor_atoms: Flat array [total_neighbors] of atom indices
 * - neighbor_counts: Number of neighbors per atom [natoms]
 * - neighbor_offsets: Start index in neighbor_atoms for each atom [natoms]
 *
 * Example for 3 atoms:
 *   Atom 0 has neighbors [1, 2]
 *   Atom 1 has neighbors [0, 2, 3]
 *   Atom 2 has neighbors [1]
 *
 *   neighbor_atoms   = [1, 2, 0, 2, 3, 1]
 *   neighbor_counts  = [2, 3, 1]
 *   neighbor_offsets = [0, 2, 5]
 */
struct NeighborList {
    // Device arrays
    int* neighbor_atoms;      // Flat array of neighbor indices [total_neighbors]
    int* neighbor_counts;     // Number of neighbors per atom [natoms]
    int* neighbor_offsets;    // Start index for each atom [natoms]

    // Build parameters
    int natoms;               // Number of atoms
    int max_neighbors;        // Maximum neighbors per atom (buffer size)
    float cutoff;             // Interaction cutoff (Å)
    float skin;               // Verlet skin/buffer (Å)
    float build_cutoff;       // cutoff + skin (Å)

    // Rebuild tracking
    double* build_coords;     // Coordinates at last build [natoms, 3]
    float max_displacement;   // Maximum atom movement since build
    float rebuild_threshold;  // Rebuild when max_displacement > this

    // Statistics
    int total_neighbors;      // Total neighbors across all atoms
    int overflow_count;       // Number of atoms that exceeded max_neighbors
    bool needs_rebuild;       // Flag: needs rebuild before next use
};

/**
 * Build neighbor list on GPU.
 *
 * Algorithm:
 * - Each thread handles one atom i
 * - Find all atoms j within build_cutoff of i
 * - Store neighbor indices in flat array
 *
 * Complexity: O(N²) but only once every 10-20 steps
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param build_cutoff Cutoff + skin distance
 * @param max_neighbors Maximum neighbors per atom
 * @param neighbor_atoms Output: flat array of neighbor indices
 * @param neighbor_counts Output: number of neighbors per atom
 * @param neighbor_offsets Output: start index for each atom
 * @param overflow_count Output: number of atoms that exceeded max_neighbors
 */
__global__ void build_neighborlist_kernel(
    int natoms,
    const double* __restrict__ coords,
    float build_cutoff,
    int max_neighbors,
    int* __restrict__ neighbor_atoms,
    int* __restrict__ neighbor_counts,
    int* __restrict__ neighbor_offsets,
    int* __restrict__ overflow_count
);

/**
 * Check if neighbor list needs rebuilding.
 *
 * Computes maximum displacement since last build:
 *   max_disp = max_i ||coords[i] - build_coords[i]||
 *
 * If max_disp > rebuild_threshold, set needs_rebuild flag.
 *
 * Uses parallel reduction to find maximum displacement.
 *
 * @param natoms Number of atoms
 * @param coords Current coordinates [natoms, 3]
 * @param build_coords Coordinates at last build [natoms, 3]
 * @param max_displacement Output: maximum displacement
 */
__global__ void check_rebuild_kernel(
    int natoms,
    const double* __restrict__ coords,
    const double* __restrict__ build_coords,
    float* __restrict__ max_displacement
);

/**
 * Copy coordinates for rebuild tracking.
 *
 * @param natoms Number of atoms
 * @param coords Source coordinates [natoms, 3]
 * @param build_coords Destination [natoms, 3]
 */
__global__ void copy_coords_kernel(
    int natoms,
    const double* __restrict__ coords,
    double* __restrict__ build_coords
);

} // namespace neighborlist
} // namespace cuda
} // namespace fennol
