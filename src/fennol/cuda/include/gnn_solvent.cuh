#ifndef FENNOL_GNN_SOLVENT_CUH
#define FENNOL_GNN_SOLVENT_CUH

namespace fennol {
namespace cuda {
namespace gnn {

/**
 * Build neighbor list for GNN message passing.
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param cutoff Cutoff distance for edges
 * @param edge_src Output: source indices [max_edges]
 * @param edge_dst Output: destination indices [max_edges]
 * @param num_edges Output: actual number of edges
 * @param max_edges Maximum number of edges to store
 */
void build_neighborlist(
    int natoms,
    const double* coords,
    double cutoff,
    int* edge_src,
    int* edge_dst,
    int* num_edges,
    int max_edges
);

/**
 * Compute RBF expansion of edge distances.
 *
 * @param nedges Number of edges
 * @param distances Edge distances [nedges]
 * @param cutoff Cutoff distance
 * @param num_rbf Number of RBF functions
 * @param rbf_features Output: RBF features [nedges, num_rbf]
 */
void compute_rbf_features(
    int nedges,
    const double* distances,
    double cutoff,
    int num_rbf,
    double* rbf_features
);

/**
 * Aggregate messages to nodes (scatter-reduce).
 *
 * @param natoms Number of atoms
 * @param nedges Number of edges
 * @param edge_dst Destination node indices [nedges]
 * @param messages Edge messages [nedges, msg_dim]
 * @param msg_dim Message dimension
 * @param aggregated Output: aggregated messages [natoms, msg_dim]
 */
void aggregate_messages(
    int natoms,
    int nedges,
    const int* edge_dst,
    const double* messages,
    int msg_dim,
    double* aggregated
);

/**
 * Full GNN inference forward pass (CUDA-optimized).
 *
 * Note: This is a simplified version. Full implementation would
 * include all GNN layers and force prediction.
 *
 * @param natoms Number of atoms
 * @param coords Atomic coordinates [natoms, 3]
 * @param atomic_numbers Atomic numbers [natoms]
 * @param solvent_id Solvent identifier
 * @param cutoff Edge cutoff distance
 * @param forces Output: predicted forces [natoms, 3]
 */
void gnn_predict_forces(
    int natoms,
    const double* coords,
    const int* atomic_numbers,
    int solvent_id,
    double cutoff,
    double* forces
);

} // namespace gnn
} // namespace cuda
} // namespace fennol

#endif // FENNOL_GNN_SOLVENT_CUH
