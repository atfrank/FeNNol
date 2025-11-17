# GNN-Based Implicit Solvent Model - Design Document

## Overview

Implementation of a Graph Neural Network (GNN) based implicit solvent model inspired by the Riniker lab's work. This model learns to predict solvation forces directly from molecular geometry, achieving explicit-solvent accuracy with significant speedup.

## Architecture

### High-Level Design

```
Molecular Structure
        ↓
  Graph Construction (atoms → nodes, bonds → edges)
        ↓
  Message Passing GNN (learn interaction patterns)
        ↓
  Force Prediction (per-atom solvation forces)
        ↓
  Add to MM forces → Total forces
```

### GNN Architecture Details

**Model Type**: Message Passing Neural Network (MPNN)

**Input Features**:
- **Node features** (per atom):
  - Atomic number (embedded)
  - Partial charge
  - Atom type features
  - Local geometry (coordination number, hybridization)

- **Edge features** (per pair):
  - Distance (r_ij)
  - Bond type (if bonded)
  - Radial basis function (RBF) expansion of distance

**Message Passing Layers** (3-5 layers):
```python
# For each layer l:
# 1. Edge update
e_ij^(l+1) = MLP_edge([h_i^(l), h_j^(l), e_ij^(l)])

# 2. Message aggregation
m_i^(l+1) = Σ_j e_ij^(l+1)

# 3. Node update
h_i^(l+1) = MLP_node([h_i^(l), m_i^(l+1)])
```

**Output Layer**:
```python
# Per-atom force prediction
F_i = MLP_force(h_i^(final)) ∈ ℝ³
```

### CUDA Optimization Strategy

#### Critical Operations to Optimize

1. **Graph Construction** (O(N²) or O(N) with cutoff)
   - CUDA kernel for neighbor list construction
   - Cutoff-based to reduce O(N²) → O(N)

2. **RBF Distance Expansion** (O(E))
   - CUDA kernel for batch RBF computation
   - Fused with distance calculation

3. **Message Passing** (O(E × D), where D = hidden dim)
   - CUDA kernel for edge message computation
   - Segment reduction for aggregation
   - Fused operations to reduce memory traffic

4. **Force Prediction** (O(N × D))
   - Batched MLP forward pass
   - Use cuBLAS for matrix multiplications

### Model Hyperparameters

```python
config = {
    # Architecture
    "num_layers": 4,
    "hidden_dim": 128,
    "edge_dim": 64,
    "num_rbf": 20,
    "rbf_cutoff": 5.0,  # Angstroms

    # Optimization
    "learning_rate": 1e-4,
    "batch_size": 32,
    "weight_decay": 1e-5,

    # Training
    "max_epochs": 100,
    "early_stopping_patience": 10,

    # Multi-solvent
    "num_solvents": 5,  # Water, Methanol, Acetonitrile, etc.
    "solvent_embedding_dim": 16,
}
```

## Training Procedure

### Data Requirements

1. **Training Data**:
   - Molecular conformations (coordinates)
   - Reference forces from explicit solvent MD
   - Solvent identity labels
   - ~1-3M training samples

2. **Data Generation**:
   ```python
   # For each molecule:
   # 1. Run explicit solvent MD (e.g., 10 ns)
   # 2. Sample conformations (e.g., every 100 fs → 100k frames)
   # 3. Extract forces on solute atoms
   # 4. Store: (coords, forces, solvent_id)
   ```

### Training Loop

```python
for epoch in epochs:
    for batch in dataloader:
        # Forward pass
        coords, forces_target, solvent_id = batch
        forces_pred = model(coords, solvent_id)

        # Loss: MSE on forces
        loss = mse_loss(forces_pred, forces_target)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Loss Function

**Primary**: Mean Squared Error on forces
```python
L = (1/N) Σ_i ||F_i^pred - F_i^target||²
```

**Optional additions**:
- Energy conservation penalty
- Force magnitude regularization
- Per-solvent weighting

## Integration with FeNNol

### Workflow

```python
# 1. Initialize GNN solvent model
gnn_solvent = GNNImplicitSolvent(checkpoint_path="trained_model.pt")

# 2. In MD loop:
for step in md_steps:
    # Compute MM forces
    f_mm = compute_mm_forces(coords)

    # Compute GNN solvation forces
    f_solv = gnn_solvent(coords, solvent="water")

    # Total forces
    f_total = f_mm + f_solv

    # Integrate
    coords, vels = integrate(coords, vels, f_total)
```

### Model Loading

```python
class GNNImplicitSolvent(ImplicitSolventModel):
    def __init__(self, checkpoint_path, use_cuda=True):
        super().__init__(parameters={})

        # Load trained model
        self.model = load_gnn_model(checkpoint_path)

        # Move to GPU if available
        if use_cuda and cuda_available():
            self.model = self.model.cuda()

    def compute_energy_forces(self, coords, charges, atomic_numbers, ...):
        # Predict forces directly (no energy)
        forces = self.model(coords, atomic_numbers, self.solvent_id)

        # Energy = 0 (forces-only model)
        energy = 0.0

        return energy, forces
```

## CUDA Implementation Details

### Kernel 1: Neighbor List Construction

```cuda
__global__ void build_neighborlist_kernel(
    int natoms,
    const double* coords,
    double cutoff,
    int* neighbor_list,
    int* num_neighbors
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    int count = 0;
    for (int j = 0; j < natoms; j++) {
        if (i == j) continue;

        double dx = coords[i*3+0] - coords[j*3+0];
        double dy = coords[i*3+1] - coords[j*3+1];
        double dz = coords[i*3+2] - coords[j*3+2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);

        if (r < cutoff) {
            neighbor_list[i * MAX_NEIGHBORS + count] = j;
            count++;
        }
    }
    num_neighbors[i] = count;
}
```

### Kernel 2: RBF Expansion

```cuda
__global__ void rbf_expansion_kernel(
    int nedges,
    const double* distances,
    double cutoff,
    int num_rbf,
    double* rbf_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nedges) return;

    double r = distances[idx];

    // Gaussian RBF: exp(-γ(r - μ_k)²)
    for (int k = 0; k < num_rbf; k++) {
        double mu = cutoff * k / (num_rbf - 1);
        double gamma = 10.0 / cutoff;
        rbf_features[idx * num_rbf + k] = exp(-gamma * (r - mu) * (r - mu));
    }
}
```

### Kernel 3: Message Passing (Edge Update)

```cuda
__global__ void edge_message_kernel(
    int nedges,
    const int* edge_src,
    const int* edge_dst,
    const double* node_features,  // [natoms, node_dim]
    const double* edge_features,  // [nedges, edge_dim]
    const double* weights,        // MLP weights
    double* messages              // [nedges, msg_dim]
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nedges) return;

    int i = edge_src[e];
    int j = edge_dst[e];

    // Concatenate: [h_i, h_j, e_ij]
    // Apply MLP
    // Store message

    // (Implementation with shared memory for efficiency)
}
```

### Kernel 4: Message Aggregation (Scatter-Reduce)

```cuda
__global__ void aggregate_messages_kernel(
    int natoms,
    int nedges,
    const int* edge_dst,
    const double* messages,
    double* node_messages
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    // Aggregate all messages for node i
    for (int e = 0; e < nedges; e++) {
        if (edge_dst[e] == i) {
            for (int d = 0; d < MSG_DIM; d++) {
                atomicAddDouble(&node_messages[i * MSG_DIM + d],
                               messages[e * MSG_DIM + d]);
            }
        }
    }
}
```

## Performance Targets

Based on Riniker lab results:

| Metric | Target |
|--------|--------|
| Speedup vs explicit solvent | 10-20x |
| Force accuracy (RMSE) | < 1 kcal/mol/Å |
| Throughput | > 5000 atoms/s |
| Memory overhead | < 500 MB for 5000 atoms |

## Advantages over Traditional Implicit Solvent

1. **Learned from explicit solvent**: Captures all solvation effects
2. **Molecular specificity**: Adapts to different molecular topologies
3. **Multi-solvent**: Single model handles multiple solvents
4. **No analytical approximations**: No Born radii, no SASA calculations

## Limitations

1. **Requires training data**: Need expensive explicit solvent reference
2. **Transferability**: May not generalize to very different molecules
3. **No energy**: Predicts forces only (unless trained on energies too)
4. **Model size**: Neural network checkpoint (~10-100 MB)

## Future Extensions

1. **Energy prediction**: Train on energies for thermodynamic properties
2. **Uncertainty quantification**: Ensemble or Bayesian models
3. **Active learning**: Identify poorly-predicted regions
4. **Multi-scale**: Combine with coarse-grained representations
5. **Reaction fields**: Extend to charged systems

## References

- Riniker lab: https://github.com/rinikerlab/GNNImplicitSolvent
- SchNet: Schütt et al., J. Chem. Phys. 2017
- DimeNet: Klicpera et al., ICLR 2020
- E(3)-equivariant GNNs: Satorras et al., ICML 2021
