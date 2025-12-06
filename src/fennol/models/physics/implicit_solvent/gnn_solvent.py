"""
GNN-based implicit solvent model.

Based on the Riniker lab's GNNImplicitSolvent approach.
Uses message-passing neural networks to predict solvation forces.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, List
from functools import partial
import flax.linen as nn

from .base import ImplicitSolventModel
# Note: ATOMIC_MASSES not needed for GNN model


class RBFExpansion(nn.Module):
    """Radial Basis Function expansion for distances."""

    num_rbf: int = 20
    cutoff: float = 5.0

    @nn.compact
    def __call__(self, distances):
        """
        Expand distances using Gaussian RBF.

        Args:
            distances: [num_edges] pairwise distances

        Returns:
            rbf_features: [num_edges, num_rbf]
        """
        # RBF centers
        centers = jnp.linspace(0, self.cutoff, self.num_rbf)

        # RBF width
        gamma = 10.0 / self.cutoff

        # Gaussian RBF: exp(-γ(r - μ)²)
        distances = distances[:, None]  # [num_edges, 1]
        rbf = jnp.exp(-gamma * (distances - centers[None, :])**2)

        # Apply cutoff
        cutoff_values = 0.5 * (jnp.cos(jnp.pi * distances / self.cutoff) + 1.0)
        cutoff_values = jnp.where(distances < self.cutoff, cutoff_values, 0.0)

        return rbf * cutoff_values


class EdgeUpdate(nn.Module):
    """Edge update layer in message passing."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, node_features_i, node_features_j, edge_features):
        """
        Update edge features based on node features.

        Args:
            node_features_i: [num_edges, node_dim] source node features
            node_features_j: [num_edges, node_dim] target node features
            edge_features: [num_edges, edge_dim] current edge features

        Returns:
            updated_edges: [num_edges, hidden_dim]
        """
        # Concatenate features
        x = jnp.concatenate([node_features_i, node_features_j, edge_features], axis=-1)

        # MLP
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)

        return x


class NodeUpdate(nn.Module):
    """Node update layer in message passing."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, node_features, aggregated_messages):
        """
        Update node features based on aggregated messages.

        Args:
            node_features: [num_nodes, node_dim] current node features
            aggregated_messages: [num_nodes, msg_dim] aggregated edge messages

        Returns:
            updated_nodes: [num_nodes, hidden_dim]
        """
        # Concatenate
        x = jnp.concatenate([node_features, aggregated_messages], axis=-1)

        # MLP with residual
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.silu(h)
        h = nn.Dense(self.hidden_dim)(h)

        # Residual connection if dimensions match
        if node_features.shape[-1] == self.hidden_dim:
            h = h + node_features

        h = nn.silu(h)

        return h


class MessagePassingLayer(nn.Module):
    """Single message passing layer."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, node_features, edge_features, edge_src, edge_dst):
        """
        Perform one round of message passing.

        Args:
            node_features: [num_nodes, node_dim]
            edge_features: [num_edges, edge_dim]
            edge_src: [num_edges] source node indices
            edge_dst: [num_edges] destination node indices

        Returns:
            updated_node_features: [num_nodes, hidden_dim]
            updated_edge_features: [num_edges, hidden_dim]
        """
        num_nodes = node_features.shape[0]

        # Gather node features for edges
        node_features_i = node_features[edge_src]
        node_features_j = node_features[edge_dst]

        # Update edges (compute messages)
        messages = EdgeUpdate(self.hidden_dim)(
            node_features_i, node_features_j, edge_features
        )

        # Aggregate messages to nodes (sum over incoming edges)
        aggregated = jnp.zeros((num_nodes, self.hidden_dim))
        aggregated = aggregated.at[edge_dst].add(messages)

        # Update nodes
        updated_nodes = NodeUpdate(self.hidden_dim)(node_features, aggregated)

        return updated_nodes, messages


class GNNForcePredictor(nn.Module):
    """
    GNN model for predicting solvation forces.

    Architecture:
    1. Node/edge feature construction
    2. Message passing layers (3-5 layers)
    3. Force prediction head (per-atom 3D forces)
    """

    num_layers: int = 4
    hidden_dim: int = 128
    num_rbf: int = 20
    rbf_cutoff: float = 5.0
    num_solvents: int = 5

    @nn.compact
    def __call__(
        self,
        coords: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        solvent_id: int = 0,
        edge_src: Optional[jnp.ndarray] = None,
        edge_dst: Optional[jnp.ndarray] = None,
    ):
        """
        Predict solvation forces.

        Args:
            coords: [num_atoms, 3] atomic coordinates
            atomic_numbers: [num_atoms] atomic numbers
            solvent_id: int, solvent identifier
            edge_src: [num_edges] source indices (optional, computed if None)
            edge_dst: [num_edges] destination indices (optional)

        Returns:
            forces: [num_atoms, 3] predicted solvation forces
        """
        num_atoms = coords.shape[0]

        # Build graph if not provided
        if edge_src is None:
            edge_src, edge_dst = self._build_edges(coords, self.rbf_cutoff)

        # Compute distances
        dr = coords[edge_src] - coords[edge_dst]
        distances = jnp.linalg.norm(dr, axis=-1)

        # Node features: atomic embeddings
        atom_embeddings = nn.Embed(
            num_embeddings=100,  # Up to atomic number 100
            features=self.hidden_dim
        )(atomic_numbers)

        # Solvent embedding
        solvent_embedding = nn.Embed(
            num_embeddings=self.num_solvents,
            features=16
        )(solvent_id)

        # Add solvent context to each atom
        solvent_features = jnp.tile(solvent_embedding[None, :], (num_atoms, 1))
        node_features = jnp.concatenate([atom_embeddings, solvent_features], axis=-1)

        # Edge features: RBF-expanded distances
        edge_features = RBFExpansion(self.num_rbf, self.rbf_cutoff)(distances)

        # Message passing layers
        for i in range(self.num_layers):
            node_features, edge_features = MessagePassingLayer(self.hidden_dim)(
                node_features, edge_features, edge_src, edge_dst
            )

        # Force prediction head
        forces = self._predict_forces(node_features, coords, edge_src, edge_dst, dr)

        return forces

    def _build_edges(self, coords, cutoff):
        """
        Build edge list based on cutoff distance.

        Args:
            coords: [num_atoms, 3]
            cutoff: float

        Returns:
            edge_src: [num_edges] source indices
            edge_dst: [num_edges] destination indices
        """
        num_atoms = coords.shape[0]

        # Compute all pairwise distances
        dr = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
        distances = jnp.linalg.norm(dr, axis=-1)  # [N, N]

        # Find pairs within cutoff
        mask = (distances < cutoff) & (distances > 0.0)  # Exclude self

        # Get edge indices
        edge_src, edge_dst = jnp.where(mask)

        return edge_src, edge_dst

    def _predict_forces(self, node_features, coords, edge_src, edge_dst, dr):
        """
        Predict forces from node features.

        Two approaches:
        1. Direct: f_i = MLP(h_i)
        2. Pairwise: f_ij = MLP([h_i, h_j, r_ij]), f_i = Σ_j f_ij

        Using approach 2 for better physics (pair additivity).
        """
        num_atoms = coords.shape[0]

        # Gather features for edges
        h_i = node_features[edge_src]
        h_j = node_features[edge_dst]

        # Normalize displacement vectors
        r_ij = jnp.linalg.norm(dr, axis=-1, keepdims=True)
        r_ij_safe = jnp.maximum(r_ij, 1e-6)
        dr_norm = dr / r_ij_safe

        # Predict force magnitude for each pair
        edge_input = jnp.concatenate([h_i, h_j], axis=-1)
        force_magnitudes = nn.Dense(64)(edge_input)
        force_magnitudes = nn.silu(force_magnitudes)
        force_magnitudes = nn.Dense(1)(force_magnitudes)  # [num_edges, 1]

        # Force vectors (magnitude * direction)
        pairwise_forces = force_magnitudes * dr_norm  # [num_edges, 3]

        # Aggregate to atoms
        forces = jnp.zeros((num_atoms, 3))
        forces = forces.at[edge_src].add(pairwise_forces)

        return forces


class GNNImplicitSolvent(ImplicitSolventModel):
    """
    GNN-based implicit solvent model.

    This model uses a trained graph neural network to predict
    solvation forces directly from molecular geometry.
    """

    def __init__(self, parameters: Dict):
        """
        Initialize GNN solvent model.

        Args:
            parameters: dict with keys:
                - checkpoint_path: path to trained model weights
                - num_layers: number of message passing layers (default: 4)
                - hidden_dim: hidden dimension (default: 128)
                - rbf_cutoff: cutoff for RBF expansion (default: 5.0 Å)
                - solvent: solvent name (default: "water")
        """
        super().__init__(parameters)

        self.checkpoint_path = parameters.get("checkpoint_path", None)
        self.num_layers = parameters.get("num_layers", 4)
        self.hidden_dim = parameters.get("hidden_dim", 128)
        self.rbf_cutoff = parameters.get("rbf_cutoff", 5.0)
        self.solvent = parameters.get("solvent", "water")

        # Solvent ID mapping
        self.solvent_map = {
            "water": 0,
            "methanol": 1,
            "acetonitrile": 2,
            "dmso": 3,
            "chloroform": 4,
        }
        self.solvent_id = self.solvent_map.get(self.solvent.lower(), 0)

        # Initialize model
        self.model = GNNForcePredictor(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            rbf_cutoff=self.rbf_cutoff,
        )

        # Initialize parameters
        self.params = self._initialize_params()

        # Load checkpoint if provided
        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

        print(f"# Initialized GNN Implicit Solvent Model")
        print(f"#   Solvent: {self.solvent}")
        print(f"#   Layers: {self.num_layers}")
        print(f"#   Hidden dim: {self.hidden_dim}")
        print(f"#   RBF cutoff: {self.rbf_cutoff} Å")
        if self.checkpoint_path:
            print(f"#   Checkpoint: {self.checkpoint_path}")
        else:
            print(f"#   WARNING: No checkpoint loaded - using random initialization!")

    def _initialize_params(self):
        """Initialize model parameters."""
        # Dummy input for initialization
        dummy_coords = jnp.zeros((10, 3))
        dummy_atomic_numbers = jnp.ones(10, dtype=jnp.int32)

        # Initialize
        rng = jax.random.PRNGKey(0)
        params = self.model.init(
            rng, dummy_coords, dummy_atomic_numbers, self.solvent_id
        )

        return params

    def _load_checkpoint(self, checkpoint_path):
        """Load trained model weights."""
        import pickle
        from pathlib import Path

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_file, 'rb') as f:
            self.params = pickle.load(f)

        print(f"# Loaded checkpoint from: {checkpoint_path}")

    def compute_energy_forces(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        box: Optional[jnp.ndarray] = None,
        neighborlist: Optional[Tuple] = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute solvation forces using GNN.

        Note: This is a forces-only model (energy = 0).

        Args:
            coords: [natoms, 3] atomic coordinates
            charges: [natoms] (not used in GNN, kept for interface)
            atomic_numbers: [natoms] atomic numbers
            box: optional box vectors (PBC not yet supported)
            neighborlist: optional precomputed neighbor list

        Returns:
            energy: float (0.0 for forces-only model)
            forces: [natoms, 3] solvation forces
        """
        # Use CUDA if available
        if self.has_cuda:
            return self._compute_cuda(coords, atomic_numbers)
        else:
            return self._compute_jax(coords, atomic_numbers)

    def _compute_jax(
        self,
        coords: jnp.ndarray,
        atomic_numbers: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """JAX implementation."""

        # Predict forces
        forces = self.model.apply(
            self.params,
            coords,
            atomic_numbers,
            self.solvent_id
        )

        # Energy is not predicted (forces-only model)
        energy = 0.0

        return energy, forces

    def _compute_cuda(
        self,
        coords: jnp.ndarray,
        atomic_numbers: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray]:
        """CUDA implementation using optimized kernels."""
        try:
            from fennol import cuda as fennol_cuda
            import numpy as np

            # Convert to numpy
            coords_np = np.array(coords)
            atomic_numbers_np = np.array(atomic_numbers)

            # Call CUDA kernel for GNN inference
            forces = fennol_cuda.gnn_predict_forces(
                coords_np,
                atomic_numbers_np,
                self.params,  # Model weights
                self.solvent_id,
                self.rbf_cutoff
            )

            # Convert back
            forces_jax = jnp.array(forces)
            energy = 0.0

            return energy, forces_jax

        except (ImportError, AttributeError) as e:
            print(f"# Warning: CUDA backend not available ({e}), falling back to JAX")
            return self._compute_jax(coords, atomic_numbers)

    def _check_cuda_available(self) -> bool:
        """Check if CUDA kernels are available."""
        try:
            from fennol import cuda
            return hasattr(cuda, "gnn_predict_forces")
        except (ImportError, AttributeError):
            return False
