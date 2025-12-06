#ifndef FENNOL_CUDA_GNN_MLP_CUH
#define FENNOL_CUDA_GNN_MLP_CUH

#include <cublas_v2.h>

namespace fennol {
namespace cuda {
namespace gnn {

/**
 * MLP (Multi-Layer Perceptron) operations for GNN.
 *
 * Uses cuBLAS for efficient dense matrix operations and custom kernels
 * for activation functions.
 */

/**
 * Initialize cuBLAS handle (call once at startup).
 */
void init_cublas();

/**
 * Cleanup cuBLAS handle (call at shutdown).
 */
void cleanup_cublas();

/**
 * Dense layer: Y = activation(X * W + b)
 *
 * Uses cuBLAS for matrix multiplication and custom kernel for activation.
 *
 * @param batch Batch size (number of nodes/edges)
 * @param in_dim Input dimension
 * @param out_dim Output dimension
 * @param input Input matrix [batch, in_dim] (row-major)
 * @param weights Weight matrix [in_dim, out_dim] (column-major for cuBLAS)
 * @param bias Bias vector [out_dim]
 * @param output Output matrix [batch, out_dim] (row-major)
 * @param activation Activation function: "silu", "relu", "none"
 */
void dense_layer(
    int batch,
    int in_dim,
    int out_dim,
    const double* input,
    const double* weights,
    const double* bias,
    double* output,
    const char* activation = "silu"
);

/**
 * Multi-layer MLP: Y = MLP(X)
 *
 * Applies a sequence of dense layers with activations.
 *
 * @param batch Batch size
 * @param layer_dims Array of layer dimensions [num_layers + 1]
 *                   e.g., [64, 128, 128, 64] for 3-layer MLP
 * @param num_layers Number of layers (transitions between dimensions)
 * @param input Input matrix [batch, layer_dims[0]]
 * @param weights Array of weight matrices (concatenated)
 * @param biases Array of bias vectors (concatenated)
 * @param output Output matrix [batch, layer_dims[num_layers]]
 * @param temp_buffer Temporary buffer for intermediate activations
 *                    Size: batch * max(layer_dims)
 * @param activation Activation function for hidden layers
 */
void mlp_forward(
    int batch,
    const int* layer_dims,
    int num_layers,
    const double* input,
    const double** weights,
    const double** biases,
    double* output,
    double* temp_buffer,
    const char* activation = "silu"
);

} // namespace gnn
} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_GNN_MLP_CUH
