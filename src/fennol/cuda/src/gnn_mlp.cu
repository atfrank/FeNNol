#include "gnn_mlp.cuh"
#include "common.cuh"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace fennol {
namespace cuda {
namespace gnn {

// Global cuBLAS handle
static cublasHandle_t cublas_handle = nullptr;
static bool cublas_initialized = false;

/**
 * Initialize cuBLAS handle.
 */
void init_cublas() {
    if (!cublas_initialized) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize cuBLAS");
        }
        cublas_initialized = true;
    }
}

/**
 * Cleanup cuBLAS handle.
 */
void cleanup_cublas() {
    if (cublas_initialized) {
        cublasDestroy(cublas_handle);
        cublas_initialized = false;
    }
}

/**
 * SiLU (Swish) activation kernel: y = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * SiLU is smooth and non-monotonic, better than ReLU for GNNs.
 */
__global__ void silu_activation_kernel(
    int n,
    const double* __restrict__ x,
    double* __restrict__ y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        double val = x[idx];
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        double sigmoid = 1.0 / (1.0 + exp(-val));
        y[idx] = val * sigmoid;
    }
}

/**
 * ReLU activation kernel: y = max(0, x)
 */
__global__ void relu_activation_kernel(
    int n,
    const double* __restrict__ x,
    double* __restrict__ y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = fmax(0.0, x[idx]);
    }
}

/**
 * Add bias kernel: y = x + b (broadcast bias across batch dimension)
 */
__global__ void add_bias_kernel(
    int batch,
    int dim,
    const double* __restrict__ x,
    const double* __restrict__ bias,
    double* __restrict__ y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch && col < dim) {
        int idx = row * dim + col;
        y[idx] = x[idx] + bias[col];
    }
}

/**
 * Dense layer implementation using cuBLAS.
 *
 * Computes: Y = activation(X * W + b)
 *
 * Matrix dimensions:
 * - X: [batch, in_dim] (row-major)
 * - W: [in_dim, out_dim] (column-major for cuBLAS)
 * - b: [out_dim]
 * - Y: [batch, out_dim] (row-major)
 *
 * cuBLAS expects column-major matrices, so we need to transpose:
 * Y^T = (X * W)^T = W^T * X^T
 *
 * In column-major: C = alpha * op(A) * op(B) + beta * C
 * We want: Y^T [out_dim, batch] = W^T [out_dim, in_dim] * X^T [in_dim, batch]
 */
void dense_layer(
    int batch,
    int in_dim,
    int out_dim,
    const double* input,
    const double* weights,
    const double* bias,
    double* output,
    const char* activation
) {
    // Initialize cuBLAS if needed
    if (!cublas_initialized) {
        init_cublas();
    }

    // Allocate temporary buffer for matrix multiplication result
    double* temp;
    CUDA_CHECK(cudaMalloc(&temp, batch * out_dim * sizeof(double)));

    // Perform matrix multiplication using cuBLAS
    // Y = X * W where X is [batch, in_dim], W is [in_dim, out_dim], Y is [batch, out_dim]
    //
    // cuBLAS uses column-major, so we compute: Y^T = W^T * X^T
    // This gives us [out_dim, batch] = [out_dim, in_dim] * [in_dim, batch]
    //
    // Row-major storage means:
    // - X [batch, in_dim] row-major: element X[i,j] at offset i*in_dim + j
    // - Viewed as column-major [in_dim, batch]: element [j,i] at offset i*in_dim + j
    // - Leading dimension (column-major) = batch
    //
    // - W [in_dim, out_dim] row-major: element W[i,j] at offset i*out_dim + j
    // - Viewed as column-major [out_dim, in_dim]: element [j,i] at offset i*out_dim + j
    // - Leading dimension (column-major) = in_dim
    //
    // cublasDgemm: C = alpha * op(A) * op(B) + beta * C
    // We want: Y^T = W^T * X^T (no-transpose version)
    //
    // A = W^T in column-major = W in row-major [in_dim, out_dim], ld = out_dim
    // B = X^T in column-major = X in row-major [batch, in_dim], ld = in_dim
    // C = Y^T in column-major = Y in row-major [batch, out_dim], ld = out_dim

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t status = cublasDgemm(
        cublas_handle,
        CUBLAS_OP_N,  // No transpose on W (already transposed by row/col major difference)
        CUBLAS_OP_N,  // No transpose on X (already transposed by row/col major difference)
        out_dim,      // m: rows of C (Y^T in column-major = rows of Y in row-major)
        batch,        // n: cols of C (Y^T in column-major = cols of Y in row-major)
        in_dim,       // k: inner dimension
        &alpha,
        weights,      // A: W [in_dim, out_dim] row-major, ld = out_dim
        out_dim,      // lda: leading dimension of W in row-major
        input,        // B: X [batch, in_dim] row-major, ld = in_dim
        in_dim,       // ldb: leading dimension of X in row-major
        &beta,
        temp,         // C: Y [batch, out_dim] row-major, ld = out_dim
        out_dim       // ldc: leading dimension of Y in row-major
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(temp);
        throw std::runtime_error("cuBLAS matrix multiplication failed");
    }

    // Add bias
    dim3 threads(16, 16);
    dim3 blocks(
        (batch + threads.x - 1) / threads.x,
        (out_dim + threads.y - 1) / threads.y
    );

    add_bias_kernel<<<blocks, threads>>>(batch, out_dim, temp, bias, output);
    CUDA_CHECK(cudaGetLastError());

    // Apply activation function
    int n = batch * out_dim;
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    if (strcmp(activation, "silu") == 0) {
        silu_activation_kernel<<<num_blocks, threads_per_block>>>(
            n, output, output
        );
    } else if (strcmp(activation, "relu") == 0) {
        relu_activation_kernel<<<num_blocks, threads_per_block>>>(
            n, output, output
        );
    } else if (strcmp(activation, "none") != 0) {
        cudaFree(temp);
        throw std::runtime_error("Unknown activation function");
    }
    // If "none", skip activation

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(temp));
}

/**
 * Multi-layer MLP forward pass.
 *
 * Applies a sequence of dense layers with activations.
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
    const char* activation
) {
    // Input to first layer
    const double* current_input = input;
    double* current_output = temp_buffer;

    for (int layer = 0; layer < num_layers; layer++) {
        int in_dim = layer_dims[layer];
        int out_dim = layer_dims[layer + 1];

        // Last layer: output directly to final output, no activation
        if (layer == num_layers - 1) {
            current_output = output;
            dense_layer(
                batch, in_dim, out_dim,
                current_input, weights[layer], biases[layer],
                current_output,
                "none"  // No activation on output layer
            );
        } else {
            // Hidden layer: use activation
            dense_layer(
                batch, in_dim, out_dim,
                current_input, weights[layer], biases[layer],
                current_output,
                activation
            );

            // Swap buffers for next layer
            current_input = current_output;

            // Alternate between two buffers to avoid extra copy
            if (layer < num_layers - 2) {
                // Point to different part of temp buffer
                current_output = temp_buffer + batch * out_dim;
            }
        }
    }
}

} // namespace gnn
} // namespace cuda
} // namespace fennol
