#ifndef FENNOL_CUDA_COMMON_CUH
#define FENNOL_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace fennol {
namespace cuda {

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// Common constants
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

#ifdef __CUDACC__
// Device function for atomic add (double precision)
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif // __CUDACC__

// Vector operations
struct Vec3 {
    double x, y, z;

    __device__ __host__ Vec3() : x(0.0), y(0.0), z(0.0) {}
    __device__ __host__ Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    __device__ __host__ Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    __device__ __host__ Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    __device__ __host__ Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    __device__ __host__ double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __device__ __host__ double norm() const {
        return sqrt(x * x + y * y + z * z);
    }

    __device__ __host__ double norm_squared() const {
        return x * x + y * y + z * z;
    }

    __device__ __host__ Vec3 normalized() const {
        double n = norm();
        return (n > 1e-10) ? Vec3(x / n, y / n, z / n) : Vec3(0.0, 0.0, 0.0);
    }

    __device__ __host__ Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
};

#ifdef __CUDACC__
// Reduction helper
template<typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}
#endif // __CUDACC__

} // namespace cuda
} // namespace fennol

#endif // FENNOL_CUDA_COMMON_CUH
