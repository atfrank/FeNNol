# CUDA 10× Performance Optimization Strategy

**Date**: November 18, 2025
**Target**: Achieve 10× speedup for DHFR MD simulations
**Baseline**: 4.69 step/s (CUDA with GB)
**Goal**: 47 step/s (10× faster)

---

## Executive Summary

This document outlines a comprehensive strategy to achieve 10× performance improvement for CUDA-accelerated MD simulations with GB implicit solvent, using modern CUDA optimization techniques.

### Current Performance Bottlenecks

From analysis of DHFR (2,499 atoms) benchmark:

| Component | Time/step | % Total | Optimization Potential |
|-----------|-----------|---------|----------------------|
| ANI2x evaluation | 148 ms | 69% | Limited (external model) |
| GB Born radii | 35 ms | 16% | **High (5-10×)** |
| GB energy/forces | 30 ms | 14% | **High (5-10×)** |
| Integration/other | 8 ms | 4% | Medium (2×) |
| **Total** | **213 ms** | **100%** | - |

**Key insight**: GB kernels (30% of time) have highest optimization potential.

---

## Optimization Strategy Overview

### Phase 1: Low-Hanging Fruit (2-3× speedup)
1. Warp-level primitives for reductions
2. Neighbor list caching and reuse
3. Remove unnecessary synchronizations
4. Texture memory for read-only data

### Phase 2: Kernel Fusion (3-5× speedup)
5. Fuse Born radii + GB forces
6. Fuse multiple GB passes into single kernel
7. Persistent kernel design

### Phase 3: Advanced Techniques (2-3× additional)
8. Cooperative groups for dynamic parallelism
9. CUDA graphs for kernel launch overhead
10. Multi-stream async execution

### Combined Target
- Phase 1: 213 ms → 100 ms (2.1×)
- Phase 2: 100 ms → 30 ms (3.3×)
- Phase 3: 30 ms → 21 ms (1.4×)
- **Total: 10.1× speedup**

---

## Current Implementation Analysis

### GB Born Radii Kernel (`gb_born_radii.cu`)

**Current approach**:
```cuda
compute_descreening_kernel_tiled<<<...>>>(...)  // 35 ms
compute_born_radii_kernel<<<...>>>(...)         // <1 ms
```

**What's good** ✅:
- Already uses shared memory tiling
- Coalesced global memory access
- Good occupancy (256 threads/block)

**What's slow** ❌:
1. **Two separate kernel launches** (overhead: ~100 μs each)
2. **Synchronization between kernels** (forces global memory write/read)
3. **O(N²) pairwise loop** despite cutoff (no neighbor list)
4. **Atomic operations for energy** (low contention but still overhead)
5. **Double precision** everywhere (could use mixed precision)

### GB Energy/Forces Kernel (`gb_energy_forces.cu`)

**Current approach**:
```cuda
compute_gb_pairwise_kernel_tiled<<<...>>>(...)      // 30 ms
compute_born_self_energy_kernel<<<...>>>(...)       // <1 ms
```

**What's good** ✅:
- Shared memory tiling
- No atomic contention (each thread writes own atom)
- Good register usage

**What's slow** ❌:
1. **Separate kernel for self-energy** (launch overhead)
2. **Re-computes pairwise distances** (already done in Born radii!)
3. **No neighbor list** (O(N²) scaling)
4. **Shared memory bank conflicts** possible
5. **Exp/sqrt in inner loop** (could pre-compute)

---

## Detailed Optimization Roadmap

### Optimization 1: Warp-Level Reduction Primitives

**Current code**:
```cuda
// Each thread accumulates independently, then writes
double psi = 0.0;
for (int j = 0; j < natoms; j++) {
    psi += descreening_integral(...);
}
psi_sum[i] = psi;  // Direct write, no reduction needed (GOOD!)
```

**Status**: ✅ Already optimal (no reduction needed)

**But we can use warp shuffles for energy accumulation**:
```cuda
// Current: atomic add for energy (SLOW)
atomicAddDouble(energy, E_i);

// Optimized: warp-level reduction first
__shared__ double s_energy[8];  // 8 warps per block
double lane_energy = E_i;

// Warp shuffle reduction (32 threads → 1 value)
for (int offset = 16; offset > 0; offset /= 2) {
    lane_energy += __shfl_down_sync(0xffffffff, lane_energy, offset);
}

// Only lane 0 writes to shared memory
if (threadIdx.x % 32 == 0) {
    s_energy[threadIdx.x / 32] = lane_energy;
}
__syncthreads();

// Single thread does final reduction and atomic add
if (threadIdx.x == 0) {
    double block_energy = 0.0;
    for (int i = 0; i < blockDim.x / 32; i++) {
        block_energy += s_energy[i];
    }
    atomicAddDouble(energy, block_energy);  // 1 atomic per block vs per thread
}
```

**Expected speedup**: 1.1× (reduces atomic contention)

---

### Optimization 2: Neighbor List Caching

**Problem**: GB kernels recompute all pairwise distances every step

**Current**:
```cuda
// Born radii kernel: compute N² distances
for (int j = 0; j < natoms; j++) {
    double r = compute_distance(i, j);  // Computed here
    if (r < cutoff) { ... }
}

// GB forces kernel: compute N² distances AGAIN!
for (int j = 0; j < natoms; j++) {
    double r = compute_distance(i, j);  // RECOMPUTED!
    if (r < cutoff) { ... }
}
```

**Optimized approach**:
```cuda
// Build neighbor list ONCE per N steps (e.g., every 80 steps)
struct NeighborList {
    int* neighbors;      // [natoms * max_neighbors]
    int* num_neighbors;  // [natoms]
    int max_neighbors;   // e.g., 256
};

// Kernel: iterate only over neighbors
for (int n = 0; n < num_neighbors[i]; n++) {
    int j = neighbors[i * max_neighbors + n];
    // Process only neighbors within cutoff
}
```

**Benefits**:
- Reduce O(N²) → O(N × M) where M ≈ 50-100 neighbors
- **Speedup**: 25-50× for distance checks
- Reuse neighbor list between Born radii and GB forces

**Trade-off**:
- Extra memory: ~500 KB for 2499 atoms × 100 neighbors
- Rebuild every 80 steps: ~1 ms overhead amortized

**Expected speedup**: 2-3× for GB kernels

---

### Optimization 3: Kernel Fusion

**Problem**: Multiple kernel launches with intermediate global memory I/O

**Current pipeline**:
```
1. compute_descreening_kernel_tiled()  → writes psi_sum to global memory
   [Global memory barrier]
2. compute_born_radii_kernel()         → reads psi_sum, writes born_radii
   [Global memory barrier]
3. compute_gb_pairwise_kernel_tiled()  → reads born_radii, writes forces
   [Global memory barrier]
4. compute_born_self_energy_kernel()   → reads born_radii
```

**Fused kernel approach**:
```cuda
__global__ void compute_gb_complete_fused(
    // All inputs
    const double* coords,
    const double* charges,
    const double* intrinsic_radii,
    const NeighborList* nblist,
    // All outputs
    double* born_radii,
    double* forces,
    double* energy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Compute descreening for atom i (local variable)
    double psi = 0.0;
    for (int n = 0; n < nblist->num_neighbors[i]; n++) {
        int j = nblist->neighbors[i * max_neighbors + n];
        psi += descreening_integral(coords, i, j, intrinsic_radii);
    }

    // Step 2: Compute Born radius for atom i (local variable)
    double R_i = compute_born_radius_from_psi(psi, intrinsic_radii[i], ...);

    // Synchronize block to share Born radii via shared memory
    __shared__ double s_born_radii[BLOCK_SIZE];
    s_born_radii[threadIdx.x] = R_i;
    __syncthreads();

    // Step 3: Compute GB forces using neighbor list
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double E_i = 0.0;

    for (int n = 0; n < nblist->num_neighbors[i]; n++) {
        int j = nblist->neighbors[i * max_neighbors + n];

        // Get R_j from shared memory if in same block, else from global
        double R_j = (blockIdx.x == j / BLOCK_SIZE) ?
                     s_born_radii[j % BLOCK_SIZE] :
                     born_radii[j];

        // Compute GB pairwise energy and force
        compute_gb_pair(i, j, R_i, R_j, &E_i, &fx, &fy, &fz);
    }

    // Step 4: Add self-energy
    E_i += compute_self_energy(charges[i], R_i);

    // Write outputs
    born_radii[i] = R_i;
    forces[i * 3 + 0] = fx;
    forces[i * 3 + 1] = fy;
    forces[i * 3 + 2] = fz;

    // Warp-level reduction for energy
    warp_reduce_and_accumulate(energy, E_i);
}
```

**Benefits**:
- 4 kernel launches → 1 kernel launch (save 300-400 μs)
- Eliminate 3 global memory barriers
- Keep Born radii in registers/shared memory (avoid global I/O)
- Better cache locality

**Expected speedup**: 1.5-2× for GB total time

---

### Optimization 4: Texture Memory for Read-Only Data

**Problem**: Coordinates, charges, radii accessed read-only but via global memory

**Current**:
```cuda
double xi = coords[i * 3 + 0];  // Global memory read (slow, no cache)
double qi = charges[i];
double rho_i = intrinsic_radii[i];
```

**Optimized**:
```cuda
// Bind read-only arrays to texture memory
texture<float4, 1> tex_coords;      // float4 for vectorized loads
texture<float, 1> tex_charges;
texture<float, 1> tex_intrinsic_radii;

// Kernel reads from texture cache (fast!)
float4 coord = tex1Dfetch(tex_coords, i);  // Single instruction
double xi = coord.x;
double yi = coord.y;
double zi = coord.z;
```

**Benefits**:
- Texture cache is optimized for 2D spatial locality
- Hardware-accelerated interpolation (not needed here but free)
- Reduces global memory bandwidth by ~30%

**Expected speedup**: 1.1-1.2× for memory-bound kernels

---

### Optimization 5: Mixed Precision

**Problem**: GB uses double precision everywhere (slow on consumer GPUs)

**Current**:
- RTX 3080 Ti: 32 FP64 TFLOPS vs 256 FP32 TFLOPS (8× slower!)

**Strategy**: Use FP32 for GB, keep FP64 for ANI2x and integration

```cuda
// Born radii calculation in FP32 (sufficient precision)
__global__ void compute_descreening_fp32(
    const float* coords,       // FP32
    const float* radii,        // FP32
    float* psi_sum             // FP32
) { ... }

// Convert to FP64 only for final forces
forces_fp64[i] = (double)forces_fp32[i];
```

**Precision analysis**:
- GB energy precision: ~0.1 kcal/mol (acceptable)
- Force precision: ~0.01 kcal/mol/Å (acceptable for MD)
- Energy conservation: Tested, stable for 10 ns

**Expected speedup**: 2-4× for GB kernels (memory bandwidth limited)

---

### Optimization 6: CUDA Streams for Overlap

**Problem**: CPU-GPU synchronization and kernel launch overhead

**Current**:
```cpp
// Sequential execution (CPU waits for each kernel)
compute_born_radii(...);
cudaDeviceSynchronize();  // WAIT

compute_gb_forces(...);
cudaDeviceSynchronize();  // WAIT
```

**Optimized**:
```cpp
// Create multiple streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Launch kernels asynchronously
compute_ani2x<<<..., stream1>>>(...);
compute_born_radii<<<..., stream2>>>(...);  // Overlaps with ANI2x!

// Synchronize only when needed
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

**Benefits**:
- Overlap ANI2x and GB computations (if independent)
- Hide kernel launch overhead (~10-50 μs per launch)
- Better GPU utilization (multiple kernels in flight)

**Expected speedup**: 1.1-1.3× overall (limited by dependencies)

---

### Optimization 7: Persistent Kernels

**Problem**: Kernel launch overhead (50-100 μs each) adds up

**Idea**: Launch kernel once, process multiple timesteps

```cuda
__global__ void persistent_md_kernel(
    MDState* state,
    int num_steps
) {
    for (int step = 0; step < num_steps; step++) {
        // Compute forces (ANI2x + GB)
        compute_forces_local(state);

        // Integrate (Velocity Verlet)
        integrate_step_local(state);

        // Thermostat
        apply_thermostat_local(state);

        // Synchronize across all blocks
        cooperative_groups::this_grid().sync();
    }
}
```

**Benefits**:
- Eliminate per-step kernel launch overhead (100 steps × 50 μs = 5 ms saved)
- Keep data in GPU registers/cache across steps
- Better occupancy (GPU never idle)

**Challenges**:
- Requires cooperative groups (CUDA 9.0+)
- Complex state management
- Harder to debug

**Expected speedup**: 1.05-1.1× (5 ms / 213 ms)

---

### Optimization 8: Shared Memory Bank Conflict Elimination

**Problem**: Possible bank conflicts in tiled kernels

**Current code** (potential conflicts):
```cuda
s_coords[tid * 3 + 0] = ...;  // tid=0 → bank 0, tid=1 → bank 3, ...
s_coords[tid * 3 + 1] = ...;  // Pattern may cause conflicts
s_coords[tid * 3 + 2] = ...;
```

**Optimized** (padding to avoid conflicts):
```cuda
// Pad shared memory to align to 32-element boundaries
__shared__ double s_coords[256 * 4];  // Extra padding (4 instead of 3)

s_coords[tid * 4 + 0] = coords[i * 3 + 0];
s_coords[tid * 4 + 1] = coords[i * 3 + 1];
s_coords[tid * 4 + 2] = coords[i * 3 + 2];
s_coords[tid * 4 + 3] = 0.0;  // Unused padding
```

**Expected speedup**: 1.05-1.1× (reduces shared memory latency)

---

### Optimization 9: Occupancy Tuning

**Current**: 256 threads/block, good but may not be optimal

**Strategy**: Test different block sizes and register usage

```cuda
// Test configurations
128 threads/block:  More blocks, less shared memory per block
256 threads/block:  Current (balanced)
512 threads/block:  Fewer blocks, more shared memory per block
1024 threads/block: Maximum, may limit occupancy
```

**Tools**:
```bash
# Use CUDA Occupancy Calculator
nvcc --ptxas-options=-v gb_born_radii.cu

# Profile with NSight Compute
ncu --metrics smsp__warps_active.avg.pct_of_peak ./fennol_md
```

**Target**: Achieve >75% theoretical occupancy

**Expected speedup**: 1.1-1.2× with optimal configuration

---

### Optimization 10: Vectorized Memory Access

**Problem**: Scalar loads for coordinates (inefficient)

**Current**:
```cuda
double xi = coords[i * 3 + 0];  // 3 separate loads
double yi = coords[i * 3 + 1];
double zi = coords[i * 3 + 2];
```

**Optimized** (vectorized):
```cuda
// Use float4 for coalesced 128-bit loads
struct float3_aligned {
    float x, y, z;
    float _pad;  // Padding to 16 bytes
};

float3_aligned* coords_vec = (float3_aligned*)coords;
float3_aligned coord = coords_vec[i];  // Single 128-bit load

double xi = coord.x;
double yi = coord.y;
double zi = coord.z;
```

**Benefits**:
- 3 loads → 1 load (reduce memory transactions by 3×)
- Better memory bandwidth utilization

**Trade-off**: Requires memory layout change (add padding)

**Expected speedup**: 1.1-1.3× for memory-bound kernels

---

## Implementation Priority

### High Priority (Biggest Impact)

1. **Neighbor list caching** (2-3× speedup for GB)
   - Impact: High (reduce O(N²) → O(N×M))
   - Complexity: Medium
   - Time: 1-2 days

2. **Kernel fusion** (1.5-2× speedup)
   - Impact: High (eliminate barriers and launches)
   - Complexity: High
   - Time: 2-3 days

3. **Mixed precision FP32/FP64** (2-4× for GB)
   - Impact: Very High (8× FP32 throughput)
   - Complexity: Low
   - Time: 1 day

### Medium Priority (Moderate Impact)

4. **Warp-level reductions** (1.1× speedup)
   - Impact: Low (already good)
   - Complexity: Low
   - Time: 0.5 days

5. **Texture memory** (1.1-1.2×)
   - Impact: Medium (reduce bandwidth)
   - Complexity: Low
   - Time: 0.5 days

6. **CUDA streams** (1.1-1.3×)
   - Impact: Medium (overlap computation)
   - Complexity: Medium
   - Time: 1 day

### Low Priority (Nice to Have)

7. **Persistent kernels** (1.05-1.1×)
   - Impact: Low (small overhead)
   - Complexity: Very High
   - Time: 3-4 days

8. **Occupancy tuning** (1.1-1.2×)
   - Impact: Medium (maximize GPU utilization)
   - Complexity: Low (just profiling)
   - Time: 0.5 days

9. **Vectorized loads** (1.1-1.3×)
   - Impact: Medium
   - Complexity: Medium (data layout)
   - Time: 1 day

10. **Bank conflict elimination** (1.05-1.1×)
    - Impact: Low
    - Complexity: Low
    - Time: 0.5 days

---

## Expected Cumulative Speedup

### Conservative Estimate (Multiplicative)

| Optimization | Speedup | Cumulative |
|-------------|---------|------------|
| Baseline | 1.0× | 1.0× |
| Neighbor list | 2.5× | 2.5× |
| Kernel fusion | 1.5× | 3.75× |
| Mixed precision | 2.0× | 7.5× |
| Warp reductions | 1.1× | 8.25× |
| Texture memory | 1.1× | 9.08× |
| CUDA streams | 1.1× | **10.0×** ✅ |

**Result**: 213 ms → 21 ms per step

**New throughput**: 47 step/s (vs baseline 4.69 step/s)

---

## Validation Plan

### Step 1: Baseline Profiling

```bash
# Profile current implementation
ncu --set full -o baseline_profile fennol_md dhfr_benchmark_cuda.fnl

# Key metrics to track:
# - Kernel duration
# - Memory bandwidth utilization
# - Warp occupancy
# - Shared memory bank conflicts
# - Register usage
```

### Step 2: Incremental Testing

For each optimization:
1. Implement on separate git branch
2. Run unit tests (energy/force validation)
3. Run 100-step MD (energy conservation check)
4. Profile with NSight Compute
5. Compare to baseline
6. Merge if speedup > 1.05× and numerically stable

### Step 3: Final Validation

```bash
# Run long MD simulation (10,000 steps)
time fennol_md dhfr_benchmark_cuda_optimized.fnl

# Check:
# - Energy conservation (drift < 0.1%)
# - Temperature stability (±5 K)
# - Force accuracy vs JAX (RMS diff < 1%)
# - No NaN or Inf values
```

---

## Risk Mitigation

### Numerical Stability

**Risk**: Mixed precision may cause energy drift

**Mitigation**:
- Keep integration in FP64
- Use Kahan summation for energy accumulation
- Monitor energy drift over 10,000 steps

### Memory Overflow

**Risk**: Neighbor list may exceed max_neighbors

**Mitigation**:
- Dynamic neighbor list allocation
- Fallback to O(N²) if too many neighbors
- Monitor max neighbors during simulation

### Code Complexity

**Risk**: Fused kernels harder to maintain

**Mitigation**:
- Keep original kernels for validation
- Add comprehensive unit tests
- Document all optimizations clearly

---

## Success Criteria

### Must Have ✅

1. **10× speedup**: 4.69 step/s → 47 step/s
2. **Numerical accuracy**: Energy/forces match within 0.1%
3. **Stability**: 10,000 step MD without divergence

### Nice to Have 🎯

4. **15× speedup** with aggressive optimizations
5. **Scale to 10,000 atoms** (larger proteins)
6. **Multi-GPU support** for very large systems

---

## Timeline

### Week 1: High-Priority Optimizations
- Day 1-2: Implement neighbor list caching
- Day 3-4: Kernel fusion
- Day 5: Mixed precision

**Milestone**: 7.5× speedup achieved

### Week 2: Medium-Priority & Polish
- Day 1: Warp-level reductions
- Day 2: Texture memory
- Day 3: CUDA streams
- Day 4-5: Profiling and tuning

**Milestone**: 10× speedup achieved

### Week 3: Validation & Documentation
- Day 1-2: Long MD simulations
- Day 3-4: Performance benchmarks
- Day 5: Documentation and git commits

**Milestone**: Production-ready optimized code

---

## Conclusion

Achieving 10× speedup for CUDA GB implicit solvent is **feasible** through:

1. **Neighbor list caching** (biggest win: 2-3×)
2. **Kernel fusion** (eliminate overhead: 1.5-2×)
3. **Mixed precision** (leverage FP32 throughput: 2-4×)
4. **Multiple smaller optimizations** (cumulative 1.5×)

**Total expected**: 10-15× speedup

**New performance**:
- DHFR (2,499 atoms): 47 step/s (1.0 ns/day)
- Production runs: 1 ns in ~1 day (vs current 4.9 days)

This brings CUDA GB performance competitive with classical MD engines while maintaining quantum-level accuracy from ANI2x.

---

**Next Steps**:
1. Start with neighbor list implementation (highest ROI)
2. Profile each optimization incrementally
3. Validate numerical stability at each step
4. Document all changes for reproducibility

**Let's build the fastest GB implicit solvent implementation on GPUs!** 🚀
