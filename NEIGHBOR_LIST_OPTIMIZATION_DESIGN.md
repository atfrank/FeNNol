# Neighbor List Optimization Design

**Date**: November 18, 2025
**Target**: 29.8× speedup (actual potential)
**Conservative estimate**: 2.5× speedup (accounting for overhead)

---

## Problem Analysis

### Current Implementation (Baseline)

**Algorithm**: Evaluate ALL pairs
- DHFR (2,499 atoms): 3,121,251 pair evaluations
- Only 104,636 pairs within 12 Å cutoff (3.4% efficiency)
- **96.6% of work is wasted!**

**Performance**:
- Born radii: 177 ms
- Energy/forces: 36 ms
- Total: 213 ms/step
- **4.69 steps/second**

### Theoretical Speedup

With neighbor list (only evaluate useful pairs):
- Reduce from 3.1M → 105K pairs (29.8× reduction)
- Born radii: 177 ms → 6 ms (theoretical)
- Energy/forces: 36 ms → 1.2 ms (theoretical)
- Total: 213 ms → 7 ms (theoretical)
- **142 steps/second (30× speedup)**

### Conservative Estimate (Accounting for Overhead)

Neighbor list has costs:
- **Build cost**: ~5-10 ms (once every 10-20 steps)
- **Memory access**: Less cache-friendly (indirect indexing)
- **Load imbalance**: Some atoms have more neighbors

**Realistic speedup**: 2.5-5× (accounting for all overheads)
- Total: 213 ms → 43-85 ms
- **12-23 steps/second**

Still excellent!

---

## Design: Verlet Neighbor List

### Data Structure

```cpp
struct NeighborList {
    // Compact storage: flat array + offsets
    int* neighbor_atoms;      // Flat array of neighbor atom indices
    int* neighbor_counts;     // Number of neighbors for each atom [natoms]
    int* neighbor_offsets;    // Start index in neighbor_atoms [natoms]

    // Build parameters
    float cutoff;             // Interaction cutoff
    float skin;               // Extra buffer (typically 1-2 Å)
    float build_cutoff;       // cutoff + skin

    // Rebuild trigger
    float max_displacement;   // Maximum atom movement since last build
    float rebuild_threshold;  // Rebuild when max_displacement > skin/2
};
```

**Key features**:
1. **Verlet buffer** (skin): Build list with cutoff+skin, rebuild only when atoms move > skin/2
2. **Compact storage**: Flat array (cache-friendly) with offsets
3. **Asymmetric storage**: Store j for each i (avoid redundancy issues)

### Build Algorithm

```cuda
__global__ void build_neighbor_list_kernel(
    int natoms,
    const double* coords,
    float build_cutoff,
    int max_neighbors_per_atom,  // Estimated (e.g., 100)
    int* neighbor_atoms,         // Output: flat array
    int* neighbor_counts,        // Output: counts per atom
    int* neighbor_offsets        // Output: offsets
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    double xi = coords[i * 3 + 0];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];

    int count = 0;
    int offset = i * max_neighbors_per_atom;  // Pre-allocated space

    for (int j = 0; j < natoms; j++) {
        if (i == j) continue;

        double dx = xi - coords[j * 3 + 0];
        double dy = yi - coords[j * 3 + 1];
        double dz = zi - coords[j * 3 + 2];
        double r_sq = dx*dx + dy*dy + dz*dz;

        if (r_sq < build_cutoff * build_cutoff) {
            if (count < max_neighbors_per_atom) {
                neighbor_atoms[offset + count] = j;
                count++;
            } else {
                // Buffer overflow - need larger max_neighbors
                // Atomic flag to trigger rebuild with larger buffer
            }
        }
    }

    neighbor_counts[i] = count;
    neighbor_offsets[i] = offset;
}
```

**Complexity**: Still O(N²) for build, but:
- Only build every 10-20 steps (amortized cost)
- Subsequent force evaluations are O(N×M) where M << N

### Modified Born Radii Kernel

```cuda
__global__ void compute_born_radii_neighborlist(
    int natoms,
    const double* coords,
    const double* radii,
    const double* b_params,
    const double* c_params,
    const int* neighbor_atoms,
    const int* neighbor_counts,
    const int* neighbor_offsets,
    double cutoff,
    double* born_radii,
    double* psi_sum
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    double xi = coords[i * 3 + 0];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];
    double rho_i = radii[i];

    double psi_i = 0.0;

    // Only loop over NEIGHBORS (not all atoms!)
    int num_neighbors = neighbor_counts[i];
    int offset = neighbor_offsets[i];

    for (int idx = 0; idx < num_neighbors; idx++) {
        int j = neighbor_atoms[offset + idx];

        double dx = xi - coords[j * 3 + 0];
        double dy = yi - coords[j * 3 + 1];
        double dz = zi - coords[j * 3 + 2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);

        // Still check cutoff (neighbor list has buffer)
        if (r > cutoff) continue;

        double rho_j = radii[j];
        double I_ij = compute_descreening_integral_HCT(r, rho_i, rho_j);
        psi_i += I_ij;
    }

    // Apply OBC formula
    double psi_scaled = 0.5 * rho_i * psi_i;
    double b_i = b_params[i];
    double c_i = c_params[i];

    double tanh_arg = psi_scaled - b_i * psi_scaled * psi_scaled
                      + c_i * psi_scaled * psi_scaled * psi_scaled;
    double tanh_val = tanh(tanh_arg);

    double R_inv = 1.0 / rho_i - tanh_val / rho_i;
    double R = 1.0 / R_inv;

    born_radii[i] = fmax(R, rho_i);
    psi_sum[i] = psi_i;
}
```

**Key change**: Loop over `num_neighbors` instead of `natoms`!

---

## Implementation Plan

### Phase 1: Basic Neighbor List (Week 3)

1. **Implement neighbor list builder** (1 day)
   - CUDA kernel for building list
   - Host-side wrapper
   - Test on small systems

2. **Modify Born radii kernel** (1 day)
   - Use neighbor list
   - Validate against baseline
   - Ensure Newton's 3rd law still satisfied

3. **Modify energy/forces kernels** (1 day)
   - Update pairwise force kernel
   - Update Born radii force kernel
   - Validate all tests pass

4. **Benchmark** (0.5 day)
   - Measure speedup on DHFR
   - Profile with nvprof
   - Identify remaining bottlenecks

5. **Optimize rebuild logic** (0.5 day)
   - Track displacements
   - Auto-rebuild when needed
   - Tune skin parameter

**Deliverable**: 2-3× speedup on DHFR

### Phase 2: Advanced Optimizations (Week 4)

1. **Cell list for O(N) build** (1 day)
   - Spatial hashing
   - Only check nearby cells
   - Build time: O(N²) → O(N)

2. **Binning/sorting** (1 day)
   - Sort atoms by cell
   - Better cache locality
   - Coalesced memory access

3. **Dynamic load balancing** (1 day)
   - Some atoms have many neighbors (up to 200)
   - Some have few (20-30)
   - Use dynamic parallelism or warp aggregation

**Deliverable**: 3-5× speedup on DHFR

---

## Validation Strategy

### Test Suite

All existing tests must PASS with neighbor list:

1. **Physics tests** (CRITICAL):
   - Newton's 3rd law (net force = 0)
   - Energy matches baseline
   - Forces match baseline within tolerance

2. **Numerical tests**:
   - Born radii match baseline (1e-6)
   - Energy matches baseline (1e-5)
   - Forces match baseline (1e-4)

3. **Edge cases**:
   - Empty neighbor list (atoms far apart)
   - Full neighbor list (all atoms within cutoff)
   - Atoms exactly at cutoff boundary

### Validation Command

```bash
# After implementing neighbor list optimization:
pytest tests/validation/test_gb_cuda_baseline.py -v

# All 9 tests must PASS ✅
```

**Success criteria**:
- All validation tests PASS
- Performance improves by ≥2×
- No regression in accuracy

---

## Memory Requirements

### Current (No neighbor list)
- Minimal: Just atom data

### With Neighbor List
- `neighbor_atoms`: natoms × max_neighbors × 4 bytes
  - 2499 atoms × 100 neighbors × 4 = 1 MB
- `neighbor_counts`: natoms × 4 bytes = 10 KB
- `neighbor_offsets`: natoms × 4 bytes = 10 KB
- **Total**: ~1 MB (negligible on modern GPUs with 8-24 GB)

### Optimizations
- Use `int16_t` for neighbor indices (if natoms < 32K): 50% savings
- Compact storage: Only allocate what's needed
- Reuse buffer across timesteps

---

## Performance Model

### Build Cost (Amortized)

**Simple build** (current plan):
- O(N²) neighbor search
- Time: ~10 ms for 2499 atoms
- Frequency: Every 10-20 steps
- **Amortized cost**: 0.5-1 ms/step

**Cell list build** (future):
- O(N) neighbor search
- Time: ~1 ms for 2499 atoms
- **Amortized cost**: 0.05-0.1 ms/step

### Force Evaluation Cost

**Baseline**: 213 ms/step
- Born radii: 177 ms (3.1M pairs)
- Energy/forces: 36 ms

**With neighbor list**: 43-85 ms/step (2.5-5× speedup)
- Born radii: 35-70 ms (105K pairs)
- Energy/forces: 7-14 ms
- Neighbor list build (amortized): 0.5-1 ms

**Breakdown of remaining time**:
- Memory transfers: 20%
- Kernel launch overhead: 10%
- Actual computation: 70%

### Bottleneck Analysis

After neighbor list, bottleneck shifts to:
1. **Memory bandwidth** (loading coordinates)
2. **Transcendental functions** (exp, sqrt, tanh)
3. **Atomic operations** (force accumulation)

Next optimizations will target these:
- Mixed precision (reduce memory bandwidth)
- Kernel fusion (reduce kernel launches)
- Warp primitives (reduce atomic contention)

---

## Risk Mitigation

### Risk 1: Load Imbalance

**Problem**: Some atoms have 200 neighbors, others have 20
**Impact**: GPU threads idle waiting for slowest
**Solution**:
- Dynamic work distribution
- Warp-aggregated neighbor processing
- Sort by neighbor count

### Risk 2: Neighbor List Overflow

**Problem**: Underestimate max_neighbors, buffer overflow
**Impact**: Missing interactions, wrong physics
**Solution**:
- Conservative max_neighbors estimate (150)
- Atomic counter to detect overflow
- Auto-rebuild with larger buffer

### Risk 3: Rebuild Too Frequent

**Problem**: Atoms move fast, need frequent rebuilds
**Impact**: Build cost dominates
**Solution**:
- Tune skin parameter (1-2 Å is typical)
- Track actual displacement
- Adaptive rebuild threshold

---

## Success Metrics

### Minimum Viable Product (MVP)

- ✅ All validation tests pass
- ✅ 2× speedup on DHFR
- ✅ Correct physics (Newton's 3rd law)
- ✅ Memory usage < 100 MB

### Target Performance

- 🎯 2.5-3× speedup on DHFR
- 🎯 Rebuild every 10-20 steps
- 🎯 Build cost < 10 ms

### Stretch Goals

- 🚀 5× speedup with cell list
- 🚀 Rebuild every 50 steps (better skin/rebuild logic)
- 🚀 Build cost < 1 ms

---

## Timeline

| Task | Duration | Dependencies |
|------|----------|-------------|
| **Phase 1: Basic Neighbor List** | **3 days** | |
| Design data structure | 0.5 day | - |
| Implement builder kernel | 1 day | Data structure |
| Modify Born radii kernel | 1 day | Builder |
| Modify force kernels | 0.5 day | Builder |
| Validation & debug | 0.5 day | All kernels |
| Benchmark | 0.5 day | Validation |
| **Phase 2: Optimizations** | **3 days** | Phase 1 |
| Cell list O(N) build | 1 day | Phase 1 |
| Sorting & binning | 1 day | Cell list |
| Load balancing | 1 day | - |
| **Total** | **6 days** | |

**Week 3**: Phase 1 (basic neighbor list, 2-3× speedup)
**Week 4**: Phase 2 (advanced optimizations, 3-5× speedup)

---

## Next Actions

1. ✅ Design complete (this document)
2. ⏭️ Implement neighbor list builder
3. ⏭️ Modify Born radii kernel
4. ⏭️ Validate physics
5. ⏭️ Benchmark performance

Let's build it! 🚀
