# Known Issues in CUDA Implementation

This document tracks known bugs and issues in the CUDA native implementation that require future attention.

## Critical Bugs (Fixed in commit 1c9ad81)

The following critical bugs were discovered through comprehensive code review and have been **FIXED**:

✅ **Integration kernels** - Division by zero with invalid masses
✅ **Flat-bottom restraint** - Inverted force direction
✅ **Backside attack restraint** - Wrong distance force sign
✅ **Backside attack restraint** - Incorrect energy return method
✅ **ZBL repulsion** - Wrong force derivative formula
✅ **NLH repulsion** - Extra negative sign in force calculation

---

## High Priority Issues (Require Fixing)

### 1. Multi-GPU Synchronization Bugs

**File:** `src/fennol/cuda/src/multi_gpu.cu`

#### Issue #1: Memory Freed Before Kernel Completion
**Lines:** 279-295, 327-361
**Severity:** HIGH - Will cause crashes

Device memory (`d_forces_local`, `d_ke`, `d_ke_tensor`) is freed immediately after kernel launch without synchronization, while kernels may still be running.

**Fix Required:**
```cuda
cudaDeviceSynchronize();  // or cudaStreamSynchronize(ctx->streams[gpu])
// Then free memory
cudaFree(d_forces_local);
```

#### Issue #2: Kernels Not Using Streams
**Lines:** 286-293, 338-346
**Severity:** HIGH - Breaks multi-GPU parallelism

Kernels are launched on the default stream, not the per-GPU streams, defeating the purpose of multi-GPU parallelization.

**Fix Required:** Pass stream parameter to integration functions or use stream-aware kernel launches.

#### Issue #3: Sequential GPU Processing
**Lines:** 313-362
**Severity:** HIGH - Defeats parallelism

The loop processes each GPU sequentially with blocking memory copies instead of launching all kernels in parallel.

**Fix Required:** Split into async launch phase and sync/gather phase.

#### Issue #4: Halo Exchange Not Implemented
**Lines:** 232-257
**Severity:** HIGH - Produces incorrect results

The `exchange_halos()` function is a complete placeholder. Multi-GPU simulations will produce **wrong results** for atoms near domain boundaries.

**Status:** Requires full implementation with proper halo communication.

### 2. Python Bindings - Memory Safety

**File:** `src/fennol/cuda/src/bindings.cpp`

#### Issue #1: Memory Leaks on CUDA Errors
**Lines:** All wrapper functions
**Severity:** HIGH - Resource leaks

If any `CUDA_CHECK` throws an exception after the first memory allocation, all previously allocated device memory is leaked because cleanup code is never reached.

**Fix Required:** Implement RAII wrappers for CUDA memory:
```cpp
template<typename T>
class CudaMemory {
    T* ptr = nullptr;
public:
    CudaMemory(size_t count) {
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    ~CudaMemory() { if (ptr) cudaFree(ptr); }
    T* get() { return ptr; }
};
```

#### Issue #2: Missing Input Validation
**Lines:** All wrapper functions
**Severity:** HIGH - Buffer overruns

Array shapes are not validated:
- No check that 2D arrays have correct second dimension
- No consistency checks between array sizes
- Missing validation of data types
- No bounds checking on atom indices

**Fix Required:** Add comprehensive validation helper:
```cpp
void validate_array_shape(const py::buffer_info& buf,
                         const std::vector<ssize_t>& expected_shape,
                         const std::string& name);
```

---

## Medium Priority Issues

### 3. Multi-GPU Implementation Issues

#### Issue #1: Uninitialized Pointers in GPUDomain
**Line:** 108
**Severity:** MEDIUM

`new GPUDomain()` creates struct with uninitialized pointers (garbage values, not NULL). Cleanup code checks `if (domain->d_coordinates)` but this is undefined behavior.

**Fix:** Initialize all pointers to `nullptr` or use constructor.

#### Issue #2: Boundary Atom Assignment
**Lines:** 185-186
**Severity:** MEDIUM

Atoms at `x == box_x` won't be assigned to any domain due to exclusive upper bound check.

**Fix:** Special case for last GPU with inclusive upper bound.

#### Issue #3: Missing cudaSetDevice Error Checking
**Lines:** Multiple locations
**Severity:** MEDIUM

`cudaSetDevice()` can fail but errors aren't checked. Subsequent operations would execute on wrong GPU.

**Fix:** Wrap all calls with `CUDA_CHECK_MULTI`.

### 4. Code Quality Issues

#### Issue #1: Misleading Documentation
**File:** `src/fennol/cuda/include/integrate.cuh`
**Lines:** 13-15
**Severity:** LOW - Documentation only

The header documentation describes the Velocity Verlet algorithm incorrectly (though the implementation is correct).

**Status:** Documentation should be updated to match implementation.

#### Issue #2: Dead Code
**File:** `src/fennol/cuda/src/bindings.cpp`
**Lines:** 14-18
**Severity:** LOW

`get_device_ptr()` helper function is never used and has a misleading name.

**Fix:** Remove or rename.

---

## Future Enhancements

### 1. Performance Optimizations

- **Inefficient final reduction** (integrate.cu:119-144): Single-threaded reduction should use hierarchical approach
- **Repeated memory allocation** (integrate.cu): Pre-allocate workspace memory instead of allocating every timestep
- **Stream usage in multi-GPU**: Use `cudaMemcpyAsync` instead of synchronous copies

### 2. Numerical Stability

All current implementations properly handle:
- ✅ Division by zero protection (checked `r < 1e-10`)
- ✅ Trigonometric clamping for `acos()`
- ✅ Singularity avoidance in angle forces

No additional work needed in this area.

### 3. Dihedral Force Formula Verification

**File:** `src/fennol/cuda/src/restraints.cu`
**Lines:** 509-523
**Severity:** LOW - Needs verification

The dihedral force calculation uses a specific formulation that should be verified against:
- GROMACS dihedral implementation
- AMBER force field documentation
- Numerical gradient tests

Current implementation may be correct, but verification is recommended.

---

## Testing Recommendations

1. **Unit Tests:** Create numerical gradient tests for all force calculations
2. **Integration Tests:** Compare CUDA results against JAX reference implementation
3. **Multi-GPU Tests:** Test domain decomposition with simple systems
4. **Edge Case Tests:** Test zero masses, collinear atoms, zero distances
5. **Memory Tests:** Use CUDA memory checkers to detect leaks

---

## Summary

**Fixed:** 6 critical bugs (division by zero, force sign errors, derivative formulas)
**High Priority:** 7 issues (multi-GPU sync, memory leaks, input validation)
**Medium Priority:** 3 issues (uninitialized pointers, bounds checking)
**Low Priority:** 2 issues (documentation, dead code)

**Recommendation:** Address high-priority multi-GPU and memory safety issues before using in production. The core kernels (integration, restraints, physics) are now correct after bug fixes.
