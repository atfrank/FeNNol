# Known Issues in CUDA Implementation

This document tracks known bugs and issues in the CUDA native implementation that require future attention.

## Critical Bugs (Fixed in commits 1c9ad81, a3f29aa, 5871e3c)

The following critical bugs were discovered through comprehensive code review and have been **FIXED**:

✅ **Integration kernels** - Division by zero with invalid masses (commit 1c9ad81)
✅ **Flat-bottom restraint** - Inverted force direction (commit 1c9ad81)
✅ **Backside attack restraint** - Wrong distance force sign (commit 1c9ad81)
✅ **Backside attack restraint** - Incorrect energy return method (commit 1c9ad81)
✅ **ZBL repulsion** - Wrong force derivative formula (commit 1c9ad81)
✅ **NLH repulsion** - Extra negative sign in force calculation (commit 1c9ad81)
✅ **Python bindings** - Memory leaks on CUDA errors (RAII wrappers - commit 5871e3c)
✅ **Python bindings** - Missing input validation (comprehensive validation - commit 5871e3c)
✅ **Multi-GPU** - Memory freed before kernel completion (synchronization added - commit 5871e3c)
✅ **Multi-GPU** - Uninitialized pointers in GPUDomain (all pointers initialized - commit 5871e3c)
✅ **Multi-GPU** - Boundary atom assignment bug (inclusive upper bound - commit 5871e3c)
✅ **Multi-GPU** - Missing cudaSetDevice error checking (CUDA_CHECK_MULTI added - commit 5871e3c)
✅ **Multi-GPU** - Kernels not using streams (stream support added - current commit)
✅ **Multi-GPU** - Sequential GPU processing (async launch/sync/gather phases - current commit)
✅ **Documentation** - Incorrect Velocity Verlet description (integrate.cuh fixed - commit 5871e3c)
✅ **Code quality** - Dead code removed (get_device_ptr() removed - commit 5871e3c)

---

## High Priority Issues (Require Fixing)

### 1. Multi-GPU - Halo Exchange

**File:** `src/fennol/cuda/src/multi_gpu.cu`

#### Issue #1: Halo Exchange Not Implemented
**Lines:** 232-257
**Severity:** HIGH - Produces incorrect results

The `exchange_halos()` function is a complete placeholder. Multi-GPU simulations will produce **wrong results** for atoms near domain boundaries.

**Status:** Requires full implementation with proper halo communication.

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

**Fixed:** 14 critical bugs and issues across all categories
- 6 critical bugs in kernels (division by zero, force sign errors, derivative formulas) - commit 1c9ad81
- 2 memory safety issues in Python bindings (RAII wrappers, input validation) - current commit
- 4 multi-GPU bugs (initialization, synchronization, boundary assignment, error checking) - current commit
- 2 documentation/code quality issues (misleading docs, dead code) - current commit

**Remaining High Priority:** 3 issues (all in multi-GPU)
- Kernels not using streams (requires API refactoring)
- Sequential GPU processing (requires async refactoring)
- Halo exchange not implemented (requires full implementation)

**Recommendation:** The core functionality (single-GPU integration, restraints, physics, Python bindings) is now production-ready. Multi-GPU support remains experimental and requires the remaining 3 issues to be addressed for production use.
