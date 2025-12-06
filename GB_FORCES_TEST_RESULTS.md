# GB Forces Numerical Gradient Test Results

## Executive Summary

**STATUS**: CRITICAL BUG FOUND IN `gb_compute_forces_complete` implementation

The comprehensive numerical gradient test revealed that the "complete" GB forces implementation (`gb_compute_forces_complete`) has a **critical bug** that causes it to produce incorrect forces. Ironically, the NEW implementation performs WORSE than the OLD (incomplete) implementation.

## Test Configuration

### Test System
- **System**: Water dimer (6 atoms: 2 water molecules)
- **Separation**: 2.8 Å
- **Total charge**: 0.000 e
- **Parameters**:
  - Dielectric: 80.0
  - Cutoff: 12.0 Å
  - Finite difference step: 1×10⁻⁵ Å

### Acceptance Criteria
- Max absolute error < 0.01 kcal/(mol·Å)
- Max relative error < 5%

## Test Results

### OLD Implementation (gb_compute_energy_forces)
**Known limitation**: Missing Born radii derivatives (∂R/∂x terms)

| Metric | Value | Status |
|--------|-------|--------|
| Max absolute error | 19.88 kcal/(mol·Å) | **FAIL** |
| RMS error | 7.80 kcal/(mol·Å) | **FAIL** |
| Max relative error | 229.59% | **FAIL** |
| Mean relative error | 98.24% | **FAIL** |

**Assessment**: Expected failure - implementation is known to be incomplete.

### NEW Implementation (gb_compute_forces_complete)
**Expected**: Complete chain rule with Born radii derivatives

| Metric | Value | Status |
|--------|-------|--------|
| Max absolute error | 139.52 kcal/(mol·Å) | **FAIL** (7× worse!) |
| RMS error | 47.90 kcal/(mol·Å) | **FAIL** (6× worse!) |
| Max relative error | 6726.97% | **FAIL** (29× worse!) |
| Mean relative error | 1437.89% | **FAIL** (15× worse!) |

**Assessment**: UNEXPECTED FAILURE - NEW implementation is significantly WORSE than OLD implementation!

## Diagnostic Analysis

### Energy Comparison
- OLD energy: -106.807461 kcal/mol
- NEW energy: -106.807461 kcal/mol
- **Difference: 0.000000 kcal/mol** ✓

**Conclusion**: Both implementations compute identical energies (correct).

### Force Comparison
| Component | OLD Force | NEW Force | Numerical Gradient | OLD Error | NEW Error |
|-----------|-----------|-----------|-------------------|-----------|-----------|
| Atom 0, x | 7.302 | 22.060 | 27.184 | 19.88 | 5.12 |
| Atom 0, y | -2.688 | **141.595** | 2.074 | 4.76 | **139.52** |
| Atom 1, y | 0.765 | 14.023 | -8.794 | 9.56 | 22.82 |

**Observations**:
1. The NEW implementation produces absurdly large forces (141.6 kcal/(mol·Å))
2. The y-component forces are especially problematic
3. Forces are completely unrealistic for a water dimer system

## Root Cause Analysis

### Bug Location
**File**: `src/fennol/cuda/src/gb_born_radii_forces.cu`
**Function**: `compute_born_radii_forces_tiled` kernel
**Lines**: 134-147

### The Bug
```cuda
// Compute ∂E/∂Rᵢ (energy derivative w.r.t. Born radius of atom i)
double dE_dR_i = 0.0;

if (i < natoms) {
    // GB factor: -0.5 * (1 - 1/ε) * COULOMB
    double gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * COULOMB_CONST;

    // Self-energy contribution: ∂/∂Rᵢ (qᵢ²/Rᵢ) = -qᵢ²/Rᵢ²
    dE_dR_i = gb_factor * (-qi * qi / (R_i * R_i));

    // Pairwise energy contributions: ∂/∂Rᵢ (qᵢqⱼ/f_GB)
    // This requires computing ∂f_GB/∂Rᵢ for all pairs (i,j)
    // For now, we'll compute this in the tile loop below to avoid redundant work
}
```

**Problem**: The comment says "For now, we'll compute this in the tile loop below" but this is **never implemented**!

The kernel only computes the **self-energy** contribution to ∂E/∂Rᵢ, completely omitting the **pairwise energy** contributions. This is a critical omission because the pairwise terms dominate the Born radii derivatives.

### What's Missing

The complete ∂E/∂Rᵢ should include:

1. **Self-energy** (currently implemented):
   ```
   ∂E/∂Rᵢ = gb_factor × (-qᵢ²/Rᵢ²)
   ```

2. **Pairwise contributions** (MISSING):
   ```
   ∂E/∂Rᵢ += Σⱼ gb_factor × qᵢqⱼ × ∂f_GB/∂Rᵢ
   ```

   where:
   ```
   f_GB = 1/√(rᵢⱼ² + RᵢRⱼexp(-rᵢⱼ²/(4RᵢRⱼ)))

   ∂f_GB/∂Rᵢ = complex expression involving Rᵢ, Rⱼ, rᵢⱼ
   ```

The pairwise contributions are typically 5-10× larger than the self-energy term, which explains why the forces are so wrong.

## Impact Assessment

### Severity
**CRITICAL** - The "complete" forces implementation produces completely incorrect forces.

### Affected Components
1. `gb_compute_forces_complete()` CUDA kernel
2. Any MD simulations using the "complete" GB forces
3. The `OBC` implicit solvent model when using CUDA backend

### Current Status
- The OLD implementation (incomplete) is actually MORE accurate than the NEW implementation (buggy)
- Neither implementation passes numerical gradient tests
- **Recommendation**: Do NOT use `gb_compute_forces_complete` until this bug is fixed

## Required Fix

The `compute_born_radii_forces_tiled` kernel needs to be modified to:

1. Add a second tile loop BEFORE the force calculation to compute pairwise ∂E/∂Rᵢ contributions
2. Load additional data (charges, Born radii) into shared memory for this calculation
3. Accumulate pairwise contributions to `dE_dR_i`
4. Then use the complete `dE_dR_i` (self + pairwise) in the force calculation

This is a significant implementation effort requiring:
- Additional shared memory
- Additional tile loops
- Careful handling of symmetry (i↔j pairs)
- Potential performance implications

## Test Files Created

1. **`test_gb_forces_gradient.py`** - Comprehensive numerical gradient test comparing OLD vs NEW implementations
2. **`test_gb_diagnostic.py`** - Diagnostic tool for debugging GB force calculations

Both scripts can be run to verify the bug and validate any future fixes.

## Recommendations

1. **Immediate**: Revert to using OLD implementation or JAX backend for GB calculations
2. **Short-term**: Fix the bug in `compute_born_radii_forces_tiled` kernel
3. **Long-term**: Add numerical gradient tests to CI/CD to catch these issues automatically
4. **Documentation**: Update docs to warn users about the current state of the CUDA GB implementation

---

**Test Date**: 2025-11-17
**Tester**: Claude (Comprehensive Analysis)
**Test Scripts**:
- `/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/test_gb_forces_gradient.py`
- `/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol/test_gb_diagnostic.py`
