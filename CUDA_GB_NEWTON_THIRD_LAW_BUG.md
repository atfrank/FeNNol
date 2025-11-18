# CRITICAL BUG: CUDA GB Forces Violate Newton's 3rd Law

**Date**: November 18, 2025
**Severity**: CRITICAL (blocks 10× optimization)
**Status**: IDENTIFIED - Must fix before optimization

---

## Executive Summary

The current CUDA GB implementation produces forces that **violate Newton's 3rd law** (momentum conservation). Net force is non-zero (~1-2 kcal/mol/Å for small systems), indicating a fundamental error in the Born radii force derivatives.

This bug would cause:
- ❌ Energy drift in MD simulations
- ❌ Invalid dynamics (unphysical trajectories)
- ❌ Cannot validate optimizations (baseline is wrong!)

**This bug MUST be fixed before attempting any performance optimizations.**

---

## Evidence

### Test Results

```
tests/validation/test_gb_cuda_baseline.py::TestCUDAvsReference::test_water_molecule_forces_newton_third_law
FAILED

❌ Newton's 3rd law check FAILED:
  Net force: [0.         0.98701916 0.        ]
  Magnitude: 9.8701916172e-01 kcal/(mol·Å) (tol: 1.0000000000e-06)
```

For a water molecule (3 atoms), the net force should be EXACTLY zero (within numerical precision). Instead, we see ~1 kcal/mol/Å net force in the y-direction.

### Comparison to Reference

| Test | Born Radii | Energy | Forces (Newton 3rd) |
|------|-----------|--------|-------------------|
| Single atom | ✅ PASS | ✅ PASS | ✅ PASS (trivial) |
| Two atoms | ✅ PASS | ✅ PASS | ✅ PASS |
| Water (3 atoms) | ✅ PASS | ✅ PASS | ❌ **FAIL** |
| Two waters (6 atoms) | ✅ PASS | ✅ PASS | ❌ **FAIL** |

**Pattern**: Newton's 3rd law violation appears when system has **asymmetric geometry** (water molecule). Born radii and energy are CORRECT, so the bug is specifically in **force calculation**.

---

## Root Cause Analysis

### Where the Bug Is

The bug is in **Born radii derivative forces**, not the direct pairwise forces:

```
F_total = F_direct + F_born_radii
```

- ✅ `F_direct`: Pairwise GB forces (appears correct - energy matches)
- ❌ `F_born_radii`: Chain rule term `(∂E/∂R_i) * (∂R_i/∂r_ij)` (violates Newton's 3rd law)

### Why Newton's 3rd Law Fails

For Newton's 3rd law to hold:
```
F_ij = -F_ji  (force on i from j equals negative of force on j from i)
```

This requires:
1. **Direct forces**: Symmetric by construction ✅
2. **Born radii forces**: Must account for ALL derivative contributions ❌

The chain rule for Born radii forces is:
```
F_i^born = Σ_j (∂E/∂R_i) * (∂R_i/∂r_ij) * r̂_ij
         + Σ_j (∂E/∂R_j) * (∂R_j/∂r_ij) * r̂_ij
```

**Hypothesis**: The CUDA implementation may be missing the second term (contribution from ∂R_j/∂r_ij).

### Code Location

File: `src/fennol/cuda/src/gb_born_radii_forces.cu`

Functions to check:
- `descreening_integral_derivative()` - Computes ∂I/∂r
- `born_radius_derivative_wrt_psi()` - Computes ∂(1/R)/∂ψ
- `gb_compute_forces_complete()` - Main force kernel

The bug is likely in how these derivatives are accumulated across atom pairs.

---

## Impact on 10× Optimization Plan

### BLOCKER Issues

1. **Cannot validate optimizations**
   If baseline is wrong, we can't tell if optimizations preserve physics!

2. **MD simulations invalid**
   Newton's 3rd law violation → energy drift → unphysical trajectories

3. **Performance benchmarks misleading**
   Comparing broken baseline to optimized version is meaningless

### Required Fix Before Optimization

```
Week 1: ✅ Validation framework (DONE)
Week 2: ❌ FIX THIS BUG FIRST! (CRITICAL)
Week 3: Validate fixed baseline
Week 4+: Begin optimizations
```

**Cannot proceed to optimization until this is fixed.**

---

## Fix Strategy

### Step 1: Reproduce with Minimal Case

Create the simplest failing case:
- Single water molecule (3 atoms)
- TIP3P charges and radii
- Verify net force ≠ 0

### Step 2: Compare to Reference Line-by-Line

Implement NumPy version of Born radii forces (analytical, not numerical):
```python
def compute_born_radii_forces_reference(coords, charges, born_radii, radii, ...):
    """
    Reference implementation matching OpenMM's ReferenceObc.cpp exactly.
    """
    # For each pair (i,j):
    for i in range(N):
        for j in range(N):
            if i == j: continue

            # Compute ∂I_ij/∂r (descreening derivative)
            dI_dr = descreening_integral_derivative(r, rho_i, rho_j)

            # Chain rule: ∂R_i/∂r_ij = (∂R_i/∂ψ_i) * (∂ψ_i/∂I_ij) * (∂I_ij/∂r)
            dR_i_dr = born_derivative[i] * dI_dr

            # Energy derivative: ∂E/∂R_i
            dE_dR_i = energy_derivative_wrt_born[i]

            # Force contribution: F_i += (∂E/∂R_i) * (∂R_i/∂r_ij) * r̂_ij
            F_i += dE_dR_i * dR_i_dr * r_vec / r

            # CRITICAL: Also account for ∂R_j/∂r_ij (Newton's 3rd law!)
            dR_j_dr = born_derivative[j] * descreening_integral_derivative(r, rho_j, rho_i)
            dE_dR_j = energy_derivative_wrt_born[j]
            F_i += dE_dR_j * dR_j_dr * r_vec / r
```

### Step 3: Verify Reference Satisfies Newton's 3rd Law

Validate that NumPy reference has net force = 0:
```python
forces_ref = compute_born_radii_forces_reference(...)
assert np.linalg.norm(np.sum(forces_ref, axis=0)) < 1e-10
```

### Step 4: Fix CUDA Implementation

Once reference is validated, compare CUDA kernel line-by-line and fix discrepancies.

### Step 5: Re-validate

Run full validation suite:
```bash
pytest tests/validation/test_gb_cuda_baseline.py -v
```

All tests must PASS before proceeding to optimization.

---

## Validation Test to Add

After fix, add regression test:

```python
def test_newton_third_law_asymmetric_molecule():
    """
    Regression test for Newton's 3rd law bug.

    Water molecule is a good test case because:
    - Asymmetric geometry (not all atoms equivalent)
    - 3 atoms (non-trivial but fast to compute)
    - Well-defined TIP3P parameters
    """
    coords = np.array([
        [0.0, 0.0, 0.0],        # O
        [0.757, 0.586, 0.0],    # H1
        [-0.757, 0.586, 0.0]    # H2
    ])
    charges = np.array([-0.834, 0.417, 0.417])

    _, forces = compute_gb_forces_cuda(coords, charges, ...)

    net_force = np.sum(forces, axis=0)
    net_force_magnitude = np.linalg.norm(net_force)

    # Must be within numerical precision (FP64)
    assert net_force_magnitude < 1e-10, \
        f"Newton's 3rd law violated: net force = {net_force_magnitude}"
```

---

## OpenMM Reference Code

The correct implementation is in OpenMM `ReferenceObc.cpp`:

Key functions:
- `computeBornRadii()` - Born radii calculation
- `computeGBForces()` - GB electrostatic forces
- `computeBornEnergyForces()` - Born radii derivative forces

OpenMM correctly implements Newton's 3rd law by accumulating BOTH:
1. Force on atom i due to ∂R_i/∂r_ij
2. Force on atom i due to ∂R_j/∂r_ij

Our CUDA implementation may only have (1), missing (2).

---

## Timeline

| Task | Estimate | Priority |
|------|----------|----------|
| Create analytical reference for Born radii forces | 4-6 hours | HIGH |
| Validate reference satisfies Newton's 3rd law | 1 hour | HIGH |
| Debug CUDA kernel | 4-8 hours | CRITICAL |
| Validate fix | 1 hour | HIGH |
| Add regression tests | 2 hours | MEDIUM |
| **Total** | **12-18 hours** | **~2 days** |

---

## Success Criteria

Before declaring bug fixed:

1. ✅ NumPy analytical reference satisfies Newton's 3rd law (net force < 1e-10)
2. ✅ CUDA matches NumPy reference within tolerance (1e-6)
3. ✅ All validation tests pass
4. ✅ Water molecule net force < 1e-10
5. ✅ Two water molecules net force < 1e-10
6. ✅ 100-atom system net force < 1e-8
7. ✅ Regression test added to prevent re-introduction

---

## Next Actions

**IMMEDIATE (before any optimization)**:

1. ⏸️ **STOP** all optimization work
2. 🔴 **FIX** Newton's 3rd law bug
3. ✅ **VALIDATE** fix with comprehensive tests
4. 📝 **DOCUMENT** fix and add regression tests
5. ✅ **VERIFY** baseline is correct
6. 🚀 **THEN** proceed with 10× optimization plan

**The validation framework caught a critical bug before we wasted time optimizing broken code. This is exactly why we built it!** 🎯

---

## Conclusion

**This bug is a validation framework success story!**

- ✅ Framework detected critical bug in existing code
- ✅ Prevented wasted optimization effort on broken baseline
- ✅ Ensures future optimizations validate against correct physics
- ✅ Demonstrates value of multi-level validation hierarchy

Once fixed, we can proceed with confidence to 10× speedup optimization.
