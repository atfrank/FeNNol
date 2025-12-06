# Debugging Plan: Multi-Pass GB Force Calculation

## Problem Statement

Multi-pass implementation gives forces that are:
- **Sign**: Wrong (positive instead of negative)
- **Magnitude**: ~28× too large
- Numerical gradient: F_y = -4.616 kcal/(mol·Å)
- Multi-pass result: F_y = +127.67 kcal/(mol·Å)

## Hypothesis

Both old and new implementations have similar ~14-30× errors, suggesting:
1. **Not a multi-pass specific bug**
2. **Likely a fundamental error in force formula or constants**
3. **Possible issues:**
   - Wrong sign convention in descreening derivative
   - Missing factor (0.5? 2.0?)
   - Unit conversion error
   - Double-counting in pair loop

## Debugging Strategy

### Phase 1: Manual Calculation Validation (30 min)

**Goal**: Calculate expected force by hand for single water molecule, compare with kernel output.

**Steps:**

1. **Extract kernel output for pair (O, H1)**:
   ```
   dE_dR[O] = 34.094
   dE_dR[H] = 5.402
   dE_dpsi[O] = -63.198 (with negative sign fix)
   dE_dpsi[H] = -2.719
   dpsi_O_dr = -1.710
   dpsi_H_dr = -1.368
   ```

2. **Manual calculation**:
   ```python
   # Force magnitude
   F_mag = -dE_dpsi[O] * dpsi_O_dr - dE_dpsi[H] * dpsi_H_dr
         = -(-63.198) * (-1.710) - (-2.719) * (-1.368)
         = -108.05 - 3.72
         = -111.77

   # Direction: (O - H) / r = (0, -0.586, 0) / 0.957 = (0, -0.612, 0)
   F_y = -111.77 * (-0.612) = +68.4

   # But we process each pair twice (once from O, once from H)
   # So total might be: 2 × 68.4 = 136.8  ← matches current output!
   ```

3. **Expected from numerical gradient**:
   ```
   F_y should be -4.616
   Ratio: 68.4 / 4.616 = 14.8×  (or 136.8 / 4.616 = 29.6×)
   ```

**Conclusion from Phase 1**: Find the source of the 15× factor!

### Phase 2: Check for Systematic Errors (20 min)

**Test 1: Check if we're double-counting pairs**

Create modified kernel that only processes pairs where `i < j`:

```cuda
for (int t = 0; t < tile_size; t++) {
    int j = tile_start + t;
    if (i >= j) continue;  // Only process each pair once

    // ... rest of calculation ...

    // Apply force to BOTH atoms using Newton's 3rd law
    atomicAdd(&born_forces[i*3 + 0], fx_i);
    atomicAdd(&born_forces[i*3 + 1], fy_i);
    atomicAdd(&born_forces[i*3 + 2], fz_i);

    atomicAdd(&born_forces[j*3 + 0], -fx_i);  // Equal and opposite
    atomicAdd(&born_forces[j*3 + 1], -fy_i);
    atomicAdd(&born_forces[j*3 + 2], -fz_i);
}
```

**Expected**: If double-counting is the issue, forces should be cut in half (but still ~15× too large).

**Test 2: Check descreening integral formula**

Compare our `descreening_integral_derivative` with OpenMM's implementation:
- Extract exact formula from OpenMM source
- Verify signs and factors
- Test on simple case: two atoms at specific distance

**Test 3: Check OBC chain derivative**

Verify `born_radius_derivative_wrt_psi` calculation:
- Manually compute for atom O with known R, ρ, ψ values
- Compare with kernel output
- Check if there's a missing factor

### Phase 3: Compare with OpenMM Source (30 min)

**Action**: Fetch OpenMM's complete force kernel and compare line-by-line.

**Files to examine**:
1. `platforms/cuda/src/kernels/gbsaObc2.cu` - Force kernel
2. `platforms/reference/src/ReferenceObc*` - Reference implementation

**What to check**:
- Exact formula for `∂ψ/∂r`
- Exact formula for `∂(1/R)/∂ψ`
- Sign conventions at each step
- Any scaling factors or constants

### Phase 4: Test on Simpler Systems (20 min)

**Test A: Two identical atoms**
- Should work (worked before)
- Validate that multi-pass gives same result as before

**Test B: Two atoms with different charges, same radii**
- Simpler than full water molecule
- Easier to trace calculation by hand

**Test C: Frozen Born radii test**
- Set all `dE_dpsi = 0` except for one atom
- Set that one to a known value (e.g., 1.0)
- Calculate expected force analytically
- Compare with kernel output

### Phase 5: Check for Missing Factors (15 min)

**Common culprits**:

1. **Factor of 0.5**: GB energy has a 0.5 factor. Does the derivative?
   ```
   E_GB = -0.5 * (1 - 1/ε) * Σ q_i q_j / f_GB
   ∂E/∂R might have different prefactor?
   ```

2. **Self-energy contribution**: Are we handling it correctly?
   ```
   Current: dE_dR includes self-energy
   Check: Should self-energy contribute to Born forces?
   ```

3. **Pair symmetry**:
   ```
   Current: Each pair (i,j) processed by both threads
   Check: Should we divide by 2 somewhere?
   ```

### Phase 6: Instrument Kernels with Detailed Output (30 min)

Add comprehensive debug output for ONE atom (O) and ONE pair (O,H1):

```cuda
if (i == 0) {
    printf("=== ATOM 0 (O) ===\n");
    printf("dE_dR[0] = %.10f\n", dE_dR[0]);
    printf("R[0] = %.10f\n", R[0]);
    printf("rho[0] = %.10f\n", rho[0]);
    printf("psi[0] = %.10f\n", psi[0]);
    printf("obcChain[0] = %.10f\n", obcChain[0]);
    printf("dE_dpsi[0] = %.10f\n", dE_dpsi[0]);
}

if (i == 0 && j == 1) {
    printf("=== PAIR (0,1) ===\n");
    printf("r = %.10f\n", r);
    printf("rho_i = %.10f, rho_j = %.10f\n", rho_i, rho_j);
    printf("dpsi_i_dr = %.10f\n", dpsi_i_dr);
    printf("dpsi_j_dr = %.10f\n", dpsi_j_dr);
    printf("dE_dpsi_i = %.10f\n", dE_dpsi_i);
    printf("dE_dpsi_j = %.10f\n", dE_dpsi_j);
    printf("force_mag = %.10f\n", force_mag);
    printf("dx = %.10f, dy = %.10f, r_inv = %.10f\n", dx, dy, r_inv);
    printf("Force contribution: fy = %.10f\n", force_mag * dy * r_inv);
}
```

Then manually verify each value matches hand calculation.

## Expected Timeline

- **Phase 1**: 30 min - Manual calculation
- **Phase 2**: 20 min - Systematic error checks
- **Phase 3**: 30 min - OpenMM comparison
- **Phase 4**: 20 min - Simpler test systems
- **Phase 5**: 15 min - Check for missing factors
- **Phase 6**: 30 min - Detailed instrumentation

**Total**: ~2.5 hours

## Decision Points

After **Phase 1**: If manual calculation identifies the 15× factor source → go directly to fix
After **Phase 2, Test 1**: If double-counting confirmed → fix pair processing
After **Phase 3**: If OpenMM formula differs → update our formula
After **Phase 5**: If missing factor found → add it

## Success Criteria

✅ Multi-pass forces match numerical gradient within 1% error
✅ Sign is correct (negative for this test case)
✅ All test cases pass (2-atom, water, water dimer)

## Most Likely Issue (Hypothesis)

Based on the fact that **both old and new implementations have ~15× error**, my hypothesis is:

**The `descreening_integral_derivative` function is missing a factor or has wrong sign.**

The descreening integral itself is:
```
ψ = Σ_j I(r, ρ_i, ρ_j)
```

And we're computing `∂ψ/∂r`, which current gives `∂I/∂r = -ρ_i/r³` for the overlap region.

But maybe the correct formula has an additional factor? Let me check OpenMM's exact implementation in Phase 3.

## Immediate Next Step

**Start with Phase 1**: Print out all intermediate values and do the hand calculation to verify where the 15× factor comes from.
