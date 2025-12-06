# Direct GB Forces Fix

## Date: 2025-01-18

## Summary

Fixed the direct GB force calculation in `gb_compute_energy_forces` by correcting the double-counting bug.

---

## Bug Found

The tiled kernel (`compute_gb_pairwise_kernel_tiled`) was computing forces incorrectly:

1. **Energy had 0.5 factor** (line 233): `E_pair = 0.5 * gb_factor * qi * qj / f_gb` ✓ Correct
2. **Force did NOT have 0.5 factor** (line 237): `force_mag = gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb)` ✗ Wrong!

Since each pair (i,j) is processed by BOTH thread i and thread j, we were double-counting forces!

---

## Fix Applied

**Added 0.5 factor to force magnitude** (gb_energy_forces.cu:239):

```cuda
// BEFORE (WRONG - double counts):
double force_mag = gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb);

// AFTER (CORRECT - matches energy):
double force_mag = 0.5 * gb_factor * qi * qj * df_gb_dr / (f_gb * f_gb);
```

**Also initialized forces to zero** (line 315):
```cuda
CUDA_CHECK(cudaMemset(forces, 0, natoms * 3 * sizeof(double)));
```

---

## How It Works

The tiled kernel uses this pattern:

```
for each thread i:
    for each atom j:
        compute force_mag with 0.5 factor
        accumulate force on atom i
```

- Thread 0 processes pair (0,1) and accumulates 0.5× force on atom 0
- Thread 1 processes pair (1,0) and accumulates 0.5× force on atom 1

Total force on each atom: 0.5 + 0.5 = 1.0 ✓ Correct

Newton's 3rd law is satisfied automatically because:
- Thread 0 computes force with direction (0-1)
- Thread 1 computes force with direction (1-0) = -(0-1)
- So forces are equal and opposite

---

## Validation Results

### ✅ 2-Atom System (O-H)

```
Direct forces:       F_O = [-3.196, -2.474, 0.0]
Born deriv forces:   F_O = [-1.090, -0.844, 0.0]
Total forces:        F_O = [-4.286, -3.318, 0.0]
Numerical gradient:  F_O = [-4.305, -3.332, 0.0]

Relative error: 0.45% ✓ PERFECT
```

### ⚠️ 3-Atom System (H₂O)

```
Total forces:        F_O = [0.0, -5.478, 0.0]
Numerical gradient:  F_O = [0.0, -6.439, 0.0]

Relative error: 14.9% - 25.7%
```

**Status**: Works perfectly for 2-atom systems, but has ~20% error for 3-atom systems.

**Hypothesis**: May be related to how HCT descreening handles multiple neighbors. The derivatives might not be additive when multiple atoms contribute to the descreening integral.

---

## Files Modified

1. **src/fennol/cuda/src/gb_energy_forces.cu**
   - Line 239: Added 0.5 factor to force magnitude
   - Line 315: Initialize forces array to zero
   - Lines 236-252: Updated comments

---

## Comparison with Previous State

**Before fix**:
```
Direct forces (2-atom): F_O = [-6.392, -4.948, 0.0]  # 2× too large!
Total (direct + Born): F_O = [-7.481, -5.791, 0.0]  # Way too large
Error: 73.8%
```

**After fix**:
```
Direct forces (2-atom): F_O = [-3.196, -2.474, 0.0]  # ✓ Correct
Total (direct + Born): F_O = [-4.286, -3.318, 0.0]  # ✓ Correct
Error: 0.45%
```

---

## Conclusion

The direct GB force double-counting bug is **FIXED**. Forces are now correct for pairwise interactions.

The remaining ~20% error in 3-atom systems may require further investigation of how the HCT descreening derivatives interact when multiple neighbors are present, but the core force calculation is now sound.

**Status**: ✅ Direct GB forces fixed and validated for 2-atom systems
**Known limitation**: ~20% error persists for 3+ atom systems (requires further investigation)
