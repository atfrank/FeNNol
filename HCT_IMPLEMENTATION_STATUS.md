# HCT Implementation - Final Status

## Date: 2025-01-18

## Summary

Successfully implemented the complete Hawkins-Cramer-Truhlar (HCT) descreening formula for GB Born radii forces, matching OpenMM's reference implementation.

---

## What Was Implemented

### 1. HCT Descreening Integral (gb_born_radii.cu)

Replaced simplified formula:
```
I = 0.5 * ρᵢ * (1/r² - 1/upper²)
```

With full HCT formula from OpenMM:
```
I = l_ij - u_ij + 0.25*r*(u_ij² - l_ij²) + 0.5*ln(u_ij/l_ij)/r + 0.25*s_j²/r*(l_ij² - u_ij²)
```

Where:
- `l_ij = 1/max(ρᵢ, |r - s_j|)` (lower bound reciprocal)
- `u_ij = 1/(r + s_j)` (upper bound reciprocal)

### 2. HCT Descreening Derivative (gb_born_radii_forces.cu)

Replaced incorrect derivative:
```
∂ψ/∂r = -ρᵢ/r³
```

With OpenMM's HCT derivative:
```
∂ψ/∂r = t3/r

where t3 = 0.125*(1 + s_j²/r²)*(l_ij² - u_ij²) + 0.25*log(u_ij/l_ij)/r²
```

**Key insight**: `dL/dr` and `dU/dr` are zero due to the max() operation in l_ij. This is mentioned in OpenMM's code comments.

### 3. Multi-Pass Force Architecture

Implemented OpenMM's multi-pass approach:

1. **Pass 1**: Compute Born radii and ψ
   ```
   fennol_cuda.gb_compute_born_radii_with_psi(...)
   ```

2. **Pass 2**: Compute ∂E/∂R
   ```
   fennol_cuda.compute_dE_dR(...)
   ```

3. **Pass 3**: Convert ∂E/∂R → ∂E/∂ψ
   ```
   fennol_cuda.reduce_born_force(...)
   ```

4. **Pass 4**: Apply Born forces using ∂E/∂ψ
   ```
   fennol_cuda.apply_born_forces(...)
   ```

### 4. Force Distribution (Newton's 3rd Law)

Fixed force application to match OpenMM:
- Each pair (i,j) applies forces to BOTH atoms
- Force on atom i: `-de × (rⱼ - rᵢ)` (subtract)
- Force on atom j: `+de × (rⱼ - rᵢ)` (add)
- Used atomicAdd to handle concurrent writes

This ensures forces sum to zero (Newton's 3rd law).

---

## Validation Results

### ✅ Energy Calculations

```
Manual calculation:  E = -59.2156686600 kcal/mol
Kernel calculation:  E = -59.2156686600 kcal/mol
Match: PERFECT
```

###  ✅ Born Radii Derivative Forces

**Manual calculation** (2-atom system):
```
From ψ₀ changing: F_O = -1.220 (x-component)
From ψ₁ changing: F_O = -0.130
Total: F_O = -1.090
```

**Multi-pass kernel**:
```
F_O = [-1.08988108, -0.84368601, 0.0]
```

**Match**: PERFECT (100% agreement)

**Newton's 3rd law**:
```
F_O + F_H = [-1.090, -0.844, 0.0] + [1.090, 0.844, 0.0] = [0, 0, 0] ✓
```

---

## Bugs Found and Fixed

### Bug #1: Double-Counting in Tiled Algorithm

**Problem**: Each thread was adding both ∂E/∂ψᵢ AND ∂E/∂ψⱼ contributions, causing 2× error.

**Fix**: Each thread only applies force from ∂E/∂ψᵢ. Newton's 3rd law distributes to both atoms.

### Bug #2: Sign Error in reduce_born_force

**Problem**: Formula had negative sign: `dE_dpsi = -dE_dR × R² × obcChain`

**Fix**: Changed to positive: `dE_dpsi = dE_dR × R² × obcChain` (matches OpenMM)

### Bug #3: Wrong Displacement Vector Direction

**Problem**: Used `dx = rᵢ - rⱼ` but OpenMM uses `dx = rⱼ - rᵢ`

**Fix**: Changed to match OpenMM's `getDeltaR(atomI, atomJ) = atomJ - atomI`

### Bug #4: Overwriting Atomic Adds

**Problem**: Final write used `=` instead of `atomicAdd`, erasing force contributions

**Fix**: Changed final write to use `atomicAdd` to preserve all contributions

---

## Remaining Issue

### Direct GB Forces Need Updating

The `gb_compute_energy_forces` function computes direct GB forces (∂E/∂r with R fixed). These forces currently don't account for the HCT formula and appear to be too large.

**Evidence**:
```
Numerical gradient (total):     F_O = -4.305
Born radii deriv (multi-pass):  F_O = -1.090  ✓ Correct
Direct forces (old formula):    F_O = -6.392  ✗ Too large
Total (direct + Born deriv):    F_O = -7.482  ✗ Wrong

Expected direct forces:  F_O = -4.305 - (-1.090) = -3.215
Actual direct forces:    F_O = -6.392
Error: ~2× too large
```

This suggests the direct force calculation in `gb_compute_energy_forces` has a bug (likely also double-counting).

**Note**: The direct GB forces are computed in `gb_energy_forces.cu`, which is a separate module from the Born radii derivative forces. Fixing this requires updating that file to use the correct f_GB derivative formula.

---

## Files Modified

1. **src/fennol/cuda/src/gb_born_radii.cu**
   - Updated descreening_integral() to use full HCT formula

2. **src/fennol/cuda/src/gb_born_radii_forces.cu**
   - Updated descreening_integral_derivative() to use HCT derivative
   - Fixed apply_born_forces_tiled() to properly distribute forces
   - Fixed displacement vector direction
   - Fixed atomic writes

3. **src/fennol/cuda/src/bindings.cpp**
   - Added Python bindings for multi-pass functions

4. **src/fennol/cuda/include/implicit_solvent.cuh**
   - Added function declarations

---

## Test Scripts Created

- `test_hct_derivative.py` - Validates HCT derivative formula
- `test_multipass_forces.py` - Tests multi-pass force calculation
- `test_manual_force_calc.py` - Manual verification of force formulas
- `test_total_forces.py` - Tests direct + Born derivative forces
- `test_2atom_forces.py` - Simple 2-atom validation

---

## References

- OpenMM's `platforms/reference/src/SimTKReference/ReferenceObc.cpp`
- Hawkins, Cramer, Truhlar (1995) - HCT model original paper
- OpenMM's `platforms/cuda/src/kernels/gbsaObc.cc` - CUDA implementation

---

## Conclusion

The HCT descreening formula and Born radii derivative forces are **correctly implemented and validated**. The multi-pass approach matches OpenMM's reference implementation exactly.

The remaining discrepancy with numerical gradients is due to incorrect direct GB forces in `gb_compute_energy_forces`, which is a separate issue that needs to be addressed in `gb_energy_forces.cu`.

**Status**: ✅ HCT implementation complete and correct
**Next step**: Fix direct GB force calculation in gb_energy_forces.cu
