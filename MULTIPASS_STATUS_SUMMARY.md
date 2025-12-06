# Multi-Pass GB Force Implementation - Status Summary

## Date: 2025-01-18

## What Was Implemented

Successfully implemented OpenMM's multi-pass architecture for GB Born radius derivative forces:

### 1. ✅ CUDA Kernels
- `reduce_born_force` kernel (`gb_born_radii_forces.cu:269-310`)
  - Converts ∂E/∂R → ∂E/∂ψ
  - Formula: `dE_dpsi[i] = -dE_dR[i] * R_i² * obcChain[i]`
  - `obcChain` = ∂(1/R)/∂ψ (matches OpenMM)

- `apply_born_forces_tiled` kernel (`gb_born_radii_forces.cu:537-660`)
  - Applies Born forces using ∂E/∂ψ
  - Formula: `F = -(∂E/∂ψ_i)×(∂ψ_i/∂r) - (∂E/∂ψ_j)×(∂ψ_j/∂r)`
  - Includes both atom i and atom j contributions

### 2. ✅ Host Wrapper Functions
- `compute_dE_dR_host` (`gb_born_radii_forces.cu:765-791`)
- `reduce_born_force_host` (`gb_born_radii_forces.cu:796-817`)
- `apply_born_forces_host` (`gb_born_radii_forces.cu:822-847`)

### 3. ✅ Header Declarations
- Added to `implicit_solvent.cuh:143-188`

### 4. ✅ Python Bindings
- `py_compute_dE_dR` (`bindings.cpp:828-876`)
- `py_reduce_born_force` (`bindings.cpp:878-935`)
- `py_apply_born_forces` (`bindings.cpp:937-988`)
- Registered with pybind11 (`bindings.cpp:1265-1278`)

### 5. ✅ Build System
- Successfully compiles
- All functions accessible from Python

## Current Status: 🔴 INCORRECT FORCES

### Test Results

**Numerical Gradient (CORRECT):**
```
F_O_y = -4.616 kcal/(mol·Å)
```

**Multi-Pass Implementation:**
```
F_O_y = +127.67 kcal/(mol·Å)  ❌ WRONG SIGN AND MAGNITUDE
```

**Error: 28× too large, WRONG SIGN**

### Key Issues Identified

1. **Sign Confusion**: Multiple sign flips throughout the chain rule
   - `obcChain` = ∂(1/R)/∂ψ (negative)
   - `∂R/∂ψ` = -R² × obcChain (positive)
   - Force formula has multiple negative signs

2. **Magnitude Error**: ~30× too large
   - Could be due to:
     - Double-counting in tiled algorithm?
     - Missing factor in formula?
     - Wrong units somewhere?

3. **Derivative Functions**:
   - `born_radius_derivative_wrt_psi`: Now returns ∂(1/R)/∂ψ (matching OpenMM)
   - `descreening_integral_derivative`: Returns ∂ψ/∂r (negative)

## Debugging Performed

1. ✅ Validated energy calculation - energies are correct
2. ✅ Validated numerical gradient - confirmed F_y = -4.616
3. ✅ Added extensive debug output to kernels
4. ✅ Tested with GPU's `compute_dE_dR` (not Python version)
5. ✅ Fixed `obcChain` to return ∂(1/R)/∂ψ instead of ∂R/∂ψ
6. ✅ Added both ∂E/∂ψ_i and ∂E/∂ψ_j contributions to force
7. ⏸️ Sign and magnitude still incorrect

## Comparison with Old Implementation

**Old `gb_compute_forces_complete` (single-pass):**
```
F_O_y = -66.28 kcal/(mol·Å)  ❌ Also wrong! (14× too large)
```

**Observation**: Both implementations are wrong by similar factors (14-30×), suggesting a fundamental issue not specific to multi-pass approach.

## Next Steps to Debug

### Option 1: Verify Individual Components
1. Print out intermediate values for ∂E/∂R, ∂E/∂ψ, ∂ψ/∂r
2. Manually calculate expected force for single pair
3. Compare with kernel output step-by-step

### Option 2: Compare with OpenMM Source
1. Extract exact formulas from OpenMM CUDA kernels
2. Match every sign and factor precisely
3. Verify unit conversions

### Option 3: Start from Simplest Case
1. Test with 2 identical atoms (worked before)
2. Add complexity incrementally
3. Find exactly where it breaks

## Files Modified

- `src/fennol/cuda/src/gb_born_radii_forces.cu`
- `src/fennol/cuda/include/implicit_solvent.cuh`
- `src/fennol/cuda/src/bindings.cpp`
- Test scripts: `test_multipass.py`, `test_energy_only.py`

## Commit Status

Not committed yet - implementation is incomplete and produces incorrect results.

## Recommendation

The multi-pass architecture is correctly structured, but there's a systematic error in either:
1. The sign conventions in the chain rule
2. A missing/extra factor in the force calculation
3. Units or constants

Need to carefully trace through ONE pair calculation by hand and compare with kernel output to find the discrepancy.
