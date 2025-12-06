# GB Multi-Pass Force Implementation - Final Debug Report

## Date: 2025-01-18

## Executive Summary

Successfully implemented OpenMM's multi-pass architecture for GB Born radius derivative forces and identified/fixed multiple bugs. **Current status**: Sign is correct, but magnitude is still ~30× too large.

---

## Bugs Found and Fixed

### ✅ Bug #1: Double-Counting in Tiled Algorithm (FIXED)

**Problem**: Each pair (i,j) was being processed by both thread i and thread j. Each thread was adding contributions from both ∂E/∂ψ_i AND ∂E/∂ψ_j, resulting in a 2× error.

**Location**: `apply_born_forces_tiled` kernel (line 634-642)

**Fix**: Changed from:
```cuda
double force_mag = -dE_dpsi_i * dpsi_i_dr - dE_dpsi_j * dpsi_j_dr;  // WRONG: double-counts
```

To:
```cuda
double force_mag = -dE_dpsi_i * dpsi_i_dr;  // CORRECT: only i's contribution
```

**Validation**: Kernel output now matches manual calculation exactly (132.29 vs 132.29).

### ✅ Bug #2: Wrong Sign in reduce_born_force (FIXED)

**Problem**: Formula had negative sign when converting ∂E/∂R → ∂E/∂ψ, causing force direction to be backwards.

**Location**: `reduce_born_force` kernel (line 302)

**Fix**: Changed from:
```cuda
double dE_dpsi_i = -dE_dR_i * R_i * R_i * obcChain;  // WRONG sign
```

To:
```cuda
double dE_dpsi_i = dE_dR_i * R_i * R_i * obcChain;  // CORRECT (matches OpenMM)
```

**Validation**: Force direction is now correct (negative F_y pushes O away from H when they're too close).

---

## Remaining Issue

###❌ Bug #3: Magnitude 30× Too Large (NOT FIXED)

**Current Status**:
- Numerical gradient: F_y = -4.616 kcal/(mol·Å) ✓ Correct
- Multi-pass forces:  F_y = -141.45 kcal/(mol·Å) ❌ Wrong magnitude
- **Error factor: 30.6×**

**Evidence**:
1. ✅ Energy calculation is CORRECT (validated manually)
2. ✅ Descreening integral formula is CORRECT (validated)
3. ✅ ∂E/∂R calculation is CORRECT (validated)
4. ✅ Force SIGN is CORRECT (after fix #2)
5. ❌ Force MAGNITUDE is WRONG

**Hypothesis**: The bug is in the **descreening integral DERIVATIVE** formula.

### Current Descreening Derivative Formula

In `descreening_integral_derivative` (gb_born_radii_forces.cu:14-41):

```cuda
// For partial overlap region:
double deriv = -rho_i / (r * r * r);  // = -ρ_i/r³
```

This comes from differentiating:
```
I = 0.5 * ρ_i * (1/r² - 1/upper²)
∂I/∂r = 0.5 * ρ_i * (-2/r³) = -ρ_i/r³  ✓ Math is correct
```

### Why This Might Be Wrong

**OpenMM uses HCT (Hawkins-Cramer-Truhlar) formula** which is much more complex:

```cpp
// From OpenMM ReferenceObc.cpp:
t3 = 0.125*(1 + s_j²*r⁻²)*(l_ij² - u_ij²) + 0.25*ln(u_ij/l_ij)*r⁻²
```

Where:
- `l_ij = 1/max(R_offset_i, |r - s_j|)`
- `u_ij = 1/(r + s_j)`

Our simplified formula `-ρ_i/r³` is NOT equivalent to this!

**Possible factor mismatch**: The HCT formula has factors like 0.125, 0.25, and logarithmic terms that our formula doesn't have. This could easily account for a 30× discrepancy.

---

## What Was Validated

### ✓ Energies Are Correct

```
Manual calculation: E = -60.0651103 kcal/mol
Kernel calculation: E = -60.0651103 kcal/mol
Match: PERFECT
```

This confirms:
- Descreening integral formula is correct
- Born radius calculation is correct
- Energy formula is correct

### ✓ Psi Sum Is Correct

```
Manual psi[0] (simplified formula): 1.4309981177
Kernel psi[0]:                       1.4309981177
Match: PERFECT
```

### ✓ Force Sign Is Correct (After Fix #2)

```
Energy increases when O moves toward H → Force should push O away
Numerical gradient: F_y = -4.616 (negative, correct direction)
Multi-pass forces:  F_y = -141.45 (negative, correct direction) ✓
```

---

## Detailed Findings

### Chain Rule Validation

The force calculation uses:
```
F = -(∂E/∂ψ) × (∂ψ/∂r)
```

Which should equal:
```
F = -(∂E/∂R) × (∂R/∂r)
```

Where:
```
∂E/∂ψ = (∂E/∂R) × (∂R/∂ψ)
∂R/∂r = (∂R/∂ψ) × (∂ψ/∂r)
```

**Validation**:
```
F (via R) = -19.664184 × 0.684747 = -13.464986
F (via ψ) = +7.875429 × (-1.709746) = -13.464986
Match: PERFECT ✓
```

This confirms the multi-pass approach is mathematically sound.

### OBC Chain Derivative Validation

The `obcChain = ∂(1/R)/∂ψ` calculation appears correct based on the OBC formula:

```
1/R = 1/ρ - tanh(ψ - b*ψ² + c*ψ³) / ρ

∂(1/R)/∂ψ = -sech²(...) * d(...)/dψ / ρ
```

For the test case:
```
obcChain = 0.088032
∂R/∂ψ = -R² × obcChain = -0.400496
```

With b=0.8, the tanh argument decreases as ψ increases, so R decreases with ψ for this system. This is physically reasonable.

---

## Files Modified

1. `src/fennol/cuda/src/gb_born_radii_forces.cu`
   - Line 302: Fixed sign in reduce_born_force
   - Line 642: Fixed double-counting in apply_born_forces_tiled
   - Line 52-76: Updated born_radius_derivative_wrt_psi to return ∂(1/R)/∂ψ

2. `src/fennol/cuda/src/bindings.cpp`
   - Added Python bindings for compute_dE_dR, reduce_born_force, apply_born_forces

3. `src/fennol/cuda/include/implicit_solvent.cuh`
   - Added declarations for new multi-pass functions

---

## Next Steps to Fix Remaining Bug

### Option A: Implement HCT Descreening Derivative (Recommended)

Replace the simplified `-ρ_i/r³` formula with OpenMM's HCT formula:

**Files to examine**:
1. `openmm/platforms/reference/src/ReferenceObc.cpp` - computeBornForces function
2. `openmm/platforms/cuda/src/kernels/gbsaObc2.cc` - CUDA implementation

**Implementation**:
1. Extract exact HCT derivative formula from OpenMM
2. Implement in `descreening_integral_derivative` function
3. Ensure all factors (0.125, 0.25, ln terms) are included
4. Validate against OpenMM reference implementation

### Option B: Check for Missing Scaling Factors

Investigate if there's a systematic factor missing:
- Factor of 32 = 2^5?
- Factor related to units (Å vs nm)?
- Factor related to energy units (kcal/mol vs kJ/mol)?

Current ratio is ~30.6×, which doesn't match any obvious factor.

### Option C: Compare with OpenMM Step-by-Step

Run OpenMM's GB/OBC implementation on same test case and compare:
1. Born radii
2. ∂E/∂R values
3. ∂E/∂ψ values
4. ∂ψ/∂r values
5. Final forces

Identify exactly where the discrepancy appears.

---

## Test Cases

### Water Molecule Test

**System**: Single H₂O molecule
- O at (0, 0, 0)
- H1 at (0.757, 0.586, 0)
- H2 at (-0.757, 0.586, 0)

**Parameters**:
- Radii: [1.5, 1.2, 1.2] Å
- Charges: [-0.834, 0.417, 0.417] e
- OBC b: [0.8, 0.85, 0.85]
- OBC c: [0.0, 0.0, 0.0]
- Dielectric: 80.0
- Cutoff: 12.0 Å

**Results**:
```
Numerical gradient: F_O_y = -4.616 kcal/(mol·Å)
Multi-pass (current): F_O_y = -141.45 kcal/(mol·Å)
Error: 30.6× too large
```

---

## Conclusions

1. **Multi-pass architecture is correctly implemented** - chain rule works, no mathematical errors
2. **Two bugs were found and fixed** - double-counting and sign error
3. **One bug remains** - 30× magnitude error, likely in descreening derivative formula
4. **Energy calculations are perfect** - the simplified descreening integral is fine for energies
5. **Forces require HCT formula** - the derivative needs the more complex OpenMM formula

**Recommendation**: Implement the HCT descreening derivative formula from OpenMM to fix the remaining magnitude error.

---

## Time Spent

- Phase 1 (Manual calculation validation): 45 min
- Bug fixes (double-counting, sign): 30 min
- Deep investigation (descreening formula): 45 min
- **Total**: ~2 hours

## Estimated Time to Complete

- Implement HCT derivative: 1-2 hours
- Testing and validation: 1 hour
- **Total remaining**: 2-3 hours
