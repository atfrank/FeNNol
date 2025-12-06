# GB Forces Debugging - Progress Update

**Date**: 2025-11-17
**Status**: SIGNIFICANT PROGRESS - Error reduced from 25× to 3.8×

## Summary

We've successfully debugged and fixed a major issue in the GB forces implementation. The analytical forces were initially 25× too large (242 kcal/(mol·Å) vs 27 numerical), and are now only 3.8× too large (98 kcal/(mol·Å) vs 27 numerical).

## Root Cause Found

**The original bug**: The pairwise ∂E_pair/∂R_i contributions were being computed on-the-fly in the force kernel, which meant they were being added ONCE per neighbor pair, not accumulated over ALL pairs before applying the Born radius derivative chain rule.

**The fix**: Pre-compute the FULL ∂E/∂R_i (self-energy + ALL pairwise terms) in the `compute_dE_dR` kernel, then use that pre-computed value in the force kernel.

## Progress Timeline

### Initial State
- Analytical forces: 242.40 kcal/(mol·Å)
- Numerical gradient: 27.18 kcal/(mol·Å)
- Error: **25× too large**

### After removing pairwise from pre-computation (WRONG)
- Analytical forces: 122.90 kcal/(mol·Å)
- Error: **4.5× too large**
- This was WRONG because we were only using self-energy

### After re-adding pairwise pre-computation (CORRECT approach)
- Analytical forces: 98.12 kcal/(mol·Å)
- Error: **3.8× too large**
- This is better! We're pre-computing the full dE/dR correctly

## Current Implementation

### `compute_dE_dR` kernel
Computes for each atom i:
```
∂E/∂R_i = ∂E_self/∂R_i + Σⱼ ∂E_pair(i,j)/∂R_i
```

Where:
- Self-energy: `∂E_self/∂R_i = -gb_factor * q_i² / R_i²`
- Pairwise: `∂E_pair(i,j)/∂R_i = gb_factor * q_i * q_j * (-1/f_GB²) * ∂f_GB/∂R_i`

**Current values for 2-oxygen test**:
- Self-energy derivative: 50.68
- Total (self + pairwise): 35.79
- Pairwise contribution: 35.79 - 50.68 = -14.89

### `compute_born_radii_forces_tiled` kernel
Applies chain rule:
```
F_i = -Σⱼ (∂E/∂R_i) * (∂R_i/∂r_ij) * (r_ij/|r_ij|)
```

Where:
- `∂E/∂R_i` is pre-computed (loaded from dE_dR array)
- `∂R_i/∂r_ij = (∂R_i/∂ψ_i) * (∂ψ_i/∂r_ij)`

## Remaining Issues

### Current Error: 3.8× too large

**Possible causes**:

1. **Missing ∂E/∂R_j contribution?**
   - Currently we only compute force from ∂R_i/∂r
   - Do we also need ∂R_j/∂r for the same pair?
   - Each thread handles one atom, so thread i computes R_i derivatives, thread j computes R_j derivatives
   - This SHOULD work... unless there's an asymmetry

2. **Double-counting pairs?**
   - We iterate over ALL j (not just i<j)
   - Each pair (i,j) is processed by both thread i and thread j
   - Is this correct or are we double-counting?

3. **Sign error or factor of 2?**
   - The remaining error is ~4×, which is suspiciously close to 2²
   - Could there be a missing factor of 0.5 somewhere?

4. **Units or constants?**
   - gb_factor = -163.956... (should be -0.5 * (1-1/80) * 332.0636)
   - Double-check this is correct

## Test System

**Simple 2-oxygen**:
- coords = [[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]]
- charges = [-0.834, -0.834]
- Born radii: [1.52, 1.52]
- Energy: -188.587 kcal/mol (CORRECT)

**Water dimer (6 atoms)**:
- Numerical gradient max: 27.18 kcal/(mol·Å)
- Analytical force max: 98.12 kcal/(mol·Å)
- Error: 3.8×

## Next Steps

1. **Check if we need R_j derivative contribution**
   - Review OpenMM implementation for comparison
   - Check if forces should include both ∂R_i/∂r AND ∂R_j/∂r

2. **Verify no double-counting**
   - Trace through exactly which pairs are processed by which threads
   - Check if we need i<j restriction

3. **Check for missing factors**
   - Factor of 0.5 for pair counting?
   - Sign conventions?

4. **Compare intermediate values with OpenMM**
   - Compute same test system in OpenMM
   - Compare dE/dR values, dR/dr values, etc.

## Files Modified

- `src/fennol/cuda/src/gb_born_radii_forces.cu`
  - `compute_dE_dR` kernel: Pre-computes full ∂E/∂R_i
  - `compute_born_radii_forces_tiled` kernel: Applies chain rule
  - Added charges to shared memory (10 doubles per tile now)

- `src/fennol/cuda/include/implicit_solvent.cuh`
  - No changes needed

## Debug Output

GPU kernel printf shows:
```
dE_self/dRi = 50.68482644
TOTAL dE_dR[0] = 35.79332898 (self + pairwise)
LOADED dE_dR_i = 35.79332898
force_mag_i = -81.54869658
FINAL born_forces = [10.42, 101.89, 0.00]
```

## Success Criteria

- Max absolute error < 0.01 kcal/(mol·Å)
- Max relative error < 5%

Currently:
- Max error: 96.04 kcal/(mol·Å) ❌
- Max relative error: 4630% ❌

We've made excellent progress (from 25× to 3.8× error), but still have work to do!
