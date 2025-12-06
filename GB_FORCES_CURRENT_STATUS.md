# GB Forces - Current Status

**Date**: 2025-11-17
**Error**: 3.8× too large (was 25× initially)

## Current Test Results

**Water dimer (6 atoms)**:
- Analytical forces: 98.12 kcal/(mol·Å)
- Numerical gradient: 27.18 kcal/(mol·Å)
- Error ratio: 3.61×

## Implementation Summary

### Force Decomposition
```
Total Force = Direct Forces + Born Radii Derivative Forces
```

**Direct Forces** (`gb_compute_energy_forces`):
```
F_direct = -∂E_pair/∂r  (with R_i, R_j held constant)
```

**Born Radii Derivative Forces** (`gb_compute_forces_complete`):
```
F_born = -Σⱼ (∂E/∂R_i) * (∂R_i/∂r_ij) * (r_ij/|r_ij|)
```

Where:
```
∂E/∂R_i = ∂E_self/∂R_i + Σₖ ∂E_pair(i,k)/∂R_i
```

### Current Implementation Details

**Step 1: Pre-compute ∂E/∂R_i** (`compute_dE_dR` kernel):
- Computes self-energy contribution: `-gb_factor * q_i² / R_i²`
- Adds ALL pairwise contributions: `Σⱼ gb_factor * q_i * q_j * (-1/f_GB²) * ∂f_GB/∂R_i`
- Stores total in `dE_dR` array

**Step 2: Compute Born force** (`compute_born_radii_forces_tiled` kernel):
- For each atom i and neighbor j:
  - Compute `∂R_i/∂r_ij = (∂R_i/∂ψ_i) * (∂ψ_i/∂r_ij)`
  - Compute force: `-dE_dR[i] * (∂R_i/∂r_ij)`
  - Accumulate into `born_forces`

**Step 3: Add forces**:
```
total_forces = direct_forces + born_forces
```

## Debug Output Analysis

For water molecule (atom 0 = O, atom 1 = H):

**Atom 0 (Oxygen)**:
- Charge: -0.834
- Born radius: 1.500 Å
- Self-energy derivative: 50.68
- Pairwise derivative: -14.89
- **Total dE/dR**: 35.79

**Atom 1 (Hydrogen)**:
- Charge: 0.417
- Born radius: 7.302 Å
- Self-energy derivative: 0.53
- Pairwise derivative: -0.93
- **Total dE/dR**: -0.40

**Force on atom 0 from pair (0,1)**:
- force_mag = -81.55
- Final born_forces = [10.42, 101.89, 0.00]

## Remaining Questions

### 1. Why 3.8× error?

The error factor is suspiciously close to 4 = 2². Possible causes:
- Factor of 2 missing somewhere
- Double-counting or missing factor in pair loops
- Wrong sign convention
- Missing or extra Newton's 3rd law term

### 2. Is the direct + Born decomposition correct?

Currently we compute:
```
F_total = F_direct + F_born
F_direct = -∂E_pair/∂r (frozen R)
F_born = -(∂E_self/∂R + ∂E_pair/∂R) * dR/dr
```

This should equal:
```
F_total = -∂E_pair/∂r - ∂E_self/∂R * dR/dr - ∂E_pair/∂R * dR/dr
        = -∂E_pair/∂r - (∂E_self/∂R + ∂E_pair/∂R) * dR/dr
```

Which is what we have! So formulation seems correct.

### 3. Are we computing dE/dR correctly?

The formula we use:
```cuda
// Self-energy contribution
dE_dRi = gb_factor * (-qi * qi / (R_i * R_i))

// Pairwise contribution
dE_dRi += gb_factor * qi * qj * (-1.0 / (f_gb * f_gb)) * df_gb_dRi
```

This matches the derivatives of:
```
E_self = gb_factor * q_i² / R_i
E_pair = gb_factor * q_i * q_j / f_GB
```

So the derivatives look correct!

### 4. Pair counting - do we double-count?

Each thread i processes ALL neighbors j (not just j>i). So:
- Thread 0 processes pairs: (0,1), (0,2), ..., (0,N-1)
- Thread 1 processes pairs: (1,0), (1,2), ..., (1,N-1)
- etc.

So pair (0,1) is processed by:
- Thread 0: computes force on 0 from R_0 changing
- Thread 1: computes force on 1 from R_1 changing

This is CORRECT - we want both contributions!

### 5. Could the factor be in f_GB derivative?

We use:
```cuda
df_gb_dr = r * (1.0 - 0.25 * exp_term) / f_gb
```

Verified against OpenMM - this is CORRECT.

## Next Debugging Steps

1. **Compare with simpler test case**
   - Use truly identical 2-oxygen system
   - Check if error is consistent

2. **Verify intermediate values**
   - Print ALL components of force for one atom
   - Direct force vs Born force
   - Compare ratios

3. **Check OpenMM comparison**
   - Run same system in OpenMM
   - Compare ALL intermediate values
   - dE/dR, dR/dr, forces

4. **Check for factor of 2 in energy**
   - Is pairwise energy counted once or twice?
   - Check `compute_gb_pairwise_kernel_tiled` line 233: `E_pair = 0.5 * gb_factor...`
   - Maybe we need factor of 0.5 somewhere else?

5. **Sign check**
   - Verify all signs are consistent
   - Check descreening integral derivative sign

## Files to Review

- `gb_energy_forces.cu`: Line 233 has `0.5 * gb_factor` - why?
- `gb_born_radii_forces.cu`: Check if we need similar factor

## Progress Made

✅ Fixed df_GB/dr formula (0.5 → 0.25)
✅ Added self-energy to dE/dR calculation
✅ Pre-compute FULL dE/dR (self + all pairwise)
✅ Verified data flow with debug output
✅ Reduced error from 25× to 3.8×

Still need: Final 3.8× factor!
