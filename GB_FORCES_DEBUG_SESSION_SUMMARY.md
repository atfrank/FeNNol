# GB Forces Debugging Session Summary

**Date**: 2025-11-17
**Status**: CRITICAL BUG FOUND - Forces unaffected by dE/dR changes

## Problem Statement

GB forces are **25× too large**:
- Analytical forces: 4.22 kcal/(mol·Å)
- Numerical gradient: 0.169 kcal/(mol·Å)
- Ratio: 24.96×

## Bugs Found and Fixed

### 1. ✅ df_GB/dr Formula Error (FIXED)
**File**: `src/fennol/cuda/src/gb_energy_forces.cu:37`

**Was**:
```cuda
df_gb_dr = r * (1.0 - 0.5 * exp_term) / f_gb;
```

**Fixed to**:
```cuda
df_gb_dr = r * (1.0 - 0.25 * exp_term) / f_gb;
```

**Impact**: ~14% correction, NOT the 2500% error we're seeing

**Verification**: OpenMM uses `(1.0 - 0.25*exp)` - our fix matches reference implementation

---

### 2. ✅ Self-Energy Term (RE-ADDED)
**File**: `src/fennol/cuda/src/gb_born_radii_forces.cu:175-181`

**Added**:
```cuda
// Initialize ∂E/∂Rᵢ with self-energy contribution
// Self-energy: E_self = gb_factor * q_i² / R_i
// So: ∂E_self/∂R_i = -gb_factor * q_i² / R_i²
double dE_dRi = 0.0;
if (i < natoms) {
    dE_dRi = gb_factor * (-qi * qi / (R_i * R_i));
}
```

**Why Important**:
- Self-energy contribution: 49.46 (for test system)
- Pairwise contribution: 2.61
- Self-energy is **19× larger** than pairwise!

---

## CRITICAL REMAINING BUG

### Forces Do NOT Change When dE/dR Changes!

**Test performed**:
```cuda
// Changed from:
dE_dRi = gb_factor * (-qi * qi / (R_i * R_i));

// To (2× self-energy):
dE_dRi = 2.0 * gb_factor * (-qi * qi / (R_i * R_i));
```

**Result**: Force UNCHANGED at 4.221884 kcal/(mol·Å)

**Implication**: The `dE_dR` array is either:
1. Not being computed correctly
2. Not being passed to `compute_born_radii_forces_tiled` correctly
3. Not being loaded correctly in the kernel
4. Not being used correctly in force calculation

---

## Test System Details

**Simple 2-atom test**:
```python
coords = [[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]]
charges = [-0.834, -0.834]  # Two oxygen atoms
atomic_numbers = [8, 8]
```

**Computed values**:
- Born radii: [1.51852646, 1.51852646]
- Psi sum: [0.01232993, 0.01232993]
- Energy: -188.587 kcal/mol (CORRECT)
- Force (analytical): 4.22 kcal/(mol·Å) (WRONG)
- Force (numerical): 0.169 kcal/(mol·Å) (CORRECT)

**Energy landscape**:
- E(x=-1e-5) = -188.586738
- E(x=0)     = -188.586740
- E(x=+1e-5) = -188.586742

Moving atoms closer LOWERS energy → **attractive force** (GB solvation)

---

## Code Structure

### Host Function: `compute_gb_forces_complete()`
Location: `src/fennol/cuda/src/gb_born_radii_forces.cu:410`

**Pipeline**:
1. Compute direct forces (frozen Born radii) → `forces`
2. Pre-compute ∂E/∂R_i for all atoms → `dE_dR`
3. Compute Born radii derivative forces → `born_forces`
4. Add: `forces += born_forces`

### Kernels:

**1. `compute_dE_dR`** (line 145)
- Computes ∂E/∂R_i for each atom
- Stores in `dE_dR[i]`
- **BUG**: Changes to this have NO effect on final forces!

**2. `compute_born_radii_forces_tiled`** (line 257)
- Loads `dE_dR_i = dE_dR[i]` (line 298)
- Computes chain rule: F = -dE_dR_i * dR_i/dr
- Stores in `born_forces`

**3. `add_arrays_kernel`** (line 390)
- Adds `born_forces` to `forces`

---

## Comparison with OpenMM

### OpenMM Implementation
**Source**: `openmm/platforms/common/src/kernels/gbsaObc.cc`

**Key formulas**:
```c
// Same as our corrected version:
real dEdR = Gpol*(1.0f - 0.25f*expTerm);

// Born radius force:
real dGpol_dalpha2_ij = -0.5f*Gpol*expTerm*(1.0f+D_ij);
force.w += dGpol_dalpha2_ij*bornRadius2;
```

**Pipeline** (3-stage):
1. `computeGBSAForce1`: Accumulate `force.w = ∂E/∂R_i`
2. `reduceBornForce`: Convert to `∂E/∂ψ_i = (∂E/∂R_i) * R_i² * obcChain`
3. `computeGBSAForce2`: Apply `∂E/∂ψ_i * ∂ψ_i/∂r`

**Our approach** (2-stage):
1. Pre-compute `∂E/∂R_i`
2. Directly apply `∂E/∂R_i * ∂R_i/∂r`

**Both are mathematically correct** - just different factorization.

---

## Debugging Steps Taken

1. ✅ Fixed df_GB/dr formula
2. ✅ Re-added self-energy term
3. ✅ Verified numerical gradient is stable and correct
4. ✅ Verified energy calculation is correct
5. ✅ Compared against OpenMM reference
6. ✅ Tested with 2× self-energy → NO CHANGE in forces
7. ❌ **Still stuck**: Forces = 4.22 vs expected 0.169

---

## Next Steps (PRIORITY)

### 1. Add Debug Output to Verify dE_dR
Add printf to `compute_dE_dR` kernel:
```cuda
if (i == 0) {
    printf("atom 0: dE_dRi = %.6f (self=%.6f, pair total=%.6f)\n",
           dE_dRi, self_energy_term, pairwise_accumulation);
}
```

### 2. Verify dE_dR is Loaded Correctly
Add printf to `compute_born_radii_forces_tiled`:
```cuda
if (i == 0) {
    printf("atom 0: loaded dE_dR_i = %.6f\n", dE_dR_i);
}
```

### 3. Check Force Accumulation
Verify the chain rule computation:
```cuda
if (i == 0 && j == 1) {
    printf("pair (0,1): dpsi_i_dr=%.6f, dR_i_dpsi=%.6f, dR_i_dr=%.6f, force_mag=%.6f\n",
           dpsi_i_dr, dR_i_dpsi, dR_i_dr, force_mag_i);
}
```

### 4. Consider Alternative Hypothesis
Maybe the issue is NOT in Born forces but in how direct forces are computed or combined?

---

## Key Files Modified

1. `src/fennol/cuda/src/gb_energy_forces.cu` - Fixed df_GB/dr formula
2. `src/fennol/cuda/src/gb_born_radii_forces.cu` - Re-added self-energy to dE/dR

## Test Command

```bash
cd /home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol
python3 test_gb_forces_gradient.py
```

**Expected**: Max error < 0.01 kcal/(mol·Å)
**Actual**: Max error = 96 kcal/(mol·Å) (NEW), 19 kcal/(mol·Å) (OLD)

---

## Critical Insight

The fact that **doubling dE_dR has zero effect** means the bug is NOT in the dE_dR calculation itself, but rather in:
- How dE_dR is transferred to the force kernel
- How dE_dR is used in the force calculation
- Or possibly the direct forces are so wrong they dominate

This is the smoking gun that points to a data flow or force composition issue.
