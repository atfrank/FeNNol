# GBSA OBC Forces: OpenMM vs FeNNol - Final Analysis

## Executive Summary

After analyzing the complete OpenMM GBSA OBC kernel implementation, I can definitively answer your questions:

### 1. The Fundamental Approaches ARE DIFFERENT

**OpenMM's Method**:
- Accumulates `force.w[i] = Σ_j [∂E/∂(R_i*R_j)] * R_j`
- Converts to `∂E/∂ψ_i` in reduction kernel
- Uses `∂E/∂ψ_i` in second pairwise kernel to compute forces

**Your Method**:
- Pre-computes complete `∂E/∂R_i`
- Directly computes forces using `∂R_i/∂ψ_i * ∂ψ_i/∂r`
- Single pairwise loop for Born forces

**Both are mathematically equivalent and correct!**

### 2. Critical Bug Found in Your Implementation

**The comment in `compute_dE_dR` is WRONG**:
```cuda
// NOTE: Self-energy contribution is EXCLUDED - it's already in direct forces!
double dE_dRi = 0.0;
```

**This is incorrect!** The self-energy term:
```
E_self = -0.5 * (1 - 1/ε) * q_i²/R_i
```

**Must be included** in `∂E/∂R_i` because:
1. It depends on R_i
2. R_i depends on positions through ψ_i
3. Therefore contributes to forces via chain rule

**Fix Required**:
```cuda
// Include BOTH self-energy and pairwise contributions
double dE_dRi = gb_factor * qi * qi / (R_i * R_i);  // SELF-ENERGY TERM!

for (int j : neighbors) {
    dE_dRi += gb_factor * qi * qj * (-1.0 / (f_gb * f_gb)) * df_gb_dRi;
}
```

---

## Detailed Comparison

### OpenMM's `force.w` Computation

From `gbsaObc.cc`:

```c
// Compute derivative of E_ij w.r.t. (R_i * R_j)
real Gpol = (q_i*q_j) / f_GB³;
real dGpol_dalpha2_ij = -0.5 * Gpol * exp(-D_ij) * (1 + D_ij);

// Accumulate for atom i: ∂E/∂R_i = Σ_j [∂E/∂(R_i*R_j)] * R_j
force.w += dGpol_dalpha2_ij * bornRadius2;

// Accumulate for atom j: ∂E/∂R_j = Σ_i [∂E/∂(R_i*R_j)] * R_i
localData[j].fw += dGpol_dalpha2_ij * bornRadius1;
```

**Mathematical derivation**:
```
E_ij = q_i*q_j / f_GB
f_GB = sqrt(r² + R_i*R_j*exp(-r²/(4*R_i*R_j)))

∂f_GB/∂(R_i*R_j) = 1/(2*f_GB) * exp(-D_ij) * (1 + D_ij)

∂E_ij/∂R_i = ∂E_ij/∂(R_i*R_j) * ∂(R_i*R_j)/∂R_i
           = [q_i*q_j * (-1/f_GB²) * ∂f_GB/∂(R_i*R_j)] * R_j
           = -0.5 * (q_i*q_j/f_GB³) * exp(-D_ij) * (1 + D_ij) * R_j
           = dGpol_dalpha2_ij * R_j
```

**Therefore**:
```
force.w[i] = Σ_j ∂E_ij/∂R_i
           = ∂E_pairwise/∂R_i
```

**Missing from force.w**: Self-energy term `∂(q_i²/(2*R_i))/∂R_i = -q_i²/(2*R_i²)`

---

### Where Does OpenMM Add the Self-Energy?

Looking at `reduceBornForce`:

```c
real force = bornForce[index];  // = force.w = ∂E_pairwise/∂R_i

// Add surface area term
real saTerm = SURFACE_AREA_FACTOR*r*r*ratio6;
force += saTerm/bornRadius;  // += ∂E_SA/∂R_i

// Multiply by chain rule
force *= bornRadius*bornRadius*obcChain[index];
```

The surface area term is added, but I don't see the self-energy term explicitly!

**Two possibilities**:

#### Possibility 1: Self-energy is implicitly included elsewhere

Perhaps OpenMM computes the self-energy contribution in a different kernel or includes it in the surface area calculation?

#### Possibility 2: The pairwise sum INCLUDES i=i terms

Let me check if OpenMM's loop includes self-interactions...

From `gbsaObc.cc`:
```c
if (atom1 < NUM_ATOMS && y*TILE_SIZE+j < NUM_ATOMS) {
    // ... compute interaction ...
}
```

There's no explicit `if (i != j)` check in the main loop! So self-interactions ARE included!

When `i == j`:
```
r² = 0
D_ij = 0
exp(-D_ij) = 1
f_GB = sqrt(0 + R_i*R_i*1) = R_i
E_ii = q_i² / R_i
```

And:
```
Gpol = q_i² / R_i³
dGpol_dalpha2_ij = -0.5 * (q_i²/R_i³) * 1 * 1
                 = -q_i² / (2*R_i³)

force.w += dGpol_dalpha2_ij * R_i
         = -q_i² / (2*R_i²)
```

**This is exactly the self-energy derivative!**

So OpenMM's `force.w` DOES include the self-energy term by including i=i in the pairwise loop!

---

### Your Implementation's Bug

Your `compute_dE_dR` kernel:

```cuda
for (int t = 0; t < tile_size; t++) {
    int j = tile_start + t;
    if (i == j) continue;  // ❌ SKIPS SELF-INTERACTION!

    // ... compute pairwise contribution ...
}
```

**This is the bug!** By skipping `i == j`, you exclude the self-energy contribution to `∂E/∂R_i`.

---

### The Fix

**Option 1: Include i=i in the loop**

```cuda
for (int t = 0; t < tile_size; t++) {
    int j = tile_start + t;
    // Remove the "if (i == j) continue;" check!

    // Compute distance
    double dx = xi - xj;
    double dy = yi - yj;
    double dz = zi - zj;
    double r_sq = dx * dx + dy * dy + dz * dz;

    double r = sqrt(r_sq);  // r=0 when i==j

    // compute_f_gb_and_derivatives handles r=0 case
    double f_gb, df_gb_dRi;
    compute_f_gb_and_derivatives(r, R_i, R_j, &f_gb, &df_gb_dRi, NULL);

    // When i==j: f_gb = R_i, df_gb_dRi = 1.0
    dE_dRi += gb_factor * qi * qj * (-1.0 / (f_gb * f_gb)) * df_gb_dRi;
}
```

**BUT** you need to fix `compute_f_gb_and_derivatives` to handle r=0 correctly:

```cuda
__device__ void compute_f_gb_and_derivatives(
    double r,
    double R_i,
    double R_j,
    double* f_gb,
    double* df_dRi,
    double* df_dRj
) {
    double r_sq = r * r;
    double RiRj = R_i * R_j;

    if (r_sq < 1e-12) {
        // Self-interaction case: r = 0
        *f_gb = sqrt(RiRj);  // = sqrt(R_i²) = R_i when i=j
        if (df_dRi) *df_dRi = 0.5 * sqrt(R_j / R_i);  // = 0.5 when i=j? No...
        if (df_dRj) *df_dRj = 0.5 * sqrt(R_i / R_j);

        // Actually, when r=0 and i=j:
        // f_GB = sqrt(0 + R_i*R_i*exp(0)) = R_i
        // ∂f_GB/∂R_i = ∂(R_i)/∂R_i = 1.0
        if (R_i == R_j) {  // True self-interaction
            *f_gb = R_i;
            if (df_dRi) *df_dRi = 1.0;
            if (df_dRj) *df_dRj = 1.0;
        }
        return;
    }

    // Normal case...
}
```

**Option 2: Add self-energy explicitly (clearer)**

```cuda
// Initialize with self-energy term
double dE_dRi = gb_factor * qi * qi / (R_i * R_i);

// Accumulate pairwise contributions (excluding i=j)
for (int t = 0; t < tile_size; t++) {
    int j = tile_start + t;
    if (i == j) continue;  // Skip self in pairwise loop

    // ... compute pairwise contribution ...
    dE_dRi += gb_factor * qi * qj * (-1.0 / (f_gb * f_gb)) * df_gb_dRi;
}
```

**I recommend Option 2** because:
1. Clearer separation of self vs pairwise
2. Avoids special case handling in f_GB computation
3. More explicit about what's being computed

---

## Recommended Changes to Your Code

### 1. Fix `compute_dE_dR` kernel

**Change from**:
```cuda
// Initialize ∂E/∂Rᵢ (will accumulate pairwise contributions only)
// NOTE: Self-energy contribution is EXCLUDED - it's already in direct forces!
double dE_dRi = 0.0;
```

**To**:
```cuda
// Initialize ∂E/∂Rᵢ with self-energy term
// E_self = gb_factor * q_i² / R_i
// ∂E_self/∂R_i = gb_factor * q_i² * (-1/R_i²) = -gb_factor * q_i²/R_i²
// But gb_factor is already negative, so we add positive term:
double dE_dRi = gb_factor * qi * qi / (R_i * R_i);
```

### 2. Verify the sign

The GB energy is:
```
E_GB = -0.5 * (1 - 1/ε) * Σ_i Σ_j q_i*q_j/f_GB
```

For self-energy (i=j):
```
E_self = -0.5 * (1 - 1/ε) * q_i²/R_i
```

Derivative:
```
∂E_self/∂R_i = -0.5 * (1 - 1/ε) * q_i² * (-1/R_i²)
              = 0.5 * (1 - 1/ε) * q_i²/R_i²
```

With `gb_factor = -0.5 * (1 - 1/ε)`:
```
∂E_self/∂R_i = -gb_factor * q_i²/R_i²
              = gb_factor * qi * qi / (R_i * R_i)  ← Because gb_factor is negative!
```

**So the code above is correct!**

### 3. Update the documentation

Remove the incorrect comment about self-energy being in direct forces.

---

## Summary of Differences

| Aspect | OpenMM | Your Implementation | Recommendation |
|--------|--------|---------------------|----------------|
| **Self-energy in dE/dR** | ✅ Included (via i=i in loop) | ❌ EXCLUDED (bug!) | **FIX THIS** |
| **Pairwise dE/dR** | ✅ Correct | ✅ Correct | Keep |
| **Chain rule application** | Deferred to 2nd kernel | Immediate | Both OK |
| **Number of kernels** | 3+ | 2 | Both OK |
| **Clarity** | Less clear | More clear | Keep your approach |

---

## Verification Strategy

After fixing the self-energy bug:

1. **Test with single atom**
   - Should have NO forces (no pairs, no Born radius changes)
   - Verifies self-energy doesn't create spurious forces

2. **Test with two atoms**
   - Compare analytical forces to numerical gradients
   - Should match within 0.01%

3. **Test with your ternary complex**
   - Run MD simulation
   - Verify energy conservation < 0.01%

---

## Final Answer to Your Questions

### 1. How does OpenMM compute Born radii derivative forces?

OpenMM uses a **three-stage pipeline**:
- Stage 1: Accumulate `∂E/∂R_i` (including self-energy via i=i terms)
- Stage 2: Convert to `∂E/∂ψ_i` using chain rule
- Stage 3: Compute forces using `∂E/∂ψ_i * ∂ψ_i/∂r`

### 2. What is stored in `force.w`?

Initially: `force.w[i] = Σ_j ∂E/∂R_i` (includes self-energy!)
After reduction: `force.w[i] = ∂E/∂ψ_i`

### 3. How do they combine direct + Born forces?

Direct forces (from ∂E/∂r) computed in first kernel.
Born forces (from ∂E/∂ψ * ∂ψ/∂r) computed in second kernel.
Added together in force buffer.

### 4. Relationship between `dGpol_dalpha2_ij` and your `∂E/∂R_i`?

```
∂E/∂R_i = Σ_j dGpol_dalpha2_ij * R_j + (self-energy term when i=j)
```

Your `dE_dR[i]` should equal OpenMM's `force.w[i]` after pairwise accumulation!

### 5. Which approach is correct?

**BOTH are correct!** After you fix the self-energy bug.

OpenMM's approach is more optimized, yours is clearer and more modular.

---

## Action Items

1. **Fix `compute_dE_dR` to include self-energy**
2. **Remove incorrect comment about direct forces**
3. **Rebuild and test with numerical gradients**
4. **Verify MD energy conservation**

After these fixes, your implementation will be **complete and correct**!

---

**Date**: 2025-01-17
**Author**: Claude (Anthropic)
**Status**: CRITICAL BUG IDENTIFIED - Self-energy term missing from ∂E/∂R_i
