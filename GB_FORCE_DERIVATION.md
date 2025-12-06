# GB/OBC Force Derivation - Complete Analysis

## Problem Summary

Current status:
- ✅ 2 identical O atoms: Forces match perfectly (0.00 error)
- ❌ Single water molecule (O + 2H): 27× error, WRONG SIGN
- ❌ Water dimer (6 atoms): 50× error

## Energy Formula

Total GB electrostatic energy:
```
E_total = Σᵢ E_self(i) + Σᵢ<ⱼ E_pair(i,j)

E_self(i) = gb_factor × qᵢ² / Rᵢ
E_pair(i,j) = gb_factor × qᵢ × qⱼ / f_GB(rᵢⱼ, Rᵢ, Rⱼ)

gb_factor = -0.5 × (1 - 1/ε) × 332.0636
f_GB = √(r² + Rᵢ×Rⱼ×exp(-r²/(4×Rᵢ×Rⱼ)))
```

## Force on Atom k

```
F_k = -∇_k E_total
    = -∇_k E_self(k) - Σⱼ≠k ∇_k E_pair(k,j)
```

### Self-Energy Contribution

```
∇_k E_self(k) = (∂E_self/∂Rk) × ∇_k Rk
              = (∂E_self/∂Rk) × (∂Rk/∂ψk) × ∇_k ψk
```

Where:
```
∂E_self/∂Rk = gb_factor × qk² × (-1/Rk²)
∂Rk/∂ψk = Rk² × sech²(ψ - b×ψ² + c×ψ³) × (1 - 2b×ψ + 3c×ψ²) / ρk
∇_k ψk = Σⱼ≠k ∇_k Iⱼ→k(rⱼk)
```

For descreening integral:
```
∇_k Iⱼ→k = (∂I/∂r) × ∇_k r = (∂I/∂r) × (r_k - r_j)/r
         = (-ρk/r³) × (r_k - r_j)/r
```

### Pairwise Energy Contribution

For pair (k,j), the force on k has THREE parts:

#### 1. Direct force (frozen Born radii):
```
F_direct = -(∂E_pair/∂r) × ∇_k r
         = gb_factor × qk × qⱼ × (∂f_GB/∂r) / f_GB² × (r_k - r_j)/r
```

#### 2. Born radius derivative for Rk:
```
F_Rk = -(∂E_pair/∂Rk) × (∂Rk/∂ψk) × (∂ψk/∂r) × ∇_k r
     = -dE/dRk × dRk/dψk × (-ρk/r³) × (r_k - r_j)/r
```

#### 3. Born radius derivative for Rⱼ:
```
F_Rⱼ = -(∂E_pair/∂Rⱼ) × (∂Rⱼ/∂ψⱼ) × (∂ψⱼ/∂r) × ∇_k r
     = -dE/dRⱼ × dRⱼ/dψⱼ × (-ρⱼ/r³) × (r_k - r_j)/r
```

## Key Insight: TWO Born Radius Chains!

When atom k moves, it affects:
1. Its own Born radius Rk (via changing distances to all neighbors)
2. The Born radii of all neighbors Rⱼ (since k contributes to their descreening)

**This means for each pair (k,j), the Born force on k must include BOTH the Rk and Rⱼ contributions!**

## Current Implementation Issues

### What We're Computing:
```
F_born(k) = Σⱼ≠k [-dE/dRk × dRk/dψk × ∂ψk/∂rⱼk × (r_k - r_j)/r]
```

### What We Should Compute:
```
F_born(k) = Σⱼ≠k [
    -dE/dRk × dRk/dψk × ∂ψk/∂rⱼk × (r_k - r_j)/r     [Rk chain]
    -dE/dRⱼ × dRⱼ/dψⱼ × ∂ψⱼ/∂rⱼk × (r_k - r_j)/r     [Rⱼ chain]
]
```

## Why 2-Atom Test Worked

For two IDENTICAL atoms (same q, same ρ, same R):
- dE/dRᵢ = dE/dRⱼ (by symmetry)
- dRᵢ/dψᵢ = dRⱼ/dψⱼ
- ∂ψᵢ/∂r = -ρᵢ/r³ = -ρⱼ/r³ = ∂ψⱼ/∂r

So:
```
Force_i = -(dE/dRᵢ × dRᵢ/dψᵢ × ∂ψᵢ/∂r) × direction
```

And by Newton's 3rd law:
```
Force_j = -Force_i
```

The R_j contribution is implicitly handled by the symmetry! Each thread computes its own R term, and they happen to be equal and opposite.

## Why Water Molecule Failed

For O-H pair:
- Charges different: qₒ = -0.834, q_H = +0.417
- Radii different: ρₒ = 1.5, ρ_H = 1.2
- Born radii different: Rₒ ≈ 1.5, R_H ≈ 1.64
- ∂ψₒ/∂r = -ρₒ/r³ ≠ ∂ψ_H/∂r = -ρ_H/r³

The Rⱼ contribution is NOT equal to the Rᵢ contribution, so we MUST compute both explicitly!

## Solution

Modify `compute_born_radii_forces_tiled` to include BOTH contributions for each pair.

For thread processing atom i with neighbor j:
```cuda
// R_i contribution (current)
double dpsi_i_dr = descreening_integral_derivative(r, rho_i, rho_j);
double dR_i_dr = dR_i_dpsi * dpsi_i_dr;
double force_mag_i = -dE_dR_i * dR_i_dr;

// R_j contribution (MISSING!)
double dpsi_j_dr = descreening_integral_derivative(r, rho_j, rho_i);
double dR_j_dr = dR_j_dpsi * dpsi_j_dr;
double force_mag_j = -dE_dR_j * dR_j_dr;

// Total Born force magnitude
double force_mag_total = force_mag_i + force_mag_j;

// Force vector
fx_born_i += force_mag_total * dx / r;
```

## Expected Outcome

With both R_i and R_j contributions:
- Heteroatomic pairs should work correctly
- Homoatomic pairs should still work (by symmetry)
- Sign should be correct (both chains have same sign structure)
- Magnitude should match numerical gradient

## Remaining Mystery

Previous attempt at adding R_j made error slightly WORSE (98 → 104). Possible reasons:
1. Had wrong sign on R_j term
2. Had wrong formula for ∂ψⱼ/∂r
3. Need to investigate why 4× systematic error remains
4. May be additional bug in dE/dR calculation

After adding R_j correctly, if error persists, need to check:
- Energy formula (is 0.5 factor correct?)
- dE/dR pre-computation (are we summing correctly?)
- Sign conventions throughout

