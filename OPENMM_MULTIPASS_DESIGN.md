# OpenMM Multi-Pass GB Force Implementation Design

## Current Architecture (Broken)

**Single-pass approach** in `compute_born_radii_forces_tiled`:
1. Pre-compute ∂E/∂R_i for all atoms
2. In one kernel, compute: F = -(∂E/∂R_i) × (∂R_i/∂ψ_i) × (∂ψ_i/∂r) × direction
3. Also try to add R_j contribution (causing errors)

**Problems:**
- 2-atom identical system worked perfectly (before adding R_j)
- Heteroatomic pairs have ~25× error
- Sign errors when trying to add R_j contribution

## OpenMM Architecture (Reference)

### Pass 1: Compute Born Radii
```
computeBornSum → accumulate ψ_i
reduceBornSum → compute R_i from ψ_i, store obcChain_i = ∂R_i/∂ψ_i
```

### Pass 2: Compute Energy and ∂E/∂R
```
computeGBSAForce1 →
  - Compute E_GB
  - Compute direct forces (frozen R)
  - Accumulate bornForce_i = Σ_j (∂E/∂α²_ij) × R_j  [this is ∂E/∂R_i]
```

### Pass 3: Convert ∂E/∂R → ∂E/∂ψ
```
reduceBornForce →
  bornForce_i *= R_i² × obcChain_i
  [now bornForce_i contains ∂E/∂ψ_i]
```

### Pass 4: Apply Born Forces
```
(Second invocation of computeBornSum or similar) →
  For each pair (i,j):
    dF_i = -(∂E/∂ψ_i) × (∂ψ_i/∂r_ij) × direction
```

## New FeNNol Architecture

### Existing Functions (Keep)
1. `gb_compute_born_radii_with_psi` → Computes R_i and ψ_i ✓
2. `gb_compute_energy_forces` → Computes E_GB and direct forces ✓

### New Functions Needed
3. **`compute_dE_dR`** (already exists, keep as-is)
   - Computes ∂E/∂R_i = ∂E_self/∂R_i + Σ_j ∂E_pair(i,j)/∂R_i
   - Output: array of ∂E/∂R_i for all atoms

4. **`reduce_born_force`** (NEW - to implement)
   - Input: ∂E/∂R_i, R_i, ψ_i, OBC parameters
   - Computes: obcChain_i = ∂R_i/∂ψ_i
   - Output: ∂E/∂ψ_i = (∂E/∂R_i) × R_i² × obcChain_i

5. **`apply_born_forces`** (NEW - to implement)
   - Input: ∂E/∂ψ_i for all atoms, coordinates, radii
   - For each pair (i,j):
     - Compute ∂ψ_i/∂r_ij
     - Apply force: F_i += -(∂E/∂ψ_i) × (∂ψ_i/∂r_ij) × (r_i - r_j)/r
   - Output: Born radius derivative forces

### Python Interface
```python
def gb_compute_forces_complete(coords, charges, born_radii, radii, b_params, c_params, psi_sum, dielectric, cutoff):
    # Step 1: Compute direct forces (frozen R)
    energy, direct_forces = gb_compute_energy_forces(coords, charges, born_radii, dielectric, cutoff)

    # Step 2: Compute ∂E/∂R
    dE_dR = compute_dE_dR(coords, charges, born_radii, radii, dielectric, cutoff)

    # Step 3: Convert ∂E/∂R → ∂E/∂ψ
    dE_dpsi = reduce_born_force(dE_dR, born_radii, radii, b_params, c_params, psi_sum)

    # Step 4: Apply Born forces
    born_forces = apply_born_forces(dE_dpsi, coords, radii)

    # Step 5: Combine
    total_forces = direct_forces + born_forces

    return energy, total_forces
```

## Implementation Plan

### Step 1: Implement `reduce_born_force` kernel
```cuda
__global__ void reduce_born_force_kernel(
    int natoms,
    const double* dE_dR,           // Input: ∂E/∂R_i
    const double* born_radii,      // Input: R_i
    const double* intrinsic_radii, // Input: ρ_i
    const double* b_params,        // Input: b_i (OBC)
    const double* c_params,        // Input: c_i (OBC)
    const double* psi_sum,         // Input: ψ_i
    double* dE_dpsi                // Output: ∂E/∂ψ_i
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    // Compute obcChain = ∂R/∂ψ
    double R_i = born_radii[i];
    double rho_i = intrinsic_radii[i];
    double psi = psi_sum[i];
    double b = b_params[i];
    double c = c_params[i];

    double obcChain = born_radius_derivative_wrt_psi(R_i, rho_i, psi, b, c);

    // Convert: ∂E/∂ψ = (∂E/∂R) × R² × obcChain
    dE_dpsi[i] = dE_dR[i] * R_i * R_i * obcChain;
}
```

### Step 2: Implement `apply_born_forces` kernel
```cuda
__global__ void apply_born_forces_kernel(
    int natoms,
    const double* coords,
    const double* intrinsic_radii,
    const double* dE_dpsi,  // Input: ∂E/∂ψ_i for all atoms
    double cutoff,
    double* born_forces      // Output: forces
) {
    // Tiled approach like current force kernel
    // For each pair (i,j):
    //   dpsi_i_dr = descreening_integral_derivative(r, rho_i, rho_j)
    //   force_mag = -dE_dpsi[i] * dpsi_i_dr
    //   forces[i] += force_mag * (r_i - r_j) / r
}
```

### Step 3: Update Python bindings
Add two new functions to `bindings.cpp`:
- `reduce_born_force()`
- `apply_born_forces()`

### Step 4: Update `gb_compute_forces_complete` in Python
Modify to use the new multi-pass approach.

## Key Advantages

1. **Matches OpenMM exactly** - easier to validate
2. **Clearer separation of concerns** - each kernel does one thing
3. **No confusion about R_i vs R_j** - handled naturally by the multi-pass approach
4. **Easier to debug** - can inspect intermediate values (∂E/∂R, ∂E/∂ψ)

## Expected Outcome

After implementing this:
- 2-atom test should still work (direct forces unchanged)
- Heteroatomic pairs should work (proper R_i and R_j handling)
- All numerical gradient tests should pass
