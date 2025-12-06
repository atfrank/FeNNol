# OpenMM Multi-Pass GB Forces - Implementation Status

## Summary

Implemented OpenMM's multi-pass architecture for GB Born radius derivative forces to fix the 25× error in heteroatomic force calculations.

## Implementation Completed

### 1. ✅ CUDA Kernels (`gb_born_radii_forces.cu`)

**Added `reduce_born_force` kernel** (lines 269-306):
- Converts ∂E/∂R → ∂E/∂ψ
- Formula: `∂E/∂ψᵢ = (∂E/∂Rᵢ) × Rᵢ² × obcChain`
- Matches OpenMM's `reduceBornForce`

**Added `apply_born_forces_tiled` kernel** (lines 534-657):
- Applies Born forces using ∂E/∂ψ
- Formula: `F = -(∂E/∂ψᵢ) × (∂ψᵢ/∂r) × direction`
- Much simpler than previous approach
- No need to handle R_i vs R_j separately

### 2. ✅ Host Wrapper Functions (`gb_born_radii_forces.cu`)

**Added `reduce_born_force_host`** (lines 765-784):
- Host wrapper for reduce_born_force kernel

**Added `apply_born_forces_host`** (lines 789-809):
- Host wrapper for apply_born_forces_tiled kernel

### 3. ✅ Header Declarations (`implicit_solvent.cuh`)

Added function declarations (lines 144-172)

## Still TODO

### 4. ⏳ Python Bindings (`bindings.cpp`)

Need to add two new Python-exposed functions:
```cpp
m.def("reduce_born_force", [](
    py::array_t<double> dE_dR,
    py::array_t<double> born_radii,
    py::array_t<double> intrinsic_radii,
    py::array_t<double> b_params,
    py::array_t<double> c_params,
    py::array_t<double> psi_sum
) {
    // Allocate output
    // Call reduce_born_force_host
    // Return dE_dpsi
});

m.def("apply_born_forces", [](
    py::array_t<double> coords,
    py::array_t<double> intrinsic_radii,
    py::array_t<double> dE_dpsi,
    double cutoff
) {
    // Allocate output
    // Call apply_born_forces_host
    // Return born_forces
});
```

### 5. ⏳ Update `gb_compute_forces_complete`

Modify the existing Python wrapper in `generalized_born.py` to use multi-pass:

```python
def gb_compute_forces_complete(coords, charges, born_radii, radii, b_params, c_params, psi_sum, dielectric, cutoff):
    # Step 1: Compute direct forces (frozen R)
    energy, direct_forces = fennol_cuda.gb_compute_energy_forces(
        coords, charges, born_radii, dielectric, cutoff
    )

    # Step 2: Compute ∂E/∂R (already exists)
    dE_dR = compute_dE_dR_python(coords, charges, born_radii, radii, dielectric, cutoff)

    # Step 3: Convert ∂E/∂R → ∂E/∂ψ (NEW)
    dE_dpsi = fennol_cuda.reduce_born_force(
        dE_dR, born_radii, radii, b_params, c_params, psi_sum
    )

    # Step 4: Apply Born forces (NEW)
    born_forces = fennol_cuda.apply_born_forces(
        coords, radii, dE_dpsi, cutoff
    )

    # Step 5: Combine
    total_forces = direct_forces + born_forces

    return energy, total_forces
```

### 6. ⏳ Testing

**Test sequence:**
1. Rebuild CUDA extension
2. Test `reduce_born_force` in isolation
3. Test `apply_born_forces` in isolation
4. Test complete pipeline on:
   - 2-atom identical (should still work perfectly)
   - Single water molecule (should now work!)
   - Water dimer (should now work!)
   - Full gradient test

## Key Advantages of Multi-Pass Approach

1. **Matches OpenMM exactly** - can validate step-by-step
2. **No R_i vs R_j confusion** - handled naturally through ∂E/∂R calculation
3. **Simpler kernels** - each does one thing
4. **Easier to debug** - can inspect intermediate values (∂E/∂R, ∂E/∂ψ)
5. **More modular** - can reuse components

## Expected Outcome

After completing the Python bindings and testing:
- ✅ 2-atom identical: Should still pass (0.00 error)
- ✅ Single water: Should now pass (~0.01 error)
- ✅ Water dimer: Should now pass (~0.01 error)
- ✅ All numerical gradient tests: Should pass

## Next Steps

1. Add Python bindings to `bindings.cpp`
2. Rebuild CUDA extension
3. Run all tests
4. Document results
